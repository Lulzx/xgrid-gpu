//! xgrid-gpu: drop-in replacement for STELLOPT MAKEGRID (`xgrid`).
//! This build contains the fp64 CPU reference backend only (milestone M0).

const std = @import("std");
const Io = std.Io;
const coils_mod = @import("coils.zig");
const grid_mod = @import("grid.zig");
const cpu_ref = @import("cpu_ref.zig");
const netcdf3 = @import("netcdf3.zig");
const fortran_fmt = @import("fortran_fmt.zig");
const config = @import("config.zig");
const metal_host = @import("metal_host.zig");
const cuda_host = @import("cuda_host.zig");

const Gpu = union(enum) {
    none,
    metal: metal_host.Backend,
    cuda: cuda_host.Backend,

    fn deinit(g: *Gpu) void {
        switch (g.*) {
            .none => {},
            .metal => |*m| m.deinit(),
            .cuda => |*c| c.deinit(),
        }
    }

    fn gpuSeconds(g: *const Gpu) ?f64 {
        return switch (g.*) {
            .none => null,
            .metal => |m| m.gpu_seconds,
            .cuda => |c| c.gpu_seconds,
        };
    }
};

/// Mirror, scale, byte-swap and write one group's three arrays.
const WriteJob = struct {
    io: Io,
    file: Io.File,
    grid: *const grid_mod.Grid,
    layout: *const netcdf3.Layout,
    group: usize,
    scale: f64,
    arrays: [3][]f64,
    swap: []u8,
    seconds: f64 = 0,
    failed: ?anyerror = null,

    fn run(job: *WriteJob) void {
        const t0 = Io.Timestamp.now(job.io, .awake);
        defer job.seconds = seconds(t0, Io.Timestamp.now(job.io, .awake));
        const a = job.arrays;
        job.grid.mirror(a[0], a[1], a[2]);
        if (job.scale != 1.0) for (a) |arr| {
            for (arr) |*v| v.* *= job.scale;
        };
        for (a, 0..) |arr, comp| {
            netcdf3.toBigEndian(job.swap, arr);
            job.file.writePositionalAll(job.io, job.swap, job.layout.fieldOffset(job.group, comp)) catch |e| {
                job.failed = e;
                return;
            };
        }
    }
};

pub const version = "0.9.0";

const help_text =
    \\xgrid-gpu {s}: vacuum magnetic field of filamentary coils on a cylindrical grid.
    \\
    \\Usage:
    \\  xgrid-gpu --coils coils.<ext> --rmin R --rmax R --zmin Z --zmax Z --kp N
    \\            [--mode S|R] [--sym] [--ir N] [--jz N] [--threads N]
    \\            [--out FILE] [--extcur FILE] [--quiet] [--legacy-fp32-trig]
    \\            [--backend auto|cpu|metal|cuda] [--precision fp64|ds|fp32] [--device N]
    \\  xgrid-gpu <namelist file>          (&MGRID_NLI namelist)
    \\  xgrid-gpu ext mode sym rmin rmax zmin zmax kp [ir] [jz]
    \\  xgrid-gpu < script                 (the same ten values, one per line)
    \\
    \\  mode  S: field per unit group current (default)   R: raw coil currents
    \\  sym   Y: assume stellarator symmetry              N: compute every plane
    \\  ir and jz default to 121.
    \\
    \\Reads coils.<ext> and writes mgrid_<ext>.nc and extcur.<ext> in the current
    \\directory. --legacy-fp32-trig reproduces the single-precision cos/sin of
    \\the Fortran code; without it output differs from legacy at about 1e-7.
    \\
;

const prompts = [_][]const u8{
    " Enter extension of \"coils\" file     : ",
    " Scale (S) bfield to unit current/turn OR use raw (R) currents from coils file: ",
    " Assume stellarator symmetry (Y/N)?  : ",
    " Enter rmin (min radial grid dimension)  : ",
    " Enter rmax (max radial grid dimension)  : ",
    " Enter zmin (min vertical grid dimension): ",
    " Enter zmax (max vertical grid dimension): ",
    " Enter number of toroidal planes/period  : ",
    " Enter number of r (radial) mesh points  : ",
    " Enter number of z mesh points  : ",
};

const Exit = enum(u8) { ok = 0, input = 1, io = 2, backend = 3 };

fn seconds(from: Io.Timestamp, to: Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(from.durationTo(to).nanoseconds)) * 1.0e-9;
}

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const arena = init.arena.allocator();

    var out_buf: [4096]u8 = undefined;
    var err_buf: [1024]u8 = undefined;
    var out_file = Io.File.stdout().writerStreaming(io, &out_buf);
    var err_file = Io.File.stderr().writerStreaming(io, &err_buf);
    const out = &out_file.interface;
    const err = &err_file.interface;
    defer out.flush() catch {};
    defer err.flush() catch {};

    const code = run(io, arena, init.minimal.args, out, err) catch |e| blk: {
        try err.print("xgrid-gpu: {s}\n", .{@errorName(e)});
        break :blk Exit.io;
    };
    return @intFromEnum(code);
}

fn run(io: Io, arena: std.mem.Allocator, proc_args: std.process.Args, out: *Io.Writer, err: *Io.Writer) !Exit {
    const t_start = Io.Timestamp.now(io, .awake);
    const argv = try proc_args.toSlice(arena);
    const args = try arena.alloc([]const u8, argv.len -| 1);
    for (args, 0..) |*a, i| a.* = argv[i + 1];

    if (args.len == 1 and std.mem.eql(u8, args[0], "--version")) {
        try out.print("xgrid-gpu {s}\n", .{version});
        return .ok;
    }
    if (args.len == 1 and std.mem.eql(u8, args[0], "--help")) {
        try out.print(help_text, .{version});
        return .ok;
    }

    var diag: config.Diag = .{};
    const cwd = Io.Dir.cwd();
    const action = config.fromArgs(arena, args, &diag) catch |e| switch (e) {
        error.Usage => {
            try err.print("xgrid-gpu: {s}\n", .{diag.msg});
            return .input;
        },
        else => return e,
    };
    const cfg_or_err: config.Error!config.Config = switch (action) {
        .help => {
            try out.print(help_text, .{version});
            return .ok;
        },
        .version => unreachable,
        .run => |c| c,
        .interactive => |base| blk: {
            for (prompts) |p| try out.writeAll(p);
            try out.writeAll("\n");
            try out.flush();
            var rbuf: [4096]u8 = undefined;
            var reader = Io.File.stdin().readerStreaming(io, &rbuf);
            const text = try reader.interface.allocRemaining(arena, .limited(1 << 20));
            break :blk config.fromStdin(arena, base, text, &diag);
        },
        .namelist => |n| blk: {
            const text = cwd.readFileAlloc(io, n.path, arena, .limited(16 << 20)) catch |e| {
                try err.print("xgrid-gpu: cannot read namelist file '{s}': {s}\n", .{ n.path, @errorName(e) });
                return .input;
            };
            break :blk config.fromNamelist(arena, n.base, text, &diag);
        },
    };
    const cfg = cfg_or_err catch |e| switch (e) {
        error.Usage => {
            try err.print("xgrid-gpu: {s}\n", .{diag.msg});
            return .input;
        },
        else => return e,
    };
    config.validate(arena, &cfg, &diag) catch |e| switch (e) {
        error.Usage => {
            try err.print("xgrid-gpu: {s}\n", .{diag.msg});
            return .input;
        },
        else => return e,
    };

    // ---- parse coils
    const coils_path = cfg.coils_path orelse try std.fmt.allocPrint(arena, "coils.{s}", .{cfg.ext});
    const text = cwd.readFileAlloc(io, coils_path, arena, .unlimited) catch |e| {
        try err.print("xgrid-gpu: cannot read coils file '{s}': {s}\n", .{ coils_path, @errorName(e) });
        return .input;
    };
    var cdiag: coils_mod.Diag = .{};
    const cs = coils_mod.parse(arena, text, &cdiag) catch |e| switch (e) {
        error.InvalidCoils => {
            try err.print("xgrid-gpu: {s}:{d}: {s}\n", .{ coils_path, cdiag.line, cdiag.msg });
            return .input;
        },
        else => return e,
    };
    const ng = cs.group_names.len;
    if (cs.two_line_coils > 0)
        try err.print("xgrid-gpu: warning: {d} two-line coil(s) treated as single open straight segments\n", .{cs.two_line_coils});

    const grid = try grid_mod.build(arena, .{
        .ir = @intCast(cfg.ir),
        .jz = @intCast(cfg.jz),
        .kp = @intCast(cfg.kp),
        .nfp = cs.nfp,
        .rmin = cfg.rmin,
        .rmax = cfg.rmax,
        .zmin = cfg.zmin,
        .zmax = cfg.zmax,
        .sym = cfg.sym,
        .legacy_fp32_trig = cfg.legacy_fp32_trig,
    });

    const layout = netcdf3.build(arena, .{
        .ir = @intCast(cfg.ir),
        .jz = @intCast(cfg.jz),
        .kp = @intCast(cfg.kp),
        .nfp = cs.nfp,
        .rmin = grid.p.rmin,
        .zmin = grid.p.zmin,
        .rmax = grid.p.rmax,
        .zmax = grid.p.zmax,
        .mode = cfg.mode,
        .group_names = cs.group_names,
        .group_current = cs.group_current,
    }) catch |e| switch (e) {
        error.TooManyGroups => {
            try err.print("xgrid-gpu: {d} coil groups; at most 999 are supported\n", .{ng});
            return .input;
        },
        error.FieldTooLarge => {
            try err.print("xgrid-gpu: grid exceeds the NetCDF classic per-variable limit\n", .{});
            return .input;
        },
        else => return e,
    };

    const threads = if (cfg.threads != 0) cfg.threads else std.Thread.getCpuCount() catch 1;
    const npts = grid.points();
    const computed_pts = grid.rows.len * grid.p.ir;
    if (!cfg.quiet) {
        try out.print(" coils file     : {s} ({d} groups, {d} coils, {d} segments, nfp = {d})\n", .{ coils_path, ng, cs.coils.len, cs.nseg, cs.nfp });
        try out.print(" grid           : ir = {d}, jz = {d}, kp = {d}; R = [{d}, {d}], Z = [{d}, {d}]\n", .{ cfg.ir, cfg.jz, cfg.kp, grid.p.rmin, grid.p.rmax, grid.p.zmin, grid.p.zmax });
        try out.print(" mode           : {c}, stellarator symmetry {s}{s}\n", .{ cfg.mode, if (cfg.sym) "on" else "off", if (cfg.legacy_fp32_trig) ", legacy single-precision trig" else "" });
    }

    // ---- backend selection (spec section 2.1).
    //   explicit --backend: that backend, or exit 3 if it cannot start.
    //   auto + fp64 (or no precision): CUDA if a device loads and passes its
    //     self-test, otherwise the CPU. Metal is never chosen for fp64: Apple
    //     GPUs have no fp64, and on the hardware measured the CPU beats
    //     Metal double-single (spec section 7.5).
    //   auto + ds/fp32: Metal on macOS, CUDA elsewhere; if that GPU cannot
    //     start, fall back to the CPU in fp64, which is more accurate.
    const Choice = enum { cpu, metal, cuda };
    const want: Choice = switch (cfg.backend) {
        .cpu => .cpu,
        .metal => .metal,
        .cuda => .cuda,
        .auto => switch (cfg.precision) {
            .default, .fp64 => if (cuda_host.available) .cuda else .cpu,
            .ds, .fp32 => if (metal_host.available) .metal else .cuda,
        },
    };
    const forced = cfg.backend != .auto;
    var gpu: Gpu = .none;
    defer gpu.deinit();
    switch (want) {
        .cpu => {},
        .metal => {
            const prec: metal_host.Precision = if (cfg.precision == .fp32) .fp32 else .ds;
            var merr: [512]u8 = @splat(0);
            if (metal_host.selfTest(arena, prec, cfg.device, &merr)) |st| {
                if (st.finite and st.eps <= metal_host.selfTestTolerance(prec)) {
                    if (metal_host.init(arena, &cs, &grid, prec, cfg.device, &merr)) |m| {
                        gpu = .{ .metal = m };
                        if (!cfg.quiet) try out.print(" backend        : metal {s} on {s} (self-test difference {e:.1})\n", .{ @tagName(prec), gpu.metal.deviceName(), st.eps });
                    } else |_| {}
                } else {
                    _ = std.fmt.bufPrint(&merr, "self-test failed ({s}: difference {e:.2} from the CPU reference, limit {e:.0})\x00", .{ @tagName(prec), st.eps, metal_host.selfTestTolerance(prec) }) catch {};
                }
            } else |_| {}
            if (gpu == .none) {
                try err.print("xgrid-gpu: Metal backend unavailable: {s}\n", .{std.mem.sliceTo(&merr, 0)});
                if (forced) return .backend;
                try err.print("xgrid-gpu: falling back to the CPU backend (fp64)\n", .{});
            }
        },
        .cuda => {
            const prec: cuda_host.Precision = switch (cfg.precision) {
                .ds => .ds,
                .fp32 => .fp32,
                else => .fp64,
            };
            var cerr: [1024]u8 = @splat(0);
            if (cuda_host.selfTest(arena, prec, cfg.device, &cerr)) |st| {
                if (st.finite and st.eps <= cuda_host.selfTestTolerance(prec)) {
                    if (cuda_host.init(arena, &cs, &grid, prec, cfg.device, &cerr)) |c| {
                        gpu = .{ .cuda = c };
                        if (!cfg.quiet) try out.print(" backend        : cuda {s} on {s} (self-test difference {e:.1})\n", .{ @tagName(prec), gpu.cuda.deviceName(), st.eps });
                    } else |_| {}
                } else {
                    _ = std.fmt.bufPrint(&cerr, "self-test failed ({s}: difference {e:.2} from the CPU reference, limit {e:.0})\x00", .{ @tagName(prec), st.eps, cuda_host.selfTestTolerance(prec) }) catch {};
                }
            } else |_| {}
            if (gpu == .none) {
                // In auto mode with no NVIDIA driver at all, fall back silently.
                const msg = std.mem.sliceTo(&cerr, 0);
                const no_driver = std.mem.startsWith(u8, msg, "cannot load the CUDA driver") or std.mem.startsWith(u8, msg, "the CUDA backend is not built");
                if (forced or !no_driver or cfg.precision == .ds or cfg.precision == .fp32)
                    try err.print("xgrid-gpu: CUDA backend unavailable: {s}\n", .{msg});
                if (forced) return .backend;
                if (cfg.precision == .ds or cfg.precision == .fp32)
                    try err.print("xgrid-gpu: falling back to the CPU backend (fp64)\n", .{});
            }
        },
    }
    try err.flush();
    if (gpu == .none and !cfg.quiet) try out.print(" backend        : cpu fp64, {d} thread(s)\n", .{threads});
    try out.flush();

    // ---- compute and stream to a temporary file. Group g is mirrored,
    // scaled, byte-swapped and written on a second thread while group g + 1
    // is computed (spec section 4.2), so two groups' arrays are live.
    const out_path = cfg.out_path orelse try std.fmt.allocPrint(arena, "mgrid_{s}.nc", .{cfg.ext});
    const tmp_path = try std.fmt.allocPrint(arena, "{s}.tmp", .{out_path});
    const extcur_path = cfg.extcur_path orelse try std.fmt.allocPrint(arena, "extcur.{s}", .{cfg.ext});

    const t_parsed = Io.Timestamp.now(io, .awake);
    var t_compute: f64 = 0;
    var t_write: f64 = 0;
    var on_filament: usize = 0;

    const file = cwd.createFile(io, tmp_path, .{}) catch |e| {
        try err.print("xgrid-gpu: cannot create '{s}': {s}\n", .{ tmp_path, @errorName(e) });
        return .io;
    };
    var committed = false;
    defer if (!committed) cwd.deleteFile(io, tmp_path) catch {};
    {
        defer file.close(io);
        file.writePositionalAll(io, layout.prefix, 0) catch |e| {
            try err.print("xgrid-gpu: cannot write '{s}': {s}\n", .{ tmp_path, @errorName(e) });
            return .io;
        };

        var bufs: [2][3][]f64 = undefined;
        for (&bufs) |*set| for (set) |*arr| {
            arr.* = try arena.alloc(f64, npts);
        };
        var jobs: [2]WriteJob = undefined;
        var pending: ?std.Thread = null;
        var pending_job: ?*WriteJob = null;
        var failed = false;

        for (0..ng) |g| {
            const set = &bufs[g % 2];
            const t0 = Io.Timestamp.now(io, .awake);
            const gpu_result: ?(anyerror!usize) = switch (gpu) {
                .none => null,
                .metal => |*m| m.computeGroup(g, set[0], set[1], set[2]),
                .cuda => |*c| c.computeGroup(g, set[0], set[1], set[2]),
            };
            const n_on = if (gpu_result) |r| r catch blk: {
                try err.print("xgrid-gpu: {s}\n", .{switch (gpu) {
                    .metal => |*m| m.message(),
                    .cuda => |*c| c.message(),
                    .none => unreachable,
                }});
                if (forced) {
                    failed = true;
                    break;
                }
                // auto: finish the run on the CPU, starting with this group.
                try err.print("xgrid-gpu: falling back to the CPU backend (fp64) from group {d}\n", .{g + 1});
                try err.flush();
                gpu.deinit();
                gpu = .none;
                break :blk cpu_ref.computeGroup(&cs, g, &grid, set[0], set[1], set[2], threads);
            } else cpu_ref.computeGroup(&cs, g, &grid, set[0], set[1], set[2], threads);
            on_filament += n_on;
            t_compute += seconds(t0, Io.Timestamp.now(io, .awake));

            // Wait for the previous group's write before reusing its buffers.
            if (pending) |th| {
                th.join();
                pending = null;
                t_write += pending_job.?.seconds;
                if (pending_job.?.failed) |e| {
                    try err.print("xgrid-gpu: cannot write '{s}': {s}\n", .{ tmp_path, @errorName(e) });
                    return .io;
                }
            }
            const ig = cs.group_current[g];
            if (cfg.mode == 'S' and @abs(ig) <= 1.0e-100)
                try err.print("xgrid-gpu: warning: group {d} ({s}) has zero nominal current; fields scaled by 1e100 as legacy does\n", .{ g + 1, cs.group_names[g] });
            const job = &jobs[g % 2];
            job.* = .{
                .io = io,
                .file = file,
                .grid = &grid,
                .layout = &layout,
                .group = g,
                .scale = if (cfg.mode != 'S') 1.0 else if (@abs(ig) <= 1.0e-100) 1.0e100 else 1.0 / ig,
                .arrays = set.*,
                .swap = undefined,
            };
            job.swap = try arena.alloc(u8, npts * 8);
            if (std.Thread.spawn(.{}, WriteJob.run, .{job})) |th| {
                pending = th;
                pending_job = job;
            } else |_| {
                job.run();
                t_write += job.seconds;
                if (job.failed) |e| {
                    try err.print("xgrid-gpu: cannot write '{s}': {s}\n", .{ tmp_path, @errorName(e) });
                    return .io;
                }
            }
        }
        if (pending) |th| {
            th.join();
            t_write += pending_job.?.seconds;
            if (pending_job.?.failed) |e| {
                try err.print("xgrid-gpu: cannot write '{s}': {s}\n", .{ tmp_path, @errorName(e) });
                return .io;
            }
        }
        if (failed) return .backend;
    }
    cwd.rename(tmp_path, cwd, out_path, io) catch |e| {
        try err.print("xgrid-gpu: cannot rename '{s}' to '{s}': {s}\n", .{ tmp_path, out_path, @errorName(e) });
        return .io;
    };
    committed = true;

    // ---- extcur file
    var ext_text: std.ArrayList(u8) = .empty;
    for (cs.group_current, 0..) |cur, g| {
        const idx = g + 1;
        const num = fortran_fmt.e22_14(cur);
        const head = if (idx < 10)
            try std.fmt.allocPrint(arena, "EXTCUR({d})  = ", .{idx})
        else
            try std.fmt.allocPrint(arena, "EXTCUR({d}) = ", .{idx});
        try ext_text.appendSlice(arena, head);
        try ext_text.appendSlice(arena, &num);
        if (cfg.mode == 'R') {
            try ext_text.append(arena, '/');
            try ext_text.appendSlice(arena, &num);
        }
        try ext_text.append(arena, '\n');
    }
    if (cfg.mode == 'R') try ext_text.appendSlice(arena,
        \\
        \\ THE BFIELDS HAVE BEEN STORED IN THE MGRID FILE IN RAW MODE.
        \\ THE USER MUST PROVIDE THE EXTCUR ARRAY VALUES FOR THE VMEC INPUT (INDATA) FILE.
        \\ CHOOSING EXTCUR=1 CORRESPONDS TO THE CURRENTS PRESENT IN THE COILS FILE.
        \\
    );
    cwd.writeFile(io, .{ .sub_path = extcur_path, .data = ext_text.items }) catch |e| {
        try err.print("xgrid-gpu: cannot write '{s}': {s}\n", .{ extcur_path, @errorName(e) });
        return .io;
    };
    if (on_filament != 0)
        try err.print("xgrid-gpu: warning: {d} computed (point, group) pair(s) lie on a filament; the field there is finite but not meaningful\n", .{on_filament});
    if (ng >= 100)
        try err.print("xgrid-gpu: note: {s} uses three-digit indices from group 100 (legacy prints '**')\n", .{extcur_path});

    const t_end = Io.Timestamp.now(io, .awake);
    if (!cfg.quiet) {
        const interactions = @as(f64, @floatFromInt(computed_pts)) * @as(f64, @floatFromInt(cs.nseg));
        try out.print(" wrote          : {s} ({d} bytes), {s}\n", .{ out_path, layout.total_bytes, extcur_path });
        try out.print(" points         : {d} computed of {d}; {e:.3} interactions, {e:.3} per second\n", .{ computed_pts, npts, interactions, if (t_compute > 0) interactions / t_compute else 0 });
        try out.print(" time (s)       : setup {d:.3}, compute {d:.3}, write {d:.3}, total {d:.3}\n", .{ seconds(t_start, t_parsed), t_compute, t_write, seconds(t_start, t_end) });
        if (gpu.gpuSeconds()) |gs| try out.print(" gpu            : {d:.3} s executing kernels, {e:.3} interactions per second\n", .{ gs, if (gs > 0) interactions / gs else 0 });
    }
    return .ok;
}

test {
    _ = coils_mod;
    _ = grid_mod;
    _ = cpu_ref;
    _ = netcdf3;
    _ = fortran_fmt;
    _ = config;
    _ = metal_host;
    _ = cuda_host;
}
