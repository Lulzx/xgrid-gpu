//! Compare two mgrid NetCDF files: header structure exactly, small variables
//! exactly, field arrays by the normalised tolerance of spec section 1.2.
//!
//! usage: compare_mgrid <reference.nc> <test.nc> [tolerance]
//! Exit status 0 if structure matches and every group is within tolerance.

const std = @import("std");
const Io = std.Io;
const netcdf3 = @import("netcdf3");

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const arena = init.arena.allocator();
    var buf: [4096]u8 = undefined;
    var fw = Io.File.stdout().writerStreaming(io, &buf);
    const out = &fw.interface;
    defer out.flush() catch {};

    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len < 3) {
        try out.writeAll("usage: compare_mgrid <reference.nc> <test.nc> [tolerance]\n");
        return 2;
    }
    const tol: f64 = if (argv.len > 3) try std.fmt.parseFloat(f64, argv[3]) else 1.0e-12;
    const cwd = Io.Dir.cwd();
    const ref = try netcdf3.parse(arena, try cwd.readFileAlloc(io, argv[1], arena, .unlimited));
    const tst = try netcdf3.parse(arena, try cwd.readFileAlloc(io, argv[2], arena, .unlimited));

    var bad: usize = 0;
    if (ref.version != tst.version) {
        try out.print("FAIL format version: {d} vs {d}\n", .{ ref.version, tst.version });
        bad += 1;
    }
    if (ref.ngatts != 0 or tst.ngatts != 0 or ref.numrecs != tst.numrecs) {
        try out.writeAll("FAIL global attributes or record count differ\n");
        bad += 1;
    }
    if (ref.dim_names.len != tst.dim_names.len) {
        try out.print("FAIL dimension count: {d} vs {d}\n", .{ ref.dim_names.len, tst.dim_names.len });
        bad += 1;
    } else for (ref.dim_names, ref.dim_sizes, tst.dim_names, tst.dim_sizes) |rn, rs, tn, ts| {
        if (!std.mem.eql(u8, rn, tn) or rs != ts) {
            try out.print("FAIL dimension: {s}={d} vs {s}={d}\n", .{ rn, rs, tn, ts });
            bad += 1;
        }
    }
    if (ref.vars.len != tst.vars.len) {
        try out.print("FAIL variable count: {d} vs {d}\n", .{ ref.vars.len, tst.vars.len });
        return 1;
    }
    for (ref.vars, tst.vars) |rv, tv| {
        if (!std.mem.eql(u8, rv.name, tv.name) or rv.nc_type != tv.nc_type or
            !std.mem.eql(u32, rv.dim_ids, tv.dim_ids) or rv.natts != tv.natts or rv.vsize != tv.vsize)
        {
            try out.print("FAIL variable definition: {s} vs {s}\n", .{ rv.name, tv.name });
            bad += 1;
        }
    }
    if (bad != 0) return 1;

    const header_identical = ref.header_len == tst.header_len and
        std.mem.eql(u8, ref.bytes[0..ref.header_len], tst.bytes[0..tst.header_len]);
    try out.print("header: structure identical; bytes {s} ({d} bytes)\n", .{ if (header_identical) "identical" else "DIFFER", ref.header_len });

    // Small variables must match bit for bit.
    var worst: f64 = 0;
    var field_vars: usize = 0;
    for (ref.vars, tst.vars) |*rv, *tv| {
        const is_field = rv.shape.len == 3;
        if (!is_field) {
            const n: usize = rv.vsize;
            if (!std.mem.eql(u8, ref.bytes[@intCast(rv.begin)..][0..n], tst.bytes[@intCast(tv.begin)..][0..n])) {
                try out.print("FAIL variable {s} differs\n", .{rv.name});
                bad += 1;
            }
            continue;
        }
        field_vars += 1;
    }

    // Fields: per group, max |difference| over all components divided by max |B|.
    const ngroups = field_vars / 3;
    for (0..ngroups) |g| {
        var names: [3][]const u8 = undefined;
        for ([_][]const u8{ "br", "bp", "bz" }, 0..) |c, i|
            names[i] = try std.fmt.allocPrint(arena, "{s}_{d:0>3}", .{ c, g + 1 });
        const r = [3]*const netcdf3.Var{ ref.find(names[0]).?, ref.find(names[1]).?, ref.find(names[2]).? };
        const t = [3]*const netcdf3.Var{ tst.find(names[0]).?, tst.find(names[1]).?, tst.find(names[2]).? };
        const n = r[0].count();
        var max_mag: f64 = 0;
        var max_diff: f64 = 0;
        var nonfinite: usize = 0;
        for (0..n) |i| {
            var mag2: f64 = 0;
            for (0..3) |c| {
                const a = ref.doubleAt(r[c], i);
                const b = tst.doubleAt(t[c], i);
                if (!std.math.isFinite(a) or !std.math.isFinite(b)) nonfinite += 1;
                mag2 += a * a;
                max_diff = @max(max_diff, @abs(a - b));
            }
            max_mag = @max(max_mag, @sqrt(mag2));
        }
        const eps = if (max_mag > 0) max_diff / max_mag else max_diff;
        worst = @max(worst, eps);
        if (nonfinite != 0 or !(eps <= tol)) {
            try out.print("FAIL group {d}: eps = {e:.3}, non-finite values = {d}\n", .{ g + 1, eps, nonfinite });
            bad += 1;
        }
    }
    try out.print("fields: {d} group(s), worst eps = {e:.3} (tolerance {e:.1})\n", .{ ngroups, worst, tol });
    return if (bad == 0) 0 else 1;
}
