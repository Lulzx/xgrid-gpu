//! Apple Metal backend (spec sections 4.2 to 4.5). The Objective-C side is in
//! metal_bridge.m; the kernel source is embedded and compiled at startup.

const std = @import("std");
const builtin = @import("builtin");
const coils_mod = @import("coils.zig");
const grid_mod = @import("grid.zig");
const cpu_ref = @import("cpu_ref.zig");
const CoilSet = coils_mod.CoilSet;
const Grid = grid_mod.Grid;
const Allocator = std.mem.Allocator;

pub const available = builtin.os.tag == .macos;
pub const Precision = enum { ds, fp32 };

const kernel_source = if (available) @embedFile("biot_savart.metal") else "";

const c = if (available) struct {
    const Handle = opaque {};
    extern fn xg_metal_create(source: [*]const u8, source_len: usize, ds: c_int, device: c_uint, err: [*]u8, errlen: usize) ?*Handle;
    extern fn xg_metal_destroy(m: *Handle) void;
    extern fn xg_metal_device_name(m: *const Handle) [*:0]const u8;
    extern fn xg_metal_set_geometry(m: *Handle, x: [*]const f64, y: [*]const f64, z: [*]const f64, dx: [*]const f64, dy: [*]const f64, dz: [*]const f64, dsq: [*]const f64, nvert: u32, coil_first: [*]const u32, coil_nvert: [*]const u32, coil_w: [*]const f64, coil_eps: [*]const f64, ncoil: u32, err: [*]u8, errlen: usize) c_int;
    extern fn xg_metal_set_points(m: *Handle, px: [*]const f64, py: [*]const f64, pz: [*]const f64, npoints: u32, err: [*]u8, errlen: usize) c_int;
    extern fn xg_metal_compute(m: *Handle, c0: u32, c1: u32, nseg: u64, bx: [*]f64, by: [*]f64, bz: [*]f64, flags: [*]u8, gpu_seconds: *f64, err: [*]u8, errlen: usize) c_int;
} else struct {
    // Not macOS: `init` fails before any of these is reached.
    const Handle = opaque {};
    fn xg_metal_create(_: [*]const u8, _: usize, _: c_int, _: c_uint, _: [*]u8, _: usize) ?*Handle {
        return null;
    }
    fn xg_metal_destroy(_: *Handle) void {}
    fn xg_metal_device_name(_: *const Handle) [*:0]const u8 {
        return "none";
    }
    fn xg_metal_set_geometry(_: *Handle, _: [*]const f64, _: [*]const f64, _: [*]const f64, _: [*]const f64, _: [*]const f64, _: [*]const f64, _: [*]const f64, _: u32, _: [*]const u32, _: [*]const u32, _: [*]const f64, _: [*]const f64, _: u32, _: [*]u8, _: usize) c_int {
        return -1;
    }
    fn xg_metal_set_points(_: *Handle, _: [*]const f64, _: [*]const f64, _: [*]const f64, _: u32, _: [*]u8, _: usize) c_int {
        return -1;
    }
    fn xg_metal_compute(_: *Handle, _: u32, _: u32, _: u64, _: [*]f64, _: [*]f64, _: [*]f64, _: [*]u8, _: *f64, _: [*]u8, _: usize) c_int {
        return -1;
    }
};

pub const Error = error{MetalFailed} || Allocator.Error;

pub const Backend = struct {
    handle: *c.Handle,
    precision: Precision,
    cs: *const CoilSet,
    grid: *const Grid,
    bx: []f64,
    by: []f64,
    bz: []f64,
    flags: []u8,
    /// GPU execution time accumulated over all `computeGroup` calls.
    gpu_seconds: f64 = 0,
    err: [512]u8 = @splat(0),

    pub fn message(b: *const Backend) []const u8 {
        return std.mem.sliceTo(&b.err, 0);
    }

    pub fn deviceName(b: *const Backend) []const u8 {
        return std.mem.span(c.xg_metal_device_name(b.handle));
    }

    pub fn deinit(b: *Backend) void {
        c.xg_metal_destroy(b.handle);
    }

    /// Same contract as `cpu_ref.computeGroup`.
    pub fn computeGroup(b: *Backend, group: usize, br: []f64, bp: []f64, bz: []f64) Error!usize {
        const cs = b.cs;
        const grid = b.grid;
        const c0 = cs.group_start[group];
        const c1 = cs.group_start[group + 1];
        var nseg: u64 = 0;
        for (cs.coils[c0..c1]) |coil| nseg += coil.nvert - 1;
        if (c.xg_metal_compute(b.handle, c0, c1, nseg, b.bx.ptr, b.by.ptr, b.bz.ptr, b.flags.ptr, &b.gpu_seconds, &b.err, b.err.len) != 0)
            return error.MetalFailed;
        const ir = grid.p.ir;
        var floored: usize = 0;
        for (grid.rows, 0..) |row, r| {
            const cp = grid.cosphi[row.k];
            const sp = grid.sinphi[row.k];
            const dst = (@as(usize, row.k) * grid.p.jz + row.j) * ir;
            for (0..ir) |i| {
                const s = r * ir + i;
                br[dst + i] = b.bx[s] * cp + b.by[s] * sp;
                bp[dst + i] = b.by[s] * cp - b.bx[s] * sp;
                bz[dst + i] = b.bz[s];
                floored += b.flags[s];
            }
        }
        return floored;
    }
};

/// Create the backend for a coil set and grid. On failure `err_out` holds a
/// message.
pub fn init(arena: Allocator, cs: *const CoilSet, grid: *const Grid, precision: Precision, device: u32, err_out: *[512]u8) Error!Backend {
    if (!available) {
        _ = std.fmt.bufPrint(err_out, "the Metal backend is only available on macOS\x00", .{}) catch {};
        return error.MetalFailed;
    }
    const handle = c.xg_metal_create(kernel_source.ptr, kernel_source.len, @intFromBool(precision == .ds), device, err_out, err_out.len) orelse
        return error.MetalFailed;
    errdefer c.xg_metal_destroy(handle);

    const ncoil = cs.coils.len;
    const first = try arena.alloc(u32, ncoil);
    const nvert = try arena.alloc(u32, ncoil);
    const w = try arena.alloc(f64, ncoil);
    const eps = try arena.alloc(f64, ncoil);
    for (cs.coils, 0..) |coil, i| {
        first[i] = coil.first;
        nvert[i] = coil.nvert;
        w[i] = coil.current * 1.0e-7;
        eps[i] = coil.eps;
    }
    if (c.xg_metal_set_geometry(handle, cs.x.ptr, cs.y.ptr, cs.z.ptr, cs.dx.ptr, cs.dy.ptr, cs.dz.ptr, cs.dsq.ptr, @intCast(cs.x.len), first.ptr, nvert.ptr, w.ptr, eps.ptr, @intCast(ncoil), err_out, err_out.len) != 0)
        return error.MetalFailed;

    const ir = grid.p.ir;
    const n = grid.rows.len * ir;
    const px = try arena.alloc(f64, n);
    const py = try arena.alloc(f64, n);
    const pz = try arena.alloc(f64, n);
    for (grid.rows, 0..) |row, r| {
        for (0..ir) |i| {
            px[r * ir + i] = grid.r[i] * grid.cosphi[row.k];
            py[r * ir + i] = grid.r[i] * grid.sinphi[row.k];
            pz[r * ir + i] = grid.z[row.j];
        }
    }
    if (c.xg_metal_set_points(handle, px.ptr, py.ptr, pz.ptr, @intCast(n), err_out, err_out.len) != 0)
        return error.MetalFailed;

    return .{
        .handle = handle,
        .precision = precision,
        .cs = cs,
        .grid = grid,
        .bx = try arena.alloc(f64, n),
        .by = try arena.alloc(f64, n),
        .bz = try arena.alloc(f64, n),
        .flags = try arena.alloc(u8, n),
    };
}

pub const SelfTest = struct {
    /// Worst normalised difference from the CPU reference (spec section 1.2).
    eps: f64,
    finite: bool,
};

/// Run a small fixed problem on the GPU and compare with the CPU reference.
/// This is what catches a kernel compiled with fast-math, a broken `fma`, or
/// a struct layout mismatch: any of them leaves double-single at float
/// accuracy, far outside the tolerance.
pub fn selfTest(arena: Allocator, precision: Precision, device: u32, err_out: *[512]u8) Error!SelfTest {
    const text =
        \\periods 1
        \\begin filament
        \\mirror NIL
        \\ 1.30  0.10 -0.40  2.5E+04
        \\ 1.90  0.25 -0.35  2.5E+04
        \\ 2.05 -0.15  0.45  2.5E+04
        \\ 1.20 -0.30  0.50  2.5E+04
        \\ 0.95  0.05  0.05  2.5E+04
        \\ 1.30  0.10 -0.40  0.0 1 loop
        \\ 1.60  0.00 -0.90 -4.0E+03
        \\ 1.60  0.00  0.90  0.0 1 loop
        \\end
    ;
    var diag: coils_mod.Diag = .{};
    const cs = coils_mod.parse(arena, text, &diag) catch return error.MetalFailed;
    const grid = try grid_mod.build(arena, .{ .ir = 9, .jz = 7, .kp = 3, .nfp = 1, .rmin = 1.1, .rmax = 1.8, .zmin = -0.3, .zmax = 0.3, .sym = false, .legacy_fp32_trig = false });
    const n = grid.points();
    const ref = try arena.alloc(f64, 3 * n);
    const got = try arena.alloc(f64, 3 * n);
    _ = cpu_ref.computeGroup(&cs, 0, &grid, ref[0..n], ref[n .. 2 * n], ref[2 * n ..], 1);
    var backend = try init(arena, &cs, &grid, precision, device, err_out);
    defer backend.deinit();
    _ = backend.computeGroup(0, got[0..n], got[n .. 2 * n], got[2 * n ..]) catch |e| {
        err_out.* = backend.err;
        return e;
    };
    var max_mag: f64 = 0;
    var max_diff: f64 = 0;
    var finite = true;
    for (0..n) |i| {
        var mag2: f64 = 0;
        for (0..3) |comp| {
            const a = ref[comp * n + i];
            const b = got[comp * n + i];
            if (!std.math.isFinite(b)) finite = false;
            mag2 += a * a;
            max_diff = @max(max_diff, @abs(a - b));
        }
        max_mag = @max(max_mag, @sqrt(mag2));
    }
    return .{ .eps = max_diff / max_mag, .finite = finite };
}

/// Tolerance the self-test must meet.
pub fn selfTestTolerance(precision: Precision) f64 {
    return switch (precision) {
        .ds => 1.0e-11,
        .fp32 => 1.0e-3,
    };
}
