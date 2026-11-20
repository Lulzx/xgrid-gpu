//! CUDA backend (spec sections 4.2 to 4.5).
//!
//! The CUDA driver library is opened at run time with dlopen, so the binary
//! has no link-time CUDA dependency and runs unchanged on machines without
//! an NVIDIA GPU. The kernels are embedded as PTX (kernels/biot_savart.ptx)
//! and compiled by the driver at load time.
//!
//! The library path defaults to `libcuda.so.1`; the environment variable
//! XGRID_LIBCUDA overrides it (used by the tests to load a mock driver).

const std = @import("std");
const builtin = @import("builtin");
const coils_mod = @import("coils.zig");
const grid_mod = @import("grid.zig");
const cpu_ref = @import("cpu_ref.zig");
const CoilSet = coils_mod.CoilSet;
const Grid = grid_mod.Grid;
const Allocator = std.mem.Allocator;

pub const Precision = enum { fp64, ds, fp32 };

/// Built wherever dlopen is available with libc.
pub const available = builtin.link_libc and (builtin.os.tag == .linux or builtin.os.tag == .macos);

const ptx = @embedFile("biot_savart.ptx");

// ---- driver API subset ----------------------------------------------------

const CUresult = c_int;
const CUdevice = c_int;
const CUcontext = ?*anyopaque;
const CUmodule = ?*anyopaque;
const CUfunction = ?*anyopaque;
const CUstream = ?*anyopaque;
const CUdeviceptr = u64;
const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CUDA_ERROR_UNSUPPORTED_PTX_VERSION: c_int = 222;
const CUDA_ERROR_NO_BINARY_FOR_GPU: c_int = 209;

const Api = struct {
    cuInit: *const fn (c_uint) callconv(.c) CUresult,
    cuDriverGetVersion: *const fn (*c_int) callconv(.c) CUresult,
    cuDeviceGetCount: *const fn (*c_int) callconv(.c) CUresult,
    cuDeviceGet: *const fn (*CUdevice, c_int) callconv(.c) CUresult,
    cuDeviceGetName: *const fn ([*]u8, c_int, CUdevice) callconv(.c) CUresult,
    cuDeviceGetAttribute: *const fn (*c_int, c_int, CUdevice) callconv(.c) CUresult,
    cuCtxCreate_v2: *const fn (*CUcontext, c_uint, CUdevice) callconv(.c) CUresult,
    cuCtxDestroy_v2: *const fn (CUcontext) callconv(.c) CUresult,
    cuCtxSynchronize: *const fn () callconv(.c) CUresult,
    cuModuleLoadDataEx: *const fn (*CUmodule, *const anyopaque, c_uint, ?[*]c_int, ?[*]?*anyopaque) callconv(.c) CUresult,
    cuModuleUnload: *const fn (CUmodule) callconv(.c) CUresult,
    cuModuleGetFunction: *const fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.c) CUresult,
    cuMemAlloc_v2: *const fn (*CUdeviceptr, usize) callconv(.c) CUresult,
    cuMemFree_v2: *const fn (CUdeviceptr) callconv(.c) CUresult,
    cuMemcpyHtoD_v2: *const fn (CUdeviceptr, *const anyopaque, usize) callconv(.c) CUresult,
    cuMemcpyDtoH_v2: *const fn (*anyopaque, CUdeviceptr, usize) callconv(.c) CUresult,
    cuLaunchKernel: *const fn (CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, ?[*]?*anyopaque, ?[*]?*anyopaque) callconv(.c) CUresult,
    cuGetErrorString: *const fn (CUresult, *?[*:0]const u8) callconv(.c) CUresult,
};

// ---- data layouts (must match kernels/biot_savart.cu) ----------------------

const VertD = extern struct { x: f64, y: f64, z: f64, dx: f64, dy: f64, dz: f64, dsq: f64, pad: f64 = 0 };
const VertF = extern struct { x: [2]f32, y: [2]f32, z: [2]f32, dx: [2]f32, dy: [2]f32, dz: [2]f32, dsq: [2]f32, pad: [2]f32 = .{ 0, 0 } };
const CoilD = extern struct { first: u32, nvert: u32, w: f64, eps: f64 };
const CoilF = extern struct { first: u32, nvert: u32, w: [2]f32, eps: [2]f32 };
const PointD = extern struct { x: f64, y: f64, z: f64, pad: f64 = 0 };
const PointF = extern struct { x: [2]f32, y: [2]f32, z: [2]f32, pad: [2]f32 = .{ 0, 0 } };
const Out = extern struct { bx: f64, by: f64, bz: f64, flag: u32, pad: u32 };
const Params = extern struct { c0: u32, c1: u32, offset: u32, count: u32 };

comptime {
    std.debug.assert(@sizeOf(VertD) == 64 and @sizeOf(VertF) == 64);
    std.debug.assert(@sizeOf(CoilD) == 24 and @sizeOf(CoilF) == 24);
    std.debug.assert(@sizeOf(PointD) == 32 and @sizeOf(PointF) == 32);
    std.debug.assert(@sizeOf(Out) == 32 and @sizeOf(Params) == 16);
}

extern "c" fn clock_gettime(clk: std.c.clockid_t, tp: *std.c.timespec) c_int;

fn monotonic() f64 {
    var ts: std.c.timespec = undefined;
    _ = clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1.0e-9;
}

fn split(v: f64, ds: bool) [2]f32 {
    const hi: f32 = @floatCast(v);
    return .{ hi, if (ds) @floatCast(v - @as(f64, hi)) else 0 };
}

pub const Error = error{CudaFailed} || Allocator.Error;

pub const Backend = struct {
    lib: std.DynLib,
    api: Api,
    ctx: CUcontext,
    module: CUmodule,
    func: CUfunction,
    precision: Precision,
    cs: *const CoilSet,
    grid: *const Grid,
    d_verts: CUdeviceptr = 0,
    d_coils: CUdeviceptr = 0,
    d_points: CUdeviceptr = 0,
    d_out: CUdeviceptr = 0,
    npoints: u32 = 0,
    out: []Out,
    name: [256]u8 = @splat(0),
    gpu_seconds: f64 = 0,
    err: [1024]u8 = @splat(0),

    pub fn message(b: *const Backend) []const u8 {
        return std.mem.sliceTo(&b.err, 0);
    }

    pub fn deviceName(b: *const Backend) []const u8 {
        return std.mem.sliceTo(&b.name, 0);
    }

    pub fn deinit(b: *Backend) void {
        for ([_]CUdeviceptr{ b.d_verts, b.d_coils, b.d_points, b.d_out }) |p| {
            if (p != 0) _ = b.api.cuMemFree_v2(p);
        }
        if (b.module != null) _ = b.api.cuModuleUnload(b.module);
        if (b.ctx != null) _ = b.api.cuCtxDestroy_v2(b.ctx);
        b.lib.close();
    }

    fn check(b: *Backend, r: CUresult, what: []const u8) Error!void {
        if (r == 0) return;
        var s: ?[*:0]const u8 = null;
        _ = b.api.cuGetErrorString(r, &s);
        _ = std.fmt.bufPrint(&b.err, "{s} failed: CUDA error {d} ({s})\x00", .{ what, r, if (s) |p| std.mem.span(p) else "unknown" }) catch {};
        return error.CudaFailed;
    }

    fn upload(b: *Backend, dst: *CUdeviceptr, bytes: []const u8, what: []const u8) Error!void {
        if (dst.* != 0) _ = b.api.cuMemFree_v2(dst.*);
        dst.* = 0;
        try b.check(b.api.cuMemAlloc_v2(dst, @max(bytes.len, 16)), what);
        if (bytes.len != 0) try b.check(b.api.cuMemcpyHtoD_v2(dst.*, bytes.ptr, bytes.len), what);
    }

    /// Same contract as `cpu_ref.computeGroup`.
    pub fn computeGroup(b: *Backend, group: usize, br: []f64, bp: []f64, bz: []f64) Error!usize {
        const cs = b.cs;
        const grid = b.grid;
        const c0 = cs.group_start[group];
        const c1 = cs.group_start[group + 1];
        var nseg: u64 = 0;
        for (cs.coils[c0..c1]) |coil| nseg += coil.nvert - 1;
        const segs: f64 = @floatFromInt(@max(nseg, 1));

        // Time-bounded chunks (spec section 4.2), sized from the measured rate.
        const block: u32 = 128;
        const target = 0.15;
        var rate: f64 = 0;
        var offset: u32 = 0;
        while (offset < b.npoints) {
            var want: f64 = if (rate > 0) rate * target / segs else 2.0e7 / segs;
            want = @max(want, block);
            const count: u32 = @intFromFloat(@min(want, @as(f64, @floatFromInt(b.npoints - offset))));
            var params: Params = .{ .c0 = c0, .c1 = c1, .offset = offset, .count = count };
            var args = [_]?*anyopaque{ &b.d_verts, &b.d_coils, &b.d_points, &b.d_out, &params };
            const t0 = monotonic();
            try b.check(b.api.cuLaunchKernel(b.func, (count + block - 1) / block, 1, 1, block, 1, 1, 0, null, &args, null), "cuLaunchKernel");
            try b.check(b.api.cuCtxSynchronize(), "kernel execution");
            const dt = monotonic() - t0;
            b.gpu_seconds += dt;
            if (dt > 0) rate = @as(f64, @floatFromInt(count)) * segs / dt;
            offset += count;
        }
        try b.check(b.api.cuMemcpyDtoH_v2(b.out.ptr, b.d_out, b.out.len * @sizeOf(Out)), "cuMemcpyDtoH");

        const ir = grid.p.ir;
        var floored: usize = 0;
        for (grid.rows, 0..) |row, r| {
            const cp = grid.cosphi[row.k];
            const sp = grid.sinphi[row.k];
            const dst = (@as(usize, row.k) * grid.p.jz + row.j) * ir;
            for (0..ir) |i| {
                const o = b.out[r * ir + i];
                br[dst + i] = o.bx * cp + o.by * sp;
                bp[dst + i] = o.by * cp - o.bx * sp;
                bz[dst + i] = o.bz;
                floored += o.flag;
            }
        }
        return floored;
    }
};

fn libPath(buf: *[1024]u8) [:0]const u8 {
    if (std.c.getenv("XGRID_LIBCUDA")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s.len < buf.len) {
            @memcpy(buf[0..s.len], s);
            buf[s.len] = 0;
            return buf[0..s.len :0];
        }
    }
    return "libcuda.so.1";
}

fn setErr(err_out: *[1024]u8, comptime fmt: []const u8, args: anytype) void {
    @memset(err_out, 0);
    _ = std.fmt.bufPrint(err_out[0 .. err_out.len - 1], fmt, args) catch {};
}

pub fn init(arena: Allocator, cs: *const CoilSet, grid: *const Grid, precision: Precision, device: u32, err_out: *[1024]u8) Error!Backend {
    if (!available) {
        setErr(err_out, "the CUDA backend is not built for this platform", .{});
        return error.CudaFailed;
    }
    var pbuf: [1024]u8 = undefined;
    const path = libPath(&pbuf);
    var lib = std.DynLib.openZ(path.ptr) catch {
        setErr(err_out, "cannot load the CUDA driver library '{s}' (no NVIDIA driver installed?)", .{path});
        return error.CudaFailed;
    };
    var api: Api = undefined;
    const info = @typeInfo(Api).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(api, name) = lib.lookup(T, name) orelse {
            lib.close();
            setErr(err_out, "the CUDA driver library '{s}' has no symbol {s}", .{ path, name });
            return error.CudaFailed;
        };
    }

    var b: Backend = .{
        .lib = lib,
        .api = api,
        .ctx = null,
        .module = null,
        .func = null,
        .precision = precision,
        .cs = cs,
        .grid = grid,
        .out = &.{},
    };
    errdefer {
        err_out.* = b.err;
        b.deinit();
    }

    try b.check(api.cuInit(0), "cuInit");
    var count: c_int = 0;
    try b.check(api.cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (device >= count) {
        _ = std.fmt.bufPrint(&b.err, "CUDA device {d} requested but {d} device(s) present\x00", .{ device, count }) catch {};
        return error.CudaFailed;
    }
    var dev: CUdevice = 0;
    try b.check(api.cuDeviceGet(&dev, @intCast(device)), "cuDeviceGet");
    _ = api.cuDeviceGetName(&b.name, b.name.len - 1, dev);
    try b.check(api.cuCtxCreate_v2(&b.ctx, 0, dev), "cuCtxCreate");

    // Load the embedded PTX, collecting the JIT log for diagnostics.
    var log: [4096]u8 = @splat(0);
    var opts = [_]c_int{ CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
    var vals = [_]?*anyopaque{ &log, @ptrFromInt(log.len) };
    const ptx_z = try arena.dupeSentinel(u8, ptx, 0);
    const r = api.cuModuleLoadDataEx(&b.module, ptx_z.ptr, opts.len, &opts, &vals);
    if (r != 0) {
        var ver: c_int = 0;
        _ = api.cuDriverGetVersion(&ver);
        if (r == CUDA_ERROR_UNSUPPORTED_PTX_VERSION or r == CUDA_ERROR_NO_BINARY_FOR_GPU) {
            const jit = std.mem.trim(u8, std.mem.sliceTo(&log, 0), " \n");
            _ = std.fmt.bufPrint(&b.err, "CUDA driver {d}.{d} cannot load the embedded kernel (PTX ISA 5.0 for sm_60; needs a driver from CUDA 8.0 or later and a Pascal or newer GPU){s}{s}\x00", .{ @divTrunc(ver, 1000), @divTrunc(@mod(ver, 1000), 10), if (jit.len > 0) ": " else "", jit }) catch {};
            return error.CudaFailed;
        }
        b.check(r, "loading the kernel module") catch |e| {
            const n = std.mem.indexOfScalar(u8, &b.err, 0) orelse b.err.len - 1;
            _ = std.fmt.bufPrint(b.err[n..], ": {s}\x00", .{std.mem.sliceTo(&log, 0)}) catch {};
            return e;
        };
    }
    const fname: [:0]const u8 = switch (precision) {
        .fp64 => "biot_savart_fp64",
        .ds => "biot_savart_ds",
        .fp32 => "biot_savart_fp32",
    };
    try b.check(api.cuModuleGetFunction(&b.func, b.module, fname.ptr), "cuModuleGetFunction");

    // Geometry.
    const ds = precision == .ds;
    const nv = cs.x.len;
    const ncoil = cs.coils.len;
    if (precision == .fp64) {
        const v = try arena.alloc(VertD, nv);
        for (v, 0..) |*e, i| e.* = .{ .x = cs.x[i], .y = cs.y[i], .z = cs.z[i], .dx = cs.dx[i], .dy = cs.dy[i], .dz = cs.dz[i], .dsq = cs.dsq[i] };
        try b.upload(&b.d_verts, std.mem.sliceAsBytes(v), "uploading vertices");
        const c = try arena.alloc(CoilD, ncoil);
        for (c, cs.coils) |*e, coil| e.* = .{ .first = coil.first, .nvert = coil.nvert, .w = coil.current * 1.0e-7, .eps = coil.eps };
        try b.upload(&b.d_coils, std.mem.sliceAsBytes(c), "uploading coils");
    } else {
        const v = try arena.alloc(VertF, nv);
        for (v, 0..) |*e, i| e.* = .{ .x = split(cs.x[i], ds), .y = split(cs.y[i], ds), .z = split(cs.z[i], ds), .dx = split(cs.dx[i], ds), .dy = split(cs.dy[i], ds), .dz = split(cs.dz[i], ds), .dsq = split(cs.dsq[i], ds) };
        try b.upload(&b.d_verts, std.mem.sliceAsBytes(v), "uploading vertices");
        const c = try arena.alloc(CoilF, ncoil);
        for (c, cs.coils) |*e, coil| e.* = .{ .first = coil.first, .nvert = coil.nvert, .w = split(coil.current * 1.0e-7, ds), .eps = split(coil.eps, ds) };
        try b.upload(&b.d_coils, std.mem.sliceAsBytes(c), "uploading coils");
    }

    // Directly computed points, in work-list order.
    const ir = grid.p.ir;
    const n = grid.rows.len * ir;
    if (n > std.math.maxInt(u32)) {
        _ = std.fmt.bufPrint(&b.err, "too many grid points for one dispatch\x00", .{}) catch {};
        return error.CudaFailed;
    }
    if (precision == .fp64) {
        const p = try arena.alloc(PointD, n);
        for (grid.rows, 0..) |row, rr| for (0..ir) |i| {
            p[rr * ir + i] = .{ .x = grid.r[i] * grid.cosphi[row.k], .y = grid.r[i] * grid.sinphi[row.k], .z = grid.z[row.j] };
        };
        try b.upload(&b.d_points, std.mem.sliceAsBytes(p), "uploading points");
    } else {
        const p = try arena.alloc(PointF, n);
        for (grid.rows, 0..) |row, rr| for (0..ir) |i| {
            p[rr * ir + i] = .{ .x = split(grid.r[i] * grid.cosphi[row.k], ds), .y = split(grid.r[i] * grid.sinphi[row.k], ds), .z = split(grid.z[row.j], ds) };
        };
        try b.upload(&b.d_points, std.mem.sliceAsBytes(p), "uploading points");
    }
    try b.check(api.cuMemAlloc_v2(&b.d_out, @max(n, 1) * @sizeOf(Out)), "allocating output");
    b.npoints = @intCast(n);
    b.out = try arena.alloc(Out, n);
    return b;
}

pub const SelfTest = struct { eps: f64, finite: bool };

/// Same fixed problem as the Metal self-test, compared with the CPU reference.
pub fn selfTest(arena: Allocator, precision: Precision, device: u32, err_out: *[1024]u8) Error!SelfTest {
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
    const cs = coils_mod.parse(arena, text, &diag) catch return error.CudaFailed;
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
            const v = got[comp * n + i];
            if (!std.math.isFinite(v)) finite = false;
            mag2 += a * a;
            max_diff = @max(max_diff, @abs(a - v));
        }
        max_mag = @max(max_mag, @sqrt(mag2));
    }
    return .{ .eps = max_diff / max_mag, .finite = finite };
}

pub fn selfTestTolerance(precision: Precision) f64 {
    return switch (precision) {
        .fp64 => 1.0e-13,
        .ds => 1.0e-11,
        .fp32 => 1.0e-3,
    };
}
