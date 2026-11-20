//! Reference fp64 Biot-Savart solver (spec sections 3.2, 3.3 and 4.4).
//!
//! Evaluates the Hanson-Hirshman straight-segment formula with the legacy
//! floor guards, in the legacy summation order: segments within a coil, then
//! the coil sum scaled by its current, then coils within the group.

const std = @import("std");
const coils_mod = @import("coils.zig");
const grid_mod = @import("grid.zig");
const CoilSet = coils_mod.CoilSet;
const Grid = grid_mod.Grid;

const V = 4;
const Vec = @Vector(V, f64);

inline fn splat(v: f64) Vec {
    return @splat(v);
}

/// Field of coil group `group` on one grid row, written as cylindrical
/// components into `br`, `bp`, `bz` (each `grid.p.ir` long).
///
/// Returns the number of points on the row for which a legacy floor was
/// engaged for at least one segment, i.e. `|R|^2 < eps_c` at a vertex or
/// `S^2 - |d|^2 < eps_c`: the point lies on a filament to rounding error.
pub fn evalRow(cs: *const CoilSet, group: usize, grid: *const Grid, k: usize, j: usize, br: []f64, bp: []f64, bz: []f64) usize {
    const ir = grid.p.ir;
    const c = grid.cosphi[k];
    const s = grid.sinphi[k];
    const pz = splat(grid.z[j]);
    const cv = splat(c);
    const sv = splat(s);
    const group_coils = cs.coils[cs.group_start[group]..cs.group_start[group + 1]];

    var floored: usize = 0;
    var i: usize = 0;
    while (i < ir) : (i += V) {
        var rbuf: [V]f64 = undefined;
        for (0..V) |l| rbuf[l] = grid.r[@min(i + l, ir - 1)];
        const rv: Vec = rbuf;
        const px = rv * cv;
        const py = rv * sv;

        var bx = splat(0);
        var by = splat(0);
        var bzc = splat(0);
        // Smallest (raw value - floor) seen; negative means a floor was used.
        var margin = splat(std.math.inf(f64));
        for (group_coils) |coil| {
            const eps = splat(coil.eps);
            var n: usize = coil.first;
            const last = coil.first + coil.nvert - 1;
            var rax = px - splat(cs.x[n]);
            var ray = py - splat(cs.y[n]);
            var raz = pz - splat(cs.z[n]);
            const ra2 = rax * rax + ray * ray + raz * raz;
            margin = @min(margin, ra2 - eps);
            var ra = @sqrt(@max(eps, ra2));
            var ax = splat(0);
            var ay = splat(0);
            var az = splat(0);
            while (n < last) : (n += 1) {
                const rbx = px - splat(cs.x[n + 1]);
                const rby = py - splat(cs.y[n + 1]);
                const rbz = pz - splat(cs.z[n + 1]);
                const rb2 = rbx * rbx + rby * rby + rbz * rbz;
                const rb = @sqrt(@max(eps, rb2));
                const sum = ra + rb;
                const den = sum * sum - splat(cs.dsq[n]);
                margin = @min(margin, @min(rb2, den) - eps);
                const fac = splat(2) * sum / (ra * rb * @max(eps, den));
                const dx = splat(cs.dx[n]);
                const dy = splat(cs.dy[n]);
                const dz = splat(cs.dz[n]);
                ax += (dy * raz - dz * ray) * fac;
                ay += (dz * rax - dx * raz) * fac;
                az += (dx * ray - dy * rax) * fac;
                rax = rbx;
                ray = rby;
                raz = rbz;
                ra = rb;
            }
            const w = splat(coil.current * 1.0e-7);
            bx += w * ax;
            by += w * ay;
            bzc += w * az;
        }
        const brv: [V]f64 = bx * cv + by * sv;
        const bpv: [V]f64 = by * cv - bx * sv;
        const bzv: [V]f64 = bzc;
        const mv: [V]f64 = margin;
        for (0..@min(V, ir - i)) |l| {
            floored += @intFromBool(mv[l] < 0);
            br[i + l] = brv[l];
            bp[i + l] = bpv[l];
            bz[i + l] = bzv[l];
        }
    }
    return floored;
}

const Job = struct {
    cs: *const CoilSet,
    grid: *const Grid,
    group: usize,
    br: []f64,
    bp: []f64,
    bz: []f64,
    next: std.atomic.Value(usize) = .init(0),
    floored: std.atomic.Value(usize) = .init(0),

    fn run(job: *Job) void {
        const ir = job.grid.p.ir;
        const jz = job.grid.p.jz;
        while (true) {
            const idx = job.next.fetchAdd(1, .monotonic);
            if (idx >= job.grid.rows.len) return;
            const row = job.grid.rows[idx];
            const off = (@as(usize, row.k) * jz + row.j) * ir;
            const n = evalRow(job.cs, job.group, job.grid, row.k, row.j, job.br[off..][0..ir], job.bp[off..][0..ir], job.bz[off..][0..ir]);
            if (n != 0) _ = job.floored.fetchAdd(n, .monotonic);
        }
    }
};

/// Evaluate one coil group on every directly computed row. Arrays are in
/// (k, j, i) order with the radial index fastest. The symmetry fill and the
/// S-mode scale are applied by the caller. Returns the number of directly
/// computed points that lie on a filament (see `evalRow`).
pub fn computeGroup(cs: *const CoilSet, group: usize, grid: *const Grid, br: []f64, bp: []f64, bz: []f64, threads: usize) usize {
    var job: Job = .{ .cs = cs, .grid = grid, .group = group, .br = br, .bp = bp, .bz = bz };
    var handles: [256]std.Thread = undefined;
    const want = @min(@max(threads, 1), handles.len + 1) - 1;
    var spawned: usize = 0;
    while (spawned < want) : (spawned += 1) {
        handles[spawned] = std.Thread.spawn(.{}, Job.run, .{&job}) catch break;
    }
    job.run();
    for (handles[0..spawned]) |h| h.join();
    return job.floored.load(.monotonic);
}

fn singleCoil(arena: std.mem.Allocator, text: []const u8) !CoilSet {
    var diag: coils_mod.Diag = .{};
    return coils_mod.parse(arena, text, &diag);
}

test "regular polygon on axis matches the closed form" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const radius = 1.0;
    const current = 1.0;
    for ([_]usize{ 3, 64, 10000 }) |nside| {
        var text: std.ArrayList(u8) = .empty;
        try text.appendSlice(a, "periods 1\nbegin filament\nmirror NIL\n");
        for (0..nside + 1) |n| {
            const t = 2.0 * std.math.pi * @as(f64, @floatFromInt(n % nside)) / @as(f64, @floatFromInt(nside));
            const line = if (n < nside)
                try std.fmt.allocPrint(a, "{e} {e} 0.0 {e}\n", .{ radius * @cos(t), radius * @sin(t), current })
            else
                try std.fmt.allocPrint(a, "{e} {e} 0.0 0.0 1 loop\n", .{ radius * @cos(t), radius * @sin(t) });
            try text.appendSlice(a, line);
        }
        try text.appendSlice(a, "end\n");
        const cs = try singleCoil(a, text.items);
        // Axis points: R = 0, a range of heights.
        const grid = try grid_mod.build(a, .{ .ir = 2, .jz = 5, .kp = 1, .nfp = 1, .rmin = 0, .rmax = 1.0e-3, .zmin = -0.7, .zmax = 1.3, .sym = false, .legacy_fp32_trig = false });
        var br: [2]f64 = undefined;
        var bp: [2]f64 = undefined;
        var bz: [2]f64 = undefined;
        const nf: f64 = @floatFromInt(nside);
        for (0..5) |j| {
            _ = evalRow(&cs, 0, &grid, 0, j, &br, &bp, &bz);
            const z = grid.z[j];
            const cosn = @cos(std.math.pi / nf);
            const exact = 1.0e-7 * current * nf * radius * radius * @sin(2.0 * std.math.pi / nf) /
                ((radius * radius * cosn * cosn + z * z) * @sqrt(radius * radius + z * z));
            try std.testing.expectApproxEqRel(exact, bz[0], 1.0e-12);
            try std.testing.expect(@abs(br[0]) < 1.0e-18);
        }
    }
}

test "finite straight segment matches the closed form" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    // Two-line coil: open segment on the z axis from z=-1 to z=2, 3 A.
    const cs = try singleCoil(a, "periods 1\nbegin filament\nmirror NIL\n0 0 -1 3.0\n0 0 2 0.0 1 wire\nend\n");
    const grid = try grid_mod.build(a, .{ .ir = 3, .jz = 4, .kp = 1, .nfp = 1, .rmin = 0.25, .rmax = 1.75, .zmin = -2.0, .zmax = 3.0, .sym = false, .legacy_fp32_trig = false });
    var br: [3]f64 = undefined;
    var bp: [3]f64 = undefined;
    var bz: [3]f64 = undefined;
    for (0..4) |j| {
        _ = evalRow(&cs, 0, &grid, 0, j, &br, &bp, &bz);
        for (0..3) |i| {
            const rho = grid.r[i];
            const z = grid.z[j];
            const sin2 = (2.0 - z) / @sqrt(rho * rho + (2.0 - z) * (2.0 - z));
            const sin1 = (-1.0 - z) / @sqrt(rho * rho + (-1.0 - z) * (-1.0 - z));
            const exact = 1.0e-7 * 3.0 / rho * (sin2 - sin1);
            try std.testing.expectApproxEqRel(exact, bp[i], 1.0e-13);
            try std.testing.expect(@abs(br[i]) < 1.0e-20 and @abs(bz[i]) < 1.0e-20);
        }
    }
}

test "points on a filament give finite values" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const cs = try singleCoil(a, "periods 1\nbegin filament\nmirror NIL\n1 0 -0.5 1e4\n1.25 0 -0.5 1e4\n1.25 0 0.5 1e4\n1 0 0.5 1e4\n1 0 -0.5 0 1 Rect\nend\n");
    const grid = try grid_mod.build(a, .{ .ir = 5, .jz = 5, .kp = 2, .nfp = 1, .rmin = 0.5, .rmax = 1.5, .zmin = -0.5, .zmax = 0.5, .sym = false, .legacy_fp32_trig = false });
    var br: [5]f64 = undefined;
    var bp: [5]f64 = undefined;
    var bz: [5]f64 = undefined;
    var on_wire: usize = 0;
    for (0..2) |k| for (0..5) |j| {
        on_wire += evalRow(&cs, 0, &grid, k, j, &br, &bp, &bz);
        for (0..5) |i| try std.testing.expect(std.math.isFinite(br[i]) and std.math.isFinite(bp[i]) and std.math.isFinite(bz[i]));
    };
    // phi = 0 plane: R = 1.0 and 1.25 at all five heights are on the rectangle.
    try std.testing.expectEqual(@as(usize, 10), on_wire);
}
