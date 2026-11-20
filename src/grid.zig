//! Cylindrical grid tables, the stellarator-symmetry work list and the mirror
//! fill (spec sections 3.1 and 3.4).

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Params = struct {
    ir: usize,
    jz: usize,
    kp: usize,
    nfp: i32,
    rmin: f64,
    rmax: f64,
    zmin: f64,
    zmax: f64,
    sym: bool,
    legacy_fp32_trig: bool,
};

pub const Row = struct { k: u32, j: u32 };

pub const Grid = struct {
    p: Params,
    r: []f64,
    z: []f64,
    cosphi: []f64,
    sinphi: []f64,
    /// (plane, row) pairs that are evaluated directly.
    rows: []Row,

    pub fn points(g: *const Grid) usize {
        return g.p.ir * g.p.jz * g.p.kp;
    }

    /// Fill the points not evaluated directly. No-op without symmetry.
    pub fn mirror(g: *const Grid, br: []f64, bp: []f64, bz: []f64) void {
        if (!g.p.sym) return;
        const ir = g.p.ir;
        const jz = g.p.jz;
        const kp = g.p.kp;
        const h = kp / 2;
        const m = jz / 2 + jz % 2;
        const plane = ir * jz;

        const inPlane = struct {
            fn f(k: usize, m_: usize, jz_: usize, ir_: usize, plane_: usize, a: []f64, b: []f64, c: []f64) void {
                for (0..m_) |j| {
                    const src = k * plane_ + j * ir_;
                    const dst = k * plane_ + (jz_ - 1 - j) * ir_;
                    for (0..ir_) |i| {
                        // For odd jz the middle row maps to itself and legacy
                        // negates br in place; this does the same.
                        a[dst + i] = -a[src + i];
                        b[dst + i] = b[src + i];
                        c[dst + i] = c[src + i];
                    }
                }
            }
        }.f;

        inPlane(0, m, jz, ir, plane, br, bp, bz);
        var k: usize = 1;
        while (k < h + kp % 2) : (k += 1) {
            for (0..jz) |j| {
                const src = k * plane + j * ir;
                const dst = (kp - k) * plane + (jz - 1 - j) * ir;
                for (0..ir) |i| {
                    br[dst + i] = -br[src + i];
                    bp[dst + i] = bp[src + i];
                    bz[dst + i] = bz[src + i];
                }
            }
        }
        if (kp % 2 == 0) inPlane(h, m, jz, ir, plane, br, bp, bz);
    }
};

pub fn build(arena: Allocator, params: Params) Allocator.Error!Grid {
    var p = params;
    if (p.sym) {
        // Legacy symmetrises the Z bounds; the larger magnitude wins.
        if (@abs(p.zmax) > @abs(p.zmin)) {
            p.zmax = @abs(p.zmax);
            p.zmin = -p.zmax;
        } else {
            p.zmin = -@abs(p.zmin);
            p.zmax = -p.zmin;
        }
    }
    const r = try arena.alloc(f64, p.ir);
    const z = try arena.alloc(f64, p.jz);
    const cosphi = try arena.alloc(f64, p.kp);
    const sinphi = try arena.alloc(f64, p.kp);
    const delr = (p.rmax - p.rmin) / @as(f64, @floatFromInt(p.ir - 1));
    const delz = (p.zmax - p.zmin) / @as(f64, @floatFromInt(p.jz - 1));
    const delp = (2.0 * std.math.pi) / @as(f64, @floatFromInt(@as(i64, p.nfp) * @as(i64, @intCast(p.kp))));
    for (r, 0..) |*v, i| v.* = p.rmin + @as(f64, @floatFromInt(i)) * delr;
    for (z, 0..) |*v, j| v.* = p.zmin + @as(f64, @floatFromInt(j)) * delz;
    for (0..p.kp) |k| {
        const phi = @as(f64, @floatFromInt(k)) * delp;
        var c = @cos(phi);
        var s = @sin(phi);
        if (p.legacy_fp32_trig) {
            // Legacy stores cos/sin in default-kind REAL (spec section 1.2).
            c = @as(f32, @floatCast(c));
            s = @as(f32, @floatCast(s));
        }
        cosphi[k] = c;
        sinphi[k] = s;
    }

    var rows: std.ArrayList(Row) = .empty;
    if (!p.sym) {
        for (0..p.kp) |k| for (0..p.jz) |j| try rows.append(arena, .{ .k = @intCast(k), .j = @intCast(j) });
    } else {
        const h = p.kp / 2;
        const m = p.jz / 2 + p.jz % 2;
        for (0..m) |j| try rows.append(arena, .{ .k = 0, .j = @intCast(j) });
        var k: usize = 1;
        while (k < h + p.kp % 2) : (k += 1)
            for (0..p.jz) |j| try rows.append(arena, .{ .k = @intCast(k), .j = @intCast(j) });
        if (p.kp % 2 == 0)
            for (0..m) |j| try rows.append(arena, .{ .k = @intCast(h), .j = @intCast(j) });
    }
    return .{ .p = p, .r = r, .z = z, .cosphi = cosphi, .sinphi = sinphi, .rows = rows.items };
}

test "symmetry work list covers each point exactly once after mirroring" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const cases = [_][2]usize{ .{ 1, 2 }, .{ 2, 3 }, .{ 5, 6 }, .{ 6, 7 }, .{ 4, 4 }, .{ 7, 5 } };
    for (cases) |c| {
        const g = try build(a, .{ .ir = 3, .jz = c[1], .kp = c[0], .nfp = 3, .rmin = 1, .rmax = 2, .zmin = -0.3, .zmax = 0.5, .sym = true, .legacy_fp32_trig = false });
        try std.testing.expectEqual(@as(f64, -0.5), g.p.zmin);
        const n = g.points();
        const br = try a.alloc(f64, n);
        const bp = try a.alloc(f64, n);
        const bz = try a.alloc(f64, n);
        @memset(bp, 0);
        for (g.rows) |row| {
            for (0..3) |i| bp[(row.k * c[1] + row.j) * 3 + i] = 1;
        }
        @memset(br, 0);
        @memset(bz, 0);
        g.mirror(br, bp, bz);
        for (bp) |v| try std.testing.expectEqual(@as(f64, 1), v);
    }
}
