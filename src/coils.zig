//! Parser for `coils.<ext>` filament files, reproducing the semantics of
//! STELLOPT `biotsavart.f` / `bsc_T.f` (spec section 2.4 and 2.5).

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Coil = struct {
    /// Output group index (order of first appearance of the group id).
    group: u32,
    /// Current of the coil's first vertex line; applies to every segment.
    current: f64,
    /// Index of the first stored vertex. A closed loop stores its first
    /// vertex again at the end, so there are always `nvert - 1` segments.
    first: u32,
    nvert: u32,
    /// Legacy floor: machine epsilon times the shortest squared segment length.
    eps: f64,
};

pub const CoilSet = struct {
    nfp: i32,
    group_ids: []i64,
    group_names: [][]const u8,
    /// Current of the largest-magnitude coil in each group (first on ties).
    group_current: []f64,
    /// Sorted by group; file order within a group.
    coils: []Coil,
    /// Coil index range per group: coils[group_start[g] .. group_start[g + 1]].
    group_start: []u32,
    x: []f64,
    y: []f64,
    z: []f64,
    /// Vector to the next stored vertex and its squared length. Entries at the
    /// last stored vertex of a coil are unused.
    dx: []f64,
    dy: []f64,
    dz: []f64,
    dsq: []f64,
    nseg: usize,
    two_line_coils: usize,
};

const start_marker = "** coils_dot_starts_below **";

pub const Diag = struct {
    line: usize = 0,
    msg: []const u8 = "",
};

pub const Error = error{InvalidCoils} || Allocator.Error;

fn fail(diag: *Diag, line: usize, msg: []const u8) Error {
    diag.* = .{ .line = line, .msg = msg };
    return error.InvalidCoils;
}

/// Parse a Fortran list-directed real: accepts `D`/`d` exponents.
pub fn parseReal(tok: []const u8) ?f64 {
    var buf: [64]u8 = undefined;
    if (tok.len == 0 or tok.len > buf.len) return null;
    for (tok, 0..) |c, i| buf[i] = if (c == 'd' or c == 'D') 'e' else c;
    return std.fmt.parseFloat(f64, buf[0..tok.len]) catch null;
}

const Vertex = struct { p: [3]f64, cur: f64 };

pub fn parse(arena: Allocator, text: []const u8, diag: *Diag) Error!CoilSet {
    var lines = std.mem.splitScalar(u8, text, '\n');
    var lineno: usize = 0;

    // Legacy: if a line contains the start marker, the file proper begins on
    // the line after it; otherwise at the top.
    if (std.mem.indexOf(u8, text, start_marker)) |pos| {
        var consumed: usize = 0;
        while (consumed <= pos) {
            const l = lines.next() orelse break;
            consumed += l.len + 1;
            lineno += 1;
        }
    }

    // First line: must contain "periods"; nfp is its second field.
    lineno += 1;
    const first_line = lines.next() orelse return fail(diag, lineno, "empty coils file");
    var nfp: i32 = 0;
    {
        if (std.mem.indexOf(u8, first_line, "periods") == null)
            return fail(diag, lineno, "first line must be 'periods <nfp>'");
        var it = std.mem.tokenizeAny(u8, first_line, " \t\r,");
        _ = it.next();
        const n = it.next() orelse return fail(diag, lineno, "missing number of field periods");
        nfp = std.fmt.parseInt(i32, n, 10) catch return fail(diag, lineno, "bad number of field periods");
        if (nfp < 1) return fail(diag, lineno, "number of field periods must be >= 1");
    }
    // Legacy skips the next two lines ("begin filament", "mirror NIL") unread.
    _ = lines.next();
    _ = lines.next();
    lineno += 2;

    var gids: std.ArrayList(i64) = .empty;
    var gnames: std.ArrayList([]const u8) = .empty;
    var coils: std.ArrayList(Coil) = .empty;
    var xs: std.ArrayList(f64) = .empty;
    var ys: std.ArrayList(f64) = .empty;
    var zs: std.ArrayList(f64) = .empty;
    var pending: std.ArrayList(Vertex) = .empty;
    var two_line: usize = 0;
    var nseg: usize = 0;

    while (lines.next()) |raw| {
        lineno += 1;
        if (raw.len >= 3 and std.mem.eql(u8, raw[0..3], "end")) break;
        var it = std.mem.tokenizeAny(u8, raw, " \t\r,");
        var tok: [6][]const u8 = undefined;
        var nt: usize = 0;
        while (nt < 6) : (nt += 1) tok[nt] = it.next() orelse break;
        if (nt == 0) continue; // blank line
        if (nt < 4) return fail(diag, lineno, "expected 'x y z current'");
        var v: Vertex = undefined;
        for (0..3) |c| v.p[c] = parseReal(tok[c]) orelse return fail(diag, lineno, "bad coordinate");
        v.cur = parseReal(tok[3]) orelse return fail(diag, lineno, "bad current");
        if (!std.math.isFinite(v.p[0]) or !std.math.isFinite(v.p[1]) or
            !std.math.isFinite(v.p[2]) or !std.math.isFinite(v.cur))
            return fail(diag, lineno, "non-finite value");
        try pending.append(arena, v);

        // A line with six fields (integer group id, name) terminates the coil.
        // As in legacy, a line whose fifth field is not an integer is an
        // ordinary vertex line with trailing text.
        if (nt < 6) continue;
        const gid = std.fmt.parseInt(i64, tok[4], 10) catch continue;
        // Legacy uses -1 as its "unused slot" marker, so a real group -1 is
        // silently merged or dropped there. Reject it instead.
        if (gid == -1) return fail(diag, lineno, "group id -1 is not supported");
        // The name is one list-directed item: a bare token or a quoted string.
        var gname = tok[5];
        if (gname[0] == '\'' or gname[0] == '"') {
            const start = @intFromPtr(gname.ptr) - @intFromPtr(raw.ptr);
            const close = std.mem.indexOfScalarPos(u8, raw, start + 1, gname[0]) orelse
                return fail(diag, lineno, "unterminated quote in group name");
            gname = raw[start + 1 .. close];
        }

        var g: usize = 0;
        while (g < gids.items.len and gids.items[g] != gid) g += 1;
        if (g == gids.items.len) {
            try gids.append(arena, gid);
            try gnames.append(arena, try arena.dupe(u8, gname));
        }

        const pts = pending.items;
        const current = pts[0].cur;
        const first: u32 = @intCast(xs.items.len);
        if (pts.len == 1) {
            return fail(diag, lineno, "one-line (circular) coils are not supported");
        } else if (pts.len == 2) {
            // One open straight segment between the two points.
            if (std.mem.eql(f64, &pts[0].p, &pts[1].p))
                return fail(diag, lineno, "two-line coil has zero length");
            for (pts) |p| {
                try xs.append(arena, p.p[0]);
                try ys.append(arena, p.p[1]);
                try zs.append(arena, p.p[2]);
            }
            two_line += 1;
        } else {
            // Drop the terminating vertex, remove exact consecutive duplicates
            // and a duplicated wrap-around vertex, then close the loop.
            var kept: usize = 0;
            for (pts[0 .. pts.len - 1]) |p| {
                if (kept > 0) {
                    const n = xs.items.len - 1;
                    if (xs.items[n] == p.p[0] and ys.items[n] == p.p[1] and zs.items[n] == p.p[2]) continue;
                }
                try xs.append(arena, p.p[0]);
                try ys.append(arena, p.p[1]);
                try zs.append(arena, p.p[2]);
                kept += 1;
            }
            if (kept > 1) {
                const n = xs.items.len - 1;
                if (xs.items[n] == xs.items[first] and ys.items[n] == ys.items[first] and zs.items[n] == zs.items[first]) {
                    _ = xs.pop();
                    _ = ys.pop();
                    _ = zs.pop();
                    kept -= 1;
                }
            }
            if (kept < 2) return fail(diag, lineno, "coil has fewer than two distinct vertices");
            try xs.append(arena, xs.items[first]);
            try ys.append(arena, ys.items[first]);
            try zs.append(arena, zs.items[first]);
        }
        const nvert: u32 = @intCast(xs.items.len - first);
        nseg += nvert - 1;
        try coils.append(arena, .{ .group = @intCast(g), .current = current, .first = first, .nvert = nvert, .eps = 0 });
        pending.clearRetainingCapacity();
    }
    if (pending.items.len != 0) return fail(diag, lineno, "last coil has no terminating 'x y z I group name' line");
    if (coils.items.len == 0) return fail(diag, lineno, "no coils found");

    const nv = xs.items.len;
    const dx = try arena.alloc(f64, nv);
    const dy = try arena.alloc(f64, nv);
    const dz = try arena.alloc(f64, nv);
    const dsq = try arena.alloc(f64, nv);
    @memset(dx, 0);
    @memset(dy, 0);
    @memset(dz, 0);
    @memset(dsq, 0);
    for (coils.items) |*c| {
        var min_dsq = std.math.inf(f64);
        for (c.first..c.first + c.nvert - 1) |n| {
            dx[n] = xs.items[n + 1] - xs.items[n];
            dy[n] = ys.items[n + 1] - ys.items[n];
            dz[n] = zs.items[n + 1] - zs.items[n];
            dsq[n] = dx[n] * dx[n] + dy[n] * dy[n] + dz[n] * dz[n];
            min_dsq = @min(min_dsq, dsq[n]);
        }
        c.eps = std.math.floatEps(f64) * min_dsq;
    }

    // Stable sort by group, and nominal group currents.
    const ng = gids.items.len;
    const sorted = try arena.alloc(Coil, coils.items.len);
    const group_start = try arena.alloc(u32, ng + 1);
    const group_current = try arena.alloc(f64, ng);
    var out: u32 = 0;
    for (0..ng) |g| {
        group_start[g] = out;
        var best: f64 = 0;
        var have = false;
        for (coils.items) |c| {
            if (c.group != g) continue;
            sorted[out] = c;
            out += 1;
            if (!have or @abs(c.current) > @abs(best)) {
                best = c.current;
                have = true;
            }
        }
        group_current[g] = best;
    }
    group_start[ng] = out;

    return .{
        .nfp = nfp,
        .group_ids = gids.items,
        .group_names = gnames.items,
        .group_current = group_current,
        .coils = sorted,
        .group_start = group_start,
        .x = xs.items,
        .y = ys.items,
        .z = zs.items,
        .dx = dx,
        .dy = dy,
        .dz = dz,
        .dsq = dsq,
        .nseg = nseg,
        .two_line_coils = two_line,
    };
}

test "closure, duplicates, grouping" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const text =
        \\periods 2
        \\begin filament
        \\mirror NIL
        \\0 0 0 5.0
        \\1 0 0 9.0
        \\1 0 0 9.0
        \\1 1 0 9.0
        \\7 7 7 0.0 9 B
        \\0 0 1 -8.0
        \\0 0 2 0.0 4 A
        \\0 0 0 2.0D0
        \\1 0 0 2.0
        \\1 1 0 2.0
        \\0 0 0 0.0 9 B
        \\end
    ;
    var diag: Diag = .{};
    const cs = try parse(arena_state.allocator(), text, &diag);
    try std.testing.expectEqual(@as(i32, 2), cs.nfp);
    try std.testing.expectEqual(@as(usize, 2), cs.group_names.len);
    try std.testing.expectEqualStrings("B", cs.group_names[0]);
    try std.testing.expectEqualStrings("A", cs.group_names[1]);
    try std.testing.expectEqual(@as(f64, 5.0), cs.group_current[0]);
    try std.testing.expectEqual(@as(f64, -8.0), cs.group_current[1]);
    // Group B: two closed triangles (3 segments each); group A: one open segment.
    try std.testing.expectEqual(@as(usize, 7), cs.nseg);
    try std.testing.expectEqual(@as(u32, 4), cs.coils[0].nvert);
    try std.testing.expectEqual(@as(u32, 2), cs.coils[2].nvert);
    try std.testing.expectEqual(@as(usize, 1), cs.two_line_coils);
}

test "start marker, quoted group names, group id -1" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    var diag: Diag = .{};
    const body = "periods 4\nbegin filament\nmirror NIL\n0 0 0 1\n1 0 0 1\n1 1 0 1\n0 0 0 0 3 'PF 1'\nend\n";
    const cs = try parse(arena_state.allocator(), "junk\nnotes ** coils_dot_starts_below ** here\n" ++ body, &diag);
    try std.testing.expectEqual(@as(i32, 4), cs.nfp);
    try std.testing.expectEqualStrings("PF 1", cs.group_names[0]);
    const bad = "periods 1\nbegin filament\nmirror NIL\n0 0 0 1\n1 0 0 1\n1 1 0 1\n0 0 0 0 -1 X\nend\n";
    try std.testing.expectError(error.InvalidCoils, parse(arena_state.allocator(), bad, &diag));
    try std.testing.expectEqual(@as(usize, 7), diag.line);
}

test "one-line coil is rejected" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    var diag: Diag = .{};
    const text = "periods 1\nbegin filament\nmirror NIL\n1 0 0 5.0 1 C\nend\n";
    try std.testing.expectError(error.InvalidCoils, parse(arena_state.allocator(), text, &diag));
    try std.testing.expectEqual(@as(usize, 4), diag.line);
}

test "mutated coils files never crash the parser" {
    const seed_text =
        \\periods 3
        \\begin filament
        \\mirror NIL
        \\ 1.0E+00 0.0 0.0 1.5D+03
        \\ 1.0 1.0 0.0 1.5E+03
        \\ 0.0 1.0 0.5 1.5E+03
        \\ 1.0 0.0 0.0 0.0 2 'Mod A'
        \\ 0.0 0.0 -1.0 7.0
        \\ 0.0 0.0 1.0 0.0 1 Line
        \\end
    ;
    const alphabet = " \t\n\r,.+-0123456789eEdD'\"*!/ endperiodsNIL";
    var prng: std.Random.DefaultPrng = .init(7);
    const rnd = prng.random();
    var buf: [seed_text.len + 8]u8 = undefined;
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    for (0..20000) |iter| {
        var len = seed_text.len;
        @memcpy(buf[0..len], seed_text);
        for (0..1 + rnd.uintLessThan(usize, 4)) |_| {
            const at = rnd.uintLessThan(usize, len);
            switch (rnd.uintLessThan(u8, 3)) {
                0 => buf[at] = alphabet[rnd.uintLessThan(usize, alphabet.len)],
                1 => { // delete
                    std.mem.copyForwards(u8, buf[at .. len - 1], buf[at + 1 .. len]);
                    len -= 1;
                },
                else => if (len < buf.len) { // insert
                    std.mem.copyBackwards(u8, buf[at + 1 .. len + 1], buf[at..len]);
                    buf[at] = alphabet[rnd.uintLessThan(usize, alphabet.len)];
                    len += 1;
                },
            }
        }
        var diag: Diag = .{};
        if (parse(arena_state.allocator(), buf[0..len], &diag)) |cs| {
            // Anything accepted must be internally consistent and finite.
            for (cs.coils) |c| {
                try std.testing.expect(c.nvert >= 2 and c.first + c.nvert <= cs.x.len);
                try std.testing.expect(c.eps > 0 and std.math.isFinite(c.eps));
            }
        } else |e| try std.testing.expect(e == error.InvalidCoils and diag.msg.len > 0);
        if (iter % 256 == 0) _ = arena_state.reset(.retain_capacity);
    }
}
