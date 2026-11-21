//! Dependency-free NetCDF classic 64-bit-offset (CDF-2) writer for mgrid
//! files, plus a minimal CDF-1/CDF-2 reader used by the test tools
//! (spec section 5.1).

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const NC_CHAR: u32 = 2;
pub const NC_INT: u32 = 4;
pub const NC_DOUBLE: u32 = 6;
const NC_DIMENSION: u32 = 0x0A;
const NC_VARIABLE: u32 = 0x0B;
const NC_ATTRIBUTE: u32 = 0x0C;

pub const string_size = 30;

pub const Meta = struct {
    ir: i32,
    jz: i32,
    kp: i32,
    nfp: i32,
    rmin: f64,
    zmin: f64,
    rmax: f64,
    zmax: f64,
    mode: u8,
    group_names: []const []const u8,
    group_current: []const f64,
};

pub const Layout = struct {
    /// Header followed by every non-field variable, ready to write at offset 0.
    prefix: []u8,
    /// Header length alone (the first `header_len` bytes of `prefix`).
    header_len: usize,
    /// Byte size of one br/bp/bz array.
    field_bytes: u64,
    total_bytes: u64,

    /// File offset of component `comp` (0 = br, 1 = bp, 2 = bz) of group `g`.
    pub fn fieldOffset(l: *const Layout, g: usize, comp: usize) u64 {
        return l.prefix.len + (3 * @as(u64, g) + comp) * l.field_bytes;
    }
};

const Buf = struct {
    list: std.ArrayList(u8) = .empty,
    arena: Allocator,

    fn u32be(b: *Buf, v: u32) !void {
        var tmp: [4]u8 = undefined;
        std.mem.writeInt(u32, &tmp, v, .big);
        try b.list.appendSlice(b.arena, &tmp);
    }
    fn u64be(b: *Buf, v: u64) !void {
        var tmp: [8]u8 = undefined;
        std.mem.writeInt(u64, &tmp, v, .big);
        try b.list.appendSlice(b.arena, &tmp);
    }
    fn f64be(b: *Buf, v: f64) !void {
        try b.u64be(@bitCast(v));
    }
    fn pad(b: *Buf) !void {
        while (b.list.items.len % 4 != 0) try b.list.append(b.arena, 0);
    }
    fn name(b: *Buf, s: []const u8) !void {
        try b.u32be(@intCast(s.len));
        try b.list.appendSlice(b.arena, s);
        try b.pad();
    }
};

const VarDef = struct {
    name: []const u8,
    dims: []const u32,
    nc_type: u32,
    bytes: u64,
};

fn padded(n: u64) u64 {
    return (n + 3) / 4 * 4;
}

pub const BuildError = error{ TooManyGroups, FieldTooLarge } || Allocator.Error;

/// Build the header and small-variable prefix of an mgrid file.
pub fn build(arena: Allocator, m: Meta) BuildError!Layout {
    const ng = m.group_names.len;
    if (ng > 999) return error.TooManyGroups;
    const npts: u64 = @as(u64, @intCast(m.ir)) * @as(u64, @intCast(m.jz)) * @as(u64, @intCast(m.kp));
    const field_bytes = npts * 8;
    if (field_bytes > 0xFFFF_FFFF - 3) return error.FieldTooLarge;

    // Dimension lists differ between the single-group and multi-group layouts.
    const Dim = struct { name: []const u8, size: u32 };
    var dims: std.ArrayList(Dim) = .empty;
    var coil_group_dims: []const u32 = undefined;
    var mode_dim: u32 = undefined;
    var coils_dim: u32 = undefined;
    if (ng == 1) {
        try dims.append(arena, .{ .name = "dim_00030", .size = string_size });
        try dims.append(arena, .{ .name = "dim_00001", .size = 1 });
        try dims.append(arena, .{ .name = "external_coils", .size = 1 });
        coil_group_dims = &.{0};
        mode_dim = 1;
        coils_dim = 2;
    } else {
        try dims.append(arena, .{ .name = "stringsize", .size = string_size });
        try dims.append(arena, .{ .name = "external_coil_groups", .size = @intCast(ng) });
        try dims.append(arena, .{ .name = "dim_00001", .size = 1 });
        try dims.append(arena, .{ .name = "external_coils", .size = @intCast(ng) });
        coil_group_dims = &.{ 1, 0 };
        mode_dim = 2;
        coils_dim = 3;
    }
    const rad: u32 = @intCast(dims.items.len);
    try dims.append(arena, .{ .name = "rad", .size = @intCast(m.ir) });
    try dims.append(arena, .{ .name = "zee", .size = @intCast(m.jz) });
    try dims.append(arena, .{ .name = "phi", .size = @intCast(m.kp) });
    const field_dims = try arena.dupe(u32, &.{ rad + 2, rad + 1, rad });

    var vars: std.ArrayList(VarDef) = .empty;
    for ([_][]const u8{ "ir", "jz", "kp", "nfp", "nextcur" }) |n|
        try vars.append(arena, .{ .name = n, .dims = &.{}, .nc_type = NC_INT, .bytes = 4 });
    for ([_][]const u8{ "rmin", "zmin", "rmax", "zmax" }) |n|
        try vars.append(arena, .{ .name = n, .dims = &.{}, .nc_type = NC_DOUBLE, .bytes = 8 });
    try vars.append(arena, .{ .name = "coil_group", .dims = coil_group_dims, .nc_type = NC_CHAR, .bytes = ng * string_size });
    try vars.append(arena, .{ .name = "mgrid_mode", .dims = try arena.dupe(u32, &.{mode_dim}), .nc_type = NC_CHAR, .bytes = 1 });
    try vars.append(arena, .{ .name = "raw_coil_cur", .dims = try arena.dupe(u32, &.{coils_dim}), .nc_type = NC_DOUBLE, .bytes = ng * 8 });
    const first_field = vars.items.len;
    for (0..ng) |g| {
        for ([_][]const u8{ "br", "bp", "bz" }) |c| {
            const n = try std.fmt.allocPrint(arena, "{s}_{d:0>3}", .{ c, g + 1 });
            try vars.append(arena, .{ .name = n, .dims = field_dims, .nc_type = NC_DOUBLE, .bytes = field_bytes });
        }
    }

    // Header size is independent of the offsets it contains.
    var header_len: usize = 4 + 4 + 8 + 8 + 8;
    for (dims.items) |d| header_len += 4 + padded(d.name.len) + 4;
    for (vars.items) |v| header_len += 4 + padded(v.name.len) + 4 + 4 * v.dims.len + 8 + 4 + 4 + 8;

    var b: Buf = .{ .arena = arena };
    try b.list.appendSlice(arena, "CDF\x02");
    try b.u32be(0); // numrecs
    try b.u32be(NC_DIMENSION);
    try b.u32be(@intCast(dims.items.len));
    for (dims.items) |d| {
        try b.name(d.name);
        try b.u32be(d.size);
    }
    try b.u32be(0); // no global attributes
    try b.u32be(0);
    try b.u32be(NC_VARIABLE);
    try b.u32be(@intCast(vars.items.len));
    var begin: u64 = header_len;
    for (vars.items) |v| {
        try b.name(v.name);
        try b.u32be(@intCast(v.dims.len));
        for (v.dims) |d| try b.u32be(d);
        try b.u32be(0); // no variable attributes
        try b.u32be(0);
        try b.u32be(v.nc_type);
        try b.u32be(@intCast(padded(v.bytes)));
        try b.u64be(begin);
        begin += padded(v.bytes);
    }
    std.debug.assert(b.list.items.len == header_len);

    // Small variables, in definition order.
    for ([_]i32{ m.ir, m.jz, m.kp, m.nfp, @intCast(ng) }) |v| try b.u32be(@bitCast(v));
    for ([_]f64{ m.rmin, m.zmin, m.rmax, m.zmax }) |v| try b.f64be(v);
    for (m.group_names) |n| {
        var field: [string_size]u8 = @splat(' ');
        const len = @min(n.len, string_size);
        @memcpy(field[0..len], n[0..len]);
        try b.list.appendSlice(arena, &field);
    }
    try b.pad();
    try b.list.append(arena, m.mode);
    try b.pad();
    for (m.group_current) |v| try b.f64be(v);
    _ = first_field;

    return .{
        .prefix = b.list.items,
        .header_len = header_len,
        .field_bytes = field_bytes,
        .total_bytes = b.list.items.len + 3 * @as(u64, ng) * field_bytes,
    };
}

/// Store `src` big-endian into `dst` (8 bytes per value).
pub fn toBigEndian(dst: []u8, src: []const f64) void {
    std.debug.assert(dst.len == src.len * 8);
    for (src, 0..) |v, i| std.mem.writeInt(u64, dst[i * 8 ..][0..8], @bitCast(v), .big);
}

// ---------------------------------------------------------------- reader

pub const Var = struct {
    name: []const u8,
    dim_ids: []u32,
    shape: []u32,
    nc_type: u32,
    vsize: u32,
    begin: u64,
    natts: u32,

    pub fn count(v: Var) usize {
        var n: usize = 1;
        for (v.shape) |s| n *= s;
        return n;
    }
};

pub const File = struct {
    bytes: []const u8,
    version: u8,
    numrecs: u32,
    dim_names: [][]const u8,
    dim_sizes: []u32,
    ngatts: u32,
    vars: []Var,
    header_len: usize,

    pub fn find(f: *const File, name: []const u8) ?*const Var {
        for (f.vars) |*v| if (std.mem.eql(u8, v.name, name)) return v;
        return null;
    }

    pub fn doubleAt(f: *const File, v: *const Var, idx: usize) f64 {
        const off: usize = @intCast(v.begin + idx * 8);
        return @bitCast(std.mem.readInt(u64, f.bytes[off..][0..8], .big));
    }

    pub fn intAt(f: *const File, v: *const Var, idx: usize) i32 {
        const off: usize = @intCast(v.begin + idx * 4);
        return std.mem.readInt(i32, f.bytes[off..][0..4], .big);
    }

    pub fn chars(f: *const File, v: *const Var) []const u8 {
        return f.bytes[@intCast(v.begin)..][0..v.count()];
    }
};

pub const ReadError = error{ NotNetcdfClassic, Truncated } || Allocator.Error;

const Cursor = struct {
    b: []const u8,
    pos: usize = 0,

    fn u32be(c: *Cursor) ReadError!u32 {
        if (c.pos + 4 > c.b.len) return error.Truncated;
        defer c.pos += 4;
        return std.mem.readInt(u32, c.b[c.pos..][0..4], .big);
    }
    fn u64be(c: *Cursor) ReadError!u64 {
        if (c.pos + 8 > c.b.len) return error.Truncated;
        defer c.pos += 8;
        return std.mem.readInt(u64, c.b[c.pos..][0..8], .big);
    }
    /// Rejects a count of header entries that could not fit in the bytes
    /// left, each entry taking at least `min_bytes`. Checked before
    /// allocating, so a corrupt count cannot request gigabytes.
    fn count(c: *Cursor, n: u32, min_bytes: usize) ReadError!u32 {
        if (n > (c.b.len - c.pos) / min_bytes) return error.Truncated;
        return n;
    }
    fn name(c: *Cursor) ReadError![]const u8 {
        const n = try c.u32be();
        if (c.pos + padded(n) > c.b.len) return error.Truncated;
        defer c.pos += @intCast(padded(n));
        return c.b[c.pos..][0..n];
    }
    fn skipAtts(c: *Cursor, n: u32) ReadError!void {
        for (0..n) |_| {
            _ = try c.name();
            const t = try c.u32be();
            const ne = try c.u32be();
            const size: u64 = switch (t) {
                1, 2 => 1,
                3 => 2,
                4, 5 => 4,
                else => 8,
            };
            const skip = padded(@as(u64, ne) * size);
            if (skip > c.b.len - c.pos) return error.Truncated;
            c.pos += @intCast(skip);
        }
    }
};

pub fn parse(arena: Allocator, bytes: []const u8) ReadError!File {
    if (bytes.len < 4 or !std.mem.eql(u8, bytes[0..3], "CDF") or (bytes[3] != 1 and bytes[3] != 2))
        return error.NotNetcdfClassic;
    var c: Cursor = .{ .b = bytes, .pos = 4 };
    const numrecs = try c.u32be();
    const dtag = try c.u32be();
    // Minimum sizes: dimension = name length + size (8 bytes); variable =
    // name length, ndims, attribute tag and count, type, vsize, begin (28).
    const nd = try c.count(try c.u32be(), 8);
    const dim_names = try arena.alloc([]const u8, if (dtag == NC_DIMENSION) nd else 0);
    const dim_sizes = try arena.alloc(u32, dim_names.len);
    for (dim_names, dim_sizes) |*n, *s| {
        n.* = try c.name();
        s.* = try c.u32be();
    }
    const gtag = try c.u32be();
    const ng = try c.u32be();
    if (gtag == NC_ATTRIBUTE) try c.skipAtts(ng);
    const vtag = try c.u32be();
    const nv = try c.count(try c.u32be(), 28);
    const vars = try arena.alloc(Var, if (vtag == NC_VARIABLE) nv else 0);
    for (vars) |*v| {
        v.name = try c.name();
        const ndims = try c.count(try c.u32be(), 4);
        v.dim_ids = try arena.alloc(u32, ndims);
        v.shape = try arena.alloc(u32, ndims);
        for (v.dim_ids, v.shape) |*d, *s| {
            d.* = try c.u32be();
            if (d.* >= dim_sizes.len) return error.Truncated;
            s.* = dim_sizes[d.*];
        }
        const atag = try c.u32be();
        v.natts = try c.u32be();
        if (atag == NC_ATTRIBUTE) try c.skipAtts(v.natts) else v.natts = 0;
        v.nc_type = try c.u32be();
        v.vsize = try c.u32be();
        v.begin = if (bytes[3] == 2) try c.u64be() else try c.u32be();
        // Every variable must lie inside the file so the accessors are safe.
        const elem: u64 = switch (v.nc_type) {
            1, 2 => 1,
            3 => 2,
            4, 5 => 4,
            6 => 8,
            else => return error.Truncated,
        };
        var total: u64 = elem;
        for (v.shape) |sz| total = std.math.mul(u64, total, sz) catch return error.Truncated;
        if (v.begin > bytes.len or total > bytes.len - v.begin) return error.Truncated;
    }
    return .{
        .bytes = bytes,
        .version = bytes[3],
        .numrecs = numrecs,
        .dim_names = dim_names,
        .dim_sizes = dim_sizes,
        .ngatts = if (gtag == NC_ATTRIBUTE) ng else 0,
        .vars = vars,
        .header_len = c.pos,
    };
}

test "truncated and corrupt files are rejected, not trusted" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const l = try build(a, .{ .ir = 2, .jz = 2, .kp = 2, .nfp = 1, .rmin = 1, .zmin = -1, .rmax = 2, .zmax = 1, .mode = 'R', .group_names = &.{ "A", "B" }, .group_current = &.{ 1, 2 } });
    const full = try a.alloc(u8, @intCast(l.total_bytes));
    @memset(full, 0);
    @memcpy(full[0..l.prefix.len], l.prefix);
    _ = try parse(a, full);
    // Every truncation must fail cleanly.
    for (0..full.len) |n| try std.testing.expect(std.meta.isError(parse(a, full[0..n])));
    // Single-byte corruption of the header must never crash.
    var prng: std.Random.DefaultPrng = .init(1);
    const copy = try a.dupe(u8, full);
    for (0..20000) |_| {
        const at = prng.random().uintLessThan(usize, l.header_len);
        const old = copy[at];
        copy[at] = prng.random().int(u8);
        if (parse(a, copy)) |f| {
            for (f.vars) |*v| {
                if (v.nc_type == NC_DOUBLE and v.count() > 0) _ = f.doubleAt(v, v.count() - 1);
            }
        } else |_| {}
        copy[at] = old;
    }
}

test "corrupt counts are rejected before allocating" {
    // A count of 0xFFFFFFFF used to size an allocation of tens of GB before
    // the bytes behind it were checked (killed by the OS in Debug builds).
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const l = try build(a, .{ .ir = 2, .jz = 2, .kp = 2, .nfp = 1, .rmin = 1, .zmin = -1, .rmax = 2, .zmax = 1, .mode = 'R', .group_names = &.{"A"}, .group_current = &.{1} });
    const full = try a.alloc(u8, @intCast(l.total_bytes));
    @memset(full, 0);
    @memcpy(full[0..l.prefix.len], l.prefix);
    // Bytes 12..16 hold the dimension count (after magic, numrecs, tag).
    std.mem.writeInt(u32, full[12..16], 0xFFFF_FFFF, .big);
    try std.testing.expectError(error.Truncated, parse(std.testing.failing_allocator, full));
}

test "written header parses back" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    for ([_]usize{ 1, 2, 11 }) |ng| {
        const names = try a.alloc([]const u8, ng);
        const cur = try a.alloc(f64, ng);
        for (names, cur, 0..) |*n, *c, i| {
            n.* = "AVeryLongGroupNameThatExceedsThirtyCharacters";
            c.* = @floatFromInt(i + 1);
        }
        const l = try build(a, .{ .ir = 3, .jz = 4, .kp = 5, .nfp = 2, .rmin = 1, .zmin = -1, .rmax = 2, .zmax = 1, .mode = 'S', .group_names = names, .group_current = cur });
        const full = try a.alloc(u8, @intCast(l.total_bytes));
        @memset(full, 0);
        @memcpy(full[0..l.prefix.len], l.prefix);
        const f = try parse(a, full);
        try std.testing.expectEqual(l.header_len, f.header_len);
        try std.testing.expectEqual(@as(usize, 12 + 3 * ng), f.vars.len);
        try std.testing.expectEqual(@as(usize, if (ng == 1) 6 else 7), f.dim_names.len);
        const last = f.find(try std.fmt.allocPrint(a, "bz_{d:0>3}", .{ng})).?;
        try std.testing.expectEqual(l.fieldOffset(ng - 1, 2), last.begin);
        try std.testing.expectEqual(l.total_bytes, last.begin + last.vsize);
        try std.testing.expectEqual(@as(i32, @intCast(ng)), f.intAt(f.find("nextcur").?, 0));
        try std.testing.expectEqualStrings("AVeryLongGroupNameThatExceedsT", f.chars(f.find("coil_group").?)[0..30]);
        try std.testing.expectEqual(@as(f64, @floatFromInt(ng)), f.doubleAt(f.find("raw_coil_cur").?, ng - 1));
    }
}
