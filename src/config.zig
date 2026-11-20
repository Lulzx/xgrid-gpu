//! Run configuration from the four invocation modes (spec section 2).

const std = @import("std");
const Allocator = std.mem.Allocator;
const parseReal = @import("coils.zig").parseReal;

pub const Config = struct {
    ext: []const u8 = "",
    /// Explicit coils path (modern flags only); default is `coils.<ext>`.
    coils_path: ?[]const u8 = null,
    mode: u8 = 'S',
    sym: bool = false,
    rmin: f64 = 0,
    rmax: f64 = 0,
    zmin: f64 = 0,
    zmax: f64 = 0,
    kp: i64 = 0,
    ir: i64 = 121,
    jz: i64 = 121,
    legacy_fp32_trig: bool = false,
    threads: usize = 0,
    out_path: ?[]const u8 = null,
    extcur_path: ?[]const u8 = null,
    quiet: bool = false,
    backend: Backend = .auto,
    precision: Precision = .default,
    device: u32 = 0,
};

pub const Backend = enum { auto, cpu, metal, cuda };
pub const Precision = enum { default, fp64, ds, fp32 };

pub const Action = union(enum) {
    run: Config,
    help,
    version,
    /// Read the ten-value script from stdin, then call `fromStdin`.
    interactive: Config,
    /// Read this namelist file, then call `fromNamelist`.
    namelist: struct { path: []const u8, base: Config },
};

pub const Diag = struct { msg: []const u8 = "" };
pub const Error = error{Usage} || Allocator.Error;

fn fail(arena: Allocator, diag: *Diag, comptime fmt: []const u8, args: anytype) Error {
    diag.msg = try std.fmt.allocPrint(arena, fmt, args);
    return error.Usage;
}

fn real(arena: Allocator, diag: *Diag, what: []const u8, tok: []const u8) Error!f64 {
    return parseReal(tok) orelse fail(arena, diag, "{s}: '{s}' is not a number", .{ what, tok });
}

fn int(arena: Allocator, diag: *Diag, what: []const u8, tok: []const u8) Error!i64 {
    return std.fmt.parseInt(i64, tok, 10) catch fail(arena, diag, "{s}: '{s}' is not an integer", .{ what, tok });
}

fn unquote(s: []const u8) []const u8 {
    if (s.len >= 2 and (s[0] == '\'' or s[0] == '"') and s[s.len - 1] == s[0]) return s[1 .. s.len - 1];
    return s;
}

fn logical(tok: []const u8) ?bool {
    const t = std.mem.trim(u8, tok, ".");
    if (t.len == 0) return null;
    return switch (std.ascii.toLower(t[0])) {
        't' => true,
        'f' => false,
        else => null,
    };
}

/// Classify the command line. `args` excludes the program name.
pub fn fromArgs(arena: Allocator, args_in: []const []const u8, diag: *Diag) Error!Action {
    var base: Config = .{};
    var rest: std.ArrayList([]const u8) = .empty;
    for (args_in) |a| {
        if (std.mem.eql(u8, a, "--legacy-fp32-trig")) base.legacy_fp32_trig = true else try rest.append(arena, a);
    }
    const args = rest.items;

    var modern = false;
    for (args) |a| {
        if (std.mem.startsWith(u8, a, "--")) modern = true;
    }
    if (modern) return .{ .run = try fromFlags(arena, args, base, diag) } ;

    switch (args.len) {
        0 => return .{ .interactive = base },
        1 => {
            if (std.mem.eql(u8, args[0], "-h")) return .help;
            return .{ .namelist = .{ .path = args[0], .base = base } };
        },
        8...10 => {
            var c = base;
            c.ext = args[0];
            c.mode = if (args[1].len > 0 and std.ascii.toLower(args[1][0]) == 'r') 'R' else 'S';
            c.sym = args[2].len > 0 and switch (args[2][0]) {
                'Y', 'y', 'T', 't' => true,
                else => false,
            };
            c.rmin = try real(arena, diag, "rmin", args[3]);
            c.rmax = try real(arena, diag, "rmax", args[4]);
            c.zmin = try real(arena, diag, "zmin", args[5]);
            c.zmax = try real(arena, diag, "zmax", args[6]);
            c.kp = try int(arena, diag, "kp", args[7]);
            if (args.len >= 9) c.ir = try int(arena, diag, "ir", args[8]);
            if (args.len >= 10) c.jz = try int(arena, diag, "jz", args[9]);
            return .{ .run = c };
        },
        else => return fail(arena, diag, "unknown number of arguments ({d}); run with -h for help", .{args.len}),
    }
}

fn fromFlags(arena: Allocator, args: []const []const u8, base: Config, diag: *Diag) Error!Config {
    var c = base;
    var have = struct { rmin: bool = false, rmax: bool = false, zmin: bool = false, zmax: bool = false, kp: bool = false, coils: bool = false }{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const flags_without_value = [_][]const u8{ "--sym", "--quiet", "--help", "--version" };
        var takes_value = true;
        for (flags_without_value) |f| {
            if (std.mem.eql(u8, a, f)) takes_value = false;
        }
        var val: []const u8 = "";
        if (takes_value) {
            if (i + 1 >= args.len) return fail(arena, diag, "{s} needs a value", .{a});
            // "--sym true|false" is also accepted for compatibility with v0.1 examples.
            i += 1;
            val = args[i];
        } else if (std.mem.eql(u8, a, "--sym") and i + 1 < args.len) {
            if (logical(args[i + 1])) |b| {
                if (std.ascii.eqlIgnoreCase(args[i + 1], "true") or std.ascii.eqlIgnoreCase(args[i + 1], "false") or args[i + 1].len == 1) {
                    c.sym = b;
                    i += 1;
                    continue;
                }
            }
        }
        if (std.mem.eql(u8, a, "--sym")) {
            c.sym = true;
        } else if (std.mem.eql(u8, a, "--quiet")) {
            c.quiet = true;
        } else if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "--version")) {
            return fail(arena, diag, "{s} must be used on its own", .{a});
        } else if (std.mem.eql(u8, a, "--coils")) {
            c.coils_path = val;
            const basename = std.fs.path.basename(val);
            if (!std.mem.startsWith(u8, basename, "coils.") or basename.len == 6)
                return fail(arena, diag, "--coils: file name must be 'coils.<ext>', got '{s}'", .{basename});
            c.ext = basename[6..];
            have.coils = true;
        } else if (std.mem.eql(u8, a, "--mode")) {
            if (val.len != 1 or (std.ascii.toUpper(val[0]) != 'S' and std.ascii.toUpper(val[0]) != 'R'))
                return fail(arena, diag, "--mode must be S or R", .{});
            c.mode = std.ascii.toUpper(val[0]);
        } else if (std.mem.eql(u8, a, "--rmin")) {
            c.rmin = try real(arena, diag, a, val);
            have.rmin = true;
        } else if (std.mem.eql(u8, a, "--rmax")) {
            c.rmax = try real(arena, diag, a, val);
            have.rmax = true;
        } else if (std.mem.eql(u8, a, "--zmin")) {
            c.zmin = try real(arena, diag, a, val);
            have.zmin = true;
        } else if (std.mem.eql(u8, a, "--zmax")) {
            c.zmax = try real(arena, diag, a, val);
            have.zmax = true;
        } else if (std.mem.eql(u8, a, "--kp")) {
            c.kp = try int(arena, diag, a, val);
            have.kp = true;
        } else if (std.mem.eql(u8, a, "--ir")) {
            c.ir = try int(arena, diag, a, val);
        } else if (std.mem.eql(u8, a, "--jz")) {
            c.jz = try int(arena, diag, a, val);
        } else if (std.mem.eql(u8, a, "--threads")) {
            const n = try int(arena, diag, a, val);
            if (n < 1) return fail(arena, diag, "--threads must be >= 1", .{});
            c.threads = @intCast(n);
        } else if (std.mem.eql(u8, a, "--out")) {
            c.out_path = val;
        } else if (std.mem.eql(u8, a, "--extcur")) {
            c.extcur_path = val;
        } else if (std.mem.eql(u8, a, "--backend")) {
            c.backend = std.meta.stringToEnum(Backend, val) orelse
                return fail(arena, diag, "--backend {s}: expected auto, cpu, metal or cuda", .{val});
        } else if (std.mem.eql(u8, a, "--device")) {
            const n = try int(arena, diag, a, val);
            if (n < 0 or n > 1024) return fail(arena, diag, "--device must be between 0 and 1024", .{});
            c.device = @intCast(n);
        } else if (std.mem.eql(u8, a, "--precision")) {
            c.precision = std.meta.stringToEnum(Precision, val) orelse
                return fail(arena, diag, "--precision {s}: expected fp64, ds or fp32", .{val});
            if (c.precision == .default) return fail(arena, diag, "--precision default: expected fp64, ds or fp32", .{});
        } else {
            if (!std.mem.startsWith(u8, a, "--"))
                return fail(arena, diag, "unexpected argument '{s}': the legacy positional form cannot be combined with -- options (except --legacy-fp32-trig)", .{a});
            return fail(arena, diag, "unknown option '{s}'", .{a});
        }
    }
    switch (c.backend) {
        .cpu => if (c.precision == .ds or c.precision == .fp32)
            return fail(arena, diag, "the cpu backend is fp64 only", .{}),
        .metal => if (c.precision == .fp64)
            return fail(arena, diag, "the metal backend supports --precision ds or fp32, not fp64", .{}),
        .cuda, .auto => {},
    }
    if (!have.coils) return fail(arena, diag, "--coils is required", .{});
    if (!(have.rmin and have.rmax and have.zmin and have.zmax and have.kp))
        return fail(arena, diag, "--rmin, --rmax, --zmin, --zmax and --kp are required", .{});
    return c;
}

/// Ten list-directed values, one per line: ext, S/R, Y/N, rmin, rmax, zmin,
/// zmax, kp, ir, jz.
pub fn fromStdin(arena: Allocator, base: Config, text: []const u8, diag: *Diag) Error!Config {
    var vals: [10][]const u8 = undefined;
    var n: usize = 0;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var it = std.mem.tokenizeAny(u8, line, " \t\r,");
        const tok = it.next() orelse continue;
        if (n < vals.len) vals[n] = unquote(tok);
        n += 1;
    }
    if (n < vals.len) return fail(arena, diag, "expected 10 input values on stdin, got {d}", .{n});
    var c = base;
    c.ext = try arena.dupe(u8, vals[0]);
    c.mode = if (std.ascii.toLower(vals[1][0]) == 'r') 'R' else 'S';
    c.sym = std.ascii.toLower(vals[2][0]) == 'y';
    c.rmin = try real(arena, diag, "rmin", vals[3]);
    c.rmax = try real(arena, diag, "rmax", vals[4]);
    c.zmin = try real(arena, diag, "zmin", vals[5]);
    c.zmax = try real(arena, diag, "zmax", vals[6]);
    c.kp = try int(arena, diag, "kp", vals[7]);
    c.ir = try int(arena, diag, "ir", vals[8]);
    c.jz = try int(arena, diag, "jz", vals[9]);
    return c;
}

const ignored_keys = [_][]const u8{ "rmajor", "aminor", "nphi", "ntheta", "extcur_mgrid", "l_rot_coil_center", "sym_ir", "sym_jz", "sym_kp", "sym_perform_tests" };
const transform_keys = [_][]const u8{ "cg_shift_1", "cg_shift_2", "cg_rot_xcent", "cg_rot_theta", "cg_rot_phi", "cg_rot_angle" };

fn inList(list: []const []const u8, key: []const u8) bool {
    for (list) |k| {
        if (std.mem.eql(u8, k, key)) return true;
    }
    return false;
}

/// Parse an `&MGRID_NLI` namelist with the legacy namelist-path defaults.
pub fn fromNamelist(arena: Allocator, base: Config, text: []const u8, diag: *Diag) Error!Config {
    var c = base;
    c.ext = "dummy.cth.m.d12c.f5ss";
    c.mode = 'R';
    c.sym = true;
    c.rmin = 0.45;
    c.rmax = 1.05;
    c.zmin = -0.3;
    c.zmax = 0.3;
    c.kp = 5;

    // Tokenise: quoted strings, '=', and runs of other characters. Comments
    // run from '!' to end of line.
    var toks: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    while (i < text.len) {
        const ch = text[i];
        if (ch == '!') {
            while (i < text.len and text[i] != '\n') i += 1;
        } else if (ch == ' ' or ch == '\t' or ch == '\r' or ch == '\n' or ch == ',') {
            i += 1;
        } else if (ch == '=' or ch == '/') {
            try toks.append(arena, text[i .. i + 1]);
            i += 1;
        } else if (ch == '\'' or ch == '"') {
            const start = i;
            i += 1;
            while (i < text.len and text[i] != ch) i += 1;
            if (i >= text.len) return fail(arena, diag, "unterminated string in namelist", .{});
            i += 1;
            try toks.append(arena, text[start..i]);
        } else {
            const start = i;
            while (i < text.len and std.mem.indexOfScalar(u8, " \t\r\n,=/!'\"", text[i]) == null) i += 1;
            try toks.append(arena, text[start..i]);
        }
    }

    var t: usize = 0;
    while (t < toks.items.len and !std.ascii.eqlIgnoreCase(toks.items[t], "&mgrid_nli")) t += 1;
    if (t == toks.items.len) return fail(arena, diag, "no &MGRID_NLI group found", .{});
    t += 1;
    var terminated = false;
    while (t < toks.items.len) {
        const key_raw = toks.items[t];
        if (std.mem.eql(u8, key_raw, "/") or std.ascii.eqlIgnoreCase(key_raw, "&end")) {
            terminated = true;
            break;
        }
        if (t + 1 >= toks.items.len or !std.mem.eql(u8, toks.items[t + 1], "="))
            return fail(arena, diag, "namelist: expected '=' after '{s}'", .{key_raw});
        // Values run until the next "name =" pair or the terminator.
        var end = t + 2;
        while (end < toks.items.len) : (end += 1) {
            const tk = toks.items[end];
            if (std.mem.eql(u8, tk, "/") or std.ascii.eqlIgnoreCase(tk, "&end")) break;
            if (end + 1 < toks.items.len and std.mem.eql(u8, toks.items[end + 1], "=")) break;
        }
        const vals = toks.items[t + 2 .. end];
        t = end;

        var kbuf: [64]u8 = undefined;
        const paren = std.mem.indexOfScalar(u8, key_raw, '(') orelse key_raw.len;
        if (paren > kbuf.len) return fail(arena, diag, "namelist: unknown variable '{s}'", .{key_raw});
        const key = std.ascii.lowerString(&kbuf, key_raw[0..paren]);

        if (inList(&ignored_keys, key)) continue;
        if (inList(&transform_keys, key)) {
            for (vals) |v| {
                const star = std.mem.indexOfScalar(u8, v, '*');
                const num = if (star) |s| v[s + 1 ..] else v;
                const x = parseReal(num) orelse return fail(arena, diag, "namelist: bad value '{s}' for {s}", .{ v, key });
                if (x != 0) return fail(arena, diag, "namelist: coil shifts and rotations ({s}) are not supported", .{key});
            }
            continue;
        }
        if (vals.len != 1) return fail(arena, diag, "namelist: {s} needs exactly one value", .{key});
        const v = vals[0];
        if (std.mem.eql(u8, key, "task")) {
            if (!std.ascii.eqlIgnoreCase(unquote(v), "mgrid"))
                return fail(arena, diag, "namelist: task {s} is not supported (only 'mgrid')", .{v});
        } else if (std.mem.eql(u8, key, "mgrid_ext")) {
            c.ext = try arena.dupe(u8, std.mem.trim(u8, unquote(v), " "));
        } else if (std.mem.eql(u8, key, "mgrid_mode")) {
            const m = unquote(v);
            if (m.len == 0) return fail(arena, diag, "namelist: empty mgrid_mode", .{});
            c.mode = if (std.ascii.toLower(m[0]) == 'r') 'R' else 'S';
        } else if (std.mem.eql(u8, key, "lstell_sym")) {
            c.sym = logical(v) orelse return fail(arena, diag, "namelist: bad logical '{s}' for lstell_sym", .{v});
        } else if (std.mem.eql(u8, key, "rmin")) {
            c.rmin = try real(arena, diag, key, v);
        } else if (std.mem.eql(u8, key, "rmax")) {
            c.rmax = try real(arena, diag, key, v);
        } else if (std.mem.eql(u8, key, "zmin")) {
            c.zmin = try real(arena, diag, key, v);
        } else if (std.mem.eql(u8, key, "zmax")) {
            c.zmax = try real(arena, diag, key, v);
        } else if (std.mem.eql(u8, key, "kp")) {
            c.kp = try int(arena, diag, key, v);
        } else if (std.mem.eql(u8, key, "ir")) {
            c.ir = try int(arena, diag, key, v);
        } else if (std.mem.eql(u8, key, "jz")) {
            c.jz = try int(arena, diag, key, v);
        } else {
            return fail(arena, diag, "namelist: unknown variable '{s}'", .{key_raw});
        }
    }
    if (!terminated) return fail(arena, diag, "namelist: &MGRID_NLI is not terminated by '/' or '&END'", .{});
    return c;
}

/// Checks from spec section 2.3 that do not need the coils file.
pub fn validate(arena: Allocator, c: *const Config, diag: *Diag) Error!void {
    if (c.ext.len == 0) return fail(arena, diag, "empty coils extension", .{});
    for ([_]f64{ c.rmin, c.rmax, c.zmin, c.zmax }) |v| {
        if (!std.math.isFinite(v)) return fail(arena, diag, "grid bounds must be finite", .{});
    }
    if (c.ir < 2) return fail(arena, diag, "ir must be >= 2 (got {d})", .{c.ir});
    if (c.jz < 2) return fail(arena, diag, "jz must be >= 2 (got {d})", .{c.jz});
    if (c.kp < 1) return fail(arena, diag, "kp must be >= 1 (got {d})", .{c.kp});
    if (c.rmin < 0) return fail(arena, diag, "rmin must be >= 0", .{});
    if (c.rmax <= c.rmin) return fail(arena, diag, "rmax must be greater than rmin", .{});
    if (c.zmax <= c.zmin) return fail(arena, diag, "zmax must be greater than zmin", .{});
    const limit: i64 = (0xFFFF_FFFF - 3) / 8;
    if (c.ir > limit or c.jz > limit or c.kp > limit or c.ir * c.jz > limit or c.ir * c.jz * c.kp > limit)
        return fail(arena, diag, "grid of {d} x {d} x {d} points exceeds the NetCDF classic per-variable limit", .{ c.ir, c.jz, c.kp });
}

test "invocation modes" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var diag: Diag = .{};

    const pos = try fromArgs(a, &.{ "c09", "r", "t", "0.5", "1.5", "-1", "1.D0", "36", "--legacy-fp32-trig" }, &diag);
    try std.testing.expectEqual(@as(u8, 'R'), pos.run.mode);
    try std.testing.expect(pos.run.sym and pos.run.legacy_fp32_trig);
    try std.testing.expectEqual(@as(i64, 121), pos.run.ir);
    try std.testing.expectEqual(@as(f64, 1.0), pos.run.zmax);

    const fl = try fromArgs(a, &.{ "--coils", "dir/coils.abc", "--mode", "S", "--sym", "--rmin", "1", "--rmax", "2", "--zmin", "-1", "--zmax", "1", "--kp", "4", "--ir", "9" }, &diag);
    try std.testing.expectEqualStrings("abc", fl.run.ext);
    try std.testing.expect(fl.run.sym);
    try std.testing.expectEqual(@as(i64, 9), fl.run.ir);

    try std.testing.expect((try fromArgs(a, &.{}, &diag)) == .interactive);
    try std.testing.expect((try fromArgs(a, &.{"-h"}, &diag)) == .help);
    try std.testing.expectError(error.Usage, fromArgs(a, &.{ "a", "b" }, &diag));

    const st = try fromStdin(a, .{}, "'two'\nS\ny\n0.8\n1.2\n-0.2\n0.2\n6\n9\n7\n", &diag);
    try std.testing.expectEqualStrings("two", st.ext);
    try std.testing.expect(st.sym);
    try std.testing.expectEqual(@as(i64, 7), st.jz);

    const nl = try fromNamelist(a, .{},
        \\ ! comment
        \\ &MGRID_NLI
        \\   mgrid_ext = 'two', MGRID_MODE='S'
        \\   LSTELL_SYM = .FALSE.  ! trailing
        \\   rmin = 0.8D0, rmax = 1.2
        \\   kp = 6, cg_shift_1 = 3*0.0
        \\ /
    , &diag);
    try std.testing.expectEqualStrings("two", nl.ext);
    try std.testing.expectEqual(@as(u8, 'S'), nl.mode);
    try std.testing.expect(!nl.sym);
    try std.testing.expectEqual(@as(f64, 0.8), nl.rmin);
    try std.testing.expectEqual(@as(f64, -0.3), nl.zmin);
    try std.testing.expectEqual(@as(i64, 121), nl.ir);

    try std.testing.expectError(error.Usage, fromNamelist(a, .{}, "&mgrid_nli cg_rot_angle = 0.1 /", &diag));
    try std.testing.expectError(error.Usage, fromNamelist(a, .{}, "&mgrid_nli task = 'mgrid_rs' /", &diag));
    try std.testing.expectError(error.Usage, fromNamelist(a, .{}, "&mgrid_nli bogus = 1 /", &diag));
}

test "mutated namelists and stdin scripts never crash" {
    const seeds = [_][]const u8{
        "&MGRID_NLI\n mgrid_ext='two', mgrid_mode='S'\n lstell_sym=.T. ! c\n rmin=0.8D0, rmax=1.2\n kp=6, ir=9, jz=7, cg_shift_1 = 3*0.0\n/\n",
        "'two'\nS\nY\n0.8\n1.2\n-0.2\n0.2\n6\n9\n7\n",
    };
    const alphabet = " \t\n,=/!&'\"*.+-0123456789eEdDtTfFmgrid_nlikpzs()";
    var prng: std.Random.DefaultPrng = .init(11);
    const rnd = prng.random();
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    var buf: [256]u8 = undefined;
    for (0..20000) |iter| {
        const which = iter % 2;
        var len = seeds[which].len;
        @memcpy(buf[0..len], seeds[which]);
        for (0..1 + rnd.uintLessThan(usize, 4)) |_| {
            const at = rnd.uintLessThan(usize, len);
            if (rnd.boolean()) {
                buf[at] = alphabet[rnd.uintLessThan(usize, alphabet.len)];
            } else {
                std.mem.copyForwards(u8, buf[at .. len - 1], buf[at + 1 .. len]);
                len -= 1;
            }
        }
        var diag: Diag = .{};
        const a = arena_state.allocator();
        const res = if (which == 0) fromNamelist(a, .{}, buf[0..len], &diag) else fromStdin(a, .{}, buf[0..len], &diag);
        if (res) |c| {
            validate(a, &c, &diag) catch |e| try std.testing.expect(e == error.Usage);
        } else |e| try std.testing.expect(e == error.Usage and diag.msg.len > 0);
        if (iter % 256 == 0) _ = arena_state.reset(.retain_capacity);
    }
}
