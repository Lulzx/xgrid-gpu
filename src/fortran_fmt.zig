//! Fortran edit-descriptor formatting needed for `extcur.<ext>` (spec 5.2).

const std = @import("std");

/// Format `v` as Fortran `1p,e22.14`: right-justified in 22 columns, one digit
/// before the decimal point, 14 after, exponent as `E+dd` (or `+ddd`).
pub fn e22_14(v: f64) [22]u8 {
    var out: [22]u8 = @splat(' ');
    if (!std.math.isFinite(v)) {
        const txt: []const u8 = if (std.math.isNan(v)) "NaN" else if (v < 0) "-Infinity" else "Infinity";
        @memcpy(out[out.len - txt.len ..], txt);
        return out;
    }
    var tmp: [64]u8 = undefined;
    // Zig prints e.g. "-1.23400000000000e5" / "1.00000000000000e-7".
    const s = std.fmt.bufPrint(&tmp, "{e:.14}", .{@abs(v)}) catch unreachable;
    const epos = std.mem.indexOfScalar(u8, s, 'e').?;
    const mant = s[0..epos];
    const exp = std.fmt.parseInt(i32, s[epos + 1 ..], 10) catch unreachable;
    var ebuf: [8]u8 = undefined;
    const mag: u32 = @abs(exp);
    const sign: u8 = if (exp < 0) '-' else '+';
    const etxt = if (mag < 100)
        std.fmt.bufPrint(&ebuf, "E{c}{d:0>2}", .{ sign, mag }) catch unreachable
    else
        std.fmt.bufPrint(&ebuf, "{c}{d:0>3}", .{ sign, mag }) catch unreachable;
    const neg = std.math.signbit(v);
    const len = mant.len + etxt.len + @intFromBool(neg);
    if (len > out.len) return @splat('*');
    var pos = out.len - len;
    if (neg) {
        out[pos] = '-';
        pos += 1;
    }
    @memcpy(out[pos..][0..mant.len], mant);
    @memcpy(out[pos + mant.len ..][0..etxt.len], etxt);
    return out;
}

test "matches gfortran 1p,e22.14" {
    try std.testing.expectEqualStrings("  1.00000000000000E+05", &e22_14(1.0e5));
    try std.testing.expectEqualStrings(" -2.00000000000000E+04", &e22_14(-2.0e4));
    try std.testing.expectEqualStrings("  1.10000000000000E+05", &e22_14(1.1e5));
    try std.testing.expectEqualStrings("  6.52271941985300E+05", &e22_14(6.522719419853e5));
    try std.testing.expectEqualStrings("  2.50000000000000E-07", &e22_14(2.5e-7));
    try std.testing.expectEqualStrings("  0.00000000000000E+00", &e22_14(0.0));
    try std.testing.expectEqualStrings("  1.00000000000000+100", &e22_14(1.0e100));
    try std.testing.expectEqualStrings("                   NaN", &e22_14(std.math.nan(f64)));
    try std.testing.expectEqualStrings("             -Infinity", &e22_14(-std.math.inf(f64)));
}
