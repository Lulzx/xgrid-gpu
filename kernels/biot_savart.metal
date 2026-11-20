// xgrid-gpu Biot-Savart kernel for Apple Metal (spec sections 3.2, 4.1, 4.3).
//
// Compiled twice from this one source, with XG_DS=1 (double-single: a value is
// an unevaluated sum hi + lo of two floats, about 48 significand bits) or
// XG_DS=0 (plain float). Must be compiled with fast-math disabled: the
// error-free transformations below are algebraically "zero" and a fast-math
// compiler removes them.
//
// The double-single routines are the standard Dekker/Knuth constructions in
// finite-only form (no NaN/Inf/zero special cases); the layout follows the
// fast48 mode of github.com/lulzx/vf64-metal (BSD 3-Clause).

#include <metal_stdlib>
using namespace metal;

struct Vert {            // 64 bytes; every float2 is {hi, lo}
    float2 x, y, z;      // vertex position
    float2 dx, dy, dz;   // vector to the next vertex of the same coil
    float2 dsq;          // its squared length
    float2 pad;
};
struct Coil {            // 24 bytes
    uint first, nvert;   // stored vertices [first, first + nvert); nvert - 1 segments
    float2 w;            // 1e-7 * current
    float2 eps;          // legacy floor
};
struct Point { float2 x, y, z, pad; };          // 32 bytes
struct Out { float2 bx, by, bz; float flag, pad; };  // 32 bytes
struct Params { uint c0, c1, offset, count; };

#if XG_DS

typedef float2 real;
static inline real R(float2 v) { return v; }
static inline float2 store(real v) { return v; }
static inline real r_zero() { return float2(0.0f, 0.0f); }
static inline real r_tiny() { return float2(1.0e-30f, 0.0f); }

static inline real two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return float2(s, (a - (s - bb)) + (b - bb));
}
static inline real quick_two_sum(float a, float b) {
    float s = a + b;
    return float2(s, b - (s - a));
}
static inline real r_add(real a, real b) {
    real s = two_sum(a.x, b.x);
    real t = two_sum(a.y, b.y);
    real u = quick_two_sum(s.x, s.y + t.x);
    return quick_two_sum(u.x, u.y + t.y);
}
static inline real r_sub(real a, real b) { return r_add(a, float2(-b.x, -b.y)); }
// Cheaper sum for operands of the same sign, where there is no cancellation
// for the more careful version to protect.
static inline real r_add_pos(real a, real b) {
    real s = two_sum(a.x, b.x);
    return quick_two_sum(s.x, s.y + (a.y + b.y));
}
static inline real r_twice(real a) { return float2(a.x + a.x, a.y + a.y); }
static inline real r_mul(real a, real b) {
    float p = a.x * b.x;
    float e = fma(a.x, b.x, -p);
    e = fma(a.x, b.y, e);
    e = fma(a.y, b.x, e);
    return quick_two_sum(p, e);
}
static inline real r_mulf(real a, float b) {
    float p = a.x * b;
    float e = fma(a.x, b, -p);
    e = fma(a.y, b, e);
    return quick_two_sum(p, e);
}
static inline real r_div(real a, real b) {
#if XG_DIV3
    float q1 = a.x / b.x;
    real r = r_sub(a, r_mulf(b, q1));
    float q2 = r.x / b.x;
    r = r_sub(r, r_mulf(b, q2));
    float q3 = r.x / b.x;
    return r_add(quick_two_sum(q1, q2), float2(q3, 0.0f));
#else
    // Quotient estimate plus one correction from the exact residual.
    float q1 = a.x / b.x;
    real r = r_sub(a, r_mulf(b, q1));
    return quick_two_sum(q1, (r.x + r.y) / b.x);
#endif
}
static inline real r_sqrt(real a) {
    float x = sqrt(a.x);
    float p = x * x;
    float e = fma(x, x, -p);
    real r = r_sub(a, float2(p, e));
    return quick_two_sum(x, r.x / (x + x));
}
static inline bool r_lt(real a, real b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }
static inline real r_max(real a, real b) { return r_lt(a, b) ? b : a; }

#else

typedef float real;
static inline real R(float2 v) { return v.x; }
static inline float2 store(real v) { return float2(v, 0.0f); }
static inline real r_zero() { return 0.0f; }
static inline real r_tiny() { return 1.0e-30f; }
static inline real r_add(real a, real b) { return a + b; }
static inline real r_add_pos(real a, real b) { return a + b; }
static inline real r_twice(real a) { return a + a; }
static inline real r_sub(real a, real b) { return a - b; }
static inline real r_mul(real a, real b) { return a * b; }
static inline real r_div(real a, real b) { return a / b; }
static inline real r_sqrt(real a) { return sqrt(a); }
static inline bool r_lt(real a, real b) { return a < b; }
static inline real r_max(real a, real b) { return max(a, b); }

#endif

static inline real norm2(real x, real y, real z) {
    return r_add_pos(r_add_pos(r_mul(x, x), r_mul(y, y)), r_mul(z, z));
}

kernel void biot_savart(
    device const Vert *verts [[buffer(0)]],
    device const Coil *coils [[buffer(1)]],
    device const Point *points [[buffer(2)]],
    device Out *out [[buffer(3)]],
    constant Params &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.count) return;
    const uint idx = p.offset + gid;
    const real px = R(points[idx].x);
    const real py = R(points[idx].y);
    const real pz = R(points[idx].z);

    real bx = r_zero(), by = r_zero(), bz = r_zero();
    bool floored = false;

    for (uint c = p.c0; c < p.c1; ++c) {
        const Coil coil = coils[c];
        const real eps = R(coil.eps);
        uint n = coil.first;
        const uint last = coil.first + coil.nvert - 1;

        Vert v = verts[n];
        real rax = r_sub(px, R(v.x));
        real ray = r_sub(py, R(v.y));
        real raz = r_sub(pz, R(v.z));
        real ra2 = norm2(rax, ray, raz);
        floored = floored || r_lt(ra2, eps);
        real ra = r_sqrt(r_max(eps, ra2));
        real ax = r_zero(), ay = r_zero(), az = r_zero();

        for (; n < last; ++n) {
            const Vert nx = verts[n + 1];
            const real rbx = r_sub(px, R(nx.x));
            const real rby = r_sub(py, R(nx.y));
            const real rbz = r_sub(pz, R(nx.z));
            const real rb2 = norm2(rbx, rby, rbz);
            const real rb = r_sqrt(r_max(eps, rb2));
            const real s = r_add_pos(ra, rb);
            const real den = r_sub(r_mul(s, s), R(v.dsq));
            floored = floored || r_lt(rb2, eps) || r_lt(den, eps);
            // One division. The clamp keeps the result finite when the
            // product underflows, which only happens on a filament.
            const real denom = r_max(r_tiny(), r_mul(r_mul(ra, rb), r_max(eps, den)));
            const real fac = r_div(r_twice(s), denom);
            const real dx = R(v.dx), dy = R(v.dy), dz = R(v.dz);
            ax = r_add(ax, r_mul(r_sub(r_mul(dy, raz), r_mul(dz, ray)), fac));
            ay = r_add(ay, r_mul(r_sub(r_mul(dz, rax), r_mul(dx, raz)), fac));
            az = r_add(az, r_mul(r_sub(r_mul(dx, ray), r_mul(dy, rax)), fac));
            rax = rbx; ray = rby; raz = rbz; ra = rb;
            v = nx;
        }
        const real w = R(coil.w);
        bx = r_add(bx, r_mul(w, ax));
        by = r_add(by, r_mul(w, ay));
        bz = r_add(bz, r_mul(w, az));
    }

    Out o;
    o.bx = store(bx);
    o.by = store(by);
    o.bz = store(bz);
    o.flag = floored ? 1.0f : 0.0f;
    o.pad = 0.0f;
    out[idx] = o;
}
