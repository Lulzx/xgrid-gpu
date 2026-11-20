// xgrid-gpu Biot-Savart kernels for CUDA (spec sections 3.2, 4.1, 4.3).
//
// Three entry points share one loop body:
//   biot_savart_fp64  native double
//   biot_savart_ds    double-single: hi + lo of two floats, about 48 bits
//   biot_savart_fp32  plain float (exploratory)
//
// Built two ways from this file:
//   * PTX for the GPU (kernels/build_ptx.sh, clang -x cuda --cuda-device-only).
//     No CUDA toolkit headers are used, so nvcc is not needed.
//   * Host C++ with XG_HOST_EMULATION defined, for the mock CUDA driver in
//     tests/cuda_mock/, which runs the kernels on the CPU so the host-side
//     backend can be tested without an NVIDIA GPU.
//
// Must be compiled with -ffp-contract=off: the double-single error-free
// transformations are algebraically "zero" and contraction or fast-math
// would silently reduce them to float accuracy.

#include <stdint.h>

#if defined(XG_HOST_EMULATION)
#include <math.h>
#define XG_GLOBAL
#define XG_DEVICE static inline
struct xg_dim { unsigned x; };
extern thread_local xg_dim xg_block_idx, xg_block_dim, xg_thread_idx;
static inline unsigned xg_tid() { return xg_block_idx.x * xg_block_dim.x + xg_thread_idx.x; }
#else
#define XG_GLOBAL __attribute__((global))
#define XG_DEVICE static __attribute__((device)) inline
XG_DEVICE unsigned xg_tid() {
    return __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() + __nvvm_read_ptx_sreg_tid_x();
}
#endif

// ---- data layouts (must match src/cuda_host.zig) -------------------------

struct VertD { double x, y, z, dx, dy, dz, dsq, pad; };          // 64 bytes
struct VertF { float x[2], y[2], z[2], dx[2], dy[2], dz[2], dsq[2], pad[2]; };  // 64 bytes, {hi, lo}
struct CoilD { uint32_t first, nvert; double w, eps; };           // 24 bytes
struct CoilF { uint32_t first, nvert; float w[2], eps[2]; };      // 24 bytes
struct PointD { double x, y, z, pad; };                           // 32 bytes
struct PointF { float x[2], y[2], z[2], pad[2]; };                // 32 bytes
struct Out { double bx, by, bz; uint32_t flag, pad; };           // 32 bytes
struct Params { uint32_t c0, c1, offset, count; };

// ---- arithmetic ----------------------------------------------------------

struct F64 {
    double v;
    XG_DEVICE F64 load(double x) { return F64{x}; }
    XG_DEVICE double out(F64 a) { return a.v; }
    XG_DEVICE F64 zero() { return F64{0.0}; }
    XG_DEVICE F64 tiny() { return F64{1.0e-300}; }
    XG_DEVICE F64 add(F64 a, F64 b) { return F64{a.v + b.v}; }
    XG_DEVICE F64 add_pos(F64 a, F64 b) { return F64{a.v + b.v}; }
    XG_DEVICE F64 sub(F64 a, F64 b) { return F64{a.v - b.v}; }
    XG_DEVICE F64 mul(F64 a, F64 b) { return F64{a.v * b.v}; }
    XG_DEVICE F64 div(F64 a, F64 b) { return F64{a.v / b.v}; }
    XG_DEVICE F64 sqrt_(F64 a) { return F64{__builtin_sqrt(a.v)}; }
    XG_DEVICE F64 twice(F64 a) { return F64{a.v + a.v}; }
    XG_DEVICE bool lt(F64 a, F64 b) { return a.v < b.v; }
    XG_DEVICE F64 max_(F64 a, F64 b) { return a.v < b.v ? b : a; }
};

struct F32 {
    float v;
    XG_DEVICE F32 load(const float p[2]) { return F32{p[0]}; }
    XG_DEVICE double out(F32 a) { return (double)a.v; }
    XG_DEVICE F32 zero() { return F32{0.0f}; }
    XG_DEVICE F32 tiny() { return F32{1.0e-30f}; }
    XG_DEVICE F32 add(F32 a, F32 b) { return F32{a.v + b.v}; }
    XG_DEVICE F32 add_pos(F32 a, F32 b) { return F32{a.v + b.v}; }
    XG_DEVICE F32 sub(F32 a, F32 b) { return F32{a.v - b.v}; }
    XG_DEVICE F32 mul(F32 a, F32 b) { return F32{a.v * b.v}; }
    XG_DEVICE F32 div(F32 a, F32 b) { return F32{a.v / b.v}; }
    XG_DEVICE F32 sqrt_(F32 a) { return F32{__builtin_sqrtf(a.v)}; }
    XG_DEVICE F32 twice(F32 a) { return F32{a.v + a.v}; }
    XG_DEVICE bool lt(F32 a, F32 b) { return a.v < b.v; }
    XG_DEVICE F32 max_(F32 a, F32 b) { return a.v < b.v ? b : a; }
};

// Double-single, finite-only Dekker/Knuth constructions; the same operations
// as kernels/biot_savart.metal.
struct DS {
    float hi, lo;
    XG_DEVICE DS make(float h, float l) { DS r; r.hi = h; r.lo = l; return r; }
    XG_DEVICE DS two_sum(float a, float b) {
        float s = a + b;
        float bb = s - a;
        return make(s, (a - (s - bb)) + (b - bb));
    }
    XG_DEVICE DS quick_two_sum(float a, float b) {
        float s = a + b;
        return make(s, b - (s - a));
    }
    XG_DEVICE DS load(const float p[2]) { return make(p[0], p[1]); }
    XG_DEVICE double out(DS a) { return (double)a.hi + (double)a.lo; }
    XG_DEVICE DS zero() { return make(0.0f, 0.0f); }
    XG_DEVICE DS tiny() { return make(1.0e-30f, 0.0f); }
    XG_DEVICE DS add(DS a, DS b) {
        DS s = two_sum(a.hi, b.hi);
        DS t = two_sum(a.lo, b.lo);
        DS u = quick_two_sum(s.hi, s.lo + t.hi);
        return quick_two_sum(u.hi, u.lo + t.lo);
    }
    XG_DEVICE DS add_pos(DS a, DS b) {
        DS s = two_sum(a.hi, b.hi);
        return quick_two_sum(s.hi, s.lo + (a.lo + b.lo));
    }
    XG_DEVICE DS sub(DS a, DS b) { return add(a, make(-b.hi, -b.lo)); }
    XG_DEVICE DS mul(DS a, DS b) {
        float p = a.hi * b.hi;
        float e = __builtin_fmaf(a.hi, b.hi, -p);
        e = __builtin_fmaf(a.hi, b.lo, e);
        e = __builtin_fmaf(a.lo, b.hi, e);
        return quick_two_sum(p, e);
    }
    XG_DEVICE DS mulf(DS a, float b) {
        float p = a.hi * b;
        float e = __builtin_fmaf(a.hi, b, -p);
        e = __builtin_fmaf(a.lo, b, e);
        return quick_two_sum(p, e);
    }
    XG_DEVICE DS div(DS a, DS b) {
        float q1 = a.hi / b.hi;
        DS r = sub(a, mulf(b, q1));
        return quick_two_sum(q1, (r.hi + r.lo) / b.hi);
    }
    XG_DEVICE DS sqrt_(DS a) {
        float x = __builtin_sqrtf(a.hi);
        float p = x * x;
        float e = __builtin_fmaf(x, x, -p);
        DS r = sub(a, make(p, e));
        return quick_two_sum(x, r.hi / (x + x));
    }
    XG_DEVICE DS twice(DS a) { return make(a.hi + a.hi, a.lo + a.lo); }
    XG_DEVICE bool lt(DS a, DS b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
    XG_DEVICE DS max_(DS a, DS b) { return lt(a, b) ? b : a; }
};

// ---- the loop (spec section 4.1) -----------------------------------------

template <class T, class Vert, class Coil, class Point>
XG_DEVICE void body(const Vert *verts, const Coil *coils, const Point *points, Out *out, Params p) {
    const unsigned gid = xg_tid();
    if (gid >= p.count) return;
    const unsigned idx = p.offset + gid;
    const T px = T::load(points[idx].x);
    const T py = T::load(points[idx].y);
    const T pz = T::load(points[idx].z);

    T bx = T::zero(), by = T::zero(), bz = T::zero();
    bool floored = false;

    for (unsigned c = p.c0; c < p.c1; ++c) {
        const Coil coil = coils[c];
        const T eps = T::load(coil.eps);
        unsigned n = coil.first;
        const unsigned last = coil.first + coil.nvert - 1;

        const Vert *v = &verts[n];
        T rax = T::sub(px, T::load(v->x));
        T ray = T::sub(py, T::load(v->y));
        T raz = T::sub(pz, T::load(v->z));
        T ra2 = T::add_pos(T::add_pos(T::mul(rax, rax), T::mul(ray, ray)), T::mul(raz, raz));
        floored = floored || T::lt(ra2, eps);
        T ra = T::sqrt_(T::max_(eps, ra2));
        T ax = T::zero(), ay = T::zero(), az = T::zero();

        for (; n < last; ++n) {
            const Vert *nx = &verts[n + 1];
            const T rbx = T::sub(px, T::load(nx->x));
            const T rby = T::sub(py, T::load(nx->y));
            const T rbz = T::sub(pz, T::load(nx->z));
            const T rb2 = T::add_pos(T::add_pos(T::mul(rbx, rbx), T::mul(rby, rby)), T::mul(rbz, rbz));
            const T rb = T::sqrt_(T::max_(eps, rb2));
            const T s = T::add_pos(ra, rb);
            const T den = T::sub(T::mul(s, s), T::load(v->dsq));
            floored = floored || T::lt(rb2, eps) || T::lt(den, eps);
            const T denom = T::max_(T::tiny(), T::mul(T::mul(ra, rb), T::max_(eps, den)));
            const T fac = T::div(T::twice(s), denom);
            const T dx = T::load(v->dx), dy = T::load(v->dy), dz = T::load(v->dz);
            ax = T::add(ax, T::mul(T::sub(T::mul(dy, raz), T::mul(dz, ray)), fac));
            ay = T::add(ay, T::mul(T::sub(T::mul(dz, rax), T::mul(dx, raz)), fac));
            az = T::add(az, T::mul(T::sub(T::mul(dx, ray), T::mul(dy, rax)), fac));
            rax = rbx; ray = rby; raz = rbz; ra = rb;
            v = nx;
        }
        const T w = T::load(coil.w);
        bx = T::add(bx, T::mul(w, ax));
        by = T::add(by, T::mul(w, ay));
        bz = T::add(bz, T::mul(w, az));
    }

    Out o;
    o.bx = T::out(bx);
    o.by = T::out(by);
    o.bz = T::out(bz);
    o.flag = floored ? 1u : 0u;
    o.pad = 0u;
    out[idx] = o;
}

extern "C" XG_GLOBAL void biot_savart_fp64(const VertD *verts, const CoilD *coils, const PointD *points, Out *out, Params p) {
    body<F64>(verts, coils, points, out, p);
}
extern "C" XG_GLOBAL void biot_savart_ds(const VertF *verts, const CoilF *coils, const PointF *points, Out *out, Params p) {
    body<DS>(verts, coils, points, out, p);
}
extern "C" XG_GLOBAL void biot_savart_fp32(const VertF *verts, const CoilF *coils, const PointF *points, Out *out, Params p) {
    body<F32>(verts, coils, points, out, p);
}
