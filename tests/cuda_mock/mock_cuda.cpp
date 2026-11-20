// Mock CUDA driver library for testing the CUDA backend on machines without
// an NVIDIA GPU. Implements the driver-API subset xgrid-gpu uses and runs the
// kernels from kernels/biot_savart.cu, compiled as host C++, on CPU threads.
//
// What this verifies: the host code's dlopen and symbol binding, struct
// layouts, argument marshalling, chunked launches, error reporting, and the
// kernel arithmetic (fp64, double-single and fp32 compiled with IEEE float
// semantics). What it cannot verify: PTX code generation, the real driver's
// JIT, or GPU performance.
//
// Fault injection, by environment variable XG_MOCK_FAIL:
//   nodevice   cuDeviceGetCount reports zero devices
//   ptx        cuModuleLoadDataEx fails with CUDA_ERROR_UNSUPPORTED_PTX_VERSION
//   launch     the second kernel launch fails with CUDA_ERROR_ILLEGAL_ADDRESS
//   wrong      the fp64/ds kernels return slightly wrong results (self-test check)

#define XG_HOST_EMULATION 1
#include "biot_savart.cu"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

thread_local xg_dim xg_block_idx, xg_block_dim, xg_thread_idx;

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;

namespace {
enum Kernel { K_FP64 = 1, K_DS = 2, K_FP32 = 3 };
struct Func { Kernel k; };
Func funcs[4] = {{K_FP64}, {K_FP64}, {K_DS}, {K_FP32}};
int module_token;
int context_token;
std::atomic<int> launches{0};
std::string module_text;

bool fail(const char *what) {
    const char *f = std::getenv("XG_MOCK_FAIL");
    return f && std::strcmp(f, what) == 0;
}
}  // namespace

extern "C" {

CUresult cuInit(unsigned) { return 0; }
CUresult cuDriverGetVersion(int *v) { *v = 12040; return 0; }
CUresult cuDeviceGetCount(int *n) { *n = fail("nodevice") ? 0 : 1; return 0; }
CUresult cuDeviceGet(CUdevice *d, int i) { if (i != 0) return 101; *d = 0; return 0; }
CUresult cuDeviceGetName(char *name, int len, CUdevice) {
    std::strncpy(name, "xgrid-gpu mock CUDA device (CPU)", len);
    return 0;
}
CUresult cuDeviceGetAttribute(int *v, int, CUdevice) { *v = 7; return 0; }
CUresult cuCtxCreate_v2(void **ctx, unsigned, CUdevice) { *ctx = &context_token; return 0; }
CUresult cuCtxDestroy_v2(void *) { return 0; }
CUresult cuCtxSynchronize() {
    if (fail("launch") && launches.load() >= 2) return 700;
    return 0;
}

CUresult cuModuleLoadDataEx(void **mod, const void *image, unsigned, int *, void **) {
    if (fail("ptx")) return 222;
    // The real driver JIT-compiles the PTX; here, check that it is our module.
    module_text = static_cast<const char *>(image);
    for (const char *e : {".entry biot_savart_fp64(", ".entry biot_savart_ds(", ".entry biot_savart_fp32("})
        if (module_text.find(e) == std::string::npos) return 218;  // CUDA_ERROR_INVALID_PTX
    *mod = &module_token;
    return 0;
}
CUresult cuModuleUnload(void *) { return 0; }
CUresult cuModuleGetFunction(void **f, void *, const char *name) {
    if (!std::strcmp(name, "biot_savart_fp64")) *f = &funcs[1];
    else if (!std::strcmp(name, "biot_savart_ds")) *f = &funcs[2];
    else if (!std::strcmp(name, "biot_savart_fp32")) *f = &funcs[3];
    else return 500;  // CUDA_ERROR_NOT_FOUND
    return 0;
}

CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t n) {
    void *m = std::malloc(n);
    if (!m) return 2;
    *p = reinterpret_cast<CUdeviceptr>(m);
    return 0;
}
CUresult cuMemFree_v2(CUdeviceptr p) { std::free(reinterpret_cast<void *>(p)); return 0; }
CUresult cuMemcpyHtoD_v2(CUdeviceptr d, const void *h, size_t n) { std::memcpy(reinterpret_cast<void *>(d), h, n); return 0; }
CUresult cuMemcpyDtoH_v2(void *h, CUdeviceptr d, size_t n) { std::memcpy(h, reinterpret_cast<void *>(d), n); return 0; }

CUresult cuLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                        unsigned shared, void *stream, void **params, void **extra) {
    if (!f || gy != 1 || gz != 1 || by != 1 || bz != 1 || bx == 0 || bx > 1024 || shared || stream || extra || !params)
        return 1;  // CUDA_ERROR_INVALID_VALUE
    launches++;
    const Kernel k = static_cast<Func *>(f)->k;
    void *verts = reinterpret_cast<void *>(*static_cast<CUdeviceptr *>(params[0]));
    void *coils = reinterpret_cast<void *>(*static_cast<CUdeviceptr *>(params[1]));
    void *points = reinterpret_cast<void *>(*static_cast<CUdeviceptr *>(params[2]));
    Out *out = reinterpret_cast<Out *>(*static_cast<CUdeviceptr *>(params[3]));
    const Params p = *static_cast<Params *>(params[4]);
    if (static_cast<unsigned long long>(gx) * bx < p.count) return 1;

    std::atomic<unsigned> next{0};
    auto worker = [&]() {
        for (unsigned b; (b = next++) < gx;) {
            xg_block_idx.x = b;
            xg_block_dim.x = bx;
            for (unsigned t = 0; t < bx; ++t) {
                xg_thread_idx.x = t;
                switch (k) {
                case K_FP64: biot_savart_fp64((const VertD *)verts, (const CoilD *)coils, (const PointD *)points, out, p); break;
                case K_DS: biot_savart_ds((const VertF *)verts, (const CoilF *)coils, (const PointF *)points, out, p); break;
                case K_FP32: biot_savart_fp32((const VertF *)verts, (const CoilF *)coils, (const PointF *)points, out, p); break;
                }
            }
        }
    };
    unsigned nt = std::thread::hardware_concurrency();
    if (nt == 0) nt = 4;
    std::vector<std::thread> pool;
    for (unsigned i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto &t : pool) t.join();

    if (fail("wrong") && k != K_FP32)
        for (unsigned i = 0; i < p.count; ++i) out[p.offset + i].bx *= 1.0 + 1.0e-9;
    return 0;
}

CUresult cuGetErrorString(CUresult r, const char **s) {
    switch (r) {
    case 1: *s = "CUDA_ERROR_INVALID_VALUE"; break;
    case 218: *s = "CUDA_ERROR_INVALID_PTX"; break;
    case 222: *s = "CUDA_ERROR_UNSUPPORTED_PTX_VERSION"; break;
    case 500: *s = "CUDA_ERROR_NOT_FOUND"; break;
    case 700: *s = "CUDA_ERROR_ILLEGAL_ADDRESS"; break;
    default: *s = "CUDA error (mock)"; break;
    }
    return 0;
}

}  // extern "C"
