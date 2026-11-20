// Metal backend bridge: compiles the embedded kernel at startup and runs it in
// time-bounded chunks (spec sections 4.2, 4.3, 4.5).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>
#include "metal_bridge.h"

typedef struct { float x[2], y[2], z[2], dx[2], dy[2], dz[2], dsq[2], pad[2]; } XGVert;
typedef struct { uint32_t first, nvert; float w[2], eps[2]; } XGCoil;
typedef struct { float x[2], y[2], z[2], pad[2]; } XGPoint;
typedef struct { float bx[2], by[2], bz[2]; float flag, pad; } XGOut;
typedef struct { uint32_t c0, c1, offset, count; } XGParams;

_Static_assert(sizeof(XGVert) == 64, "Vert layout");
_Static_assert(sizeof(XGCoil) == 24, "Coil layout");
_Static_assert(sizeof(XGPoint) == 32, "Point layout");
_Static_assert(sizeof(XGOut) == 32, "Out layout");

struct xg_metal {
    void *device, *queue, *pipeline;       // retained Objective-C objects
    void *verts, *coils, *points, *out;
    uint32_t npoints;
    int ds;
    char name[128];
};

static void set_err(char *err, size_t errlen, NSString *msg) {
    if (err && errlen) {
        strncpy(err, msg.UTF8String ?: "unknown error", errlen - 1);
        err[errlen - 1] = 0;
    }
}

static inline void split(double v, float out[2], int ds) {
    float hi = (float)v;
    out[0] = hi;
    out[1] = ds ? (float)(v - (double)hi) : 0.0f;
}

#define OBJ(type, field) ((__bridge type)(field))

xg_metal *xg_metal_create(const char *source, size_t source_len, int ds, unsigned index, char *err, size_t errlen) {
    @autoreleasepool {
        id<MTLDevice> device = nil;
        if (index == 0) {
            device = MTLCreateSystemDefaultDevice();
        } else {
            NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
            if (index < all.count) device = all[index];
            else {
                set_err(err, errlen, [NSString stringWithFormat:@"Metal device %u requested but %lu present", index, (unsigned long)all.count]);
                return NULL;
            }
        }
        if (!device) {
            set_err(err, errlen, @"no Metal device available");
            return NULL;
        }
        NSString *src = [[NSString alloc] initWithBytes:source length:source_len encoding:NSUTF8StringEncoding];
        MTLCompileOptions *opts = [MTLCompileOptions new];
        // Error-free transformations require strict IEEE evaluation.
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        opts.preprocessorMacros = @{ @"XG_DS" : ds ? @1 : @0 };
        NSError *e = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&e];
        if (!lib) {
            set_err(err, errlen, [NSString stringWithFormat:@"Metal kernel compilation failed: %@", e.localizedDescription]);
            return NULL;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"biot_savart"];
        id<MTLComputePipelineState> pipeline = fn ? [device newComputePipelineStateWithFunction:fn error:&e] : nil;
        if (!pipeline) {
            set_err(err, errlen, [NSString stringWithFormat:@"Metal pipeline creation failed: %@", e.localizedDescription]);
            return NULL;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        xg_metal *m = calloc(1, sizeof(*m));
        m->device = (__bridge_retained void *)device;
        m->queue = (__bridge_retained void *)queue;
        m->pipeline = (__bridge_retained void *)pipeline;
        m->ds = ds;
        strncpy(m->name, device.name.UTF8String ?: "unknown", sizeof(m->name) - 1);
        return m;
    }
}

static void release(void **p) {
    if (*p) {
        CFRelease(*p);
        *p = NULL;
    }
}

void xg_metal_destroy(xg_metal *m) {
    if (!m) return;
    release(&m->verts);
    release(&m->coils);
    release(&m->points);
    release(&m->out);
    release(&m->pipeline);
    release(&m->queue);
    release(&m->device);
    free(m);
}

const char *xg_metal_device_name(const xg_metal *m) { return m->name; }

static void *new_buffer(xg_metal *m, size_t bytes, char *err, size_t errlen) {
    id<MTLBuffer> b = [OBJ(id<MTLDevice>, m->device) newBufferWithLength:(bytes ? bytes : 16)
                                                                 options:MTLResourceStorageModeShared];
    if (!b) {
        set_err(err, errlen, [NSString stringWithFormat:@"cannot allocate a %zu-byte Metal buffer", bytes]);
        return NULL;
    }
    return (__bridge_retained void *)b;
}

int xg_metal_set_geometry(xg_metal *m,
                          const double *x, const double *y, const double *z,
                          const double *dx, const double *dy, const double *dz,
                          const double *dsq, uint32_t nvert,
                          const uint32_t *coil_first, const uint32_t *coil_nvert,
                          const double *coil_w, const double *coil_eps, uint32_t ncoil,
                          char *err, size_t errlen) {
    @autoreleasepool {
        release(&m->verts);
        release(&m->coils);
        m->verts = new_buffer(m, (size_t)nvert * sizeof(XGVert), err, errlen);
        m->coils = new_buffer(m, (size_t)ncoil * sizeof(XGCoil), err, errlen);
        if (!m->verts || !m->coils) return -1;
        XGVert *v = OBJ(id<MTLBuffer>, m->verts).contents;
        for (uint32_t i = 0; i < nvert; ++i) {
            split(x[i], v[i].x, m->ds);
            split(y[i], v[i].y, m->ds);
            split(z[i], v[i].z, m->ds);
            split(dx[i], v[i].dx, m->ds);
            split(dy[i], v[i].dy, m->ds);
            split(dz[i], v[i].dz, m->ds);
            split(dsq[i], v[i].dsq, m->ds);
            v[i].pad[0] = v[i].pad[1] = 0;
        }
        XGCoil *c = OBJ(id<MTLBuffer>, m->coils).contents;
        for (uint32_t i = 0; i < ncoil; ++i) {
            c[i].first = coil_first[i];
            c[i].nvert = coil_nvert[i];
            split(coil_w[i], c[i].w, m->ds);
            split(coil_eps[i], c[i].eps, m->ds);
        }
        return 0;
    }
}

int xg_metal_set_points(xg_metal *m, const double *px, const double *py, const double *pz,
                        uint32_t npoints, char *err, size_t errlen) {
    @autoreleasepool {
        release(&m->points);
        release(&m->out);
        m->points = new_buffer(m, (size_t)npoints * sizeof(XGPoint), err, errlen);
        m->out = new_buffer(m, (size_t)npoints * sizeof(XGOut), err, errlen);
        if (!m->points || !m->out) return -1;
        XGPoint *p = OBJ(id<MTLBuffer>, m->points).contents;
        for (uint32_t i = 0; i < npoints; ++i) {
            split(px[i], p[i].x, m->ds);
            split(py[i], p[i].y, m->ds);
            split(pz[i], p[i].z, m->ds);
            p[i].pad[0] = p[i].pad[1] = 0;
        }
        m->npoints = npoints;
        return 0;
    }
}

int xg_metal_compute(xg_metal *m, uint32_t c0, uint32_t c1, uint64_t nseg,
                     double *bx, double *by, double *bz, uint8_t *flags,
                     double *gpu_seconds, char *err, size_t errlen) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = OBJ(id<MTLComputePipelineState>, m->pipeline);
        id<MTLCommandQueue> queue = OBJ(id<MTLCommandQueue>, m->queue);
        const NSUInteger group = pipeline.maxTotalThreadsPerThreadgroup;
        const uint32_t n = m->npoints;

        // Chunks are sized to run for roughly `target` seconds each so a
        // dispatch never trips the GPU watchdog or stalls the display. The
        // first chunk is small and calibrates the rate.
        const double target = 0.15;
        double rate = 0;  // interactions per second, measured
        uint32_t offset = 0;
        const uint64_t segs = nseg ? nseg : 1;
        while (offset < n) {
            uint64_t want = rate > 0 ? (uint64_t)(rate * target / (double)segs) : (uint64_t)(2.0e7 / (double)segs);
            if (want < group) want = group;
            if (want > n - offset) want = n - offset;
            const uint32_t count = (uint32_t)want;

            XGParams params = { c0, c1, offset, count };
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:OBJ(id<MTLBuffer>, m->verts) offset:0 atIndex:0];
            [enc setBuffer:OBJ(id<MTLBuffer>, m->coils) offset:0 atIndex:1];
            [enc setBuffer:OBJ(id<MTLBuffer>, m->points) offset:0 atIndex:2];
            [enc setBuffer:OBJ(id<MTLBuffer>, m->out) offset:0 atIndex:3];
            [enc setBytes:&params length:sizeof(params) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) {
                set_err(err, errlen, [NSString stringWithFormat:@"Metal command buffer failed: %@",
                                                                cb.error.localizedDescription ?: @"unknown error"]);
                return -1;
            }
            const double dt = cb.GPUEndTime - cb.GPUStartTime;
            if (gpu_seconds) *gpu_seconds += dt;
            if (dt > 0) rate = (double)count * (double)segs / dt;
            offset += count;
        }

        const XGOut *o = OBJ(id<MTLBuffer>, m->out).contents;
        for (uint32_t i = 0; i < n; ++i) {
            bx[i] = (double)o[i].bx[0] + (double)o[i].bx[1];
            by[i] = (double)o[i].by[0] + (double)o[i].by[1];
            bz[i] = (double)o[i].bz[0] + (double)o[i].bz[1];
            flags[i] = o[i].flag != 0.0f;
        }
        return 0;
    }
}
