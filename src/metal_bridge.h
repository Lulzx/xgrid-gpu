// C interface to the Metal backend (implemented in metal_bridge.m).
#ifndef XG_METAL_BRIDGE_H
#define XG_METAL_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

typedef struct xg_metal xg_metal;

// Compile `source` for the default GPU. `ds` selects double-single (1) or
// plain float (0). Returns NULL and fills `err` on failure.
xg_metal *xg_metal_create(const char *source, size_t source_len, int ds, unsigned device, char *err, size_t errlen);
void xg_metal_destroy(xg_metal *m);
const char *xg_metal_device_name(const xg_metal *m);

// Geometry in fp64; converted to {hi, lo} float pairs internally.
int xg_metal_set_geometry(xg_metal *m,
                          const double *x, const double *y, const double *z,
                          const double *dx, const double *dy, const double *dz,
                          const double *dsq, uint32_t nvert,
                          const uint32_t *coil_first, const uint32_t *coil_nvert,
                          const double *coil_w, const double *coil_eps, uint32_t ncoil,
                          char *err, size_t errlen);

int xg_metal_set_points(xg_metal *m, const double *px, const double *py, const double *pz,
                        uint32_t npoints, char *err, size_t errlen);

// Cartesian field of coils [c0, c1) at every point. `flags[i]` is non-zero if
// a floor was engaged at point i. `gpu_seconds` accumulates GPU execution time.
int xg_metal_compute(xg_metal *m, uint32_t c0, uint32_t c1, uint64_t nseg,
                     double *bx, double *by, double *bz, uint8_t *flags,
                     double *gpu_seconds, char *err, size_t errlen);

#endif
