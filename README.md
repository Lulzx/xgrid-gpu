# xgrid-gpu

[![CI](https://github.com/Lulzx/xgrid-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/Lulzx/xgrid-gpu/actions/workflows/ci.yml)

Replacement for STELLOPT `MAKEGRID` (`xgrid`): computes the vacuum magnetic field of filamentary
coils on a cylindrical grid and writes `mgrid_<ext>.nc` and `extcur.<ext>` that VMEC, FIELDLINES
and the rest of STELLOPT read unchanged.

Status: release candidate (0.9.0). The CPU backend (fp64) and the Apple Metal backend
(double-single and single precision) pass every gate, including free-boundary VMEC and
FIELDLINES runs on the output. The CUDA backend (fp64, double-single, single) passes every gate
on an NVIDIA T4, with fp64 bit-identical to the CPU backend; it has not yet been measured on a
data-centre GPU (A100, H100). See [spec.md](spec.md) for the design, measurements and open
questions. To run the CUDA gate and benchmark on a Colab GPU, open
[`tools/colab_cuda.ipynb`](https://colab.research.google.com/github/Lulzx/xgrid-gpu/blob/main/tools/colab_cuda.ipynb).

201 x 201 x 36 grid, stellarator symmetry, end to end. Speedup is against legacy on the M4 Pro.

| Program | Hardware | NCSX | W7-X | Speedup (NCSX / W7-X) | Difference from fp64 |
| --- | --- | --- | --- | --- | --- |
| legacy `xgrid`, one process | Apple M4 Pro | 79 s | 71 s | 1x | |
| `xgrid-gpu`, CPU, 12 threads (default on macOS) | Apple M4 Pro | 2.1 s | 1.9 s | 38x / 37x | reference |
| `xgrid-gpu --backend metal` (double-single) | Apple M4 Pro | 2.3 s | 2.1 s | 34x / 34x | up to 4e-11 |
| `xgrid-gpu --precision fp32` (Metal) | Apple M4 Pro | 0.37 s | 0.35 s | 210x / 200x | up to 1e-3 |
| `xgrid-gpu --backend cuda` (fp64, default) | NVIDIA T4 | 7.0 s | 6.4 s | 11x / 11x | 0 (bit-identical) |
| `xgrid-gpu --backend cuda --precision ds` | NVIDIA T4 | 2.4 s | 2.2 s | 33x / 32x | up to 4e-11 |
| `xgrid-gpu --backend cuda --precision fp32` | NVIDIA T4 | 1.5 s | 1.1 s | 53x / 65x | up to 1e-3 |

On Apple silicon the GPU in double-single is no faster than the CPU in fp64, so the CPU is the
default; single precision is for quick exploratory grids. The T4, like other consumer and
inference cards, runs native fp64 at 1/32 of its fp32 rate, so there double-single is 3x faster
than fp64 at an accuracy of 4e-11; data-centre GPUs run fp64 at half rate. T4 times are from a
Google Colab VM (spec section 7.5).

## Download

Single binaries, no dependencies beyond libc (Linux, glibc 2.17 or newer) or the system
frameworks (macOS 13 or newer, Apple silicon):

```sh
curl -LO https://github.com/Lulzx/xgrid-gpu/releases/latest/download/xgrid-gpu-linux-x86_64
chmod +x xgrid-gpu-linux-x86_64 && ./xgrid-gpu-linux-x86_64 --version
```

Also `xgrid-gpu-linux-aarch64` and `xgrid-gpu-macos-arm64`; checksums in `SHA256SUMS` on the
[releases page](https://github.com/Lulzx/xgrid-gpu/releases).

## Build and test

Requires Zig 0.17.1. No other build dependencies. The binary needs only libc (Linux) or system
frameworks (macOS); an NVIDIA driver, if present, is loaded at run time.

```sh
zig build --release=fast      # zig-out/bin/xgrid-gpu, compare_mgrid, lib/libcuda_mock
zig build test                # unit tests, including analytical field checks and fuzzing
tests/run_reference.sh        # CPU backend against legacy outputs in reference/
tests/run_gpu.sh metal        # Metal against the CPU backend (macOS)
tests/run_gpu.sh cuda         # CUDA against the CPU backend (needs an NVIDIA GPU, or:)
XGRID_LIBCUDA=zig-out/lib/libcuda_mock.dylib tests/run_gpu.sh cuda   # mock driver, CPU only
```

Cross-compile for Linux with `zig build --release=fast -Dtarget=x86_64-linux-gnu`. The CUDA
kernels are embedded as PTX (`kernels/biot_savart.ptx`, `sm_60`, PTX ISA 5.0: Pascal or newer
GPU, driver from CUDA 8.0 or later). To regenerate it from `kernels/biot_savart.cu` you need a
clang with the NVPTX target, not the CUDA toolkit: `kernels/build_ptx.sh`.

## Use

```sh
# same arguments as legacy xgrid: ext mode sym rmin rmax zmin zmax kp [ir] [jz]
xgrid-gpu NCSX S Y 0.436 2.436 -1.0 1.0 36 201 201

xgrid-gpu --coils coils.NCSX --mode S --sym --rmin 0.436 --rmax 2.436 \
          --zmin -1.0 --zmax 1.0 --kp 36 --ir 201 --jz 201 \
          [--backend auto|cpu|metal|cuda] [--precision fp64|ds|fp32] [--device N]
```

A namelist file (`xgrid-gpu input.nli`) and the ten-line stdin script also work; `xgrid-gpu -h`
lists everything. With `--backend auto` (the default), xgrid-gpu uses CUDA fp64 if an NVIDIA
GPU is available and passes a startup self-test, and the CPU otherwise. If a GPU fails mid-run it
finishes on the CPU. Exit codes: 0 success, 1 bad input, 2 I/O error, 3 a requested backend could
not run.

## Difference from legacy output

Legacy `xgrid` rounds `cos(phi)` and `sin(phi)` to single precision, which shifts its field
points and projections. `xgrid-gpu` uses double precision by default, so its output differs from
legacy by about 1e-7 of the peak field on simple cases, 7e-6 on NCSX, and up to 2e-4 on a W7-X
grid that passes within a millimetre of the coils. Pass `--legacy-fp32-trig` to reproduce legacy;
with it the two agree to better than 1e-12 in every test case. The same rounding makes legacy's own
symmetric and full-period runs of a symmetric coil set disagree by 3.5e-5.

In a free-boundary VMEC run on NCSX, the equilibrium from the default-trig file differs from the
one from the legacy file by at most 5e-8 relative (spec section 7.4).

Coils given as a single line (legacy's analytic circular coil) are not supported and are reported
as an error. Grid points lying exactly on a filament produce a warning; the value there is finite
but meaningless, as it is in legacy.

## Repository

- `spec.md`: design, accuracy definitions, measurements and open questions. `spec-v0.1.0.md` is
  the original draft, kept for comparison.
- `src/`, `kernels/`: the program and its Metal and CUDA kernels.
- `tests/`: comparison tool, gate scripts and the mock CUDA driver.
- `tools/colab_cuda.ipynb`: CUDA gate and benchmark on a Google Colab GPU.
- `reference/`: legacy `xgrid` outputs at a pinned STELLOPT commit, test fixtures and their
  generators, and the VMEC and FIELDLINES downstream results. Third-party material there is
  listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Licence

MIT; see [LICENSE](LICENSE). Material under `reference/` from STELLOPT and simsopt keeps its own
MIT notices, reproduced in [THIRD_PARTY.md](THIRD_PARTY.md).
