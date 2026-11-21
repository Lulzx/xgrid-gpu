# xgrid-gpu

[![CI](https://github.com/Lulzx/xgrid-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/Lulzx/xgrid-gpu/actions/workflows/ci.yml)

Replacement for STELLOPT `MAKEGRID` (`xgrid`): computes the vacuum magnetic field of filamentary
coils on a cylindrical grid and writes `mgrid_<ext>.nc` and `extcur.<ext>` that VMEC, FIELDLINES
and the rest of STELLOPT read unchanged.

Status: release candidate (0.9.1). The CPU backend (fp64) and the Apple Metal backend
(double-single and single precision) pass every gate, including free-boundary VMEC and
FIELDLINES runs on the output. The CUDA backend (fp64, double-single, single) passes every gate
on an NVIDIA A100 and an NVIDIA T4, with fp64 bit-identical to the CPU backend. See [spec.md](spec.md) for the design, measurements and open
questions. To run the CUDA gate and benchmark on a Colab GPU, open
[`tools/colab_cuda.ipynb`](https://colab.research.google.com/github/Lulzx/xgrid-gpu/blob/main/tools/colab_cuda.ipynb).

201 x 201 x 36 grid, stellarator symmetry, end to end. Speedup is against legacy on the M4 Pro.

| Program | Hardware | NCSX | W7-X | Speedup (NCSX / W7-X) | Difference from fp64 |
| --- | --- | --- | --- | --- | --- |
| legacy `xgrid`, one process | Apple M4 Pro | 79 s | 71 s | 1x | |
| `xgrid-gpu`, CPU, 12 threads (default on macOS) | Apple M4 Pro | 2.1 s | 1.9 s | 38x / 37x | reference |
| `xgrid-gpu --backend metal` (double-single) | Apple M4 Pro | 2.3 s | 2.1 s | 34x / 34x | up to 4e-11 |
| `xgrid-gpu --precision fp32` (Metal) | Apple M4 Pro | 0.37 s | 0.35 s | 210x / 200x | up to 1e-3 |
| `xgrid-gpu --backend cuda` (fp64, default) | NVIDIA A100 | 1.1 s | 1.0 s | 72x / 71x | 0 (bit-identical) |
| `xgrid-gpu --backend cuda --precision ds` | NVIDIA A100 | 1.4 s | 1.3 s | 58x / 56x | up to 4e-11 |
| `xgrid-gpu --backend cuda --precision fp32` | NVIDIA A100 | 0.99 s | 0.89 s | 80x / 80x | up to 1e-3 |
| `xgrid-gpu --backend cuda` (fp64, default) | NVIDIA T4 | 7.0 s | 6.4 s | 11x / 11x | 0 (bit-identical) |
| `xgrid-gpu --backend cuda --precision ds` | NVIDIA T4 | 2.4 s | 2.2 s | 33x / 32x | up to 4e-11 |
| `xgrid-gpu --backend cuda --precision fp32` | NVIDIA T4 | 1.5 s | 1.1 s | 53x / 66x | up to 1e-3 |

On Apple silicon the GPU in double-single is no faster than the CPU in fp64, so the CPU is the
default; single precision is for quick exploratory grids. On the A100 the fp64 kernel takes
0.17 s of the 1.1 s; the rest is CUDA start-up (about 0.6 s) and writing the 349 MB file. The T4,
like other consumer and inference cards, runs native fp64 at 1/32 of its fp32 rate, so there
double-single is 3x faster than fp64 at an accuracy of 4e-11; on the A100, which runs fp64 at
half rate, fp64 is faster. NVIDIA times are from Google Colab VMs (spec section 7.5).

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
zig build --release=fast        # zig-out/bin/xgrid-gpu, compare_mgrid, lib/libcuda_mock
zig build test                  # unit tests, including analytical field checks and fuzzing
tests/run_reference.sh          # CPU backend against legacy outputs in reference/
tests/run_gpu.sh metal          # Metal against the CPU backend (macOS)
tests/run_gpu.sh cuda           # CUDA against the CPU backend (needs an NVIDIA GPU, or:)
XGRID_LIBCUDA=zig-out/lib/libcuda_mock.dylib tests/run_gpu.sh cuda   # mock driver, CPU only
```

Cross-compile for Linux with `zig build --release=fast -Dtarget=x86_64-linux-gnu`. The CUDA
kernels are embedded as PTX (`kernels/biot_savart.ptx`, `sm_60`, PTX ISA 5.0: Pascal or newer
GPU, driver from CUDA 8.0 or later). To regenerate it from `kernels/biot_savart.cu` you need a
clang with the NVPTX target, not the CUDA toolkit: `kernels/build_ptx.sh`.

### On a Google Colab GPU

[`tools/colab_cuda.ipynb`](tools/colab_cuda.ipynb) builds xgrid-gpu on a Colab VM, runs
`tests/run_gpu.sh cuda`, and times NCSX and W7-X at `201 x 201 x 36` in fp64, ds and fp32 against
the CPU backend. In the browser, open it
[in Colab](https://colab.research.google.com/github/Lulzx/xgrid-gpu/blob/main/tools/colab_cuda.ipynb),
pick a GPU runtime and run all cells. From a terminal, with the
[Colab CLI](https://github.com/googlecolab/google-colab-cli):

```sh
uv tool install google-colab-cli
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory

colab --auth=adc new -s xgrid --gpu T4                 # or L4, A100, H100 (Colab Pro)
colab --auth=adc exec -s xgrid --timeout 1200 -f tools/colab_cuda.ipynb
colab --auth=adc stop -s xgrid                         # release the VM
```

`exec` prints each cell's output and writes the executed notebook to
`tools/colab_cuda_output.ipynb`. `--timeout` is needed because the build cell is silent for
longer than the CLI's default 30 s. The VM is billed from `new` to `stop`; a full run takes about
five minutes. Executed notebooks from the runs in spec section 7.5 are in `reference/nvidia/`.

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
- `.github/workflows/`: CI on every push; `release.yml` builds and publishes the binaries when a
  `v*` tag is pushed (procedure in spec section 6).
- `reference/`: legacy `xgrid` outputs at a pinned STELLOPT commit, test fixtures and their
  generators, and the VMEC and FIELDLINES downstream results. Third-party material there is
  listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Licence

MIT; see [LICENSE](LICENSE). Material under `reference/` from STELLOPT and simsopt keeps its own
MIT notices, reproduced in [THIRD_PARTY.md](THIRD_PARTY.md).
