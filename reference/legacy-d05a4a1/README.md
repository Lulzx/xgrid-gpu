# Legacy reference outputs

Reference `mgrid_*.nc` / `extcur.*` files from STELLOPT `MAKEGRID` (`xgrid`), used to pin the
legacy behaviour quoted in `spec.md`.

- Source: `PrincetonUniversity/STELLOPT`, branch `develop`,
  commit `d05a4a11231b191544016f24be1b2ead61a12c8f` (2026-11-08).
- Built 2026-11-20 on macOS arm64 with GNU Fortran 16.2.0 (Homebrew), Open MPI, netcdf-fortran 4.6.4.
- Build config: `make_xgridref.inc`, which is STELLOPT's `SHARE/make_osx_brew.inc` with FFTW, HDF5,
  PGPLOT, DKES/NEO and ScaLAPACK switched off. Compiler flags are unchanged:
  `-O2 -g -fexternal-blas -march=native -fallow-argument-mismatch`. No real-promotion flag.
- To rebuild: copy `make_xgridref.inc` into `STELLOPT/SHARE/`, set `STELLOPT_PATH`,
  `MACHINE=xgridref`, `STELLOPT_HOME`, then `make clean_release` in `LIBSTELL/` and `MAKEGRID/`.

## Contents

- `gen_coils.py <dir>`: writes the synthetic `coils.two`, `coils.one`, `coils.eleven` fixtures.
- `gen_probes.py <dir>`: writes the edge-case fixtures `coils.onwire` (grid points on a filament),
  `coils.twovert` (a two-line coil) and `coils.many` (101 groups).
- `gen_probes.py` also writes `coils.poly`: a 10,000-sided polygon with a grid point on a vertex.
- `gen_w7x.py W7-X.dat <dir>`: writes `coils.w7x`, the W7-X coil set rebuilt from the Fourier
  coefficients in simsopt (MIT licence, `src/simsopt/configs/W7-X.dat` at commit 912bdcc). See
  the script header for conventions; it is a test fixture, not an authoritative W7-X file.
- `coils.NCSX` (in `cases/ncsx_S_Y/`) is copied from STELLOPT `BENCHMARKS/FIELDLINES_TEST/`.
- `cases/<name>/`: coils file, legacy `mgrid`, `extcur`, console output and `ncdump -h` header.
  `cases/ARGS.txt` lists the arguments for each.
- `oracle.py`: numpy `fp64` prototype of spec sections 2.4 to 3.4 with a minimal CDF reader.
  `python3 oracle.py <coils> <mgrid.nc> <S|R> <Y|N> rmin rmax zmin zmax kp ir jz` prints the
  normalised difference from the legacy file, with double trig and with emulated
  single-precision `cos`/`sin`.

## Results (2026-11-20)

Worst normalised difference (spec section 1.2) between legacy and the prototype:

| Cases | Emulated single-precision trig | Double trig |
| --- | --- | --- |
| six synthetic cases | below 1e-15 | about 1e-7 |
| `onwire`, `twovert`, `many` | below 3e-15 | same (phi = 0 and pi only) |
| NCSX, 41 x 41 x 12, S mode, symmetry on | 1.5e-14 | 7.3e-6 |

Legacy behaviour seen in the probes: output is finite when a grid point lies on a filament or
vertex; a two-line coil is one open straight segment; variable names run cleanly to `br_101`;
`extcur` prints `EXTCUR(**)` for group index 100 and above.

Timing: NCSX at 201 x 201 x 36, S mode, symmetry on, one process on an Apple M4 Pro: 79 s wall
(`cases/ncsx_full_S_Y/time.txt`). Single run, machine not otherwise idle.

W7-X (added later the same day): `cases/w7x_S_Y` and `cases/w7x_R_N` at 33 x 33 x 12. Legacy's
symmetric and full-period runs differ by 3.5e-5, entirely from its single-precision trig: with
double trig the same comparison gives 5.6e-13. Full-resolution W7-X (201 x 201 x 36, S mode,
symmetry on) took 71 s in legacy; `xgrid-gpu --legacy-fp32-trig` matches that output to 7.0e-14.
Full-resolution NCSX parity is 2.8e-13.

The LIBSTELL in this build was later rebuilt with HDF5 enabled (`make_xgridref_h5.inc`) to build
FIELDLINES for the downstream gate; VMEC2000 needed ScaLAPACK, built from source
(Reference-ScaLAPACK, commit 6849865) and added to `LIBS`. The `xgrid` binary that produced the
files here was built before either change.

Not exercised: one-line (circular) coils, quoted group names with spaces, `group_id = -1`.
