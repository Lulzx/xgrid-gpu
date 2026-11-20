# Technical Specification: `xgrid-gpu`
**GPU Drop-In Replacement for STELLOPT `MAKEGRID` (`xgrid`)**

**Version:** 0.9.0 (release candidate)  
**Status:** all milestones implemented. CPU and Metal backends pass every gate on an Apple M4 Pro, and the downstream gate passes with VMEC and FIELDLINES. The CUDA backend passes every gate through a mock driver that runs its kernels on the CPU, but has **not been run on an NVIDIA GPU**; that is the one remaining step before 1.0 (§8)
**Backends:** CPU reference (`fp64`), CUDA (`fp64` or `ds`), Apple Metal (`ds` or `fp32`)
**Implementation language:** Zig (pinned toolchain version in `build.zig.zon`) host + `.cu` / `.metal` kernels
**Runtime dependencies:** libc and the OS/driver-provided GPU runtime only (no `libnetcdf`, `libhdf5`, MPI, Python, or NVRTC)

**Legacy parity target:** `PrincetonUniversity/STELLOPT` (`MAKEGRID` + `LIBSTELL`), branch `develop`, commit `d05a4a11231b191544016f24be1b2ead61a12c8f`. Items marked **[legacy]** describe that commit. Reference outputs from a build of it, the build recipe, and a numpy prototype of §2.4–§3.4 that reproduces them to better than $10^{-15}$ are in `reference/legacy-d05a4a1/`. The items not yet exercised by a reference run are listed in §9.

---

## 1. Purpose and Design Goals

`xgrid-gpu` computes the vacuum magnetic field $\mathbf{B}(R, \phi, Z)$ of filamentary coils on a cylindrical grid and writes `mgrid_<ext>.nc` and `extcur.<ext>` files that downstream codes (`VMEC`, `FIELDLINES`, `BEAMS3D`, `STELLOPT`) accept in place of the legacy output.

### 1.1 Requirements

1. **Structural parity of outputs.** The `mgrid` header (dimensions, variable names, types, shapes, definition order) is identical to the legacy file for the same inputs. `ncdump -h` output must match exactly; a byte-identical header is the stretch check.
2. **Numerical parity of fields.** Field arrays match legacy within the tolerance in §1.2. They are not bit-identical: summation order, FMA contraction and compiler differences make that unachievable, and v0.1's "bit-for-bit" wording is withdrawn.
3. **Input parity.** All four legacy invocation paths are accepted with their legacy defaults (§2.1).
4. **Self-contained binary.** One executable with kernels embedded. It links only libc and loads the GPU runtime lazily (§4.5), so the same binary runs on a machine with no GPU.
5. **Determinism.** Same binary, backend, device and inputs give bit-identical output across runs.
6. **Bounded host memory.** Peak host memory is independent of `nextcur`: one coil group's three field arrays plus one byte-swap buffer and the segment buffer on the CPU backend, and two groups' arrays on GPU backends that overlap computing group $g+1$ with writing group $g$ (§4.2).

### 1.2 Accuracy definition

Absolute tolerances in tesla are meaningless here because `S`-mode fields are in T/A (magnitude ~$10^{-7}$) and `R`-mode fields are in T (magnitude ~1). All tolerances are therefore normalised per coil group $g$:

$$\varepsilon_g = \frac{\max_{\text{grid}} \lVert \mathbf{B}_g^{\text{test}} - \mathbf{B}_g^{\text{ref}} \rVert_\infty}{\max_{\text{grid}} \lVert \mathbf{B}_g^{\text{ref}} \rVert_2}$$

| Comparison | Target $\varepsilon_g$ |
| --- | --- |
| GPU `fp64` vs. CPU reference | $< 10^{-13}$ |
| GPU `ds` vs. CPU reference | $< 10^{-10}$. Measured on Metal: below $3 \times 10^{-13}$ on the synthetic cases, $2.6 \times 10^{-12}$ on NCSX and $4.0 \times 10^{-11}$ on W7-X at `201 × 201 × 36` (see below) |
| GPU `fp32` vs. CPU reference | reported, not gated. Measured on Metal: $10^{-7}$ to $10^{-6}$ on the synthetic cases, $3 \times 10^{-5}$ on NCSX at `41 × 41 × 12`, $6 \times 10^{-4}$ at `201 × 201 × 36` |
| CPU reference with `--legacy-fp32-trig` vs. legacy Fortran | $< 10^{-12}$. Measured with the Zig CPU reference: below $3 \times 10^{-15}$ on synthetic cases, $1.5 \times 10^{-14}$ on NCSX at `41 × 41 × 12`, $2.8 \times 10^{-13}$ on NCSX and $7.0 \times 10^{-14}$ on W7-X at `201 × 201 × 36` |
| CPU reference (default) vs. legacy Fortran | reported, not gated (measured $10^{-7}$ to $7 \times 10^{-6}$, and $2.3 \times 10^{-4}$ on W7-X at full resolution where grid points are 0.5 mm from coils; see below) |

**Why `ds` is gated at $10^{-10}$, not $10^{-11}$.** `ds` stores coordinates with a relative rounding of about $2^{-48}$, roughly $2 \times 10^{-14}$ m for a point 6 m from the origin. Near a filament the field varies as $1/d$, so the resulting relative field error is about $2 \times 10^{-14} / d$: $10^{-11}$ at $d = 2$ mm. The W7-X `201 × 201 × 36` grid passes 0.5–2 mm from coils; 10 of its 10 million (point, group) values exceed $10^{-11}$, the worst $4.0 \times 10^{-11}$, all within 2 mm of a filament. The 0.4.0 tolerance of $10^{-11}$ assumed grids stay clear of coils, which real mgrid grids need not do.

All tolerances exclude cases with grid points lying on a filament (§3.2), where only finiteness is required. The CPU reference is the oracle for all GPU backends. Only the CPU reference is compared against Fortran, so a legacy quirk is debugged once.

**Legacy single-precision trig [legacy].** `bfield` in `biotsavart.f` declares `REAL :: cosp, sinp`, and none of the 33 `SHARE/make_*.inc` build configurations promotes default reals. Legacy therefore rounds $\cos\phi_k$ and $\sin\phi_k$ to single precision and uses the rounded values both to place the field point and to project onto $B_R$, $B_\phi$. Measured on the reference cases, this puts legacy about $10^{-7}$ away from a correct `fp64` evaluation on the synthetic cases and up to $7 \times 10^{-6}$ on NCSX, where modular-coil filaments pass through the grid box and the misplaced field point matters more. Emulating the rounding reproduces legacy to $1.5 \times 10^{-14}$ or better in every case.

Policy: the default is correct double-precision trig on every backend. `--legacy-fp32-trig` rounds the host trig tables through `float` before use (§3.1), which is the only change needed since no trig runs in the kernels. It exists for the parity gate and for users who need output indistinguishable from legacy. The README states that default output differs from legacy at the $10^{-7}$ level and why.

### 1.3 Non-goals for 0.x

* Legacy tasks other than `mgrid` (`mgrid_rs`, `circ_tor_mgrid`), coil shifts/rotations (`cg_shift_*`, `cg_rot_*`), and `lgrps` per-coil grouping.
* Circular coils (`fil_circ`). **[legacy]** treats a coil consisting of a single line (the terminating line alone) as an analytic circular loop, which needs complete elliptic integrals. `xgrid-gpu` exits with an error naming the coil and its line number. Decided: stays an error for all of 0.x.
* The legacy unformatted `mgrid_<ext>.bin` output.
* Vector potential output. **[legacy]** does not write `ar_/ap_/az_` either.
* Windows.

---

## 2. Input Specification

### 2.1 Invocation modes

`--legacy-fp32-trig` is removed from the argument list first, wherever it appears. Mode is then selected from the remaining arguments by shape, exactly as **[legacy]** `makegrid.f` does, with one addition (flags):

| Arguments | Mode |
| --- | --- |
| any remaining argument beginning with `--` | Modern flags |
| none | Interactive: 10 values read from stdin |
| exactly one, `-h` | Help |
| exactly one, anything else | Path to an `&MGRID_NLI` namelist file |
| 8, 9 or 10 positional | Legacy positional |
| anything else | Error, exit code 1 |

```bash
# Modern flags
xgrid-gpu --coils coils.c09r00 --mode S --sym \
  --rmin 0.436 --rmax 2.436 --zmin -1.0 --zmax 1.0 \
  --ir 201 --jz 201 --kp 36

# Namelist file
xgrid-gpu mgrid_c09r00.nli

# Legacy positional: ext mode sym rmin rmax zmin zmax kp [ir] [jz]
xgrid-gpu c09r00 S Y 0.436 2.436 -1.0 1.0 36 201 201

# Legacy stdin script (ten lines, same order as positional)
xgrid-gpu < xgrid_c09r00.in
```

**Defaults differ by path [legacy], and are reproduced:**

| | Positional | Interactive | Namelist |
| --- | --- | --- | --- |
| `mgrid_mode` | `S` unless arg is `R`/`r` | `S` unless `R`/`r` | `'R'` |
| `lstell_sym` | true only for `Y`/`y`/`T`/`t` | true only for `Y`/`y` | `.TRUE.` |
| `ir`, `jz` | 121 if omitted | always read | 121 |
| bounds, `kp` | required | required | `rmin=.45 rmax=1.05 zmin=-.3 zmax=.3 kp=5` |

Stdin values follow Fortran list-directed rules: optional quotes around strings, `D` or `E` exponents, leading/trailing blanks ignored. Prompts are written to stdout as in legacy so existing logs look familiar.

**Additional flags (modern mode only):**

| Flag | Meaning |
| --- | --- |
| `--backend auto\|cpu\|metal\|cuda` | Default `auto`. With `fp64` (or no `--precision`): CUDA if an NVIDIA driver and device load and pass the self-test, else the CPU. Metal is never chosen for `fp64` (§7.5). With `ds` or `fp32`: Metal on macOS, CUDA elsewhere, falling back to CPU `fp64` if that GPU cannot start or fails mid-run |
| `--precision fp64\|ds\|fp32` | CPU: `fp64`. Metal: `ds` (default) or `fp32`. CUDA: `fp64` (default), `ds` or `fp32` |
| `--device N` | GPU index for Metal (`MTLCopyAllDevices` order) or CUDA |
| `--out PATH`, `--extcur PATH` | Override output paths (default `mgrid_<ext>.nc`, `extcur.<ext>` in the working directory) |
| `--threads N` | CPU backend thread count |
| `--legacy-fp32-trig` | Reproduce legacy's single-precision $\cos\phi$/$\sin\phi$ (§1.2). Valid in every invocation mode |
| `--quiet`, `--version` | |

`<ext>` is everything after `coils.` in the coils file name.

### 2.2 `&MGRID_NLI` namelist

Supported keys: `task`, `mgrid_ext`, `mgrid_mode`, `lstell_sym`, `rmin`, `rmax`, `zmin`, `zmax`, `kp`, `ir`, `jz`.

Recognised but unsupported **[legacy]** keys: `rmajor`, `aminor`, `nphi`, `ntheta`, `extcur_mgrid`, `cg_shift_1`, `cg_shift_2`, `cg_rot_xcent`, `cg_rot_theta`, `cg_rot_phi`, `cg_rot_angle`, `l_rot_coil_center`, `sym_ir`, `sym_jz`, `sym_kp`, `sym_perform_tests`. Their presence is accepted when `task` is `mgrid` and they cannot affect the result; a value that would change the field (any non-zero shift or rotation) is a hard error. Silently ignoring them would produce a plausible but wrong `mgrid`.

`task` is lowercased; anything other than `mgrid` is an error naming the unsupported task. Unknown keys are an error.

The namelist parser supports the subset actually found in the wild, stated explicitly so it can be tested: case-insensitive names, `&NAME ... /` and `&NAME ... &END` terminators, `!` comments, comma or whitespace separators, single- or double-quoted strings, logicals `T`/`F`/`.TRUE.`/`.FALSE.` (any case), and reals with `D`/`E` exponents. Array syntax and repeat counts (`3*0.0`) are needed only for the unsupported keys and are parsed just far enough to skip them.

### 2.3 Input validation

Errors (exit 1): `ir < 2`, `jz < 2`, `kp < 1`, `rmax <= rmin`, `zmax <= zmin` (checked on raw input, before symmetrisation, as **[legacy]**), `rmin < 0`, non-finite values, `ir*jz*kp*8 >= 2^32 - 4` (CDF-2 per-variable limit, §5.1), missing or unreadable coils file.

### 2.4 `coils.<ext>` filament file

```text
periods 3
begin filament
mirror NIL
  x0   y0   z0   I
  x1   y1   z1   I
  ...
  x0   y0   z0   0.0   <group_id>   <group_name>
end
```

Parsing rules, all **[legacy]** `biotsavart.f` / `bsc_T.f`:

* If any line contains the marker `** coils_dot_starts_below **`, the file proper starts on the line after it; otherwise at the top. The first line must then contain `periods`, and $N_{\text{fp}}$ is its second field. The next two lines (`begin filament`, `mirror NIL`) are skipped without being read.
* A line that parses as six fields (`x y z I group_id group_name`) ends the current coil. Termination is decided by the field count, not by the current being zero. A line whose first three characters are `end` ends the file.
* `group_name` is one list-directed item: a bare whitespace-delimited token, or a quoted string that may contain spaces. It is truncated to 30 characters in the output (confirmed on a 45-character name). A six-token line whose fifth field is not an integer is an ordinary vertex line, as in legacy.
* `group_id = -1` is an error. Legacy uses −1 as its empty-slot marker, so such a group is silently mishandled there.
* **One current per coil:** the current on the coil's *first* vertex applies to every segment. Currents on later vertices are read and ignored. v0.1's per-segment $I_k$ was wrong.
* **Closure:** for coils of more than two lines (terminating line included) the final (terminating) vertex is dropped, consecutive duplicate vertices are removed by exact equality, a duplicate wrap-around vertex is removed, and the loop is then closed by a segment from the last kept vertex back to the first. A coil of $n$ kept vertices therefore yields $n$ segments.
* A two-line coil (one vertex line plus the terminating line) is a single open straight segment between the two points, with no closing segment. Legacy prints a warning; so does `xgrid-gpu`. Confirmed by reference run. A one-line coil is an error (§1.3).
* **Group numbering:** output group index is the order of *first appearance* of each distinct `group_id`, not the numeric id and not sorted. `nextcur` is the number of distinct ids. Non-contiguous ids are fine.
* Numeric fields accept `D` and `E` exponents. Tabs count as whitespace.

### 2.5 Group currents and scaling

For each group $g$, the nominal current is the current of the coil with the largest magnitude **[legacy]**:

$$I_g = I_{c^*}, \qquad c^* = \arg\max_{c \in g} |I_c| \quad \text{(first such coil on ties)}$$

* `raw_coil_cur(g)` $= I_g$ in both modes.
* **`R` mode:** fields are computed with file currents and written unscaled.
* **`S` mode:** fields are computed with file currents and multiplied by $1/I_g$. If $|I_g| \le 10^{-100}$ the multiplier is $10^{100}$ **[legacy]**; `xgrid-gpu` additionally prints a warning naming the group.

The kernel never sees the mode. Each coil carries the weight $w = 10^{-7} \cdot I_c$, applied once to the coil's summed segment contributions (§4.1), and the host multiplies the read-back field arrays by $s_g$ in `fp64`, with $s_g = 1/I_g$ in `S` mode and $1$ in `R` mode. This is the order legacy uses, and it keeps $s_g$ out of the `float`-based kernels, where the $10^{100}$ multiplier and very small group currents (NCSX has $I_g = 2.5 \times 10^{-7}$ A for two PF groups) would overflow or lose range. The constant $\mu_0/4\pi$ is the hard-coded $10^{-7}$ that legacy uses; it must not be replaced by the post-2019 CODATA value.

---

## 3. Mathematical Formulation

### 3.1 Grid

For $i \in [0, \text{IR}-1]$, $j \in [0, \text{JZ}-1]$, $k \in [0, \text{KP}-1]$:

$$R_i = R_{\min} + i \cdot \frac{R_{\max} - R_{\min}}{\text{IR} - 1}, \qquad Z_j = Z_{\min} + j \cdot \frac{Z_{\max} - Z_{\min}}{\text{JZ} - 1}, \qquad \phi_k = k \cdot \frac{2\pi}{N_{\text{fp}} \cdot \text{KP}}$$

$$\mathbf{r} = (R_i \cos\phi_k,\; R_i \sin\phi_k,\; Z_j)$$

When `lstell_sym` is true the Z bounds are first symmetrised, larger magnitude winning **[legacy]**: if $|Z_{\max}| > |Z_{\min}|$ then $Z_{\max} \leftarrow |Z_{\max}|$, $Z_{\min} \leftarrow -Z_{\max}$; otherwise $Z_{\min} \leftarrow -|Z_{\min}|$, $Z_{\max} \leftarrow -Z_{\min}$. The symmetrised values are what get written to the file. Without this the mirror in §3.4 maps onto the wrong Z rows; v0.1 omitted it.

All grid coordinates, $\cos\phi_k$ and $\sin\phi_k$ are computed on the host in `fp64` and passed to the kernel as small lookup tables (as `{hi, lo}` pairs for `ds`). No trigonometry runs on the GPU. With `--legacy-fp32-trig`, each $\cos\phi_k$ and $\sin\phi_k$ is computed in `fp64`, rounded to `float`, and widened back before the tables are built; everything downstream is unchanged.

### 3.2 Straight-segment field (Hanson–Hirshman)

For a segment from $\mathbf{r}_a$ to $\mathbf{r}_b$ with $\Delta = \mathbf{r}_b - \mathbf{r}_a$:

$$\mathbf{R}_a = \mathbf{r} - \mathbf{r}_a, \quad \mathbf{R}_b = \mathbf{r} - \mathbf{r}_b, \quad R_a = \lVert\mathbf{R}_a\rVert, \quad R_b = \lVert\mathbf{R}_b\rVert, \quad S = R_a + R_b$$

$$\mathbf{B}_{\text{seg}} = w \cdot \frac{2S}{R_a R_b \left(S^2 - \lVert\Delta\rVert^2\right)} \left(\Delta \times \mathbf{R}_a\right)$$

This is algebraically the v0.1 formula but written with the segment vector $\Delta$ rather than $L\hat{\mathbf{e}}$, which is the form legacy evaluates. It needs only $\lVert\Delta\rVert^2$ (precomputed on the host), so there is no per-segment $L$, no unit vector, and no extra square root or division.

**Where precision is lost.** $S^2 - \lVert\Delta\rVert^2$ cancels only when the field point is close to the segment between its endpoints ($S \to \lVert\Delta\rVert$). It is well conditioned in the far field; v0.1 had this backwards. The real reasons `fp32` is inadequate are (a) cancellation in $\mathbf{r} - \mathbf{r}_a$ when the point is near a coil, (b) cancellation *between* segment contributions when summing $10^4$–$10^6$ terms, and (c) `fp32` rounding of the inputs themselves.

**Guard [legacy].** Legacy uses floors rather than a skip, with a per-coil constant $\epsilon_c = \epsilon_{\text{mach}} \cdot \min_{\text{seg} \in c} \lVert\Delta\rVert^2$ where $\epsilon_{\text{mach}} = 2^{-52}$:

$$R_{a,b} \leftarrow \sqrt{\max(\epsilon_c,\; \lVert\mathbf{R}_{a,b}\rVert^2)}, \qquad S^2 - \lVert\Delta\rVert^2 \leftarrow \max(\epsilon_c,\; S^2 - \lVert\Delta\rVert^2)$$

`xgrid-gpu` reproduces these floors, implemented with `max` so the kernel stays branch-free. The result is always finite. Confirmed on a reference case with grid points on segment interiors and on vertices: legacy output is finite everywhere and the prototype matches it to $4 \times 10^{-16}$. v0.1's absolute $10^{-24}\,\text{m}^2$ skip threshold was below `fp64` rounding for metre-scale geometry and would not have matched legacy. **On-filament warning.** A computed point is *on a filament* for a group if a floor was engaged for at least one of that group's segments, that is $\lVert\mathbf{R}\rVert^2 < \epsilon_c$ at a vertex or $S^2 - \lVert\Delta\rVert^2 < \epsilon_c$. The backend returns the count of such (point, group) pairs and the host prints one warning with the total, since a grid point sitting on a filament is almost always an input mistake. On the CPU this is two extra `min` operations per segment. A GPU kernel writes one flag per thread alongside its field values; no atomics are needed.

**On-filament points are excluded from every tolerance.** The value there is rounding noise multiplied by a factor of order $1/\epsilon_c$, so it depends on details such as whether the compiler contracts `a*b - c*d` into a fused multiply-add. Measured on a 10,000-gon with a grid point exactly on a vertex: legacy, the CPU reference and Metal `ds` return three unrelated finite values at that one point ($B_Z$ of $8.8 \times 10^{-4}$ on the CPU against 253 in `ds`), and agree everywhere else. A case in which any point is flagged on-filament passes its gate if all output is finite. The `onwire` case, whose coordinates are exact binary fractions, happens to agree to $10^{-15}$ on every backend.

### 3.3 Cylindrical projection

$$B_R = B_x \cos\phi_k + B_y \sin\phi_k, \qquad B_\phi = -B_x \sin\phi_k + B_y \cos\phi_k, \qquad B_Z = B_z$$

### 3.4 Stellarator symmetry (`lstell_sym`)

$$B_R(R, -\phi, -Z) = -B_R(R, \phi, Z), \qquad B_\phi(R, -\phi, -Z) = B_\phi(R, \phi, Z), \qquad B_Z(R, -\phi, -Z) = B_Z(R, \phi, Z)$$

The flag *assumes* the coil set is stellarator-symmetric; it does not check. To stay within tolerance of legacy even for imperfectly symmetric coil sets, the set of points computed directly is exactly the legacy set. With $h = \lfloor\text{KP}/2\rfloor$, $m = \lfloor\text{JZ}/2\rfloor + (\text{JZ} \bmod 2)$ and $j' = \text{JZ} - 1 - j$:

| Plane | Computed directly | Filled by symmetry |
| --- | --- | --- |
| $k = 0$ | rows $j < m$ | rows $j'$ of the same plane |
| $1 \le k \le h - 1 + (\text{KP} \bmod 2)$ | all rows | plane $\text{KP} - k$, row $j'$ |
| $k = h$, only if KP is even | rows $j < m$ | rows $j'$ of the same plane |

Fill rule in every case: $B_R \leftarrow -B_R$, $B_\phi$ and $B_Z$ copied. This is exactly half the grid. v0.1 computed the two self-conjugate planes in full, which costs more and differs from legacy when the coils are not perfectly symmetric.

Odd-JZ edge case **[legacy]**: on self-conjugate planes the middle row maps to itself, so legacy negates the computed $B_R$ in place. The exact value is zero for a symmetric coil set, so this only flips the sign of rounding noise. `xgrid-gpu` does the same.

Symmetry is not a speedup over legacy, which already uses it; do not count it in speedup claims.

---

## 4. Compute Architecture

### 4.1 Segment buffer and loop structure

Every backend uses the same data and the same loop nest, which is the one `cpu_ref.zig` implements and the parity gate has validated:

```text
for each coil c of the group:            # same trip count for every thread
    R_a, r_a  <- first vertex of c       # one sqrt
    acc = 0
    for each segment n of c:
        R_b, r_b <- vertex n+1           # one sqrt, shared with the next segment
        acc += (d_n x R_a) * 2S / (r_a r_b max(eps_c, S^2 - dsq_n))
        R_a, r_a <- R_b, r_b
    B += (1e-7 * I_c) * acc              # weight applied once per coil
```

* **Vertex arrays** `x, y, z` hold each coil's vertices contiguously; a closed loop stores its first vertex again at the end, so a coil with $n$ stored vertices has $n-1$ segments. `dx, dy, dz, dsq` hold the vector to the next vertex and its squared length.
* **Coil table:** first vertex index, vertex count, weight $10^{-7} I_c$, floor $\epsilon_c$. Coils are sorted by output group, in file order within a group.
* The outer loop over coils has data-independent bounds, so it causes no divergence between GPU threads. v0.2 instead proposed a single flat chain with zero-weight link segments between coils. That is withdrawn: a link segment's factor can be `Inf` or `NaN` (coincident coil ends, underflow in `ds`), and `0 × Inf` would poison the accumulator.
* **The weight is applied once per coil, not per segment.** This is legacy's order, saves three multiplies per segment, and in `ds` keeps small currents from pushing per-segment terms toward the `float` underflow range before they are summed.
* For `ds`, every array entry is a `{hi, lo}` pair produced on the host by splitting the `fp64` value.

### 4.2 Dispatch

* One thread per directly computed grid point (§3.4), in a compact work list; the host scatters and mirrors results into the `(k, j, i)` output layout.
* One coil group at a time. Results for group $g$ are read back, mirrored, scaled, byte-swapped and written (§5.1) on a second thread while group $g+1$ computes. Implemented for all backends; it cut the end-to-end Metal `fp32` NCSX time from 0.45 s to 0.37 s.
* Work is split into chunks sized so one dispatch takes roughly 50–200 ms. A single multi-second dispatch risks driver watchdog resets and freezes the desktop on machines where the GPU also drives the display.
* Thread groups are padded to the group size with an in-kernel bounds guard.

**Tiling: not implemented.** v0.1 mandated 256-segment shared-memory tiles. Every thread reads the same vertex at the same time, which GPU caches serve as a broadcast, so tiling saves memory traffic the kernels are not limited by. The measured Metal `fp32` kernel reaches $4.4 \times 10^{10}$ interactions per second, and the `ds` kernel spends its time in the double-single divide, square root and multiply chains. Tiling would also complicate the per-coil loop, whose bounds differ between coils. Revisit only if a profile on real hardware shows memory stalls.

**Summation.** Each thread accumulates segments in buffer order, which gives the determinism in §1.1. Compensated (Kahan) accumulation for `fp64` is an option to evaluate against the parity gate; it is not on by default because it moves results away from legacy's plain summation.

### 4.3 Precision modes

Precision is a property of the kernel, not of the backend.

| Mode | Backends | Notes |
| --- | --- | --- |
| `fp64` | CPU, CUDA | Native IEEE double. |
| `ds` | Metal, CUDA | Double-single `{hi, lo}` in `float`, ~48-bit significand. |
| `fp32` | Metal, CUDA | Exploratory only. Output is still written as `NC_DOUBLE`. |

**`ds` requirements.**

* Operations needed by the kernel: `add`, `sub`, `mul`, `div`, `sqrt`, `max`, built on `two_sum`, `quick_two_sum` and `two_prod` (via `fma(a, b, -p)`). They are written in finite-only form in `kernels/biot_savart.metal`, following the layout of the `fast48` mode of `lulzx/vf64-metal`. One source is compiled twice, with `XG_DS=1` or `0`.
* **Fast-math must be off** for the kernel translation unit (Metal: `MTLCompileOptions` fast-math disabled / safe math mode; CUDA: no `--use_fast_math`, contraction controlled explicitly). With fast-math on, the compiler legally simplifies error-free transformations to zero and `ds` silently degrades to `fp32`.
* **Startup self-test.** Before any real work the GPU backend runs one small fixed Biot–Savart problem (a non-planar five-vertex loop plus an open segment, 189 points) and compares it with the CPU reference: limit $10^{-11}$ for `ds`, $10^{-3}$ for `fp32`. A kernel compiled with fast-math, a broken `fma`, or a host/kernel struct layout mismatch all leave `ds` at `float` accuracy or worse and fail this by orders of magnitude. Failure is a hard error (exit code 3), never a silent fallback to lower precision. Measured self-test difference on the M4 Pro: about $10^{-14}$ in `ds`.
* Accumulation across segments is done in `ds`.

**`ds` range hazards.** `ds` has the significand of roughly a 48-bit float but only the exponent range of `float` (smallest normal about $1.2 \times 10^{-38}$), and its rounding unit is about 16 times coarser than `fp64`'s. Both interact badly with the legacy floor $\epsilon_c = 2^{-52}\min\lVert\Delta\rVert^2$:

* For a finely discretised coil ($\lVert\Delta\rVert \approx 6 \times 10^{-4}$ m, as in the 10,000-gon test) $\epsilon_c \approx 9 \times 10^{-23}$, and at a point on a vertex the product $r_a r_b \epsilon_c$ is about $5 \times 10^{-38}$: at the edge of `float` underflow, where error-free products stop being error-free and the division can return `Inf`.
* Near a filament, the rounding noise of $S^2 - \lVert\Delta\rVert^2$ in `ds` exceeds $\epsilon_c$, so whether the floor engages differs from `fp64`.

Consequences for the `ds` kernels:

1. The denominator $r_a r_b \max(\epsilon_c, \cdot)$ is formed as a product and clamped below at $10^{-30}$ before the single division. Staged division would avoid the underflow without a clamp but costs two more `ds` divisions per segment, and underflow only occurs at on-filament points, which carry no accuracy requirement.
2. Output must be finite everywhere; the self-test includes the `onwire` case and the 10,000-gon with a grid point on a vertex.
3. The `ds` tolerance ($10^{-10}$, §1.2) applies to cases with no point on a filament by the CPU reference's count (§3.2). On-filament points are only required to be finite.
4. Both cases were run in M1 and are stored as reference cases (`onwire_R_N`, `poly_vertex_R_N`): output is finite in `ds` and `fp32`, and all three backends flag the same points.

**`ds` cost reductions adopted** (accuracy unchanged at $2.6 \times 10^{-12}$ on full NCSX, kernel 19% faster): a cheaper sum for same-sign operands (the squared norms and $S$), exact doubling for $2S$, and a division with one residual correction instead of two.

**`ds` on CUDA.** Consumer NVIDIA cards run native `fp64` at a small fraction of their `fp32` rate (roughly 1/64 on recent GeForce generations, versus about 1/2 on data-centre parts). On those cards the `ds` kernel may outrun native `fp64`. Both are built for CUDA; the default stays `fp64` until the benchmark says otherwise.

### 4.4 CPU reference

`cpu_ref.zig` is a straightforward `fp64` implementation of §3 using the same segment buffer and the same summation order as the kernels, vectorised over grid points and threaded over rows. It is the fallback backend and the oracle for every GPU test, so it is written for clarity first.

### 4.5 Linking and runtime loading

* **macOS:** links system frameworks (`Metal`, `Foundation`). The Metal source is embedded in the binary and compiled at startup with `newLibraryWithSource`, so there is no build-time dependency on the Xcode `metal` compiler.
* **Linux:** dynamically linked against glibc only (`libc.so.6`, `libpthread.so.0`, `libdl.so.2`; verified on cross-built x86-64 and arm64 binaries). A fully static (musl) binary cannot `dlopen` the glibc-linked `libcuda.so.1`, so v0.1's "single static binary" and "CUDA" were mutually exclusive. The CUDA driver library is opened with `dlopen` at runtime; if it is absent the CUDA backend reports unavailable and `auto` falls back to CPU without a message.
* **CUDA kernels** (`kernels/biot_savart.cu`) are compiled to PTX by `kernels/build_ptx.sh` using clang's CUDA device compilation with no CUDA headers or libdevice, so neither nvcc nor the CUDA toolkit is needed; any clang with the NVPTX target works (Homebrew LLVM 23 was used). The script passes `-ffp-contract=off`, refuses output containing approximate or flush-to-zero instructions, and checks the three entry points exist. The generated PTX is checked in and embedded. NVRTC is not used: it is not part of a driver install.
* **PTX compatibility.** The PTX targets `sm_60` and is ISA version 5.0, so it loads on Pascal and newer GPUs with any driver from CUDA 8.0 onwards and is JIT-compiled for the actual GPU at load time. Every arithmetic instruction carries an explicit IEEE rounding mode (`add.rn`, `mul.rn`, `fma.rn`, `div.rn`, `sqrt.rn`), which also prevents `ptxas` from contracting separate multiplies and adds. A load failure with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` or `CUDA_ERROR_NO_BINARY_FOR_GPU` is reported with the driver version and these requirements.
* **`XGRID_LIBCUDA`** overrides the driver library path. Its purpose is testing: `tests/cuda_mock/mock_cuda.cpp` builds a library implementing the driver-API subset that `xgrid-gpu` uses, which runs `biot_savart.cu` compiled as host C++ on CPU threads and can inject failures (no device, PTX rejected, launch error, wrong results). It exercises the host code, data layouts, argument passing, chunking, error handling and the kernel arithmetic, but not PTX code generation, the driver's JIT, or GPU performance.

---

## 5. Output Writers

### 5.1 `mgrid_<ext>.nc` (NetCDF classic, 64-bit offset)

**[legacy]** creates the file with `NF_CLOBBER | NF_64BIT_OFFSET`, so the format is CDF-2 (`"CDF\x02"`), written here directly with big-endian serialisation.

Everything in this section except the format rules was confirmed with `ncdump -h` / `ncdump -k` on reference files (`reference/legacy-d05a4a1/cases/*/header.cdl`).

**Dimensions**, in definition order, for `nextcur > 1`:

| Name | Size |
| --- | --- |
| `stringsize` | `30` |
| `external_coil_groups` | `nextcur` |
| `dim_00001` | `1` |
| `external_coils` | `nextcur` |
| `rad` | `IR` |
| `zee` | `JZ` |
| `phi` | `KP` |

**Variables**, in definition order **[legacy]** `write_mgrid.f`. v0.1 listed `mgrid_mode` before `coil_group`; legacy defines `coil_group` first.

| # | Variable | Type | Dimensions (C order) |
| --- | --- | --- | --- |
| 1–5 | `ir`, `jz`, `kp`, `nfp`, `nextcur` | `NC_INT` | scalar |
| 6–9 | `rmin`, `zmin`, `rmax`, `zmax` | `NC_DOUBLE` | scalar |
| 10 | `coil_group` | `NC_CHAR` | `(external_coil_groups, stringsize)` |
| 11 | `mgrid_mode` | `NC_CHAR` | `(dim_00001)` |
| 12 | `raw_coil_cur` | `NC_DOUBLE` | `(external_coils)` |
| then, per group | `br_NNN`, `bp_NNN`, `bz_NNN` | `NC_DOUBLE` | `(phi, zee, rad)` |

* `NNN` is the 1-based group index, zero-padded to three digits (confirmed through `br_101`). More than 999 groups is an error.
* Field arrays are in `(k, j, i)` order with `rad` varying fastest.
* `coil_group` entries are space-padded to 30 bytes; longer names are truncated.
* **`nextcur == 1` special case [legacy]:** the dimension list is `dim_00030 = 30`, `dim_00001 = 1`, `external_coils = 1`, `rad`, `zee`, `phi`. There is no `stringsize` or `external_coil_groups`, and `coil_group` is `char coil_group(dim_00030)`. Variable order is unchanged.
* No global or per-variable attributes.
* The header golden tests in §7.2 are generated from the reference `header.cdl` files.

**Format rules.** `numrecs = 0`; absent attribute lists are written as `ABSENT` (two zero words); names and variable data are padded to 4-byte boundaries; `vsize` is the padded size; `begin` offsets are 64-bit. Every variable must be smaller than $2^{32} - 4$ bytes, so `IR*JZ*KP` is limited to about 536 million points.

**Writing.**

* The full file size is known after parsing the coils file. The header and all small variables are written first, then each group's three arrays as soon as that group is computed.
* Arrays are byte-swapped to big-endian in a reused buffer; this pass is part of the I/O cost and is included in timings.
* Output goes to a temporary file in the destination directory and is renamed into place on success, so an interrupted run never leaves a truncated `mgrid` that downstream codes might read.
* Measured on the CPU backend (M4 Pro, internal SSD): byte-swapping and writing the 349 MB NCSX file takes about 0.05 s. That is a write into the page cache with no `fsync`, so it is a lower bound on durable I/O and will be higher on network file systems. Once a GPU brings compute down to the same order, the run is I/O-bound, which is why §4.2 overlaps writing with the next group's compute.

### 5.2 `extcur.<ext>`

Written in both modes, one line per group, using the legacy Fortran formats **[legacy]** `write_mgrid.f`. v0.1's format (zero-padded index, trailing `! name` comment) does not match legacy and is replaced.

| Mode | Group index | Fortran format |
| --- | --- | --- |
| `S` | 1–9 | `'EXTCUR(', i1, ')  = ', 1p, e22.14` |
| `S` | ≥ 10 | `'EXTCUR(', i2, ') = ', 1p, e22.14` |
| `R` | 1–9 | `'EXTCUR(', i1, ')  = ', 1p, e22.14, '/', e22.14` (value written twice) |
| `R` | ≥ 10 | `'EXTCUR(', i2, ') = ', 1p, e22.14, '/', e22.14` |

In `R` mode legacy also appends a blank line and three lines of explanatory text; reproduce them from `reference/legacy-d05a4a1/cases/two_R_N/extcur.two`. Both modes and the `i1`/`i2` switch at group 10 were confirmed on reference files. The Zig formatter must reproduce Fortran `1p,e22.14` exactly (width 22, one digit before the point, 14 after, two-digit exponent with sign), and is covered by golden tests. **Deliberate deviation for group index ≥ 100.** Legacy's `i2` field overflows and it writes `EXTCUR(**) = ...`, which is not a usable namelist line. `xgrid-gpu` writes the three-digit index in the same layout (`'EXTCUR(', i3, ') = '`) and prints a one-line notice. Nothing can depend on the legacy text, so this is not gated for parity.

### 5.3 Console output and exit codes

Progress and timing go to stdout; warnings and errors to stderr. A final line reports backend, device, precision, points computed, segment count, and wall time split into parse / compute / write.

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Usage or input error |
| 2 | I/O error |
| 3 | Requested backend unavailable, failed its self-test, or failed mid-run (only when a backend was forced; `auto` falls back to the CPU, including mid-run from the failed group onwards) |

---

## 6. Repository Layout

```text
xgrid-gpu/
├── build.zig, build.zig.zon   # `zig build --release=fast`; Zig 0.17.1 pinned
├── src/
│   ├── main.zig               # Orchestration, backend selection, overlapped output, extcur
│   ├── config.zig             # Flags, positional, stdin script, &MGRID_NLI, validation
│   ├── coils.zig              # coils.<ext> parser, closure, grouping
│   ├── grid.zig               # Grid tables, symmetry work list and mirroring
│   ├── cpu_ref.zig            # fp64 reference solver
│   ├── netcdf3.zig            # CDF-2 writer and minimal reader
│   ├── fortran_fmt.zig        # 1p,e22.14
│   ├── metal_host.zig         # Metal backend: setup, self-test, projection
│   ├── metal_bridge.m, .h     # Objective-C side: compile kernel, chunked dispatch
│   └── cuda_host.zig          # CUDA backend: dlopen driver API, self-test, chunked launches
├── kernels/
│   ├── biot_savart.metal      # Metal ds and fp32 kernel, embedded
│   ├── biot_savart.cu         # CUDA fp64, ds and fp32 kernels
│   ├── biot_savart.ptx        # Generated from .cu by build_ptx.sh, checked in, embedded
│   └── build_ptx.sh           # clang -> PTX, no CUDA toolkit needed
├── tests/
│   ├── compare_mgrid.zig      # Header-exact, field-tolerant comparison (§1.2)
│   ├── run_reference.sh       # CPU reference vs. legacy outputs in reference/
│   ├── run_gpu.sh             # Metal or CUDA vs. CPU reference, same cases
│   └── cuda_mock/             # Mock CUDA driver running the kernels on the CPU
├── reference/
│   ├── legacy-d05a4a1/        # Legacy outputs, fixtures, generators, build recipe, numpy prototype
│   └── downstream/            # VMEC and FIELDLINES inputs and comparison results
└── README.md
```

Unit tests live beside the code they test (`zig build test`), including the analytical checks of §7.1 in `cpu_ref.zig`.

---

## 7. Verification and Benchmarking

A release requires gates 7.1–7.4. Gate 7.5 produces numbers; it has no pass threshold until a baseline exists.

### 7.1 Analytical gate

v0.1 compared a 10,000-segment polygon with the smooth circular-loop formula to $10^{-11}$. That cannot pass: an inscribed $N$-gon differs from the circle by about $(\pi/N)^2/3 \approx 3 \times 10^{-8}$ at $N = 10^4$. The test must use the exact polygon result.

* **Regular $N$-gon on axis.** Circumradius $a$, current $I$, field point at height $z$ on the axis:

  $$B_Z(z) = \frac{\mu_0 I}{4\pi} \cdot \frac{N a^2 \sin(2\pi/N)}{\left(a^2\cos^2(\pi/N) + z^2\right)\sqrt{a^2 + z^2}}$$

  Checked for $N \in \{3, 64, 10^4\}$ and several $z$, relative error $< 10^{-13}$ (`fp64`) and $< 10^{-11}$ (`ds`). As $N \to \infty$ this tends to $\mu_0 I a^2 / 2(a^2+z^2)^{3/2}$, which is asserted only as a convergence trend.
* **Single finite straight segment** at off-axis points, against the closed-form $\frac{\mu_0 I}{4\pi\rho}(\sin\theta_2 - \sin\theta_1)$, including points near the segment and on its extension.
* **Guard behaviour:** points exactly on a segment and on a vertex return finite values equal to the CPU reference.

### 7.2 Format and parser gates

* CDF-2 header bytes match golden files generated from legacy `ncdump -h` references for `nextcur = 1`, `nextcur = 2` and `nextcur >= 10`.
* Output opens without error in an independent reader (netCDF-C `ncdump` in CI; a test-only dependency).
* `extcur` files match legacy byte-for-byte in both modes, including a group index ≥ 10.
* Coils parser fixtures: duplicate consecutive vertices, unclosed loop, two-vertex coil, one-vertex coil (error), non-contiguous and out-of-order group ids, mixed-sign currents within a group, `D` exponents, tabs, missing `end`.
* Invocation: each of the four modes on the same case yields identical output, apart from documented default differences.
* Malformed-input fuzzing of the three parsers: no crashes, always a diagnostic and exit 1.

### 7.3 Parity gate

Legacy `xgrid` at the pinned commit, built with recorded compiler and flags, is run on the NCSX coil set shipped in STELLOPT (`BENCHMARKS/FIELDLINES_TEST/coils.NCSX`: 10 groups, 78 coils, 18,690 vertex lines) and a W7-X coils file, in `S` and `R` modes, with `lstell_sym` on and off, and with odd and even `KP` and `JZ` at reduced resolution. The cases in `reference/legacy-d05a4a1/cases/` (synthetic fixtures, edge-case probes, and NCSX at `41 × 41 × 12`) are the fast tier of this gate and run on every commit.

* Header: exact.
* CPU reference with `--legacy-fp32-trig` vs. legacy: §1.2. The default-trig difference is reported alongside.
* Each GPU backend vs. CPU reference: §1.2.
* Symmetry self-consistency: on the symmetric coil sets, `lstell_sym = T` and `F` runs agree to $10^{-10}$. Measured on W7-X: $5.6 \times 10^{-13}$ with default trig. With legacy single-precision trig the same comparison gives $3.5 \times 10^{-5}$, for legacy itself as well as for `--legacy-fp32-trig`: legacy's symmetric and full-period runs of a symmetric coil set disagree at that level.

**W7-X coils.** No W7-X coils file is in STELLOPT. `reference/legacy-d05a4a1/gen_w7x.py` rebuilds one from the Fourier coefficients distributed with simsopt (MIT), which simsopt documents as reproducing the CAD-coil xgrid file `coils.w7x_v001` to about $10^{-13}$ m: 70 coils in 7 groups, 16,800 segments. Planar coils are given $10^5$ A so that `S`-mode files contain a usable field for them. It is a test fixture, not an authoritative W7-X file.

### 7.4 Downstream gate

A free-boundary `VMEC` run and a `FIELDLINES` run using the `xgrid-gpu` file complete without format errors, and the VMEC result agrees with the run using the legacy file to within VMEC's own convergence tolerance. "Zero warnings" alone, as in v0.1, would not detect a wrong field.

**Result: passed.** VMEC2000 and FIELDLINES were built from the pinned STELLOPT commit. Inputs and comparison output are in `reference/downstream/`.

*VMEC, free boundary.* NCSX equilibrium `input.NCSX_plasma` from STELLOPT's FIELDLINES benchmark, switched to free boundary: `NZETA = 18`, `EXTCUR` from `extcur.NCSX`, `NS_ARRAY = 17 33 49`, `FTOL_ARRAY` down to $10^{-11}$, using the `201 × 201 × 36` mgrid file. All five runs converged normally (`ier_flag = 0`) in the same 981 iterations at the finest grid, with no warnings. Worst relative difference from the run on the legacy file, over `iotaf`, `volume_p`, axis, `rmnc`, `zmns`, `bmnc`, `b0`, `wb`, `betatotal` and boundary extents:

| mgrid file | Worst difference | In |
| --- | --- | --- |
| `xgrid-gpu` CPU, `--legacy-fp32-trig` | $1.5 \times 10^{-9}$ | `zaxis_cs` |
| `xgrid-gpu` CPU, default trig | $5.3 \times 10^{-8}$ | `rmin_surf` |
| `xgrid-gpu` Metal `ds` | $8.7 \times 10^{-8}$ | `iotaf` |
| `xgrid-gpu` Metal `fp32` | $3.2 \times 10^{-6}$ | `zmns` |

The first row is at the level of the convergence tolerance. The second and third reflect the $10^{-7}$–$10^{-6}$ field difference from legacy's single-precision trig, and are far below anything physically meaningful.

*FIELDLINES.* STELLOPT's `input.NCSX_s1` (two field lines, 2,000 toroidal transits, LSODE tolerance $10^{-9}$), run with `-vmec NCSX_s1 -mgrid mgrid_NCSX.nc -vac`. All runs completed. FIELDLINES' internal field arrays agree with the legacy-file run to $1.2 \times 10^{-10}$ with `--legacy-fp32-trig`, and with a median of $10^{-7}$ with default trig. The largest default-trig outliers ($3 \times 10^{-2}$ relative) are in its stored $R B_R / B_\phi$ at points near coils where $B_\phi \approx 10^{-4}$ T, which divides a $10^{-7}$ field difference by almost zero. Poincaré points agree with the legacy-file run to under 0.9 mm for every file. That includes the legacy-trig file, whose input field matches to $10^{-10}$, so the separation reflects the sensitivity of 2,000-transit trajectories rather than the field.

### 7.5 Performance benchmark

v0.1 set a target of 500–1000× and under 200 ms for the NCSX case without a cost model. The cost is

$$T_{\text{compute}} \approx \frac{N_{\text{points}} \cdot N_{\text{seg}} \cdot c}{\text{device throughput}}$$

with $N_{\text{points}} = \tfrac12 \cdot 201 \cdot 201 \cdot 36 \approx 7.3 \times 10^5$ under symmetry and $c \approx 40$ `fp64` operations (one `sqrt` and one division included), or several hundred `fp32` operations in `ds`. With $N_{\text{seg}}$ of order $10^4$–$10^5$ that is $10^{10}$–$10^{11}$ interactions. Whether 200 ms is reachable therefore depends on $N_{\text{seg}}$ and on the device's precision-specific throughput; it is plausible on data-centre GPUs in `fp64` and should not be assumed for consumer cards in `fp64` or for Apple GPUs in `ds`.

**Measured (Apple M4 Pro, 12 CPU cores, `S` mode, symmetry on, `201 × 201 × 36`, end to end):**

| Program | NCSX: time | W7-X: time | Kernel interactions per second | Difference from CPU reference (NCSX / W7-X) |
| --- | --- | --- | --- | --- |
| Legacy `xgrid`, one process | 79 s | 71 s | $1.7 \times 10^8$ | (see §1.2) |
| `xgrid-gpu` CPU `fp64`, 1 thread | 18 s | 19 s | $7.5 \times 10^8$ | reference |
| `xgrid-gpu` CPU `fp64`, 12 threads | 2.1 s | 1.9 s | $6.5 \times 10^9$ | reference |
| `xgrid-gpu` Metal `ds` | 2.3 s | 2.1 s | $6.0 \times 10^9$ | $2.6 \times 10^{-12}$ / $4.0 \times 10^{-11}$ |
| `xgrid-gpu` Metal `fp32` | 0.37 s | 0.35 s | $4.4 \times 10^{10}$ | $5.8 \times 10^{-4}$ / $1.1 \times 10^{-3}$ |

NCSX: `coils.NCSX`, R 0.436–2.436 m, Z ±1 m, $1.35 \times 10^{10}$ interactions, 349 MB output. W7-X: `coils.w7x` (§7.3), R 4.2–6.8 m, Z ±1.3 m, $1.22 \times 10^{10}$ interactions, 244 MB output. Write is overlapped with compute (§4.2).

Wall time is end to end including setup and the 349 MB write. Single runs on a machine that was not otherwise idle; treat as ±10%. The first Metal run after a kernel change adds about 0.6 s of shader compilation, which the OS caches afterwards.

What this says about Apple silicon:

* **`ds` on the GPU is not faster than `fp64` on the CPU.** The M4 Pro's GPU runs the double-single kernel at about the same rate as its 12 CPU cores run native doubles. `auto` therefore selects the CPU backend, which is also the more accurate one.
* **`fp32` is 5× faster than either but only accurate to $6 \times 10^{-4}$ of the peak field on NCSX,** worse than v0.1's assumed $10^{-5}$, because grid points within millimetres of the modular coils dominate the error. It is an exploratory mode.
* **The 500× target is met only in a narrow sense:** legacy single-process to Metal `fp32` is about 210×, and to either full-precision path about 35–40×. Reaching 500× at full precision needs roughly $10^{11}$ `fp64` interactions per second, which is data-centre GPU territory (M2).
* Not implemented: running the CPU and GPU concurrently on disjoint points, which should roughly halve the full-precision time on this machine (Q2, §9).

**Legacy baseline detail.** Legacy `xgrid` at the pinned commit (gfortran 16.2.0, `-O2 -march=native`, one process) on an Apple M4 Pro, `coils.NCSX`, `S` mode, symmetry on, `201 × 201 × 36`: **79 s** wall, 52 MB peak memory, 349 MB output. That is about $1.35 \times 10^{10}$ interactions (727,218 computed points × about 18,600 segments) at $1.7 \times 10^8$ interactions per second. It is a single run with other light load on the machine, so treat it as ±10%. v0.1's "~400 s" figure does not hold on this hardware, so a 500× speedup means about 160 ms end to end and requires roughly $10^{11}$ interactions per second from the kernel.

Targets are set after the first `xgrid-gpu` measurement. The README table reports, per case and device:

* $N_{\text{seg}}$, $N_{\text{points}}$, and interactions per second (the number that transfers between cases);
* wall time split into parse / compute / write, cold and warm, including Metal shader compilation;
* legacy `xgrid` on one core, stating compiler, flags and CPU, and on 16 MPI ranks if the pinned build supports it;
* `xgrid-gpu` CPU backend at 1 and all threads;
* Apple silicon (Metal `ds`, `fp32`), a consumer NVIDIA card (`fp64` and `ds`), a data-centre NVIDIA card (`fp64`).

Comparisons against `SIMSOPT` / `DESC` are optional and, if included, state grid, precision and whether JIT warm-up is counted.

---

## 8. Milestones

1. **M0, correctness on CPU: done.** Parsers (with fuzz tests), grid and symmetry, CPU reference, CDF-2 writer, `extcur` writer. Unit tests including the analytical gate (7.1), byte-identical headers and `extcur` files, legacy parity on 14 reference cases including NCSX and W7-X, full-resolution NCSX parity, and agreement between all four invocation modes.
2. **M1, Metal: done.** `ds` and `fp32` kernels, startup self-test, time-bounded chunked dispatch, `--device`. `tests/run_gpu.sh metal` passes on every reference case.
3. **M2, CUDA: done, except running on NVIDIA hardware.** `fp64`, `ds` and `fp32` kernels from one source; PTX built without the CUDA toolkit; `dlopen` loader; self-test; chunked dispatch; mid-run fallback. `tests/run_gpu.sh cuda` passes on every reference case through the mock driver, with `fp64` bit-identical to the CPU reference. Linux x86-64 and arm64 builds link against glibc only.
4. **M3, release: done for Apple silicon.** Downstream gate passed (§7.4); write/compute overlap; benchmark table in §7.5 and the README.

**Before 1.0:** run `tests/run_gpu.sh cuda` (without `XGRID_LIBCUDA`) and the NCSX benchmark on at least one NVIDIA GPU, ideally one consumer card and one data-centre card, and record the results in §7.5. Then decide whether `auto` should keep preferring CUDA `fp64` on consumer cards, whose `fp64` rate is about 1/64 of `fp32`, or prefer CUDA `ds` there.

---

## 9. Open Questions

| # | Question | Why it matters |
| --- | --- | --- |
| Q1 | CUDA backend on real NVIDIA hardware: correctness of the PTX path, performance, and the `auto` choice between `fp64` and `ds` on consumer cards. | The one unverified component (§8). |
| Q2 | Should `xgrid-gpu` split points between CPU and GPU when both are available? | The only identified route to a faster full-precision result on Apple silicon. Mixes `fp64` and `ds` points in one file, each within tolerance. |
| Q3 | Metal results are from one machine (M4 Pro). Other Apple GPUs are unmeasured. | Whether `auto` should ever prefer Metal. |
| Q4 | Quoted group names with spaces are parsed as one item here; legacy's handling was not run. Is `coils.NCSX` identical to `coils.c09r00`? Is the MPI build of legacy in scope for the benchmark? | Minor parity and benchmark details. |

Resolved in 0.9.0: W7-X parity case added (from simsopt's published coil data); downstream gate passed with VMEC and FIELDLINES; write/compute overlap implemented; tiling decided against.

Resolved in 0.4.0: full-resolution NCSX parity ($2.8 \times 10^{-13}$) is explained. The worst points are 2 to 7 mm from a modular-coil filament, where the field peaks; the seven PF and TF groups, whose coils are at least 44 mm from any grid point, agree with legacy to about $10^{-15}$. `ds` meets its $10^{-11}$ tolerance on the same case with a factor of four to spare.

Resolved in 0.3.0: the NetCDF header is byte-identical to the one libnetcdf writes for every reference case, not merely `ncdump -h`-equal.

Resolved in 0.2.1: commit pinned; legacy single-precision trig confirmed and handled by `--legacy-fp32-trig`; NetCDF dimensions, types and the `nextcur = 1` layout confirmed; group-name truncation confirmed; one-line circular coils stay an error.

Resolved in 0.2.2: floor guard, two-line coils and more than 99 groups confirmed by reference run; NCSX parity measured at reduced resolution; legacy single-core baseline measured (§7.5).

---

## 10. Changes

### 0.9.0

* M2 implemented: CUDA backend (`fp64`, `ds`, `fp32`), PTX via clang without the CUDA toolkit, `dlopen` loader, mock driver and `tests/run_gpu.sh` (replaces `run_metal.sh`). Untested on NVIDIA hardware.
* Backend selection rewritten: `auto` prefers CUDA `fp64`, never Metal for `fp64`, and falls back to the CPU at startup or mid-run.
* Group writes overlapped with the next group's compute; `--device` implemented for Metal and CUDA.
* Downstream gate run and passed (VMEC free boundary, FIELDLINES).
* W7-X reference cases added; legacy single-precision trig shown to break legacy's own symmetric/full-period agreement at $3.5 \times 10^{-5}$.
* Tiling decided against, with reasons.

### 0.4.0

* M1 implemented: Metal `ds` and `fp32` backend with startup self-test and time-bounded chunked dispatch; gate script `tests/run_metal.sh` (now `tests/run_gpu.sh metal`).
* Measured Metal against the CPU backend; `auto` now means CPU on Apple silicon, with the reasoning in §7.5.
* On-filament points excluded from all tolerances after finding that legacy, CPU and `ds` disagree there; `poly_vertex_R_N` reference case added.
* `ds` kernel uses one clamped division instead of staged division; cheaper same-sign sums and one-correction division adopted after measurement.
* Parser fuzz tests added for coils, namelist and stdin input.

### 0.3.0

* M0 implemented in `src/` and `tests/`; status, layout and measured CPU-backend results recorded.
* Kernel structure changed to coil-by-coil with one weight per coil; zero-weight link segments withdrawn.
* `ds` exponent-range and floor hazards documented, with required handling and an adjusted tolerance scope.
* On-filament warning given an exact predicate; mode detection wording fixed; host-memory requirement corrected for overlapped I/O; PTX driver compatibility addressed; measured I/O cost added.
* Coils parsing: start marker, quoted group names, `group_id = -1` rejected.

### 0.2.2

* Reference runs added for grid points on a filament, two-line coils, 101 groups, and the NCSX coil set.
* Coil termination is by field count; two-line coils are open segments; `extcur` deviates from legacy for group index ≥ 100.
* Legacy parity gate set to $10^{-12}$ with measured values recorded; measured legacy baseline added to §7.5.

### 0.2.1

* Pinned the legacy commit and added reference outputs under `reference/legacy-d05a4a1/`.
* Confirmed by reference run: coil parsing and closure, group ordering, nominal group current, `S`/`R` scaling, Z-bound symmetrisation, the symmetry point set for odd and even `KP`/`JZ`, `extcur` formats, and the NetCDF layout.
* Legacy's single-precision trig confirmed; added `--legacy-fp32-trig` and split the legacy tolerance into a gated compatibility comparison and a reported default comparison.
* Documented the `nextcur = 1` dimension layout.

### 0.2.0 (from 0.1.0)

* Replaced "bit-for-bit" with header-exact plus a normalised field tolerance; absolute tesla tolerances removed.
* Corrected legacy semantics: one current per coil from the first vertex, automatic loop closure and duplicate removal, groups ordered by first appearance, nominal group current is the largest-magnitude coil, Z-bound symmetrisation, legacy symmetry point set, floor-based singularity guard, `extcur` format, variable definition order.
* Added the legacy positional invocation and per-path defaults; resolved the ambiguity between namelist-file and stdin modes.
* Fixed the analytical gate (exact polygon formula) and the precision rationale for `ds`.
* Rewrote the kernel formulation around the segment vector and a shared-vertex chain; made tiling an experiment rather than a requirement.
* Made precision independent of backend; added fast-math and self-test requirements for `ds`.
* Replaced "single static binary" and NVRTC with a libc-only dynamic binary, `dlopen`ed CUDA driver and embedded PTX.
* Added input validation, exit codes, atomic and streamed output, CDF-2 size limits, non-goals, milestones and open questions.
* Replaced unmeasured performance figures with a cost model and a measurement protocol.
