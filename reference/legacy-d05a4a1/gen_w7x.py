"""Write coils.w7x: the Wendelstein 7-X coil set as single filaments, in
MAKEGRID format, from the Fourier coefficients distributed with simsopt.

Source: simsopt (MIT licence), src/simsopt/configs/W7-X.dat at commit
912bdcc. simsopt documents these coefficients as a Fourier transform of the
xgrid file coils.w7x_v001 (W7-X CAD coils, provided by J. Geiger, 2022) that
reproduces its coordinates to about 1e-13 m. This script reverses that
transform; it is a test fixture, not an authoritative W7-X coils file.

Conventions follow simsopt's get_data("w7x") and coils_via_symmetries:
  * 7 base coils (5 non-planar, 2 planar), nfp = 5, stellarator symmetric:
    70 coils, ordered k (period) -> flip -> base coil, as simsopt orders them.
  * A flipped copy is the rotated coil mapped (x, y, z) -> (x, -y, -z) and
    carries the negated current.
  * Non-planar coils carry 15000 A x 108 turns (simsopt's standard
    configuration). simsopt gives the planar coils zero current; here they get
    1e5 A so that S-mode mgrid files contain a usable unit-current field for
    them. In S mode the field is normalised per group, so this value only
    affects raw_coil_cur and extcur.

usage: python3 gen_w7x.py W7-X.dat out_dir [points_per_coil]
"""
import sys
import numpy as np

src, out = sys.argv[1], sys.argv[2]
npts = int(sys.argv[3]) if len(sys.argv) > 3 else 240

data = np.loadtxt(src, delimiter=",")
order = data.shape[0] - 1
ncoil = data.shape[1] // 6
t = np.arange(npts) / npts
base = []
for ic in range(ncoil):
    xyz = []
    for d in range(3):
        s_col, c_col = data[:, 6 * ic + 2 * d], data[:, 6 * ic + 2 * d + 1]
        v = np.full(npts, c_col[0])
        for m in range(1, order + 1):
            v += s_col[m] * np.sin(2 * np.pi * m * t) + c_col[m] * np.cos(2 * np.pi * m * t)
        xyz.append(v)
    base.append(np.stack(xyz, 1))

nfp = 5
currents = [15000.0 * 108] * 5 + [1.0e5, 1.0e5]
names = ["NPC1", "NPC2", "NPC3", "NPC4", "NPC5", "PCA", "PCB"]

with open(f"{out}/coils.w7x", "w") as f:
    f.write(f"periods {nfp}\nbegin filament\nmirror NIL\n")
    for k in range(nfp):
        phi = 2 * np.pi * k / nfp
        c, s = np.cos(phi), np.sin(phi)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        for flip in (False, True):
            for i, pts in enumerate(base):
                p = pts @ rot.T
                cur = currents[i]
                if flip:
                    p = p * np.array([1.0, -1.0, -1.0])
                    cur = -cur
                for q in p:
                    f.write(f"{q[0]:24.16E}{q[1]:24.16E}{q[2]:24.16E}{cur:24.16E}\n")
                q = p[0]
                f.write(f"{q[0]:24.16E}{q[1]:24.16E}{q[2]:24.16E}{0.0:24.16E} {i + 1} {names[i]}\n")
    f.write("end\n")
