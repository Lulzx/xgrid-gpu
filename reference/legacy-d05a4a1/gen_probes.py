"""Edge-case coils fixtures: grid points on a filament, two-line coils, >99 groups."""
import sys
import numpy as np

out = sys.argv[1]
HDR = "periods 1\nbegin filament\nmirror NIL\n"


def coil(f, pts, cur, gid, name, term=None):
    for p in pts:
        f.write(f"{p[0]:.14E} {p[1]:.14E} {p[2]:.14E} {cur:.14E}\n")
    t = pts[0] if term is None else term
    f.write(f"{t[0]:.14E} {t[1]:.14E} {t[2]:.14E} {0.0:.14E} {gid} {name}\n")


# Rectangle in the y=0 plane whose edges and corners coincide with grid points of
# rmin=0.5 rmax=1.5 ir=5, zmin=-0.5 zmax=0.5 jz=5 on the phi=0 plane (all dyadic).
with open(f"{out}/coils.onwire", "w") as f:
    f.write(HDR)
    coil(f, [(1.0, 0, -0.5), (1.25, 0, -0.5), (1.25, 0, 0.5), (1.0, 0, 0.5)], 1.0e4, 1, "Rect")
    f.write("end\n")

# Group 1: an ordinary loop. Group 2: a coil given as one vertex line + terminator line.
with open(f"{out}/coils.twovert", "w") as f:
    f.write(HDR)
    t = np.arange(16) * 2 * np.pi / 16
    coil(f, np.stack([2 * np.cos(t), 2 * np.sin(t), 0 * t + 0.9], 1), 1.0e4, 1, "Loop")
    coil(f, [(0.0, 0.0, -3.0)], 2.0e4, 2, "Line", term=(0.0, 0.0, 3.0))
    f.write("end\n")

# 101 groups, one small square loop each.
with open(f"{out}/coils.many", "w") as f:
    f.write(HDR)
    for g in range(101):
        z = 2.0 + 0.01 * g
        coil(f, [(1, 1, z), (-1, 1, z), (-1, -1, z), (1, -1, z)], 1.0e3 * (g + 1), g + 1, f"C{g + 1:03d}")
    f.write("end\n")

# 10,000-sided polygon of radius 1 with a vertex at (1, 0, 0), which is a grid
# point of rmin=0.5 rmax=1.5 ir=3, zmin=-0.5 zmax=0.5 jz=3, kp=1.
with open(f"{out}/coils.poly", "w") as f:
    f.write(HDR)
    n = 10000
    for k in range(n):
        t = 2 * np.pi * k / n
        f.write(f"{np.cos(t) if k else 1.0:.17E} {np.sin(t) if k else 0.0:.17E} 0.0 1.0E+03\n")
    f.write("1.0 0.0 0.0 0.0 1 poly\nend\n")
