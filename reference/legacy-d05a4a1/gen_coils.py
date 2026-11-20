"""Generate small synthetic coils.<ext> fixtures for legacy-parity checks."""
import sys
import numpy as np

NFP = 2


def modular(phi0, nv=24, R0=1.0, a=0.5, tilt=0.15, wob=0.06):
    t = np.arange(nv) * 2 * np.pi / nv
    r = R0 + a * np.cos(t)
    z = a * np.sin(t)
    ph = phi0 + tilt * np.sin(t) + wob * np.cos(2 * t)
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)


def circle(rad, z, nv=40):
    t = np.arange(nv) * 2 * np.pi / nv
    return np.stack([rad * np.cos(t), rad * np.sin(t), np.full(nv, z)], 1)


def stell_image(p):  # (x, y, z) -> (x, -y, -z), a proper rotation about x
    return p * np.array([1.0, -1.0, -1.0])


def rot(p, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]], 1)


def sym_set(p):
    out = []
    for q in (p, stell_image(p)):
        for n in range(NFP):
            out.append(rot(q, 2 * np.pi * n / NFP))
    return out


def write(path, coils):
    """coils: list of (points, current, gid, name, opts)"""
    with open(path, "w") as f:
        f.write(f"periods {NFP}\nbegin filament\nmirror NIL\n")
        for pts, cur, gid, name, opts in coils:
            pts = list(pts)
            if opts.get("dup"):  # duplicate a consecutive vertex
                pts.insert(5, pts[5])
            for i, p in enumerate(pts):
                # later-vertex currents deliberately differ when opts['varcur']
                c = cur * (1.0 + 0.5 * i) if opts.get("varcur") and i > 0 else cur
                f.write(f"{p[0]:.14E} {p[1]:.14E} {p[2]:.14E} {c:.14E}\n")
            term = opts.get("term", pts[0])
            f.write(f"{term[0]:.14E} {term[1]:.14E} {term[2]:.14E} {0.0:.14E} {gid} {name}\n")
        f.write("end\n")


def case_two():
    coils = []
    mods = sym_set(modular(0.35))
    for n, p in enumerate(mods):
        o = {}
        if n == 0:
            o["dup"] = True
        if n == 1:
            o["term"] = (9.0, 9.0, 9.0)  # terminator that is NOT the first vertex
        if n == 2:
            o["varcur"] = True
        coils.append((p, 1.0e5, 7, "ModA", o))
    # group id 3 appears after id 7; mixed-magnitude, mixed-sign currents
    for p in sym_set(circle(1.9, 0.7))[::NFP]:
        coils.append((p, 5.0e3, 3, "PF1", {}))
    for p in sym_set(circle(0.3, 0.8))[::NFP]:
        coils.append((p, -2.0e4, 3, "PF1", {}))
    return coils


def case_one():
    return [(p, -3.0e4, 1, "AVeryLongGroupNameThatExceedsThirtyCharacters", {}) for p in sym_set(modular(0.35))]


def case_eleven():
    coils = []
    for g in range(11):
        for p in sym_set(modular(0.1 + 0.06 * g, a=0.45 + 0.01 * g)):
            coils.append((p, 1.0e4 * (g + 1) * (-1) ** g, 20 - g, f"G{g:02d}", {}))
    return coils


if __name__ == "__main__":
    out = sys.argv[1]
    write(f"{out}/coils.two", case_two())
    write(f"{out}/coils.one", case_one())
    write(f"{out}/coils.eleven", case_eleven())
