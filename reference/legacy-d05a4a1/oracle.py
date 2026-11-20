"""fp64 prototype of spec v0.2.0 sections 2.4-3.4, plus a minimal CDF-1/2 reader.

usage: oracle.py coils_file mgrid.nc MODE(S|R) SYM(Y|N) rmin rmax zmin zmax kp ir jz
Compares legacy mgrid.nc field arrays with this implementation, with double
trig and with emulated single-precision cos/sin.
"""
import struct
import sys
import numpy as np


# ---------------------------------------------------------------- CDF reader
def read_cdf(path):
    b = open(path, "rb").read()
    assert b[:3] == b"CDF", "not classic netcdf"
    ver = b[3]
    pos = [4]

    def u32():
        v = struct.unpack_from(">I", b, pos[0])[0]
        pos[0] += 4
        return v

    def u64():
        v = struct.unpack_from(">Q", b, pos[0])[0]
        pos[0] += 8
        return v

    def name():
        n = u32()
        s = b[pos[0]:pos[0] + n].decode()
        pos[0] += (n + 3) // 4 * 4
        return s

    numrecs = u32()
    tag, n = u32(), u32()
    dims = [(name(), u32()) for _ in range(n)] if tag == 10 else []
    gtag, gn = u32(), u32()
    tsize = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8}

    def atts(n):
        out = []
        for _ in range(n):
            nm, t, ne = name(), u32(), u32()
            pos[0] += (ne * tsize[t] + 3) // 4 * 4
            out.append(nm)
        return out

    gatts = atts(gn) if gtag == 12 else []
    vtag, vn = u32(), u32()
    vars_ = {}
    order = []
    for _ in range(vn):
        nm = name()
        nd = u32()
        dimids = [u32() for _ in range(nd)]
        atag, an = u32(), u32()
        va = atts(an) if atag == 12 else []
        t, vsize = u32(), u32()
        begin = u64() if ver == 2 else u32()
        shape = [dims[d][1] for d in dimids]
        vars_[nm] = dict(dims=[dims[d][0] for d in dimids], shape=shape, type=t,
                         vsize=vsize, begin=begin, atts=va)
        order.append(nm)
    hdr_len = pos[0]

    def get(nm):
        v = vars_[nm]
        cnt = int(np.prod(v["shape"])) if v["shape"] else 1
        dt = {2: "S1", 4: ">i4", 5: ">f4", 6: ">f8"}[v["type"]]
        a = np.frombuffer(b, dt, cnt, v["begin"])
        return a.reshape(v["shape"]) if v["shape"] else a[0]

    return dict(version=ver, numrecs=numrecs, dims=dims, gatts=gatts, vars=vars_,
                order=order, hdr_len=hdr_len, get=get, size=len(b))


# ------------------------------------------------------------- coils parser
def parse_coils(path):
    lines = open(path).read().splitlines()
    assert "periods" in lines[0]
    nfp = int(lines[0].split()[1])
    coils, gids, names = [], [], []
    pts, cur_first = [], None
    for ln in lines[3:]:
        if ln[:3] == "end":
            break
        f = ln.split()
        if len(f) < 4:
            continue
        p = [float(x.replace("D", "E")) for x in f[:3]]
        c = float(f[3].replace("D", "E"))
        pts.append(p)
        if len(pts) == 1:
            cur_first = c
        if len(f) >= 6:
            gid = int(f[4])
            if gid not in gids:
                gids.append(gid)
                names.append(f[5])
            assert len(pts) > 1, "one-line (circular) coils are not supported"
            if len(pts) == 2:                 # two lines: one open straight segment
                coils.append((gids.index(gid), cur_first, np.array(pts), False))
                pts = []
                continue
            v = pts[:-1]                      # terminating vertex dropped
            kept = [v[0]]
            for q in v[1:]:                   # exact-duplicate consecutive removal
                if q != kept[-1]:
                    kept.append(q)
            if kept[-1] == kept[0]:           # duplicate wrap-around vertex
                kept.pop()
            coils.append((gids.index(gid), cur_first, np.array(kept), True))
            pts = []
    return nfp, names, coils


# ------------------------------------------------------------------- field
def field_group(coils, g, X):
    """Sum of file-current fields of group g at points X (n,3), fp64."""
    B = np.zeros_like(X)
    for gi, cur, v, closed in coils:
        if gi != g:
            continue
        a = v
        d = np.roll(v, -1, 0) - v             # closing segment included
        nseg = len(a) if closed else len(a) - 1
        dsq = (d * d).sum(1)
        eps = np.finfo(float).eps * dsq[:nseg].min()
        acc = np.zeros_like(X)
        nxt = np.roll(a, -1, 0)
        for s in range(nseg):                 # buffer order
            Ra = X - a[s]
            Rb = X - nxt[s]
            ra = np.sqrt(np.maximum(eps, (Ra * Ra).sum(1)))
            rb = np.sqrt(np.maximum(eps, (Rb * Rb).sum(1)))
            S = ra + rb
            fac = 2 * S / (ra * rb * np.maximum(eps, S * S - dsq[s]))
            acc += np.cross(d[s], Ra) * fac[:, None]
        B += 1e-7 * cur * acc
    return B


def compute(coils_path, mode, sym, rmin, rmax, zmin, zmax, kp, ir, jz, f32trig):
    nfp, names, coils = parse_coils(coils_path)
    ng = len(names)
    if sym:
        if abs(zmax) > abs(zmin):
            zmax = abs(zmax); zmin = -zmax
        else:
            zmin = -abs(zmin); zmax = -zmin
    R = rmin + np.arange(ir) * ((rmax - rmin) / (ir - 1))
    Z = zmin + np.arange(jz) * ((zmax - zmin) / (jz - 1))
    out, Ig = [], []
    for g in range(ng):
        cs = [c for gi, c, _, _ in coils if gi == g]
        Ig.append(cs[int(np.argmax(np.abs(cs)))])
        scale = 1.0 / Ig[-1] if mode == "S" else 1.0
        br = np.zeros((kp, jz, ir)); bp = np.zeros_like(br); bz = np.zeros_like(br)

        def plane(k, rows):
            phi = k * (2 * np.pi / (nfp * kp))
            c, s = np.cos(phi), np.sin(phi)
            if f32trig:
                c, s = float(np.float32(c)), float(np.float32(s))
            jj, ii = np.meshgrid(rows, np.arange(ir), indexing="ij")
            X = np.stack([R[ii] * c, R[ii] * s, Z[jj]], -1).reshape(-1, 3)
            B = field_group(coils, g, X).reshape(len(rows), ir, 3) * scale
            br[k, rows] = B[..., 0] * c + B[..., 1] * s
            bp[k, rows] = -B[..., 0] * s + B[..., 1] * c
            bz[k, rows] = B[..., 2]

        allrows = np.arange(jz)
        if not sym:
            for k in range(kp):
                plane(k, allrows)
        else:
            h, m = kp // 2, jz // 2 + jz % 2
            half = np.arange(m)

            def inplane(k):
                plane(k, half)
                for j in half:
                    jp = jz - 1 - j
                    br[k, jp] = -br[k, j]; bp[k, jp] = bp[k, j]; bz[k, jp] = bz[k, j]

            inplane(0)
            for k in range(1, h + kp % 2):
                plane(k, allrows)
                br[kp - k] = -br[k, ::-1]; bp[kp - k] = bp[k, ::-1]; bz[kp - k] = bz[k, ::-1]
            if kp % 2 == 0:
                inplane(h)
        out.append((br, bp, bz))
    return names, Ig, out, (zmin, zmax)


if __name__ == "__main__":
    a = sys.argv
    coils_path, nc_path, mode, sym = a[1], a[2], a[3].upper(), a[4].upper() == "Y"
    rmin, rmax, zmin, zmax = map(float, a[5:9])
    kp, ir, jz = map(int, a[9:12])
    nc = read_cdf(nc_path)
    for f32 in (False, True):
        names, Ig, out, zb = compute(coils_path, mode, sym, rmin, rmax, zmin, zmax, kp, ir, jz, f32)
        worst = 0.0
        for g, (br, bp, bz) in enumerate(out):
            ref = [nc["get"](f"{c}_{g + 1:03d}") for c in ("br", "bp", "bz")]
            mag = np.sqrt(ref[0] ** 2 + ref[1] ** 2 + ref[2] ** 2).max()
            e = [np.abs(x - y).max() / mag for x, y in zip((br, bp, bz), ref)]
            worst = max(worst, *e)
            print(f"  trig={'f32' if f32 else 'f64'} g{g + 1}: eps br/bp/bz = "
                  f"{e[0]:.2e} {e[1]:.2e} {e[2]:.2e}")
        print(f"  trig={'f32' if f32 else 'f64'} WORST eps = {worst:.2e}")
    rc = nc["get"]("raw_coil_cur")
    print("  raw_coil_cur legacy:", np.atleast_1d(rc).tolist())
    print("  raw_coil_cur oracle:", Ig)
    print("  zmin/zmax legacy:", nc["get"]("zmin"), nc["get"]("zmax"), " oracle:", zb)
    cg = nc["get"]("coil_group")
    print("  coil_group:", [x.tobytes().decode() for x in np.atleast_2d(cg)], " oracle order:", names)
