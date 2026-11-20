#!/bin/sh
# GPU backend gate (spec section 1.2): run every stored case on the CPU
# reference and on a GPU backend, and compare.
#
#   fp64: normalised difference <= 1e-13
#   ds:   normalised difference <= 1e-10 (spec section 1.2 explains why)
#   fp32: reported, not gated
#   Cases with grid points on a filament only have to be finite.
#
# usage: tests/run_gpu.sh metal|cuda [reference-dir]     (after `zig build --release=fast`)
#   metal: macOS only; tests ds and fp32
#   cuda:  tests fp64, ds and fp32. Without an NVIDIA GPU, set
#          XGRID_LIBCUDA=zig-out/lib/libcuda_mock.dylib (or .so) to run the
#          kernels on the CPU through the mock driver (tests/cuda_mock/).
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
backend=${1:?usage: run_gpu.sh metal|cuda [reference-dir]}
ref=${2:-$root/reference/legacy-d05a4a1}
case $backend in
  metal) precs="ds fp32" ;;
  cuda)  precs="fp64 ds fp32" ;;
  *) echo "unknown backend $backend" >&2; exit 2 ;;
esac
bin=$root/zig-out/bin
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
fail=0

while read -r name ext mode sym rmin rmax zmin zmax kp ir jz rest; do
  case "$name" in ''|\#*) continue ;; esac
  [ -f "$ref/cases/$name/coils.$ext" ] || continue
  d=$work/$name
  mkdir -p "$d"
  cp "$ref/cases/$name/coils.$ext" "$d/"
  symflag=; case "$sym" in Y|y) symflag=--sym ;; esac
  run() { # run <backend> <precision> <out>
    (cd "$d" && "$bin/xgrid-gpu" --coils "coils.$ext" --mode "$mode" $symflag --rmin "$rmin" --rmax "$rmax" \
        --zmin "$zmin" --zmax "$zmax" --kp "$kp" --ir "$ir" --jz "$jz" --backend "$1" --precision "$2" \
        --out "$3" --extcur "$3.extcur" --quiet 2>"$3.err")
  }
  run cpu fp64 cpu.nc || { echo "FAIL $name: cpu run failed"; fail=1; continue; }
  onwire=no; grep -q 'lie on a filament' "$d/cpu.nc.err" && onwire=yes
  line=$(printf '%-16s' "$name"); res=PASS
  for p in $precs; do
    if ! run "$backend" "$p" "$p.nc"; then
      line="$line $p FAILED TO RUN: $(cat "$d/$p.nc.err")"; res=FAIL; continue
    fi
    case $p in fp64) tol=1e-13 ;; ds) tol=1e-10 ;; *) tol=1e300 ;; esac
    out=$("$bin/compare_mgrid" "$d/cpu.nc" "$d/$p.nc" "$tol"); status=$?
    eps=$(echo "$out" | sed -n 's/.*worst eps = \([^ ]*\).*/\1/p')
    if [ $status -ne 0 ]; then
      if [ $onwire = yes ] && ! echo "$out" | grep -q 'non-finite values = [1-9]'; then
        eps="$eps*"
      else
        eps="$eps!"; res=FAIL
      fi
    fi
    line="$line $p $(printf '%-11s' "$eps")"
  done
  [ $res = PASS ] || fail=1
  echo "$res $line"
done <"$ref/cases/ARGS.txt"
echo "(* on-filament case: tolerance not applied, output finite; ! over tolerance)"
exit $fail
