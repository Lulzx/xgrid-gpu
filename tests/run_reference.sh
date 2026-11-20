#!/bin/sh
# Parity gate, fast tier (spec section 7.3): run xgrid-gpu on every stored
# legacy reference case and compare with the legacy output.
#
#   mgrid header structure and small variables: exact
#   field arrays, with --legacy-fp32-trig:      normalised difference <= 1e-12
#   extcur file:                                byte-identical
#     (cases with >= 100 groups: first 99 lines only; see spec section 5.2)
#
# usage: tests/run_reference.sh [reference-dir]     (run after `zig build --release=fast`)
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
ref=${1:-$root/reference/legacy-d05a4a1}
bin=$root/zig-out/bin
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
fail=0

while read -r name ext mode sym rmin rmax zmin zmax kp ir jz rest; do
  case "$name" in ''|\#*) continue ;; esac
  [ -f "$ref/cases/$name/mgrid_$ext.nc" ] || { echo "SKIP $name (no stored mgrid)"; continue; }
  d=$work/$name
  mkdir -p "$d"
  cp "$ref/cases/$name/coils.$ext" "$d/"
  if ! (cd "$d" && "$bin/xgrid-gpu" "$ext" "$mode" "$sym" "$rmin" "$rmax" "$zmin" "$zmax" "$kp" "$ir" "$jz" --legacy-fp32-trig >run.log 2>&1); then
    echo "FAIL $name: xgrid-gpu exited with an error"; sed 's/^/     /' "$d/run.log"; fail=1; continue
  fi
  cmp_out=$("$bin/compare_mgrid" "$ref/cases/$name/mgrid_$ext.nc" "$d/mgrid_$ext.nc" 1e-12)
  status=$?
  # Default (double-precision) trig, reported but not gated.
  (cd "$d" && "$bin/xgrid-gpu" "$ext" "$mode" "$sym" "$rmin" "$rmax" "$zmin" "$zmax" "$kp" "$ir" "$jz" >run2.log 2>&1)
  dflt=$("$bin/compare_mgrid" "$ref/cases/$name/mgrid_$ext.nc" "$d/mgrid_$ext.nc" 1 | sed -n 's/.*worst eps = \([^ ]*\).*/\1/p')
  ngroups=$(grep -c . "$ref/cases/$name/extcur.$ext")
  if grep -q '(\*\*)' "$ref/cases/$name/extcur.$ext"; then
    head -99 "$ref/cases/$name/extcur.$ext" >"$d/e.ref"; head -99 "$d/extcur.$ext" >"$d/e.tst"
  else
    cp "$ref/cases/$name/extcur.$ext" "$d/e.ref"; cp "$d/extcur.$ext" "$d/e.tst"
  fi
  if cmp -s "$d/e.ref" "$d/e.tst"; then ext_ok=identical; else ext_ok=DIFFERS; status=1; fi
  hdr=$(echo "$cmp_out" | sed -n 's/^header: structure identical; bytes \([A-Za-z]*\).*/\1/p')
  eps=$(echo "$cmp_out" | sed -n 's/.*worst eps = \([^ ]*\).*/\1/p')
  # Points lying exactly on a filament are not comparable: the result there is
  # rounding noise amplified by the floored denominator. Such cases only have
  # to be finite (spec section 3.2).
  note=
  if [ $status -ne 0 ] && [ "$ext_ok" = identical ] && grep -q 'lie on a filament' "$d/run.log" \
     && ! echo "$cmp_out" | grep -q -e 'non-finite values = [1-9]' -e 'FAIL variable' -e 'FAIL dim' -e 'FAIL format' -e 'FAIL global'; then
    status=0; note=' (on-filament points: finite, tolerance not applied)'
  fi
  if [ $status -eq 0 ]; then res=PASS; else res=FAIL; fail=1; fi
  printf '%s %-14s eps %-10s (default trig %-10s) header bytes %-9s extcur %s%s\n' "$res" "$name" "$eps" "$dflt" "$hdr" "$ext_ok" "$note"
  [ $status -eq 0 ] || echo "$cmp_out" | sed 's/^/     /'
done <"$ref/cases/ARGS.txt"

exit $fail
