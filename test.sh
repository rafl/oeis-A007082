#!/bin/bash

set -euo pipefail

BFILE='b007082.txt'
MAX_N="${1:-7}"
fail=0

declare -a VAL
while read -r idx val; do
  [[ -z "$idx" || "$idx" == \#* ]] && continue
  VAL[idx]=$val
done < "$BFILE"

for (( n = 1; n <= MAX_N; n++ )); do
  k=$(( 2*n + 1 ))
  expected="${VAL[n]:-}"

  if [[ -z "$expected" ]]; then
    echo "⚠️  Missing value for OEIS n=$n in $BFILE"
    fail=1
    continue
  fi

  got="$(./oeis -q "$k")"

  if [[ "$got" != "$expected" ]]; then
    printf '❌  n=%d (k=%d): expected %s, got %s\n' "$n" "$k" "$expected" "$got"
    fail=1
  else
    printf '✅  n=%d OK\n' "$n"
  fi
done

if (( fail )); then
  echo "Some tests FAILED"
  exit 1
else
  echo "All tests passed ✔"
fi
