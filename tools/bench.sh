#!/usr/bin/env bash
# End-to-end speed at 720p and 1080p: encode and decode, one thread and
# many, SIMD and scalar kernels. Usage:
#
#   tools/bench.sh SOURCE.y4m [THREADS] [OUTDIR]
#
# SOURCE is any 8-bit 4:2:0 Y4M clip (it is scaled to each size; 30 frames
# are used, repeating if the clip is shorter). THREADS defaults to the
# number of CPUs. Encoded streams go to OUTDIR (default: a temporary
# directory). Each figure is the fastest of several runs; run on an idle
# machine. To compare with another revision, set BASE to that revision's
# vp8bench binary (built from this same example) and its figures are
# printed alongside, interleaved run by run.
set -euo pipefail
src=${1:?usage: tools/bench.sh SOURCE.y4m [THREADS] [OUTDIR]}
threads=${2:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo "${NUMBER_OF_PROCESSORS:-8}")}
out=${3:-$(mktemp -d)}
cd "$(dirname "$0")/.."
cargo build --release --example vp8bench >/dev/null
target=$(cargo metadata --format-version 1 --no-deps | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')
target=${target:-target}
bin="$target/release/examples/vp8bench"
[ -x "$bin" ] || bin="$bin.exe"
run() { "$@" | sed 's/^/  /'; }
for size in 1280x720 1920x1080; do
  echo "== $size"
  for t in 1 "$threads"; do
    p=$([ "$t" = 1 ] && echo 1 || echo 8)
    ivf="$out/vp8-$size-p$p.ivf"
    echo "-- $t thread(s), $p token partition(s)"
    run "$bin" "$src" --size "$size" --frames 30 --threads "$t" --partitions "$p" --enc-reps 3 --reps 10 --out "$ivf"
    if [ "$t" = 1 ]; then
      echo "-- scalar kernels"
      VP8_FORCE_SCALAR=1 run "$bin" "$src" --size "$size" --frames 30 --threads 1 --enc-reps 1 --reps 5
    fi
    if [ -n "${BASE:-}" ] && [ "$t" = 1 ]; then
      echo "-- BASE ($BASE)"
      run "$BASE" "$src" --size "$size" --frames 30 --enc-reps 1 --reps 5
    fi
  done
  echo "-- decode the 1-partition stream on $threads threads"
  run "$bin" --ivf "$out/vp8-$size-p1.ivf" --threads "$threads" --reps 10
done
