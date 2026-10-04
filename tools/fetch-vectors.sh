#!/usr/bin/env bash
# Downloads the rest of the public VP8 test vectors (bitstreams plus the MD5
# of every decoded frame) into tests/vectors/: the vp80-01 to vp80-06 sets
# (intra, inter, segmentation, partitions, sharpness, small sizes), 44
# streams, about 3 MB. The eighteen vp80-00-comprehensive streams are
# committed under tests/data/. Data published by the WebM project for
# decoder conformance testing:
#   https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/
set -euo pipefail
cd "$(dirname "$0")/.."
base=https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx
mkdir -p tests/vectors
while read -r name; do
  name="${name%$'\r'}"
  [ -z "$name" ] && continue
  for f in "$name.ivf" "$name.ivf.md5"; do
    [ -s "tests/vectors/$f" ] || curl -sSfL --retry 3 -o "tests/vectors/$f" "$base/$f"
  done
done < tools/vectors.txt
echo "tests/vectors: $(ls tests/vectors | grep -vc '\.md5$') streams"
