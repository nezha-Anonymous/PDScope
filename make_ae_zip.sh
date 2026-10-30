#!/usr/bin/env bash
# Build the Artifact-Evaluation archive: the shipped artifact plus the paper PDF and the
# AE guide, without model weights, build products, run outputs or local-only notes.
#
# Usage: ./make_ae_zip.sh [output.zip]
set -euo pipefail
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=${1:-"$root_dir/pdscope-artifact.zip"}
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

mkdir -p "$stage/pdscope"
tar -cf - -C "$root_dir" \
    --exclude='./.git' --exclude='./.cache' --exclude='./runs' --exclude='./research' \
    --exclude='./expertbench/native-build' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='*.so' --exclude='./README_ZH_new.md' --exclude='./AE.md' \
    --exclude='./AE_HotCRP_fields.md' --exclude='./pdscope-artifact.zip' \
    . | tar -xf - -C "$stage/pdscope"

rm -f "$out"
(cd "$stage" && zip -qr "$out" pdscope)
size=$(stat -c %s "$out")
echo "wrote $out ($size bytes, $(du -h "$out" | cut -f1))"
if [ "$size" -gt 600000000 ]; then
  echo "ERROR: archive exceeds the 600 MB AE limit" >&2
  exit 2
fi
unzip -l "$out" | tail -3
