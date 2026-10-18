#!/usr/bin/env bash
# Register the SO-101 adapter in a Bot Crossing checkout (idempotent).
#   integrations/bot-crossing/install.sh [path/to/bot-crossing]   (default ~/tools/bot-crossing)
set -euo pipefail
BC="${1:-$HOME/tools/bot-crossing}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDX="$BC/server/harnesses/index.mjs"
[ -f "$IDX" ] || { echo "no Bot Crossing checkout at $BC" >&2; exit 1; }
ln -sfn "$HERE/so101.mjs" "$BC/server/harnesses/so101.mjs"   # symlink: repo edits apply live
if ! grep -q "from './so101.mjs'" "$IDX"; then
  sed -i "s|^import opencode from './opencode.mjs'$|&\nimport so101 from './so101.mjs'|" "$IDX"
  sed -i "s|^export const HARNESSES = \[\(.*\)\]$|export const HARNESSES = [\1, so101]|" "$IDX"
fi
grep -q "so101 \]\|, so101\]" "$IDX" && grep -q "from './so101.mjs'" "$IDX" \
  || { echo "could not register the adapter in $IDX; add it by hand" >&2; exit 1; }
node --check "$BC/server/harnesses/so101.mjs"
echo "SO-101 adapter registered in $BC"
