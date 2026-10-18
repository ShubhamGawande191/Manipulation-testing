#!/usr/bin/env bash
# Register the Lab board adapter in a Bot Crossing checkout (idempotent; replaces the old
# SO-101 adapter registration if present).
#   integrations/bot-crossing/install.sh [path/to/bot-crossing]   (default ~/tools/bot-crossing)
set -euo pipefail
BC="${1:-$HOME/tools/bot-crossing}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDX="$BC/server/harnesses/index.mjs"
[ -f "$IDX" ] || { echo "no Bot Crossing checkout at $BC" >&2; exit 1; }
# drop the old SO-101 adapter (renamed to lab.mjs)
sed -i "/^import so101 from '.\/so101.mjs'$/d; s/, so101\]$/]/" "$IDX"
rm -f "$BC/server/harnesses/so101.mjs"
ln -sfn "$HERE/lab.mjs" "$BC/server/harnesses/lab.mjs"   # symlink: repo edits apply live
if ! grep -q "from './lab.mjs'" "$IDX"; then
  sed -i "s|^import opencode from './opencode.mjs'$|&\nimport lab from './lab.mjs'|" "$IDX"
  sed -i "s|^export const HARNESSES = \[\(.*\)\]$|export const HARNESSES = [\1, lab]|" "$IDX"
fi
grep -q ", lab\]" "$IDX" && grep -q "from './lab.mjs'" "$IDX" \
  || { echo "could not register the adapter in $IDX; add it by hand" >&2; exit 1; }
node --check "$BC/server/harnesses/lab.mjs"
echo "Lab board adapter registered in $BC"
