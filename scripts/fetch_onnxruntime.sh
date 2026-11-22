#!/usr/bin/env bash
# Fetch Microsoft.ML.OnnxRuntime NuGet (CPU) for Linux x64 into third_party/onnxruntime.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-1.32.0}"
DOWNLOADS="$ROOT/third_party/downloads"
DEST="$ROOT/third_party/onnxruntime"
mkdir -p "$DOWNLOADS"
NUPKG="$DOWNLOADS/microsoft.ml.onnxruntime.${VERSION}.nupkg"
URL="https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${VERSION}/microsoft.ml.onnxruntime.${VERSION}.nupkg"
if [[ ! -f "$NUPKG" ]]; then
  curl -fsSL "$URL" -o "$NUPKG"
fi
EXTRACT="$DOWNLOADS/ort-nupkg-${VERSION}"
rm -rf "$EXTRACT"
mkdir -p "$EXTRACT"
unzip -q "$NUPKG" -d "$EXTRACT"
rm -rf "$DEST"
mkdir -p "$DEST/include" "$DEST/lib" "$DEST/bin"
cp -a "$EXTRACT"/build/native/include/. "$DEST/include/"
cp -a "$EXTRACT"/runtimes/linux-x64/native/. "$DEST/lib/"
# Ensure an unversioned .so name exists for CMake linking.
if [[ ! -e "$DEST/lib/libonnxruntime.so" ]]; then
  ver_so="$(ls "$DEST"/lib/libonnxruntime.so.* 2>/dev/null | head -n 1 || true)"
  if [[ -n "$ver_so" ]]; then
    ln -s "$(basename "$ver_so")" "$DEST/lib/libonnxruntime.so"
  fi
fi
cp -a "$DEST"/lib/libonnxruntime.so* "$DEST/bin/" 2>/dev/null || true
cat > "$DEST/VERSION.txt" <<EOF
name=Microsoft.ML.OnnxRuntime
version=$VERSION
source=$URL
ep=CPU
platform=linux-x64
EOF
echo "Staged ORT $VERSION at $DEST"
