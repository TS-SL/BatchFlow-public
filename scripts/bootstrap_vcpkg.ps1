# Bootstrap local vcpkg under tools/vcpkg and install manifest deps.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VcpkgDir = Join-Path $Root "tools\vcpkg"
if (-not (Test-Path (Join-Path $VcpkgDir ".git"))) {
  New-Item -ItemType Directory -Force -Path (Join-Path $Root "tools") | Out-Null
  git clone --depth 1 https://github.com/microsoft/vcpkg.git $VcpkgDir
}
Push-Location $VcpkgDir
& .\bootstrap-vcpkg.bat -disableMetrics
Pop-Location

$env:VCPKG_ROOT = $VcpkgDir
& "$VcpkgDir\vcpkg.exe" install --triplet x64-windows
Write-Host "vcpkg install finished. Configure with: cmake --preset default"
