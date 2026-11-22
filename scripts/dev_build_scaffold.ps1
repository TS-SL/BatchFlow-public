# Configure, build and test the dependency-free subset (dev-no-deps preset): the core
# library and smoke test only. Assumes Visual Studio 2022 Build Tools at the default path.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path","User")

$VsDev = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $VsDev)) {
  throw "VsDevCmd.bat not found at $VsDev"
}

cmd /c "call `"$VsDev`" -arch=x64 >nul && cmake --preset dev-no-deps && cmake --build --preset dev-no-deps && ctest --test-dir build-scaffold --output-on-failure"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
