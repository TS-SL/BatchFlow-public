# Fetch Microsoft.ML.OnnxRuntime NuGet (CPU) into third_party/onnxruntime.
param(
  [string]$Version = "1.32.0"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$downloads = Join-Path $Root "third_party\downloads"
$dest = Join-Path $Root "third_party\onnxruntime"
New-Item -ItemType Directory -Force -Path $downloads | Out-Null

$nupkg = Join-Path $downloads "microsoft.ml.onnxruntime.$Version.nupkg"
$url = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$Version/microsoft.ml.onnxruntime.$Version.nupkg"
if (-not (Test-Path $nupkg)) {
  Write-Host "Downloading $url"
  Invoke-WebRequest -Uri $url -OutFile $nupkg -UseBasicParsing
}

$extract = Join-Path $downloads "ort-nupkg-$Version"
if (Test-Path $extract) { Remove-Item $extract -Recurse -Force }
Copy-Item $nupkg "$nupkg.zip" -Force
Expand-Archive -Path "$nupkg.zip" -DestinationPath $extract -Force

if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
New-Item -ItemType Directory -Force -Path "$dest\include","$dest\lib","$dest\bin" | Out-Null
Copy-Item (Join-Path $extract "build\native\include\*") "$dest\include\" -Force
Copy-Item (Join-Path $extract "runtimes\win-x64\native\*") "$dest\lib\" -Force
Copy-Item (Join-Path $extract "runtimes\win-x64\native\onnxruntime.dll") "$dest\bin\" -Force
if (Test-Path (Join-Path $extract "runtimes\win-x64\native\onnxruntime_providers_shared.dll")) {
  Copy-Item (Join-Path $extract "runtimes\win-x64\native\onnxruntime_providers_shared.dll") "$dest\bin\" -Force
}

@"
name=Microsoft.ML.OnnxRuntime
version=$Version
source=$url
ep=CPU
"@ | Set-Content (Join-Path $dest "VERSION.txt")

Write-Host "Staged ORT $Version at $dest"
