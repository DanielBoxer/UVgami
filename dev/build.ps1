# stages the same file set as release.yml, then validates the manifest
param(
    [string]$Blender,
    [string]$Output
)

$ErrorActionPreference = "Stop"

$repo = Split-Path $PSScriptRoot -Parent

if (-not $Blender) {
    $cmd = Get-Command blender -ErrorAction SilentlyContinue
    if ($cmd) {
        $Blender = $cmd.Source
    } else {
        $found = Get-ChildItem "$env:ProgramFiles\Blender Foundation\Blender *\blender.exe" -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | Select-Object -First 1
        if (-not $found) { throw "blender.exe not found, pass -Blender" }
        $Blender = $found.FullName
    }
}

$staging = Join-Path ([System.IO.Path]::GetTempPath()) "uvgami-build"
if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
New-Item -ItemType Directory $staging | Out-Null

# must match INCLUDED_FILES in release.yml
foreach ($item in "src", "LICENSE", "__init__.py", "blender_manifest.toml") {
    Copy-Item (Join-Path $repo $item) $staging -Recurse
}

if (-not $Output) {
    $manifest = Join-Path $repo "blender_manifest.toml"
    $version = (Select-String '^version = "(.*)"' $manifest).Matches[0].Groups[1].Value
    $dist = Join-Path $repo "dist"
    New-Item -ItemType Directory -Force $dist | Out-Null
    $Output = Join-Path $dist "UVgami-$version.zip"
}

& $Blender --factory-startup --command extension build --source-dir $staging --output-filepath $Output
if ($LASTEXITCODE -ne 0) { throw "extension build failed" }

Remove-Item -Recurse -Force $staging
Write-Host "built $Output"
