param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
# Keep the build usable in an offline development checkout.  PyInstaller is
# optional; the source fallback below still produces a runnable distribution.
& $Python -m pip install -e . --no-build-isolation
if (Get-Command pyinstaller -ErrorAction SilentlyContinue) {
  & pyinstaller --noconfirm --clean --onefile --name AgentArena --paths . agent_arena/__main__.py
  Write-Host "Built dist/AgentArena.exe"
} else {
  Write-Warning "PyInstaller is not installed; creating a source distribution instead."
  New-Item -ItemType Directory -Force -Path dist\AgentArena | Out-Null
  Copy-Item -Recurse -Force agent_arena dist\AgentArena\
  Copy-Item -Force README.md, pyproject.toml dist\AgentArena\
  Write-Host "Source runtime copied to dist/AgentArena"
}
