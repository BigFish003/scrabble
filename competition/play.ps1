[CmdletBinding(DefaultParameterSetName = 'Tournament')]
param(
  [Parameter(Mandatory = $true, ParameterSetName = 'Tournament')][ValidatePattern('^\d{1,20}$')][string]$TournamentId,
  [Parameter(Mandatory = $true, ParameterSetName = 'Casual')][switch]$Casual,
  [ValidateSet('expert', 'calibrated', 'mcts')][string]$Policy = 'expert'
)
$ErrorActionPreference = 'Stop'
$nodeBinary = (Get-Command node -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1).Source
if (-not $nodeBinary -and (Get-Command Get-AppxPackage -ErrorAction SilentlyContinue)) {
  $runtime = Get-AppxPackage -Name 'OpenAI.CodexPrimaryRuntime*' | Sort-Object Version -Descending | Select-Object -First 1
  if ($runtime) {
    $candidate = Join-Path $runtime.InstallLocation 'dependencies\node\bin\node.exe'
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { $nodeBinary = $candidate }
  }
}
if (-not $nodeBinary) { throw 'Node 22.18 or newer is required. Install Node or launch from the project terminal.' }
$previousToken = $env:BOT_TOKEN
$secret = $null
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
  if (-not $previousToken) {
    $secret = Read-Host 'Team bot token (hidden)' -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
    try { $env:BOT_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
  }
  [string[]]$modeArgs = if ($Casual) { 'casual' } else { 'play', '--tournament', $TournamentId }
  & $nodeBinary --import tsx competition/main.ts @modeArgs --policy $Policy
  $status = $LASTEXITCODE
} finally {
  $env:BOT_TOKEN = $previousToken
  if ($null -ne $secret) { $secret.Dispose() }
  Pop-Location
}
exit $status
