param([string]$Map = 'goblin_codex/showroom')
$ErrorActionPreference = 'Stop'
$worktreeRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$testScript = Join-Path $PSScriptRoot 'test_game.py'
if (-not (Test-Path -LiteralPath (Join-Path $worktreeRoot 'out\goblin-codex\game\gameinfo.txt'))) {
    throw 'Build the isolated mod first. See assets/goblin-codex/README.md.'
}
Push-Location -LiteralPath $worktreeRoot
try {
    py -3 $testScript --launch --map $Map --seconds 5
    if ($LASTEXITCODE -ne 0) { throw 'Portal 2 test launch failed.' }
    py -3 $testScript --command 'sv_cheats 1' --command 'setmodel info_character/info_character_player' --command 'give weapon_portalgun' --command 'thirdperson' --command 'cam_collision 0' --command 'cam_snapto 1' --command 'cam_idealdist 85' --command 'cam_idealyaw 135' --command 'cam_idealpitch 15' --command 'cl_drawhud 0' --command 'ent_fire goblin_display disable' --seconds 2
    if ($LASTEXITCODE -ne 0) { throw 'Goblin preview setup failed.' }
} finally {
    Pop-Location
}
