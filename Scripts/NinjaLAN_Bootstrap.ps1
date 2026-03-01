#Requires -RunAsAdministrator
<#
.SYNOPSIS
    NinjaLAN one-time bootstrap.

.DESCRIPTION
    Creates the NinjaLAN directory tree under C:\NinjaLAN, copies all scripts
    from the repository into place, and then executes the first full startup
    run.  Safe to re-run — existing files are not overwritten unless -Force is
    used.

.PARAMETER Force
    Overwrite existing files in C:\NinjaLAN\Scripts.

.EXAMPLE
    Set-ExecutionPolicy Bypass -Scope Process -Force
    C:\NinjaLAN\Scripts\NinjaLAN_Bootstrap.ps1
#>

[CmdletBinding()]
param(
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$InstallRoot = 'C:\NinjaLAN'
$ScriptsDst  = "$InstallRoot\Scripts"
$DocsDst     = "$InstallRoot\docs"
$LogDir      = "$InstallRoot\Logs"

# Resolve the source directory (the repo Scripts\ folder next to this file).
$ScriptsSrc  = $PSScriptRoot
$DocsSrc     = Join-Path (Split-Path $PSScriptRoot -Parent) 'docs'

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Host "[$ts][INFO] $Message"
}

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Log "Created: $Path"
    }
}

Write-Log '=== NinjaLAN Bootstrap starting ==='

# 1. Create directory tree
foreach ($dir in @($InstallRoot, $ScriptsDst, $DocsDst, $LogDir)) {
    Ensure-Dir $dir
}

# 2. Copy scripts
$ps1Files = Get-ChildItem -Path $ScriptsSrc -Filter '*.ps1'
foreach ($f in $ps1Files) {
    $dest = Join-Path $ScriptsDst $f.Name
    if (-not (Test-Path $dest) -or $Force) {
        Copy-Item -Path $f.FullName -Destination $dest -Force
        Write-Log "Copied: $($f.Name) -> $ScriptsDst"
    } else {
        Write-Log "Skipped (exists): $($f.Name)"
    }
}

# 3. Copy docs (CSS theme etc.)
if (Test-Path $DocsSrc) {
    $docFiles = Get-ChildItem -Path $DocsSrc
    foreach ($f in $docFiles) {
        $dest = Join-Path $DocsDst $f.Name
        if (-not (Test-Path $dest) -or $Force) {
            Copy-Item -Path $f.FullName -Destination $dest -Force
            Write-Log "Copied: $($f.Name) -> $DocsDst"
        } else {
            Write-Log "Skipped (exists): $($f.Name)"
        }
    }
}

# 4. Run the installer to register scheduled tasks
$installScript = Join-Path $ScriptsDst 'NinjaLAN_Install.ps1'
if (Test-Path $installScript) {
    Write-Log 'Running installer (scheduled task registration)...'
    & $installScript
}

# 5. Execute first full startup
Write-Log 'Running first full startup...'
& "$ScriptsDst\NinjaLAN_Startup.ps1"

Write-Log '=== Bootstrap complete. Dashboard at %USERPROFILE%\Documents\NetworkDashboard.html ==='
