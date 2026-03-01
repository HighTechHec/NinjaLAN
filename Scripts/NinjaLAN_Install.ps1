#Requires -RunAsAdministrator
<#
.SYNOPSIS
    NinjaLAN scheduled-task installer.

.DESCRIPTION
    Registers the three Windows Task Scheduler tasks that drive the NinjaLAN
    automation suite:
      - NinjaLAN-Startup   : runs at system boot (highest privileges)
      - NinjaLAN-Hourly    : maintenance run every hour
      - NinjaLAN-HealthCheck: lightweight check every 5 minutes

    Safe to re-run — existing tasks are replaced (/F flag).

.EXAMPLE
    .\Scripts\NinjaLAN_Install.ps1
#>

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptPath = 'C:\NinjaLAN\Scripts\NinjaLAN_Startup.ps1'
$PsExe      = 'powershell.exe'
$BaseArgs   = "-ExecutionPolicy Bypass -NonInteractive -File `"$ScriptPath`""

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Host "[$ts][INFO] $Message"
}

function Register-NinjaTask {
    param(
        [string]$TaskName,
        [string]$Arguments,
        [string]$ScheduleArgs
    )
    $cmd = "schtasks /Create /TN `"$TaskName`" /TR `"$PsExe $Arguments`" $ScheduleArgs /RL HIGHEST /F"
    Write-Log "Registering task: $TaskName"
    Invoke-Expression $cmd | Out-Null
    Write-Log "  OK: $TaskName"
}

Write-Log '=== NinjaLAN Install: registering scheduled tasks ==='

# Startup task — runs at every system boot
Register-NinjaTask `
    -TaskName   'NinjaLAN-Startup' `
    -Arguments  $BaseArgs `
    -ScheduleArgs '/SC ONSTART'

# Hourly maintenance task
Register-NinjaTask `
    -TaskName   'NinjaLAN-Hourly' `
    -Arguments  "$BaseArgs -Maintenance" `
    -ScheduleArgs '/SC HOURLY /MO 1'

# Health-check task every 5 minutes
Register-NinjaTask `
    -TaskName   'NinjaLAN-HealthCheck' `
    -Arguments  "$BaseArgs -HealthCheck" `
    -ScheduleArgs '/SC MINUTE /MO 5'

Write-Log '=== Installation complete. Tasks registered: ==='
schtasks /Query /TN 'NinjaLAN-Startup'   /FO LIST 2>&1 | Where-Object { $_ -match 'Task Name|Status|Next Run' } | ForEach-Object { Write-Log "  $_" }
schtasks /Query /TN 'NinjaLAN-Hourly'    /FO LIST 2>&1 | Where-Object { $_ -match 'Task Name|Status|Next Run' } | ForEach-Object { Write-Log "  $_" }
schtasks /Query /TN 'NinjaLAN-HealthCheck' /FO LIST 2>&1 | Where-Object { $_ -match 'Task Name|Status|Next Run' } | ForEach-Object { Write-Log "  $_" }
