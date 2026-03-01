#Requires -RunAsAdministrator
<#
.SYNOPSIS
    NinjaLAN main orchestration script.

.DESCRIPTION
    Full startup orchestration, hourly maintenance, and health-check modes for
    the NinjaLAN automation suite.  Manages network adapter optimisation,
    security hardening, WireGuard VPN, Hyper-V VM lifecycle, failover, and the
    live HTML dashboard.

.PARAMETER Maintenance
    Run hourly maintenance tasks only (adapter refresh, VPN check, dashboard).

.PARAMETER HealthCheck
    Run a lightweight 5-minute health check (VPN handshake, adapter status).

.EXAMPLE
    # Full startup (run at system boot via scheduled task)
    C:\NinjaLAN\Scripts\NinjaLAN_Startup.ps1

.EXAMPLE
    # Hourly maintenance
    C:\NinjaLAN\Scripts\NinjaLAN_Startup.ps1 -Maintenance

.EXAMPLE
    # 5-minute health check
    C:\NinjaLAN\Scripts\NinjaLAN_Startup.ps1 -HealthCheck
#>

[CmdletBinding(DefaultParameterSetName = 'Startup')]
param(
    [Parameter(ParameterSetName = 'Maintenance')]
    [switch]$Maintenance,

    [Parameter(ParameterSetName = 'HealthCheck')]
    [switch]$HealthCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
$ScriptRoot   = 'C:\NinjaLAN'
$LogDir       = "$ScriptRoot\Logs"
$DashboardOut = "$env:USERPROFILE\Documents\NetworkDashboard.html"
$DashboardCSS = "$ScriptRoot\docs\dashboard-theme.css"

$WgExe         = "$env:ProgramFiles\WireGuard\wg.exe"
$WgConfDir     = "$env:ProgramFiles\WireGuard\Data\Configurations"
$TunnelName    = 'NinjaLAN'
$WgServiceName = "WireGuardTunnel`$$TunnelName"

$VmName        = 'NinjaVM'
$VSwitchName   = 'NinjaLAN'

$PrimaryAdapter = 'Ethernet'
$FallbackAdapter= 'Wi-Fi'
$TargetAdapter  = 'Ethernet 3'   # NetBIOS hardening target
$PreferredDNS   = @('1.1.1.1', '8.8.8.8')

$StaleHandshakeThresholdSeconds = 180   # restart WG if handshake older than 3 min

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
function Write-Log {
    param([string]$Message, [string]$Level = 'INFO')
    $ts   = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "[$ts][$Level] $Message"
    Write-Host $line
    $logFile = Join-Path $LogDir "NinjaLAN_$(Get-Date -Format 'yyyy-MM-dd').log"
    Add-Content -Path $logFile -Value $line -ErrorAction SilentlyContinue
}

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Log "Created directory: $Path"
    }
}

# ---------------------------------------------------------------------------
# Network adapter optimisation
# ---------------------------------------------------------------------------
function Set-AdapterMetrics {
    Write-Log 'Setting interface metrics (wired first, Wi-Fi fallback)...'

    $wired = Get-NetAdapter -Name $PrimaryAdapter -ErrorAction SilentlyContinue
    if ($wired -and $wired.Status -eq 'Up') {
        Set-NetIPInterface -InterfaceIndex $wired.ifIndex -InterfaceMetric 10 -ErrorAction SilentlyContinue
        Write-Log "  $PrimaryAdapter metric -> 10"
    }

    $wifi = Get-NetAdapter -Name $FallbackAdapter -ErrorAction SilentlyContinue
    if ($wifi) {
        Set-NetIPInterface -InterfaceIndex $wifi.ifIndex -InterfaceMetric 50 -ErrorAction SilentlyContinue
        Write-Log "  $FallbackAdapter metric -> 50"
    }
}

function Set-PrivacyDNS {
    Write-Log 'Enforcing privacy DNS servers...'
    $adapters = Get-NetAdapter | Where-Object { $_.Status -eq 'Up' }
    foreach ($a in $adapters) {
        try {
            Set-DnsClientServerAddress -InterfaceIndex $a.ifIndex -ServerAddresses $PreferredDNS
            Write-Log "  $($a.Name): DNS -> $($PreferredDNS -join ', ')"
        } catch {
            Write-Log "  $($a.Name): DNS set failed — $_" 'WARN'
        }
    }
}

function Add-StaticRoutes {
    Write-Log 'Adding persistent static routes...'
    # Add routes to key IPs through the primary wired adapter.
    $wired = Get-NetAdapter -Name $PrimaryAdapter -ErrorAction SilentlyContinue
    if (-not $wired -or $wired.Status -ne 'Up') { Write-Log '  Primary adapter not up; skipping static routes.' 'WARN'; return }

    $gw = (Get-NetRoute -InterfaceIndex $wired.ifIndex -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue |
           Select-Object -First 1).NextHop
    if (-not $gw) { Write-Log '  No default gateway found on primary adapter.' 'WARN'; return }

    $staticDestinations = @('1.1.1.1/32', '8.8.8.8/32')
    foreach ($dest in $staticDestinations) {
        $existing = Get-NetRoute -DestinationPrefix $dest -ErrorAction SilentlyContinue
        if (-not $existing) {
            New-NetRoute -DestinationPrefix $dest -InterfaceIndex $wired.ifIndex -NextHop $gw -RouteMetric 1 -PolicyStore PersistentStore -ErrorAction SilentlyContinue
            Write-Log "  Added route $dest via $gw"
        }
    }
}

function Set-JumboFrames {
    Write-Log 'Setting jumbo frames on Realtek NICs...'
    $realtekAdapters = Get-NetAdapter | Where-Object { $_.InterfaceDescription -match 'Realtek' }
    foreach ($a in $realtekAdapters) {
        try {
            Set-NetAdapterAdvancedProperty -Name $a.Name -DisplayName 'Jumbo Packet' -DisplayValue '9014 Bytes' -ErrorAction SilentlyContinue
            Write-Log "  $($a.Name): Jumbo frames -> 9014"
        } catch {
            Write-Log "  $($a.Name): Jumbo frame set failed — $_" 'WARN'
        }
    }
}

function Disable-NicPowerSaving {
    Write-Log 'Disabling NIC power saving...'
    $adapters = Get-NetAdapter | Where-Object { $_.Status -eq 'Up' }
    foreach ($a in $adapters) {
        try {
            Disable-NetAdapterPowerManagement -Name $a.Name -ErrorAction SilentlyContinue
            Write-Log "  $($a.Name): power management disabled"
        } catch {
            Write-Log "  $($a.Name): power mgmt change skipped — $_" 'WARN'
        }
    }
}

function Optimize-NetworkAdapters {
    Set-AdapterMetrics
    Set-PrivacyDNS
    Add-StaticRoutes
    Set-JumboFrames
    Disable-NicPowerSaving
}

# ---------------------------------------------------------------------------
# Security hardening
# ---------------------------------------------------------------------------
function Disable-NetBIOSOnTarget {
    Write-Log "Disabling NetBIOS on $TargetAdapter..."
    $adapter = Get-WmiObject Win32_NetworkAdapterConfiguration |
               Where-Object { $_.Description -like "*$TargetAdapter*" -or
                               (Get-NetAdapter -Name $TargetAdapter -EA SilentlyContinue |
                                Select-Object -ExpandProperty InterfaceIndex) -eq $_.InterfaceIndex }
    if ($adapter) {
        $adapter | ForEach-Object { $_.SetTcpipNetbios(2) | Out-Null }
        Write-Log "  NetBIOS disabled on $TargetAdapter"
    } else {
        Write-Log "  Adapter '$TargetAdapter' not found; skipping." 'WARN'
    }
}

function Block-mDNS {
    Write-Log 'Blocking mDNS (UDP/5353) inbound and outbound...'
    $ruleNames = @('Block mDNS Inbound', 'Block mDNS Outbound')
    if (-not (Get-NetFirewallRule -DisplayName $ruleNames[0] -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $ruleNames[0] -Direction Inbound  -Protocol UDP -LocalPort 5353 -Action Block | Out-Null
        Write-Log '  Created: Block mDNS Inbound'
    }
    if (-not (Get-NetFirewallRule -DisplayName $ruleNames[1] -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $ruleNames[1] -Direction Outbound -Protocol UDP -LocalPort 5353 -Action Block | Out-Null
        Write-Log '  Created: Block mDNS Outbound'
    }
}

function Remove-SMBv1 {
    Write-Log 'Removing SMBv1...'
    $feature = Get-WindowsOptionalFeature -Online -FeatureName 'SMB1Protocol' -ErrorAction SilentlyContinue
    if ($feature -and $feature.State -eq 'Enabled') {
        Disable-WindowsOptionalFeature -Online -FeatureName 'SMB1Protocol' -NoRestart | Out-Null
        Write-Log '  SMBv1 disabled.'
    } else {
        Write-Log '  SMBv1 already disabled or feature not found.'
    }
}

function Invoke-SecurityHardening {
    Disable-NetBIOSOnTarget
    Block-mDNS
    Remove-SMBv1
}

# ---------------------------------------------------------------------------
# WireGuard VPN
# ---------------------------------------------------------------------------
function Install-WireGuardTunnel {
    Write-Log 'Ensuring WireGuard tunnel is installed...'
    $confPath = "$WgConfDir\$TunnelName.conf"
    if (-not (Test-Path $confPath)) {
        Write-Log "  WireGuard config not found at $confPath" 'WARN'
        return
    }
    $svc = Get-Service -Name $WgServiceName -ErrorAction SilentlyContinue
    if (-not $svc) {
        Write-Log '  Installing WireGuard tunnel service...'
        & "$env:ProgramFiles\WireGuard\wireguard.exe" /installtunnelservice $confPath
        Write-Log '  Tunnel service installed.'
    }
}

function Start-WireGuardTunnel {
    Write-Log 'Starting WireGuard tunnel...'
    $svc = Get-Service -Name $WgServiceName -ErrorAction SilentlyContinue
    if (-not $svc) { Write-Log '  Tunnel service not found.' 'WARN'; return }
    if ($svc.Status -ne 'Running') {
        Start-Service -Name $WgServiceName
        Start-Sleep -Seconds 3
        Write-Log '  WireGuard tunnel started.'
    } else {
        Write-Log '  WireGuard tunnel already running.'
    }
}

function Test-WireGuardHandshake {
    <#
    Returns the age of the latest handshake in seconds, or [int]::MaxValue if
    the tunnel is down or wg.exe is unavailable.
    #>
    if (-not (Test-Path $WgExe)) { return [int]::MaxValue }
    try {
        $output = & $WgExe show $TunnelName latest-handshakes 2>&1
        $epoch  = ($output -split '\s+')[1]
        if ($epoch -match '^\d+$' -and [long]$epoch -gt 0) {
            $handshakeTime = [DateTimeOffset]::FromUnixTimeSeconds([long]$epoch).UtcDateTime
            return [int]([datetime]::UtcNow - $handshakeTime).TotalSeconds
        }
    } catch {}
    return [int]::MaxValue
}

function Invoke-WireGuardHealthCheck {
    Write-Log 'WireGuard health check...'
    $svc = Get-Service -Name $WgServiceName -ErrorAction SilentlyContinue
    if (-not $svc -or $svc.Status -ne 'Running') {
        Write-Log '  Tunnel not running — restarting...' 'WARN'
        Start-WireGuardTunnel
        Write-Log 'VPN restarted (service was down).' 'WARN'
        return
    }

    $age = Test-WireGuardHandshake
    if ($age -gt $StaleHandshakeThresholdSeconds) {
        Write-Log "  Stale handshake ($age s) — restarting tunnel..." 'WARN'
        Restart-Service -Name $WgServiceName -Force
        Start-Sleep -Seconds 3
        Write-Log "VPN restarted (stale handshake ${age}s)." 'WARN'
    } else {
        Write-Log "  Handshake age: ${age}s — OK"
    }
}

# ---------------------------------------------------------------------------
# Hyper-V VM management
# ---------------------------------------------------------------------------
function Start-NinjaVM {
    Write-Log "Ensuring Hyper-V VM '$VmName' is running..."
    $vm = Get-VM -Name $VmName -ErrorAction SilentlyContinue
    if (-not $vm) { Write-Log "  VM '$VmName' not found." 'WARN'; return }

    if ($vm.State -ne 'Running') {
        Start-VM -Name $VmName
        Write-Log "  VM '$VmName' started."
    } else {
        Write-Log "  VM '$VmName' already running."
    }
}

function New-WeeklyCheckpoint {
    Write-Log "Checking weekly checkpoint for '$VmName'..."
    $vm = Get-VM -Name $VmName -ErrorAction SilentlyContinue
    if (-not $vm) { return }

    $latest = Get-VMCheckpoint -VMName $VmName -ErrorAction SilentlyContinue |
              Sort-Object CreationTime -Descending |
              Select-Object -First 1

    $threshold = (Get-Date).AddDays(-7)
    if (-not $latest -or $latest.CreationTime -lt $threshold) {
        $cpName = "NinjaLAN_Weekly_$(Get-Date -Format 'yyyyMMdd_HHmm')"
        Checkpoint-VM -Name $VmName -SnapshotName $cpName
        Write-Log "  Checkpoint created: $cpName"
    } else {
        Write-Log "  Last checkpoint $($latest.CreationTime) is recent — skipping."
    }
}

# ---------------------------------------------------------------------------
# Failover logic
# ---------------------------------------------------------------------------
function Invoke-FailoverCheck {
    Write-Log 'Checking adapter failover...'
    $primary = Get-NetAdapter -Name $PrimaryAdapter -ErrorAction SilentlyContinue
    if (-not $primary -or $primary.Status -ne 'Up') {
        Write-Log "  Primary adapter '$PrimaryAdapter' is down — activating Wi-Fi fallback." 'WARN'
        $wifi = Get-NetAdapter -Name $FallbackAdapter -ErrorAction SilentlyContinue
        if ($wifi -and $wifi.Status -ne 'Up') {
            Enable-NetAdapter -Name $FallbackAdapter -Confirm:$false
            Write-Log "  '$FallbackAdapter' enabled as fallback." 'WARN'
        }
        Set-DnsClientServerAddress -InterfaceAlias $FallbackAdapter -ServerAddresses $PreferredDNS -ErrorAction SilentlyContinue
        Write-Log "Failover to '$FallbackAdapter' activated." 'WARN'
    } else {
        Write-Log "  Primary adapter '$PrimaryAdapter' is up — no failover needed."
    }
}

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
function Get-AdapterSummary {
    $rows = foreach ($a in (Get-NetAdapter)) {
        $ip4  = (Get-NetIPAddress -InterfaceIndex $a.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
                 Select-Object -First 1).IPAddress
        $ip6  = (Get-NetIPAddress -InterfaceIndex $a.ifIndex -AddressFamily IPv6 -ErrorAction SilentlyContinue |
                 Select-Object -First 1).IPAddress
        $dns  = (Get-DnsClientServerAddress -InterfaceIndex $a.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue).ServerAddresses -join ', '
        $metric = (Get-NetIPInterface -InterfaceIndex $a.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
                   Select-Object -First 1).InterfaceMetric
        $jumbo  = (Get-NetAdapterAdvancedProperty -Name $a.Name -DisplayName 'Jumbo Packet' -ErrorAction SilentlyContinue |
                   Select-Object -First 1).DisplayValue
        $statusClass = if ($a.Status -eq 'Up') { 'up' } else { 'down' }
        @"
        <tr>
          <td>$($a.Name)</td>
          <td><span class="badge $statusClass">$($a.Status)</span></td>
          <td>$($a.LinkSpeed)</td>
          <td>$($jumbo)</td>
          <td>$ip4</td>
          <td>$ip6</td>
          <td>$dns</td>
          <td>$metric</td>
        </tr>
"@
    }
    return $rows -join "`n"
}

function Get-VpnSummary {
    if (-not (Test-Path $WgExe)) { return '<tr><td colspan="5">WireGuard not installed</td></tr>' }
    try {
        $svc    = Get-Service -Name $WgServiceName -ErrorAction SilentlyContinue
        $status = if ($svc -and $svc.Status -eq 'Running') { 'Running' } else { 'Stopped' }
        $info   = & $WgExe show $TunnelName 2>&1 | Out-String
        $ep     = if ($info -match 'endpoint: (\S+)') { $Matches[1] } else { 'N/A' }
        $txrx   = if ($info -match 'transfer: ([^\r\n]+)') { $Matches[1] } else { 'N/A' }
        $age    = Test-WireGuardHandshake
        $ageStr = if ($age -eq [int]::MaxValue) { 'N/A' } else { "${age}s ago" }
        $statusClass = if ($status -eq 'Running') { 'up' } else { 'down' }
        return "<tr><td>$TunnelName</td><td><span class='badge $statusClass'>$status</span></td><td>$ep</td><td>$ageStr</td><td>$txrx</td></tr>"
    } catch {
        return "<tr><td colspan='5'>Error reading WireGuard status</td></tr>"
    }
}

function Get-VmSummary {
    $vm = Get-VM -Name $VmName -ErrorAction SilentlyContinue
    if (-not $vm) { return "<tr><td>$VmName</td><td colspan='2'>Not found</td></tr>" }
    $cp = Get-VMCheckpoint -VMName $VmName -ErrorAction SilentlyContinue |
          Sort-Object CreationTime -Descending | Select-Object -First 1
    $cpAge = if ($cp) { "$([int]((Get-Date) - $cp.CreationTime).TotalDays)d ago" } else { 'None' }
    $statusClass = if ($vm.State -eq 'Running') { 'up' } else { 'down' }
    return "<tr><td>$VmName</td><td><span class='badge $statusClass'>$($vm.State)</span></td><td>$cpAge</td></tr>"
}

function Update-Dashboard {
    Write-Log 'Generating network dashboard...'

    $cssInline = if (Test-Path $DashboardCSS) { Get-Content $DashboardCSS -Raw } else { '' }
    $generated = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $adapterRows = Get-AdapterSummary
    $vpnRow      = Get-VpnSummary
    $vmRow       = Get-VmSummary

    $html = @"
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="15">
  <title>NinjaLAN Dashboard</title>
  <style>$cssInline</style>
</head>
<body>
  <h1>NinjaLAN Dashboard</h1>
  <p class="generated">Generated: $generated &nbsp;&bull;&nbsp; Auto-refresh: 15s</p>

  <h2>Network Adapters</h2>
  <table>
    <thead>
      <tr>
        <th>Adapter</th><th>Status</th><th>Speed</th><th>Jumbo</th>
        <th>IPv4</th><th>IPv6</th><th>DNS</th><th>Metric</th>
      </tr>
    </thead>
    <tbody>
$adapterRows
    </tbody>
  </table>

  <h2>WireGuard VPN</h2>
  <table>
    <thead>
      <tr><th>Tunnel</th><th>Status</th><th>Endpoint</th><th>Handshake</th><th>Transfer</th></tr>
    </thead>
    <tbody>
$vpnRow
    </tbody>
  </table>

  <h2>Hyper-V VM</h2>
  <table>
    <thead>
      <tr><th>VM</th><th>State</th><th>Last Checkpoint</th></tr>
    </thead>
    <tbody>
$vmRow
    </tbody>
  </table>
</body>
</html>
"@

    $html | Set-Content -Path $DashboardOut -Encoding UTF8
    Write-Log "  Dashboard saved to $DashboardOut"
}

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
function Invoke-FullStartup {
    Write-Log '=== NinjaLAN Full Startup ==='
    Ensure-Directory $LogDir
    Optimize-NetworkAdapters
    Invoke-SecurityHardening
    Install-WireGuardTunnel
    Start-WireGuardTunnel
    Start-NinjaVM
    New-WeeklyCheckpoint
    Invoke-FailoverCheck
    Update-Dashboard
    Write-Log '=== Startup complete ==='
}

function Invoke-Maintenance {
    Write-Log '=== NinjaLAN Maintenance ==='
    Ensure-Directory $LogDir
    Set-AdapterMetrics
    Set-PrivacyDNS
    Invoke-WireGuardHealthCheck
    Invoke-FailoverCheck
    Update-Dashboard
    Write-Log '=== Maintenance complete ==='
}

function Invoke-HealthCheckMode {
    Write-Log '=== NinjaLAN Health Check ==='
    Ensure-Directory $LogDir
    Invoke-WireGuardHealthCheck
    Invoke-FailoverCheck
    Write-Log '=== Health check complete ==='
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
switch ($PSCmdlet.ParameterSetName) {
    'Maintenance'  { Invoke-Maintenance }
    'HealthCheck'  { Invoke-HealthCheckMode }
    default        { Invoke-FullStartup }
}
