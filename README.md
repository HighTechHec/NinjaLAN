# NinjaLAN Automation Suite

NinjaLAN is a fully automated, self-healing network environment built around PowerShell, Hyper-V, and WireGuard. It prioritizes network adapters, enforces hardened security defaults, manages a Hyper‑V VM with snapshot logic, auto‑launches and monitors a WireGuard tunnel, generates a live dashboard, and runs 24/7 via scheduled health checks and recovery.

## Features

- Network adapter optimization
  - Interface metric prioritization (wired first, Wi‑Fi fallback)
  - Forced privacy DNS (1.1.1.1, 8.8.8.8)
  - Static routes to key IPs (persistent; configurable)
  - Jumbo frames on Realtek NICs (9014)
  - NIC power saving off (prevents wake‑related disruptions)
- Security hardening
  - Disable NetBIOS on Ethernet 3
  - Block mDNS (UDP/5353) inbound/outbound
  - Remove SMBv1
- WireGuard VPN
  - Auto‑install/start service for `NinjaLAN.conf`
  - Health checks; restart on failure or stale handshake
- Hyper‑V VM
  - Auto‑start `NinjaVM` on vSwitch `NinjaLAN`
  - Weekly checkpoints for rollback
- Failover logic
  - Detects primary adapter failure
  - Seamless Wi‑Fi fallback with logging
- Dashboard
  - Live HTML status for adapters, IPs, DNS, metrics
  - Saved to `%USERPROFILE%\Documents\NetworkDashboard.html`
- Logging
  - VPN restarts, failovers, script lifecycle to `C:\NinjaLAN\Logs`

## Requirements

- Windows 10/11 Pro or Windows Server with:
  - PowerShell 5.1+ (Windows PowerShell)
  - Hyper‑V enabled (for `NinjaVM` control)
  - WireGuard installed (service + `NinjaLAN.conf` present)
- Admin rights for network/Hyper‑V operations

## Quick Start

1) Open an elevated PowerShell session.
2) Run the bootstrap once (recommended):
   ```powershell
   Set-ExecutionPolicy Bypass -Scope Process -Force
   C:\NinjaLAN\Scripts\NinjaLAN_Bootstrap.ps1
   ```
   Or run the installer from the repo:
   ```powershell
   .\Scripts\NinjaLAN_Install.ps1
   ```
3) Verify the dashboard at `%USERPROFILE%\Documents\NetworkDashboard.html` and logs at `C:\NinjaLAN\Logs`.

## Configuration

- WireGuard
  - Ensure `C:\Program Files\WireGuard\Data\Configurations\NinjaLAN.conf` exists and is valid (keys, peer, endpoint, allowed IPs).
  - The tunnel runs as `WireGuardTunnel$NinjaLAN` (service managed by the scripts).
- Hyper‑V
  - VM name: `NinjaVM`
  - vSwitch: `NinjaLAN`
  - Recommended: 60GB dynamic VHD, 4GB RAM, 2 vCPU, Secure Boot enabled
- DNS & Routes
  - Defaults: `1.1.1.1`, `8.8.8.8`
  - Add/modify static routes within `NinjaLAN_Startup.ps1` if needed

## Scheduling (Installed Automatically)

- At startup (highest privileges): runs full orchestration
- Hourly maintenance: `-Maintenance`
- Health check every 5 minutes: `-HealthCheck`

If you prefer manual creation, examples:
```powershell
schtasks /Create /TN "NinjaLAN-Startup" /TR "powershell.exe -ExecutionPolicy Bypass -File C:\NinjaLAN\Scripts\NinjaLAN_Startup.ps1" /SC ONSTART /RL HIGHEST /F
schtasks /Create /TN "NinjaLAN-Hourly" /TR "powershell.exe -ExecutionPolicy Bypass -File C:\NinjaLAN\Scripts\NinjaLAN_Startup.ps1 -Maintenance" /SC HOURLY /MO 1 /RL HIGHEST /F
schtasks /Create /TN "NinjaLAN-HealthCheck" /TR "powershell.exe -ExecutionPolicy Bypass -File C:\NinjaLAN\Scripts\NinjaLAN_Startup.ps1 -HealthCheck" /SC MINUTE /MO 5 /RL HIGHEST /F
```

## Dashboard

- Output: `%USERPROFILE%\Documents\NetworkDashboard.html`
- Theme: `docs/dashboard-theme.css` (dark, neon accents, status badges)
- Auto-refresh: 15 seconds
- Data points:
  - Adapter: name, status, link speed, jumbo frames
  - IP: IPv4, IPv6
  - DNS: active servers
  - Routing: interface metric, key routes
  - VPN: status, endpoint, handshake age, transfer
  - VM: state, checkpoint age

## Security Posture

- NetBIOS disabled on targeted adapter(s)
- mDNS blocked
- SMBv1 removed
- No interactive prompts; non‑compliant state auto‑remediates
- All actions logged to `C:\NinjaLAN\Logs`

## Troubleshooting

- Run PowerShell as Administrator
- Check services: `Get-Service -Name 'WireGuardTunnel$NinjaLAN'`
- WireGuard CLI: `& "$env:ProgramFiles\WireGuard\wg.exe" show`
- Hyper‑V: `Get-VM NinjaVM`
- Event Logs: `Get-WinEvent -LogName Application, System | Select-Object -First 100`

---

MIT License