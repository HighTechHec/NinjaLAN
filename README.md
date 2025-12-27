# NINJALAN AGENT FACTORY V22.0 [OMNI-CORE]

> *"Zero Buffering. Zero Keys. Maximum Extravagance."*

NinjaLAN has evolved. What was once a static collection of scripts is now a **Living, Autonomous AI System** powered by Python and the `omni_core` engine. It manages your network, optimizes your hardware (Nvidia included), and syncs intelligence to your Google Drive and NotebookLM.

## 🚀 Capabilities

### 🧠 The Hyperion Singularity (Agent 99)
- **Central Orchestrator:** Manages the lifecycle of all agents.
- **Self-Healing:** Monitors agent health and restores functionality instantly.
- **Memory Stress Testing:** Ensures the system never buffers, even under load.

### 💪 The Agent Roster
1.  **NetworkSentinel:** Real-time interface monitoring and latency optimization.
2.  **SystemHardener:** Security enforcer (NetBIOS killer, SMBv1 purger).
3.  **NvidiaOverseer:** GPU Commander. Maximizes tensor throughput and manages thermals.
4.  **CreativeDirector:** Injects "magnificence" into the system stream.
5.  **KnowledgeSynapse:** Autonomous bridge to your Second Brain. Exports daily intelligence briefings.
6.  **GoogleDriveUplink:** **[NEW]** Syncs system intelligence to the cloud.
7.  **NotebookLMPreparer:** **[NEW]** Formats data specifically for `jaxPrime`, Google's AI Notebook.

### 🖥️ The Zero-Buffering Dashboard
- **Tech Stack:** Built with `rich` for a stunning, neon-drenched TUI (Terminal User Interface).
- **Features:**
    - Real-time CPU/RAM/GPU Metrics.
    - Live Agent Status Roster.
    - **Cloud Link Status** (Local Mirror vs. Active Uplink).
    - "Neural Stream" of system logs and decisions.

## 🛠️ Installation & Usage

### Requirements
- Python 3.8+
- Linux (Preferred) or Windows (via WSL)

### Quick Start

1.  **Install Dependencies:**
    ```bash
    pip install -r omni_core/requirements.txt
    ```

2.  **Ignite the Reactor:**
    ```bash
    python3 -m omni_core.main
    ```

3.  **Witness Perfection:**
    The dashboard will launch. No keys required. No cloud. Just pure, local power.

## ☁️ Cloud Integration (Google Drive & NotebookLM)

By default, the system operates in **"Local Mirror Mode"** (Zero Keys compliant). It creates a folder `jaxPrime_Local` and saves all intelligence there.

### To Activate Real Cloud Uplink:
1.  Place your `credentials.json` (Google Drive API) in the root directory.
2.  Restart the system.
3.  The **GoogleDriveUplink** agent will detect the keys, switch to "CLOUD_ACTIVE" mode, and sync files to your Google Drive folder named `jaxPrime`.

### NotebookLM (`jaxPrime`):
The **NotebookLMPreparer** agent automatically generates `jaxPrime_NotebookLM_Source.txt`.
- **Manual:** Drag this file from `jaxPrime_Local` into your NotebookLM interface.
- **Auto (with Cloud):** Point NotebookLM to the `jaxPrime` folder in your Google Drive. The system keeps the source file updated with the latest network intelligence.

---

**Original NinjaLAN (Legacy)**
*The PowerShell scripts described in previous versions have been sublimated into the Python Omni-Core.*
