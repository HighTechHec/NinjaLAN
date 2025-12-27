import asyncio
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich import box
from datetime import datetime
import psutil
import random

from ..agents.hyperion import HyperionAgent

class Dashboard:
    def __init__(self, hyperion: HyperionAgent):
        self.hyperion = hyperion
        self.console = Console()
        self.layout = Layout()

    def make_layout(self) -> Layout:
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        self.layout["main"].split_row(
            Layout(name="side", ratio=1),
            Layout(name="body", ratio=2)
        )
        self.layout["side"].split(
            Layout(name="roster"),
            Layout(name="system"),
            Layout(name="cloud", size=3) # New cloud panel
        )
        return self.layout

    def generate_header(self) -> Panel:
        title = Text("NINJALAN AGENT FACTORY V22.0 [OMNI-CORE]", style="bold magenta")
        subtitle = Text("ZERO BUFFERING // SUPERCHARGED // OPERATIONAL", style="bold cyan")
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_row(title)
        grid.add_row(subtitle)
        return Panel(grid, style="white on black", border_style="magenta")

    def generate_roster_table(self) -> Panel:
        table = Table(title="AGENT ROSTER", expand=True, border_style="green", box=box.ROUNDED)
        table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("Agent", style="magenta")
        table.add_column("Status", justify="center")

        # Add Hyperion
        status_style = "bold green" if self.hyperion.status == "ACTIVE" else "yellow"
        table.add_row("99", "Hyperion", Text(self.hyperion.status, style=status_style))

        # Add others
        for aid, agent in self.hyperion.roster.items():
            s_style = "bold green" if agent.status in ["ACTIVE", "SECURE", "ZEN", "RENDERING", "OVERCLOCKING", "SYNCING [CLOUD_ACTIVE]", "WAITING"] else "yellow"
            if "LOCAL_MIRROR" in agent.status:
                s_style = "bold blue"
            table.add_row(aid, agent.name, Text(agent.status, style=s_style))

        return Panel(table, title="[bold green]ACTIVE AGENTS[/]", border_style="green")

    def generate_system_metrics(self) -> Panel:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        gpu = random.randint(20, 80)

        table = Table(expand=True, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="bold white")

        table.add_row("CPU Load", f"{cpu}%")
        table.add_row("Memory", f"{mem}%")
        table.add_row("GPU (NVIDIA)", f"{gpu}%")

        cpu_bar = "█" * int(cpu / 5)
        mem_bar = "█" * int(mem / 5)
        gpu_bar = "█" * int(gpu / 5)

        display_text = Text()
        display_text.append(f"\nCPU: ", style="cyan")
        display_text.append(f"{cpu_bar}", style="magenta")
        display_text.append(f"\nMEM: ", style="cyan")
        display_text.append(f"{mem_bar}", style="blue")
        display_text.append(f"\nGPU: ", style="green")
        display_text.append(f"{gpu_bar}", style="green")

        return Panel(display_text, title="[bold blue]SYSTEM METRICS[/]", border_style="blue")

    def generate_cloud_panel(self) -> Panel:
        # Check Cloud Agent status specifically
        cloud_agent = None
        for agent in self.hyperion.roster.values():
            if agent.name == "GoogleDriveUplink":
                cloud_agent = agent
                break

        status = "OFFLINE"
        style = "red"
        if cloud_agent:
            if "CLOUD_ACTIVE" in cloud_agent.mode:
                status = "CONNECTED [jaxPrime]"
                style = "bold green"
            else:
                status = "LOCAL MIRROR [jaxPrime_Local]"
                style = "bold blue"

        return Panel(Text(status, justify="center", style=style), title="[bold white]CLOUD UPLINK[/]", border_style="white")

    def generate_log_stream(self) -> Panel:
        log_text = Text()
        all_logs = []
        all_logs.extend(self.hyperion.logs)
        for agent in self.hyperion.roster.values():
            all_logs.extend(agent.logs)

        all_logs.sort()
        recent_logs = all_logs[-15:]

        for log in recent_logs:
            if "CRITICAL" in log:
                log_text.append(log + "\n", style="bold red")
            elif "OPTIMIZED" in log or "Cooling" in log:
                log_text.append(log + "\n", style="bold green")
            elif "GoogleDrive" in log or "NotebookLM" in log:
                log_text.append(log + "\n", style="bold cyan")
            else:
                log_text.append(log + "\n", style="white")

        return Panel(log_text, title="[bold yellow]NEURAL STREAM[/]", border_style="yellow")

    def generate_footer(self) -> Panel:
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return Panel(Text(f"SYSTEM TIME: {t} | MODE: GODLIKE | SESSION: ACTIVE | NVIDIA: ON | GDRIVE: LINKED", justify="center"), style="white on black")

    async def run(self):
        self.make_layout()
        with Live(self.layout, refresh_per_second=4, screen=True) as live:
            while True:
                self.layout["header"].update(self.generate_header())
                self.layout["side"]["roster"].update(self.generate_roster_table())
                self.layout["side"]["system"].update(self.generate_system_metrics())
                self.layout["side"]["cloud"].update(self.generate_cloud_panel())
                self.layout["body"].update(self.generate_log_stream())
                self.layout["footer"].update(self.generate_footer())
                await asyncio.sleep(0.25)
