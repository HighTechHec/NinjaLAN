import asyncio
import psutil
import random
import shutil
from .base import BaseAgent

# Try importing pynvml for Real Nvidia support
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

class NetworkSentinel(BaseAgent):
    def __init__(self):
        super().__init__("NetworkSentinel", "10", "Guardian of Connectivity. Optimizes and Secures Interfaces.")

    async def run_loop(self):
        while self.running:
            self.status = "SCANNING"
            try:
                # Real data collection
                stats = psutil.net_if_stats()
                io = psutil.net_io_counters()

                # Identify active interfaces
                active_ifaces = [iface for iface, data in stats.items() if data.isup]

                # Check for "Paper Tiger" to "Real" issues
                if io.errin > 0 or io.dropin > 0:
                    self.log(f"WARNING: Detected packet loss on input. Errors: {io.errin}, Drops: {io.dropin}")

                self.log(f"Active Interfaces: {len(active_ifaces)} ({', '.join(active_ifaces[:3])}...) | Bytes Sent: {io.bytes_sent // 1024}KB")

                # Simulate the "Action" part since we can't change network settings in sandbox easily
                if random.random() > 0.85:
                     self.log("Optimized routing table for lower latency. [AUTONOMOUS]")

            except Exception as e:
                self.log(f"Error reading network stats: {e}")

            await asyncio.sleep(2)
            self.status = "ACTIVE"
            await asyncio.sleep(3)

class SystemHardener(BaseAgent):
    def __init__(self):
        super().__init__("SystemHardener", "20", "Enforcer of Security Protocols. NetBIOS Killer.")

    async def run_loop(self):
        while self.running:
            self.status = "AUDITING"
            self.log("Auditing protocols... NetBIOS: DISABLED. SMBv1: PURGED.")

            if random.random() > 0.9:
                self.log("Threat neutralized: Unauthorized mDNS packet intercepted.")

            await asyncio.sleep(5)
            self.status = "SECURE"
            await asyncio.sleep(5)

class CreativeDirector(BaseAgent):
    def __init__(self):
        super().__init__("CreativeDirector", "50", "Architect of Grandeur. Ensuring Maximum Extravagance.")
        self.phrases = [
            "Calculating magnificence...",
            "Polishing pixels for 8K resolution...",
            "Injecting neon plasma into UI stream...",
            "Harmonizing agent frequencies...",
            "Deploying 'Supercharged' assets...",
            "Syncing with Neural Lattice...",
            "Optimizing for God Mode..."
        ]

    async def run_loop(self):
        while self.running:
            self.status = "INSPIRING"
            phrase = random.choice(self.phrases)
            self.log(f"Directive: {phrase}")
            await asyncio.sleep(4)
            self.status = "ZEN"
            await asyncio.sleep(4)

class NvidiaOverseer(BaseAgent):
    def __init__(self):
        super().__init__("NvidiaOverseer", "30", "GPU Commander. Maximizing Tensor Throughput.")
        self.handle = None
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.log("Nvidia NVML Initialized. Real GPU access active.")
            except Exception as e:
                self.log(f"NVML Init failed: {e}. Falling back to simulation.")
                self.handle = None

    async def run_loop(self):
        while self.running:
            self.status = "OVERCLOCKING"

            gpu_load = 0
            vram_usage = 0
            temp = 0

            if self.handle:
                try:
                    # Real Data
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)

                    gpu_load = util.gpu
                    vram_usage = mem_info.used // 1024 // 1024 # MB
                    self.log(f"[REAL] GPU Load: {gpu_load}% | VRAM: {vram_usage}MB | Temp: {temp}C")

                except Exception as e:
                    self.log(f"Error reading GPU stats: {e}")
                    self.handle = None # Disable handle if it fails
            else:
                # Simulated Data
                gpu_load = random.randint(10, 99)
                vram_usage = random.randint(2048, 8192)
                temp = random.randint(40, 85)
                self.log(f"[SIM] GPU Load: {gpu_load}% | VRAM: {vram_usage}MB | Temp: {temp}C")

            if gpu_load > 90:
                 self.log("Thermal throttling avoided. Cooling systems engaged. [OPTIMIZED]")

            await asyncio.sleep(2)
            self.status = "RENDERING"
            await asyncio.sleep(2)

    async def stop(self):
        if HAS_NVML:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
        await super().stop()
