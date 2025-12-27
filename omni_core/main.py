import asyncio
import sys
import os

# Ensure we can find the modules
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from omni_core.agents.hyperion import HyperionAgent
from omni_core.agents.factory_extensions import NetworkSentinel, SystemHardener, CreativeDirector, NvidiaOverseer
from omni_core.agents.knowledge import KnowledgeSynapse
from omni_core.ui.dashboard import Dashboard

async def main():
    # 1. Initialize Hyperion (The Brain)
    hyperion = HyperionAgent()

    # 2. Register Agents (The Muscles)
    sentinel = NetworkSentinel()
    hardener = SystemHardener()
    creative = CreativeDirector()
    nvidia = NvidiaOverseer()
    knowledge = KnowledgeSynapse()

    hyperion.register_agent(sentinel)
    hyperion.register_agent(hardener)
    hyperion.register_agent(creative)
    hyperion.register_agent(nvidia)
    hyperion.register_agent(knowledge)

    # 3. Create Dashboard (The Face)
    dashboard = Dashboard(hyperion)

    # 4. Ignite!
    # We run Hyperion, Agents, and Dashboard concurrently
    tasks = [
        asyncio.create_task(hyperion.start()),
        asyncio.create_task(sentinel.start()),
        asyncio.create_task(hardener.start()),
        asyncio.create_task(creative.start()),
        asyncio.create_task(nvidia.start()),
        asyncio.create_task(knowledge.start()),
        asyncio.create_task(dashboard.run())
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("Shutdown signal received.")
    finally:
        await hyperion.stop()
        print("System Offline.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
