import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime

class BaseAgent(ABC):
    def __init__(self, name: str, agent_id: str, description: str):
        self.name = name
        self.agent_id = agent_id
        self.description = description
        self.status = "IDLE"
        self.logs = []
        self.running = False

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{self.name}] {message}"
        self.logs.append(entry)
        # Keep logs manageable
        if len(self.logs) > 100:
            self.logs.pop(0)

    async def start(self):
        self.running = True
        self.status = "ACTIVE"
        self.log("Agent initializing...")
        try:
            await self.run_loop()
        except Exception as e:
            self.log(f"CRITICAL ERROR: {e}")
            self.status = "ERROR"
        finally:
            self.running = False
            self.status = "STOPPED"

    async def stop(self):
        self.running = False
        self.log("Stopping agent...")

    @abstractmethod
    async def run_loop(self):
        """Main logic loop for the agent."""
        pass
