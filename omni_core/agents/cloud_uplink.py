import asyncio
import os
import shutil
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from .base import BaseAgent

SCOPES = ['https://www.googleapis.com/auth/drive.file']

class GoogleDriveUplink(BaseAgent):
    def __init__(self, target_folder_name="jaxPrime"):
        super().__init__("GoogleDriveUplink", "77", "Interstellar Cloud Link. Syncing to 'jaxPrime'.")
        self.target_folder_name = target_folder_name
        self.local_mirror = f"{target_folder_name}_Local"
        self.creds = None
        self.service = None
        self.mode = "LOCAL_MIRROR" # or "CLOUD_ACTIVE"

    async def start(self):
        # Initial check for credentials
        if os.path.exists('credentials.json'):
            try:
                self.authenticate()
                self.mode = "CLOUD_ACTIVE"
                self.log("Credentials detected. Authenticated with Google Drive.")
            except Exception as e:
                self.log(f"Authentication failed: {e}. Reverting to Local Mirror.")
        else:
            self.log("No credentials found. Zero Keys Mode Active. Using Local Mirror.")

        # Ensure local mirror exists regardless
        if not os.path.exists(self.local_mirror):
            os.makedirs(self.local_mirror)
            self.log(f"Created local mirror directory: {self.local_mirror}")

        await super().start()

    def authenticate(self):
        # This is a simplified auth flow. In a real headless server,
        # you'd likely use a service account or pre-generated token.json.
        # Here we check for token.json first.
        if os.path.exists('token.json'):
            self.creds = Credentials.from_authorized_user_file('token.json', SCOPES)

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                # If we are strictly headless/autonomous, we can't pop a browser.
                # So we assume if token.json isn't there, we can't auth fully interactively
                # without user present. We'll fallback to local if this fails.
                raise Exception("Interactive auth required but not supported in autonomous mode.")

        self.service = build('drive', 'v3', credentials=self.creds)

    async def run_loop(self):
        while self.running:
            self.status = f"SYNCING [{self.mode}]"

            # File to sync: second_brain_briefing.md
            source_file = "second_brain_briefing.md"

            if os.path.exists(source_file):
                if self.mode == "CLOUD_ACTIVE":
                    await self.sync_to_cloud(source_file)
                else:
                    await self.sync_to_local(source_file)

            await asyncio.sleep(60) # Sync every minute
            self.status = "IDLE"
            await asyncio.sleep(10)

    async def sync_to_local(self, filepath):
        dest = os.path.join(self.local_mirror, filepath)
        try:
            shutil.copy2(filepath, dest)
            self.log(f"Mirrored {filepath} to {dest}")
        except Exception as e:
            self.log(f"Error mirroring file: {e}")

    async def sync_to_cloud(self, filepath):
        # Pseudo-code for Drive upload since we can't actually run this in sandbox without keys
        # 1. Search for folder self.target_folder_name
        # 2. If not found, create it.
        # 3. Upload file.
        self.log(f"Uploading {filepath} to Drive/{self.target_folder_name}... [SIMULATED]")


class NotebookLMPreparer(BaseAgent):
    def __init__(self, source_file="second_brain_briefing.md", output_file="jaxPrime_NotebookLM_Source.txt"):
        super().__init__("NotebookLMPreparer", "88", "Data Alchemist. Refining Intelligence for AI Ingestion.")
        self.source_file = source_file
        self.output_file = output_file

    async def run_loop(self):
        while self.running:
            self.status = "REFINING"

            if os.path.exists(self.source_file):
                try:
                    with open(self.source_file, 'r') as f:
                        content = f.read()

                    # Transform content into "Dense" format for NotebookLM
                    # NotebookLM loves structure: "Source: ... \n Key Insight: ..."

                    refined_content = "SOURCE: NINJALAN_OMNI_CORE_V22\n"
                    refined_content += f"TIMESTAMP: {datetime.now()}\n"
                    refined_content += "CONTEXT: AUTONOMOUS NETWORK OPERATIONS\n"
                    refined_content += "-" * 20 + "\n"
                    refined_content += content.replace("# ", "SECTION: ").replace("- ", "FACT: ")

                    # Write to the local mirror folder too if it exists
                    mirror_path = os.path.join("jaxPrime_Local", self.output_file)

                    if os.path.exists("jaxPrime_Local"):
                        with open(mirror_path, 'w') as f:
                            f.write(refined_content)
                        self.log(f"Generated optimized source at {mirror_path}")

                except Exception as e:
                    self.log(f"Error refining data: {e}")

            await asyncio.sleep(30)
            self.status = "WAITING"
            await asyncio.sleep(30)
