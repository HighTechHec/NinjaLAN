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
            notebook_source = "jaxPrime_NotebookLM_Source.txt"

            # Files to sync
            targets = [source_file]
            if os.path.exists(f"jaxPrime_Local/{notebook_source}"):
                 # We assume NotebookLMPreparer writes to jaxPrime_Local/jaxPrime_NotebookLM_Source.txt
                 # We want to upload that too.
                 targets.append(os.path.join("jaxPrime_Local", notebook_source))

            if self.mode == "CLOUD_ACTIVE":
                # Ensure targets exist
                for t in targets:
                     if os.path.exists(t):
                          await self.sync_to_cloud(t)
            else:
                # Mirroring Logic:
                # We already have jaxPrime_Local.
                # NotebookLMPreparer writes directly there.
                # We just need to copy the briefing file there.
                if os.path.exists(source_file):
                    await self.sync_to_local(source_file)

            await asyncio.sleep(60) # Sync every minute
            self.status = "IDLE"
            await asyncio.sleep(10)

    async def sync_to_local(self, filepath):
        dest = os.path.join(self.local_mirror, os.path.basename(filepath))
        try:
            shutil.copy2(filepath, dest)
            self.log(f"Mirrored {os.path.basename(filepath)} to {dest}")
        except Exception as e:
            self.log(f"Error mirroring file: {e}")

    async def sync_to_cloud(self, filepath):
        """
        Executes real Google Drive API calls to upload/update the file.
        """
        try:
            filename = os.path.basename(filepath)

            # 1. Find or Create Folder
            folder_id = self._get_folder_id(self.target_folder_name)
            if not folder_id:
                folder_id = self._create_folder(self.target_folder_name)

            # 2. Check if file exists in folder
            file_id = self._get_file_id_in_folder(filename, folder_id)

            media = MediaFileUpload(filepath, mimetype='text/plain')

            if file_id:
                # Update
                self.service.files().update(
                    fileId=file_id,
                    media_body=media
                ).execute()
                self.log(f"Updated {filename} in Drive/{self.target_folder_name}")
            else:
                # Create
                file_metadata = {
                    'name': filename,
                    'parents': [folder_id]
                }
                self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                self.log(f"Uploaded {filename} to Drive/{self.target_folder_name}")

        except Exception as e:
            self.log(f"Cloud Uplink Error: {e}")

    def _get_folder_id(self, folder_name):
        query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
        results = self.service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = results.get('files', [])
        if files:
            return files[0]['id']
        return None

    def _create_folder(self, folder_name):
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        file = self.service.files().create(body=file_metadata, fields='id').execute()
        return file.get('id')

    def _get_file_id_in_folder(self, filename, folder_id):
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = self.service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = results.get('files', [])
        if files:
            return files[0]['id']
        return None


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

                    if not os.path.exists("jaxPrime_Local"):
                        os.makedirs("jaxPrime_Local")

                    with open(mirror_path, 'w') as f:
                        f.write(refined_content)
                    self.log(f"Generated optimized source at {mirror_path}")

                except Exception as e:
                    self.log(f"Error refining data: {e}")

            await asyncio.sleep(30)
            self.status = "WAITING"
            await asyncio.sleep(30)
