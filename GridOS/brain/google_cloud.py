"""
Google Cloud Platform Integration
Core Google Cloud services for the second brain

Features:
- Google Keep API integration
- Google Drive storage
- Vertex AI (Gemini) integration
- Firebase/Firestore sync
- Cloud Functions automation
- Cloud Storage backups
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import os


class SyncStatus(Enum):
    """Sync status."""
    SYNCED = "synced"
    PENDING = "pending"
    CONFLICT = "conflict"
    ERROR = "error"


@dataclass
class CloudConfig:
    """Google Cloud configuration."""
    project_id: str = ""
    credentials_path: str = ""
    keep_enabled: bool = True
    drive_enabled: bool = True
    firebase_enabled: bool = True
    vertex_ai_enabled: bool = True
    region: str = "us-central1"


@dataclass
class SyncRecord:
    """Record of sync operation."""
    local_id: str
    cloud_id: str
    service: str  # "keep", "drive", "firebase"
    status: SyncStatus
    last_synced: datetime
    conflict_data: Optional[Dict] = None


class GoogleCloudManager:
    """
    Central manager for all Google Cloud services.
    
    Coordinates:
    - Google Keep for rapid capture
    - Google Drive for document storage
    - Vertex AI for AI capabilities
    - Firebase for real-time sync
    """
    
    def __init__(self, config: Optional[CloudConfig] = None):
        """
        Initialize Google Cloud manager.
        
        Args:
            config: Cloud configuration
        """
        self.config = config or CloudConfig()
        self.sync_records: Dict[str, SyncRecord] = {}
        
        # Initialize service clients (lazy loading)
        self._keep_client = None
        self._drive_client = None
        self._vertex_client = None
        self._firebase_client = None
        
        self.stats = {
            'keep_syncs': 0,
            'drive_syncs': 0,
            'firebase_syncs': 0,
            'vertex_calls': 0,
            'sync_conflicts': 0
        }
    
    @property
    def keep_client(self):
        """Lazy load Google Keep client."""
        if self._keep_client is None and self.config.keep_enabled:
            self._keep_client = self._init_keep_client()
        return self._keep_client
    
    @property
    def drive_client(self):
        """Lazy load Google Drive client."""
        if self._drive_client is None and self.config.drive_enabled:
            self._drive_client = self._init_drive_client()
        return self._drive_client
    
    @property
    def vertex_client(self):
        """Lazy load Vertex AI client."""
        if self._vertex_client is None and self.config.vertex_ai_enabled:
            self._vertex_client = self._init_vertex_client()
        return self._vertex_client
    
    @property
    def firebase_client(self):
        """Lazy load Firebase client."""
        if self._firebase_client is None and self.config.firebase_enabled:
            self._firebase_client = self._init_firebase_client()
        return self._firebase_client
    
    def _init_keep_client(self):
        """Initialize Google Keep client."""
        try:
            # Note: gkeepapi is unofficial but widely used
            import gkeepapi
            keep = gkeepapi.Keep()
            # Would authenticate here with credentials
            return keep
        except ImportError:
            print("Google Keep API not available (install gkeepapi)")
            return None
    
    def _init_drive_client(self):
        """Initialize Google Drive client."""
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            
            credentials = service_account.Credentials.from_service_account_file(
                self.config.credentials_path,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            service = build('drive', 'v3', credentials=credentials)
            return service
        except Exception as e:
            print(f"Google Drive API initialization failed: {e}")
            return None
    
    def _init_vertex_client(self):
        """Initialize Vertex AI client."""
        try:
            import vertexai
            from vertexai.preview.generative_models import GenerativeModel
            
            vertexai.init(
                project=self.config.project_id,
                location=self.config.region
            )
            
            model = GenerativeModel("gemini-pro")
            return model
        except Exception as e:
            print(f"Vertex AI initialization failed: {e}")
            return None
    
    def _init_firebase_client(self):
        """Initialize Firebase client."""
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
            
            cred = credentials.Certificate(self.config.credentials_path)
            firebase_admin.initialize_app(cred)
            
            db = firestore.client()
            return db
        except Exception as e:
            print(f"Firebase initialization failed: {e}")
            return None
    
    async def sync_from_keep(self) -> Dict:
        """
        Sync notes from Google Keep.
        
        Returns:
            Sync results
        """
        if not self.keep_client:
            return {"error": "Google Keep not available"}
        
        synced = []
        errors = []
        
        try:
            # Get all notes from Keep
            notes = self.keep_client.all()
            
            for note in notes:
                try:
                    # Create sync record
                    record = SyncRecord(
                        local_id=f"keep_{note.id}",
                        cloud_id=note.id,
                        service="keep",
                        status=SyncStatus.SYNCED,
                        last_synced=datetime.utcnow()
                    )
                    
                    self.sync_records[record.local_id] = record
                    synced.append({
                        'id': note.id,
                        'title': note.title,
                        'text': note.text,
                        'labels': [label.name for label in note.labels.all()]
                    })
                    
                except Exception as e:
                    errors.append({'note_id': note.id, 'error': str(e)})
            
            self.stats['keep_syncs'] += len(synced)
            
        except Exception as e:
            return {"error": f"Keep sync failed: {e}"}
        
        return {
            'synced': len(synced),
            'errors': len(errors),
            'notes': synced[:10]  # Return first 10
        }
    
    async def sync_to_keep(self, note_data: Dict) -> Dict:
        """
        Create/update note in Google Keep.
        
        Args:
            note_data: Note data (title, content, labels)
            
        Returns:
            Sync result
        """
        if not self.keep_client:
            return {"error": "Google Keep not available"}
        
        try:
            # Create new note
            note = self.keep_client.createNote(
                title=note_data.get('title', ''),
                text=note_data.get('content', '')
            )
            
            # Add labels
            for label_name in note_data.get('labels', []):
                label = self.keep_client.findLabel(label_name)
                if not label:
                    label = self.keep_client.createLabel(label_name)
                note.labels.add(label)
            
            # Sync
            self.keep_client.sync()
            
            self.stats['keep_syncs'] += 1
            
            return {
                'status': 'success',
                'note_id': note.id,
                'title': note.title
            }
            
        except Exception as e:
            return {"error": f"Failed to sync to Keep: {e}"}
    
    async def sync_from_drive(self, folder_id: Optional[str] = None) -> Dict:
        """
        Sync documents from Google Drive.
        
        Args:
            folder_id: Specific folder to sync (or root if None)
            
        Returns:
            Sync results
        """
        if not self.drive_client:
            return {"error": "Google Drive not available"}
        
        synced = []
        errors = []
        
        try:
            # Query for files
            query = f"'{folder_id}' in parents" if folder_id else "mimeType='text/plain' or mimeType='application/pdf'"
            
            results = self.drive_client.files().list(
                q=query,
                pageSize=100,
                fields="files(id, name, mimeType, createdTime, modifiedTime)"
            ).execute()
            
            files = results.get('files', [])
            
            for file in files:
                try:
                    # Download file content
                    content = self.drive_client.files().get_media(
                        fileId=file['id']
                    ).execute()
                    
                    record = SyncRecord(
                        local_id=f"drive_{file['id']}",
                        cloud_id=file['id'],
                        service="drive",
                        status=SyncStatus.SYNCED,
                        last_synced=datetime.utcnow()
                    )
                    
                    self.sync_records[record.local_id] = record
                    synced.append({
                        'id': file['id'],
                        'name': file['name'],
                        'mime_type': file['mimeType'],
                        'size': len(content) if content else 0
                    })
                    
                except Exception as e:
                    errors.append({'file_id': file['id'], 'error': str(e)})
            
            self.stats['drive_syncs'] += len(synced)
            
        except Exception as e:
            return {"error": f"Drive sync failed: {e}"}
        
        return {
            'synced': len(synced),
            'errors': len(errors),
            'files': synced[:10]
        }
    
    async def sync_to_drive(self, file_data: Dict) -> Dict:
        """
        Upload file to Google Drive.
        
        Args:
            file_data: File data (name, content, folder_id)
            
        Returns:
            Upload result
        """
        if not self.drive_client:
            return {"error": "Google Drive not available"}
        
        try:
            from googleapiclient.http import MediaInMemoryUpload
            
            file_metadata = {
                'name': file_data.get('name', 'untitled.txt'),
                'mimeType': file_data.get('mime_type', 'text/plain')
            }
            
            if 'folder_id' in file_data:
                file_metadata['parents'] = [file_data['folder_id']]
            
            media = MediaInMemoryUpload(
                file_data.get('content', b''),
                mimetype=file_data.get('mime_type', 'text/plain')
            )
            
            file = self.drive_client.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name'
            ).execute()
            
            self.stats['drive_syncs'] += 1
            
            return {
                'status': 'success',
                'file_id': file.get('id'),
                'name': file.get('name')
            }
            
        except Exception as e:
            return {"error": f"Failed to upload to Drive: {e}"}
    
    async def query_gemini(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Query Google Gemini via Vertex AI.
        
        Args:
            prompt: User prompt
            context: Optional context to include
            
        Returns:
            Generated response
        """
        if not self.vertex_client:
            return "[Vertex AI not available]"
        
        try:
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
            
            response = self.vertex_client.generate_content(full_prompt)
            
            self.stats['vertex_calls'] += 1
            
            return response.text
            
        except Exception as e:
            return f"[Gemini error: {e}]"
    
    async def sync_to_firebase(self, collection: str, document_id: str, data: Dict) -> Dict:
        """
        Sync data to Firebase Firestore.
        
        Args:
            collection: Collection name
            document_id: Document ID
            data: Data to sync
            
        Returns:
            Sync result
        """
        if not self.firebase_client:
            return {"error": "Firebase not available"}
        
        try:
            doc_ref = self.firebase_client.collection(collection).document(document_id)
            doc_ref.set(data, merge=True)
            
            self.stats['firebase_syncs'] += 1
            
            return {
                'status': 'success',
                'collection': collection,
                'document_id': document_id
            }
            
        except Exception as e:
            return {"error": f"Firebase sync failed: {e}"}
    
    async def sync_from_firebase(self, collection: str, limit: int = 100) -> Dict:
        """
        Sync data from Firebase Firestore.
        
        Args:
            collection: Collection name
            limit: Maximum documents to retrieve
            
        Returns:
            Sync results
        """
        if not self.firebase_client:
            return {"error": "Firebase not available"}
        
        try:
            docs = self.firebase_client.collection(collection).limit(limit).stream()
            
            synced = []
            for doc in docs:
                synced.append({
                    'id': doc.id,
                    'data': doc.to_dict()
                })
            
            self.stats['firebase_syncs'] += len(synced)
            
            return {
                'synced': len(synced),
                'documents': synced
            }
            
        except Exception as e:
            return {"error": f"Firebase sync failed: {e}"}
    
    def get_sync_status(self, local_id: str) -> Optional[SyncRecord]:
        """Get sync status for an item."""
        return self.sync_records.get(local_id)
    
    def get_stats(self) -> Dict:
        """Get cloud service statistics."""
        return {
            'keep_syncs': self.stats['keep_syncs'],
            'drive_syncs': self.stats['drive_syncs'],
            'firebase_syncs': self.stats['firebase_syncs'],
            'vertex_calls': self.stats['vertex_calls'],
            'sync_conflicts': self.stats['sync_conflicts'],
            'total_sync_records': len(self.sync_records),
            'services': {
                'keep': self.config.keep_enabled and self.keep_client is not None,
                'drive': self.config.drive_enabled and self.drive_client is not None,
                'vertex_ai': self.config.vertex_ai_enabled and self.vertex_client is not None,
                'firebase': self.config.firebase_enabled and self.firebase_client is not None
            }
        }


# Convenience instance
google_cloud = GoogleCloudManager()


__all__ = [
    "GoogleCloudManager",
    "CloudConfig",
    "SyncRecord",
    "SyncStatus",
    "google_cloud",
]
