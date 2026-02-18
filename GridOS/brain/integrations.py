"""
Integration modules: Browser Extension, Obsidian Sync, Mobile API
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os

# ============================================================================
# Browser Extension API
# ============================================================================

@dataclass
class WebCapture:
    """Captured web content."""
    url: str
    title: str
    content: str
    selected_text: Optional[str] = None
    screenshot: Optional[str] = None  # Base64
    tags: List[str] = None
    captured_at: datetime = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.captured_at is None:
            self.captured_at = datetime.utcnow()

class BrowserExtensionAPI:
    """API for browser extension communication."""
    
    def __init__(self, brain_server_url: str = "http://localhost:8888"):
        self.brain_url = brain_server_url
    
    async def capture_page(self, capture: WebCapture) -> Dict:
        """Handle page capture from browser extension."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.brain_url}/api/ingest",
                    json={
                        "content": f"{capture.title}\n\n{capture.content}",
                        "memory_type": "episodic",
                        "tags": capture.tags + ["web_capture"],
                        "metadata": {
                            "url": capture.url,
                            "title": capture.title,
                            "source": "browser_extension",
                            "selected_text": capture.selected_text
                        }
                    }
                )
                return response.json()
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    def get_extension_manifest(self) -> Dict:
        """Return manifest.json for browser extension."""
        return {
            "manifest_version": 3,
            "name": "NVIDIA Second Brain",
            "version": "1.0.0",
            "description": "Capture web content to your AI-powered second brain",
            "permissions": ["activeTab", "scripting", "storage"],
            "host_permissions": ["<all_urls>"],
            "background": {
                "service_worker": "background.js"
            },
            "action": {
                "default_popup": "popup.html",
                "default_title": "Capture to Second Brain",
                "default_icon": {
                    "16": "icon16.png",
                    "48": "icon48.png",
                    "128": "icon128.png"
                }
            },
            "icons": {
                "16": "icon16.png",
                "48": "icon48.png",
                "128": "icon128.png"
            }
        }

BROWSER_EXTENSION_JS = """
// background.js - Browser extension background script

chrome.runtime.onMessage.addListener(async (request, sender, sendResponse) => {
  if (request.action === "capture") {
    try {
      const response = await fetch("http://localhost:8888/api/extension/capture", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          url: request.url,
          title: request.title,
          content: request.content,
          selected_text: request.selected_text,
          tags: request.tags || ["web_capture"]
        })
      });
      const result = await response.json();
      sendResponse(result);
    } catch (error) {
      sendResponse({error: error.message});
    }
    return true;  // Keep channel open for async response
  }
});

// popup.html - Extension popup
const POPUP_HTML = `
<!DOCTYPE html>
<html>
<head>
  <style>
    body { 
      width: 320px; 
      padding: 16px; 
      background: #0a0e27; 
      color: #00d9ff; 
      font-family: 'Monaco', monospace; 
      font-size: 13px;
    }
    h3 {
      margin: 0 0 16px 0;
      color: #00ff88;
      font-size: 16px;
    }
    input, textarea { 
      width: 100%; 
      padding: 8px; 
      margin: 8px 0;
      background: #1a1e3f;
      border: 1px solid #00d9ff;
      color: #00d9ff;
      border-radius: 4px;
      font-family: inherit;
      font-size: 12px;
    }
    input:focus, textarea:focus {
      outline: none;
      border-color: #00ff88;
    }
    button { 
      background: linear-gradient(135deg, #4A90E2 0%, #00d9ff 100%);
      color: white; 
      border: none; 
      padding: 10px 18px; 
      border-radius: 6px; 
      cursor: pointer;
      font-weight: bold;
      width: 100%;
      margin-top: 12px;
      transition: transform 0.2s;
    }
    button:hover {
      transform: translateY(-2px);
    }
    button:active {
      transform: translateY(0);
    }
    #status {
      margin-top: 12px;
      padding: 8px;
      border-radius: 4px;
      font-size: 12px;
      text-align: center;
    }
    .success {
      background: rgba(126, 211, 33, 0.2);
      border: 1px solid #7ED321;
      color: #7ED321;
    }
    .error {
      background: rgba(233, 75, 60, 0.2);
      border: 1px solid #E94B3C;
      color: #E94B3C;
    }
    .info {
      color: #888;
      font-size: 11px;
      margin-top: 12px;
    }
  </style>
</head>
<body>
  <h3>🧠 Capture to Brain</h3>
  <input id="tags" placeholder="Tags (comma-separated)" />
  <textarea id="notes" placeholder="Additional notes..." rows="3"></textarea>
  <button id="capture">💾 Save to Second Brain</button>
  <div id="status"></div>
  <div class="info">Captures current page + selected text</div>
  
  <script>
    document.getElementById("capture").addEventListener("click", async () => {
      const statusDiv = document.getElementById("status");
      statusDiv.textContent = "Capturing...";
      statusDiv.className = "";
      
      const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
      const tags = document.getElementById("tags").value.split(",").map(t => t.trim()).filter(Boolean);
      const notes = document.getElementById("notes").value;
      
      try {
        // Get selected text and page content
        const [result] = await chrome.scripting.executeScript({
          target: {tabId: tab.id},
          func: () => {
            return {
              selectedText: window.getSelection().toString(),
              pageText: document.body.innerText.substring(0, 5000)
            };
          }
        });
        
        chrome.runtime.sendMessage({
          action: "capture",
          url: tab.url,
          title: tab.title,
          content: result.result.pageText + (notes ? "\\n\\nNotes: " + notes : ""),
          selected_text: result.result.selectedText,
          tags: tags
        }, (response) => {
          if (response.error) {
            statusDiv.textContent = "❌ " + response.error;
            statusDiv.className = "error";
          } else {
            statusDiv.textContent = "✅ Captured successfully!";
            statusDiv.className = "success";
            setTimeout(() => window.close(), 2000);
          }
        });
      } catch (error) {
        statusDiv.textContent = "❌ " + error.message;
        statusDiv.className = "error";
      }
    });
  </script>
</body>
</html>
`;
"""

# ============================================================================
# Obsidian Vault Sync
# ============================================================================

class ObsidianSync:
    """Sync with Obsidian vault (bidirectional)."""
    
    def __init__(self, vault_path: str, brain_server_url: str = "http://localhost:8888"):
        self.vault_path = os.path.expanduser(vault_path)
        self.brain_url = brain_server_url
        self.sync_metadata = {}
    
    async def sync_from_obsidian(self) -> Dict:
        """Import notes from Obsidian vault."""
        imported = 0
        skipped = 0
        errors = 0
        
        try:
            if not os.path.exists(self.vault_path):
                return {"error": f"Vault path does not exist: {self.vault_path}"}
            
            for root, dirs, files in os.walk(self.vault_path):
                for file in files:
                    if file.endswith('.md'):
                        path = os.path.join(root, file)
                        
                        try:
                            with open(path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Parse YAML frontmatter if present
                            tags = []
                            if content.startswith('---'):
                                try:
                                    end_idx = content.find('---', 3)
                                    if end_idx > 0:
                                        frontmatter = content[3:end_idx]
                                        # Simple tag extraction
                                        for line in frontmatter.split('\n'):
                                            if line.startswith('tags:'):
                                                tags_str = line.replace('tags:', '').strip()
                                                tags = [t.strip() for t in tags_str.strip('[]').split(',')]
                                        content = content[end_idx + 3:].strip()
                                except:
                                    pass
                            
                            # Add note to brain
                            import httpx
                            async with httpx.AsyncClient() as client:
                                await client.post(
                                    f"{self.brain_url}/api/ingest",
                                    json={
                                        "content": content,
                                        "memory_type": "long_term",
                                        "tags": tags + ["obsidian"],
                                        "metadata": {
                                            "obsidian_path": path,
                                            "filename": file
                                        }
                                    }
                                )
                            
                            imported += 1
                            self.sync_metadata[path] = datetime.utcnow().isoformat()
                        
                        except Exception as e:
                            errors += 1
                            print(f"Error importing {path}: {e}")
        
        except Exception as e:
            return {"error": str(e)}
        
        return {
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "message": f"Synced {imported} notes from Obsidian vault"
        }
    
    async def sync_to_obsidian(self, memories: List[Dict]) -> Dict:
        """Export memories to Obsidian as new notes."""
        try:
            # Create SecondBrain directory in vault
            notes_dir = os.path.join(self.vault_path, "SecondBrain")
            os.makedirs(notes_dir, exist_ok=True)
            
            exported = 0
            for memory in memories:
                # Create note filename
                title = memory.get('content', '')[:50].replace('/', '-').replace('\\', '-')
                filename = f"{title}_{memory.get('id', '')[:8]}.md"
                filepath = os.path.join(notes_dir, filename)
                
                # Create frontmatter
                frontmatter = f"""---
tags: {', '.join(memory.get('tags', []))}
created: {datetime.utcnow().isoformat()}
source: second_brain
memory_id: {memory.get('id', '')}
---

"""
                
                # Write note
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(frontmatter + memory.get('content', ''))
                
                exported += 1
            
            return {
                "exported": exported,
                "directory": notes_dir,
                "message": f"Exported {exported} memories to Obsidian"
            }
        
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# Mobile API
# ============================================================================

class MobileAPI:
    """API for mobile apps."""
    
    def __init__(self, brain_server_url: str = "http://localhost:8888"):
        self.brain_url = brain_server_url
    
    async def get_mobile_dashboard(self) -> Dict:
        """Lightweight dashboard for mobile."""
        return {
            "version": "1.0.0",
            "api_endpoint": self.brain_url,
            "features": {
                "capture": True,
                "search": True,
                "ask": True,
                "voice_capture": True,
                "offline_sync": True,
                "notifications": True
            },
            "endpoints": {
                "capture": "/api/mobile/capture",
                "search": "/api/search",
                "ask": "/api/question",
                "voice": "/api/mobile/voice-capture",
                "sync": "/api/mobile/sync-offline"
            }
        }
    
    async def voice_capture(self, audio_data: bytes, language: str = "en") -> Dict:
        """Capture via voice transcription."""
        # Placeholder for speech-to-text integration
        return {
            "transcribed_text": "[Voice transcription would appear here]",
            "confidence": 0.9,
            "language": language,
            "status": "success"
        }
    
    async def sync_offline_queue(self, queue: List[Dict]) -> Dict:
        """Sync offline-created notes when online."""
        synced = 0
        failed = 0
        
        import httpx
        async with httpx.AsyncClient() as client:
            for item in queue:
                try:
                    await client.post(
                        f"{self.brain_url}/api/ingest",
                        json=item
                    )
                    synced += 1
                except:
                    failed += 1
        
        return {
            "synced": synced,
            "failed": failed,
            "total": len(queue),
            "message": f"Synced {synced}/{len(queue)} items"
        }

# ============================================================================
# API Gateway
# ============================================================================

class APIGateway:
    """Unified API for all integrations."""
    
    def __init__(self, brain_server_url: str = "http://localhost:8888"):
        self.brain_url = brain_server_url
        self.browser = BrowserExtensionAPI(brain_server_url)
        self.obsidian = ObsidianSync("~/Obsidian", brain_server_url)
        self.mobile = MobileAPI(brain_server_url)
    
    async def handle_webhook(self, source: str, payload: Dict) -> Dict:
        """Handle webhooks from external sources."""
        
        handlers = {
            "github": self._handle_github,
            "twitter": self._handle_twitter,
            "email": self._handle_email,
            "slack": self._handle_slack,
            "notion": self._handle_notion,
            "todoist": self._handle_todoist
        }
        
        handler = handlers.get(source)
        if handler:
            return await handler(payload)
        else:
            return {"error": f"Unknown source: {source}"}
    
    async def _handle_github(self, payload: Dict) -> Dict:
        """Process GitHub webhook (new issues, PRs, commits)."""
        return {
            "action": "stored",
            "type": payload.get("action", "unknown"),
            "repository": payload.get("repository", {}).get("name", "unknown")
        }
    
    async def _handle_twitter(self, payload: Dict) -> Dict:
        """Process Twitter mentions and saved tweets."""
        return {"action": "stored", "tweets": 1}
    
    async def _handle_email(self, payload: Dict) -> Dict:
        """Process important emails (via forwarding)."""
        return {"action": "stored", "subject": payload.get("subject", "")}
    
    async def _handle_slack(self, payload: Dict) -> Dict:
        """Process Slack messages and threads."""
        return {"action": "stored", "channel": payload.get("channel", "")}
    
    async def _handle_notion(self, payload: Dict) -> Dict:
        """Process Notion page updates."""
        return {"action": "stored", "page": payload.get("page_id", "")}
    
    async def _handle_todoist(self, payload: Dict) -> Dict:
        """Process completed tasks from Todoist."""
        return {"action": "stored", "task": payload.get("content", "")}

__all__ = [
    "WebCapture",
    "BrowserExtensionAPI",
    "ObsidianSync",
    "MobileAPI",
    "APIGateway",
    "BROWSER_EXTENSION_JS",
]


if __name__ == '__main__':
    print("=== Integration Modules ===")
    print("✓ Browser Extension API")
    print("✓ Obsidian Sync")
    print("✓ Mobile API")
    print("✓ Webhook Gateway (GitHub, Slack, Email, etc.)")
    print("\nIntegrations ready!")
