"""
3D Knowledge Graph Visualization Dashboard
Real-time interactive visualization of the brain
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import random
import math

@dataclass
class Vector3:
    x: float
    y: float
    z: float
    
    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}

@dataclass
class NodeVisual:
    """Visual representation of knowledge node."""
    id: str
    label: str
    position: Vector3
    size: float
    color: str
    metadata: Dict
    
    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "position": self.position.to_dict(),
            "size": self.size,
            "color": self.color,
            "metadata": self.metadata
        }

@dataclass
class EdgeVisual:
    """Visual representation of relation."""
    source_id: str
    target_id: str
    relation_type: str
    weight: float
    color: str
    dashed: bool = False
    
    def to_dict(self):
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation_type,
            "weight": self.weight,
            "color": self.color,
            "dashed": self.dashed
        }

class KnowledgeGraphVisualizer:
    """Convert knowledge graph to 3D visualization data."""
    
    def __init__(self, memory_store):
        self.memory_store = memory_store
        self.node_visuals: Dict[str, NodeVisual] = {}
        self.edge_visuals: List[EdgeVisual] = []
    
    def _assign_colors(self, memory_type: str) -> str:
        """Color nodes by type."""
        colors = {
            "semantic": "#4A90E2",      # Blue
            "episodic": "#F5A623",      # Orange
            "long_term": "#7ED321",     # Green
            "short_term": "#BD10E0"     # Purple
        }
        return colors.get(memory_type, "#808080")
    
    def _layout_force_directed(self, iterations: int = 50) -> Dict[str, Vector3]:
        """Force-directed layout for 3D positioning."""
        positions: Dict[str, Vector3] = {}
        
        # Random initial positions
        for memory_id in self.memory_store.memories.keys():
            positions[memory_id] = Vector3(
                x=random.uniform(-100, 100),
                y=random.uniform(-100, 100),
                z=random.uniform(-100, 100)
            )
        
        # Simulate forces
        for _ in range(iterations):
            forces: Dict[str, Vector3] = {
                memory_id: Vector3(0, 0, 0)
                for memory_id in positions.keys()
            }
            
            # Repulsive forces between all nodes
            for node1_id, pos1 in list(positions.items())[:100]:  # Limit for performance
                for node2_id, pos2 in list(positions.items())[:100]:
                    if node1_id != node2_id:
                        dx = pos1.x - pos2.x
                        dy = pos1.y - pos2.y
                        dz = pos1.z - pos2.z
                        dist = math.sqrt(dx**2 + dy**2 + dz**2) + 0.1
                        
                        # Repulsive force
                        force = 100 / (dist * dist)
                        forces[node1_id].x += (dx / dist) * force
                        forces[node1_id].y += (dy / dist) * force
                        forces[node1_id].z += (dz / dist) * force
            
            # Update positions
            for memory_id, force in forces.items():
                dt = 0.1
                positions[memory_id].x += force.x * dt
                positions[memory_id].y += force.y * dt
                positions[memory_id].z += force.z * dt
        
        return positions
    
    def generate_visualization(self) -> Dict:
        """Generate complete visualization data."""
        # Layout
        positions = self._layout_force_directed()
        
        # Create node visuals
        for memory_id, memory in self.memory_store.memories.items():
            size = 5 + (memory.access_count / 2)  # Size by popularity
            color = self._assign_colors(memory.memory_type)
            
            visual = NodeVisual(
                id=memory_id,
                label=memory.content[:30],
                position=positions.get(memory_id, Vector3(0, 0, 0)),
                size=size,
                color=color,
                metadata={
                    "type": memory.memory_type,
                    "access_count": memory.access_count,
                    "retention": memory.calculate_retention(),
                    "tags": memory.tags[:5]
                }
            )
            self.node_visuals[memory_id] = visual
        
        # Create edge visuals for tag relationships
        tag_groups = {}
        for memory_id, memory in self.memory_store.memories.items():
            for tag in memory.tags:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(memory_id)
        
        # Connect memories with shared tags
        for tag, memory_ids in tag_groups.items():
            if len(memory_ids) > 1:
                for i in range(len(memory_ids) - 1):
                    visual = EdgeVisual(
                        source_id=memory_ids[i],
                        target_id=memory_ids[i + 1],
                        relation_type=f"shared_tag_{tag}",
                        weight=0.5,
                        color="#999999"
                    )
                    self.edge_visuals.append(visual)
        
        # Generate Three.js compatible JSON
        return {
            "nodes": [v.to_dict() for v in self.node_visuals.values()],
            "edges": [e.to_dict() for e in self.edge_visuals[:200]],  # Limit edges
            "metadata": {
                "total_nodes": len(self.node_visuals),
                "total_edges": len(self.edge_visuals),
                "layout_type": "force_directed",
                "timestamp": str(__import__("datetime").datetime.utcnow())
            }
        }

class DashboardServer:
    """Serve visualization dashboard."""
    
    def __init__(self, memory_store):
        self.memory_store = memory_store
        self.visualizer = KnowledgeGraphVisualizer(memory_store)
    
    async def get_dashboard_data(self) -> Dict:
        """Get all dashboard data."""
        viz = self.visualizer.generate_visualization()
        
        return {
            "graph": viz,
            "stats": {
                "total_memories": len(self.memory_store.memories),
                "long_term": len(self.memory_store.long_term),
                "short_term": len(self.memory_store.short_term),
                "episodic": len(self.memory_store.episodic),
                "semantic": len(self.memory_store.semantic)
            },
            "time_generated": str(__import__("datetime").datetime.utcnow())
        }
    
    async def get_node_details(self, node_id: str) -> Dict:
        """Get details about specific node."""
        memory = self.memory_store.get_memory(node_id)
        
        if not memory:
            return {"error": "Node not found"}
        
        return {
            "id": node_id,
            "content": memory.content,
            "type": memory.memory_type,
            "tags": memory.tags,
            "access_count": memory.access_count,
            "retention": memory.calculate_retention(),
            "strength": memory.strength,
            "created_at": memory.timestamp,
            "last_accessed": memory.last_accessed
        }

# HTML/JS Dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>NVIDIA Second Brain - 3D Knowledge Graph</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            margin: 0;
            overflow: hidden;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1e3f 100%);
            font-family: 'Monaco', 'Courier New', monospace;
            color: #00d9ff;
        }
        #canvas { width: 100%; height: 100vh; }
        #info {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(10, 14, 39, 0.95);
            border: 2px solid #00d9ff;
            padding: 20px;
            border-radius: 8px;
            max-width: 350px;
            font-size: 13px;
            box-shadow: 0 8px 32px rgba(0, 217, 255, 0.3);
        }
        #info h3 {
            margin: 0 0 15px 0;
            color: #00ff88;
            font-size: 18px;
            text-shadow: 0 0 10px #00ff88;
        }
        #stats {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(10, 14, 39, 0.95);
            border: 2px solid #00d9ff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 8px 32px rgba(0, 217, 255, 0.3);
        }
        .stat { 
            margin: 8px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .stat-label { 
            color: #4a90e2;
            margin-right: 15px;
        }
        .stat-value { 
            color: #00ff88;
            font-weight: bold;
            font-size: 16px;
            text-shadow: 0 0 5px #00ff88;
        }
        .legend {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #00d9ff;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 5px 0;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(10, 14, 39, 0.95);
            border: 2px solid #00d9ff;
            padding: 30px;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
        }
        .spinner {
            border: 4px solid rgba(0, 217, 255, 0.3);
            border-top: 4px solid #00d9ff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div id="loading">
        <div>🧠 Loading NVIDIA Second Brain...</div>
        <div class="spinner"></div>
    </div>
    
    <div id="canvas"></div>
    
    <div id="info" style="display:none;">
        <h3>🧠 Second Brain Visualization</h3>
        <div id="node-info">Hover over nodes to see details</div>
        <div class="legend">
            <strong>Memory Types:</strong>
            <div class="legend-item">
                <div class="legend-color" style="background: #4A90E2;"></div>
                <span>Semantic</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #F5A623;"></div>
                <span>Episodic</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #7ED321;"></div>
                <span>Long-term</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #BD10E0;"></div>
                <span>Short-term</span>
            </div>
        </div>
        <p style="margin-top: 15px; color: #888; font-size: 11px;">
            <small>Click and drag to rotate • Scroll to zoom</small>
        </p>
    </div>
    
    <div id="stats" style="display:none;">
        <div class="stat">
            <span class="stat-label">Total Memories:</span>
            <span class="stat-value" id="total-count">0</span>
        </div>
        <div class="stat">
            <span class="stat-label">Long-term:</span>
            <span class="stat-value" id="longterm-count">0</span>
        </div>
        <div class="stat">
            <span class="stat-label">Short-term:</span>
            <span class="stat-value" id="shortterm-count">0</span>
        </div>
        <div class="stat">
            <span class="stat-label">Episodic:</span>
            <span class="stat-value" id="episodic-count">0</span>
        </div>
        <div class="stat">
            <span class="stat-label">Semantic:</span>
            <span class="stat-value" id="semantic-count">0</span>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Three.js scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0e27);
        scene.fog = new THREE.Fog(0x0a0e27, 200, 500);
        
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
        camera.position.z = 300;
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.getElementById('canvas').appendChild(renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const pointLight = new THREE.PointLight(0x00d9ff, 1, 1000);
        pointLight.position.set(100, 100, 100);
        scene.add(pointLight);
        
        const pointLight2 = new THREE.PointLight(0x00ff88, 0.5, 1000);
        pointLight2.position.set(-100, -100, -100);
        scene.add(pointLight2);
        
        // Fetch graph data
        fetch('/api/dashboard/data')
            .then(r => r.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('info').style.display = 'block';
                document.getElementById('stats').style.display = 'block';
                
                // Create nodes
                data.graph.nodes.forEach(node => {
                    const geometry = new THREE.SphereGeometry(node.size, 16, 16);
                    const material = new THREE.MeshPhongMaterial({ 
                        color: node.color,
                        emissive: node.color,
                        emissiveIntensity: 0.3
                    });
                    const mesh = new THREE.Mesh(geometry, material);
                    mesh.position.set(node.position.x, node.position.y, node.position.z);
                    mesh.userData = node;
                    scene.add(mesh);
                });
                
                // Create edges
                data.graph.edges.forEach(edge => {
                    const node1 = data.graph.nodes.find(n => n.id === edge.source);
                    const node2 = data.graph.nodes.find(n => n.id === edge.target);
                    
                    if (node1 && node2) {
                        const geometry = new THREE.BufferGeometry();
                        const positions = new Float32Array([
                            node1.position.x, node1.position.y, node1.position.z,
                            node2.position.x, node2.position.y, node2.position.z
                        ]);
                        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                        
                        const material = new THREE.LineBasicMaterial({ 
                            color: edge.color,
                            opacity: 0.3,
                            transparent: true
                        });
                        const line = new THREE.Line(geometry, material);
                        scene.add(line);
                    }
                });
                
                // Update stats
                document.getElementById('total-count').textContent = data.stats.total_memories;
                document.getElementById('longterm-count').textContent = data.stats.long_term || 0;
                document.getElementById('shortterm-count').textContent = data.stats.short_term || 0;
                document.getElementById('episodic-count').textContent = data.stats.episodic || 0;
                document.getElementById('semantic-count').textContent = data.stats.semantic || 0;
            })
            .catch(err => {
                document.getElementById('loading').innerHTML = 
                    '<div>❌ Error loading visualization</div><div style="color: #E94B3C; margin-top: 10px; font-size: 12px;">' + err.message + '</div>';
            });
        
        // Animation loop
        let autoRotate = true;
        function animate() {
            requestAnimationFrame(animate);
            if (autoRotate) {
                scene.rotation.x += 0.0005;
                scene.rotation.y += 0.001;
            }
            renderer.render(scene, camera);
        }
        animate();
        
        // Mouse controls
        let mouseDown = false;
        let mouseX = 0, mouseY = 0;
        
        renderer.domElement.addEventListener('mousedown', () => {
            mouseDown = true;
            autoRotate = false;
        });
        
        renderer.domElement.addEventListener('mouseup', () => {
            mouseDown = false;
        });
        
        renderer.domElement.addEventListener('mousemove', (e) => {
            if (mouseDown) {
                const deltaX = e.clientX - mouseX;
                const deltaY = e.clientY - mouseY;
                scene.rotation.y += deltaX * 0.01;
                scene.rotation.x += deltaY * 0.01;
            }
            mouseX = e.clientX;
            mouseY = e.clientY;
        });
        
        // Zoom
        renderer.domElement.addEventListener('wheel', (e) => {
            e.preventDefault();
            camera.position.z += e.deltaY * 0.5;
            camera.position.z = Math.max(50, Math.min(camera.position.z, 800));
        });
        
        // Handle resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>
"""

__all__ = [
    "Vector3",
    "NodeVisual",
    "EdgeVisual",
    "KnowledgeGraphVisualizer",
    "DashboardServer",
    "DASHBOARD_HTML",
]


if __name__ == '__main__':
    print("=== 3D Visualization Module ===")
    print("✓ Dashboard HTML template ready")
    print("✓ Three.js visualization configured")
    print("✓ Force-directed layout implemented")
    print("\nAccess at: http://localhost:8888/dashboard")
