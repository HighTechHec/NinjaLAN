"""
Second Brain CLI

Interactive command-line interface for knowledge management.

Commands:
- ingest <content> - Add new information
- search <query> - Semantic search
- ask <question> - Question answering
- review - Get review queue
- stats - Show statistics
- cleanup - Remove weak memories
- entity add <name> <type> - Add entity
- relation add <e1> <rel> <e2> - Add relationship
- neighbors <entity> - Show neighbors
- help - Show help
- exit - Exit CLI
"""

import cmd
import sys
import json
from typing import Optional
from datetime import datetime

from core import SecondBrain
from nvidia_inference import NVIDIAInferenceEngine
from vector_db import MilvusVectorDB
from reasoning import RetrievalPipeline, ReasoningEngine


class SecondBrainCLI(cmd.Cmd):
    """Interactive CLI for the Second Brain system."""
    
    intro = """
╔════════════════════════════════════════════════════════════════╗
║              SECOND BRAIN - Interactive CLI                    ║
║         Production-grade knowledge management system           ║
╚════════════════════════════════════════════════════════════════╝

Type 'help' for available commands or 'help <command>' for details.
Type 'exit' to quit.
"""
    
    prompt = "🧠 > "
    
    def __init__(self):
        super().__init__()
        self.brain = None
        self.vector_db = None
        self.inference_engine = None
        self.retrieval_pipeline = None
        self.reasoning_engine = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize all system components."""
        print("Initializing Second Brain system...")
        
        try:
            self.brain = SecondBrain()
            self.vector_db = MilvusVectorDB()
            self.inference_engine = NVIDIAInferenceEngine()
            self.retrieval_pipeline = RetrievalPipeline(
                self.vector_db,
                self.inference_engine,
                self.brain.knowledge_graph
            )
            self.reasoning_engine = ReasoningEngine(
                self.retrieval_pipeline,
                self.inference_engine
            )
            print("✓ System initialized successfully!\n")
        except Exception as e:
            print(f"✗ Error initializing system: {e}")
            print("Some features may be unavailable.\n")
    
    # Command: ingest
    def do_ingest(self, arg):
        """
        Ingest new content into the second brain.
        
        Usage: ingest <content>
        Example: ingest Python is a high-level programming language
        
        Options:
        - Can include tags with #tag
        - Can specify type with --type=<type>
        """
        if not arg:
            print("Error: Please provide content to ingest")
            return
        
        # Parse arguments
        content = arg
        memory_type = 'long_term'
        tags = []
        
        # Extract tags
        words = content.split()
        filtered_words = []
        for word in words:
            if word.startswith('#'):
                tags.append(word[1:])
            elif word.startswith('--type='):
                memory_type = word.split('=')[1]
            else:
                filtered_words.append(word)
        
        content = ' '.join(filtered_words)
        
        if not content:
            print("Error: No content to ingest after parsing")
            return
        
        try:
            # Ingest
            memory_id = self.brain.ingest(content, memory_type=memory_type, tags=tags)
            
            # Generate embedding and store
            embedding = self.inference_engine.embed(content)
            self.vector_db.insert(memory_id, embedding, content, {'tags': tags})
            
            print(f"✓ Content ingested successfully")
            print(f"  Memory ID: {memory_id}")
            print(f"  Type: {memory_type}")
            if tags:
                print(f"  Tags: {', '.join(tags)}")
        except Exception as e:
            print(f"✗ Error ingesting content: {e}")
    
    # Command: search
    def do_search(self, arg):
        """
        Perform semantic search.
        
        Usage: search <query>
        Example: search GPU acceleration for AI
        
        Options:
        - --top=<n> : Number of results (default: 5)
        - --no-rerank : Disable reranking
        - --no-expand : Disable graph expansion
        """
        if not arg:
            print("Error: Please provide a search query")
            return
        
        # Parse options
        parts = arg.split()
        options = {}
        query_parts = []
        
        for part in parts:
            if part.startswith('--top='):
                options['top_k'] = int(part.split('=')[1])
            elif part == '--no-rerank':
                options['use_reranking'] = False
            elif part == '--no-expand':
                options['use_expansion'] = False
            else:
                query_parts.append(part)
        
        query = ' '.join(query_parts)
        top_k = options.get('top_k', 5)
        use_reranking = options.get('use_reranking', True)
        use_expansion = options.get('use_expansion', True)
        
        try:
            results = self.retrieval_pipeline.retrieve(
                query,
                top_k=top_k,
                use_reranking=use_reranking,
                use_expansion=use_expansion
            )
            
            print(f"\n🔍 Search results for: '{query}'")
            print(f"Found {len(results)} results\n")
            
            for i, result in enumerate(results, 1):
                print(f"[{i}] Score: {result.score:.4f} | Source: {result.source}")
                print(f"    {result.content[:100]}{'...' if len(result.content) > 100 else ''}")
                print()
        except Exception as e:
            print(f"✗ Error searching: {e}")
    
    # Command: ask
    def do_ask(self, arg):
        """
        Ask a question and get a reasoned answer.
        
        Usage: ask <question>
        Example: ask What are the benefits of GPU acceleration?
        
        Options:
        - --multi-hop : Use multi-hop reasoning
        - --context=<n> : Max context documents (default: 5)
        """
        if not arg:
            print("Error: Please provide a question")
            return
        
        # Parse options
        parts = arg.split()
        options = {}
        question_parts = []
        
        for part in parts:
            if part == '--multi-hop':
                options['multi_hop'] = True
            elif part.startswith('--context='):
                options['max_context'] = int(part.split('=')[1])
            else:
                question_parts.append(part)
        
        question = ' '.join(question_parts)
        multi_hop = options.get('multi_hop', False)
        max_context = options.get('max_context', 5)
        
        try:
            print(f"\n💭 Thinking...")
            
            if multi_hop:
                trace = self.reasoning_engine.multi_hop_reasoning(question)
            else:
                trace = self.reasoning_engine.answer_question(question, max_context)
            
            print(f"\n❓ Question: {trace.query}")
            print(f"\n📊 Reasoning steps:")
            for step in trace.steps:
                print(f"   • {step}")
            
            print(f"\n✓ Answer (confidence: {trace.confidence:.2f}):")
            print(f"   {trace.answer}")
            
            if trace.sources:
                print(f"\n📚 Sources: {', '.join(trace.sources[:3])}")
            print()
        except Exception as e:
            print(f"✗ Error answering question: {e}")
    
    # Command: review
    def do_review(self, arg):
        """
        Get memories that need review (spaced repetition).
        
        Usage: review [limit]
        Example: review 10
        """
        limit = 10
        if arg:
            try:
                limit = int(arg)
            except ValueError:
                print("Error: Limit must be a number")
                return
        
        try:
            memories = self.brain.get_review_queue(limit)
            
            if not memories:
                print("✓ No memories need review right now!")
                return
            
            print(f"\n📚 Review Queue ({len(memories)} items)\n")
            
            for i, memory in enumerate(memories, 1):
                should_review, days_until = memory.should_review()
                retention = memory.calculate_retention()
                
                print(f"[{i}] ID: {memory.id}")
                print(f"    Content: {memory.content[:80]}...")
                print(f"    Retention: {retention:.2f} | Strength: {memory.strength:.2f}")
                print(f"    Access count: {memory.access_count} | Last accessed: {datetime.fromtimestamp(memory.last_accessed).strftime('%Y-%m-%d %H:%M')}")
                print()
        except Exception as e:
            print(f"✗ Error getting review queue: {e}")
    
    # Command: stats
    def do_stats(self, arg):
        """
        Show system statistics.
        
        Usage: stats
        """
        try:
            stats = self.brain.get_comprehensive_stats()
            
            if self.vector_db:
                stats['vector_db'] = self.vector_db.get_stats()
            if self.inference_engine:
                stats['inference_engine'] = self.inference_engine.get_stats()
            if self.retrieval_pipeline:
                stats['retrieval_pipeline'] = self.retrieval_pipeline.get_stats()
            
            print("\n📊 System Statistics\n")
            print(json.dumps(stats, indent=2))
            print()
        except Exception as e:
            print(f"✗ Error getting statistics: {e}")
    
    # Command: cleanup
    def do_cleanup(self, arg):
        """
        Remove weak memories below retention threshold.
        
        Usage: cleanup [threshold]
        Example: cleanup 0.1
        """
        threshold = 0.1
        if arg:
            try:
                threshold = float(arg)
            except ValueError:
                print("Error: Threshold must be a number between 0 and 1")
                return
        
        try:
            removed = self.brain.cleanup(threshold)
            print(f"✓ Removed {removed} weak memories (threshold: {threshold})")
        except Exception as e:
            print(f"✗ Error during cleanup: {e}")
    
    # Command: entity
    def do_entity(self, arg):
        """
        Manage entities in the knowledge graph.
        
        Usage: entity add <name> <type>
        Example: entity add Python programming_language
        """
        if not arg:
            print("Error: Please provide entity command")
            print("Usage: entity add <name> <type>")
            return
        
        parts = arg.split()
        if len(parts) < 3:
            print("Error: Please provide entity name and type")
            print("Usage: entity add <name> <type>")
            return
        
        command = parts[0]
        name = parts[1]
        entity_type = parts[2]
        
        if command == 'add':
            try:
                self.brain.knowledge_graph.add_entity(name, entity_type)
                print(f"✓ Added entity: {name} ({entity_type})")
            except Exception as e:
                print(f"✗ Error adding entity: {e}")
        else:
            print(f"Unknown entity command: {command}")
    
    # Command: relation
    def do_relation(self, arg):
        """
        Manage relationships in the knowledge graph.
        
        Usage: relation add <entity1> <relation> <entity2>
        Example: relation add Python uses_for machine_learning
        """
        if not arg:
            print("Error: Please provide relation command")
            print("Usage: relation add <entity1> <relation> <entity2>")
            return
        
        parts = arg.split()
        if len(parts) < 4:
            print("Error: Please provide entity1, relation, and entity2")
            print("Usage: relation add <entity1> <relation> <entity2>")
            return
        
        command = parts[0]
        entity1 = parts[1]
        relation = parts[2]
        entity2 = parts[3]
        
        if command == 'add':
            try:
                self.brain.knowledge_graph.add_relationship(entity1, relation, entity2)
                print(f"✓ Added relationship: {entity1} -{relation}-> {entity2}")
            except Exception as e:
                print(f"✗ Error adding relationship: {e}")
        else:
            print(f"Unknown relation command: {command}")
    
    # Command: neighbors
    def do_neighbors(self, arg):
        """
        Get neighbors of an entity in the knowledge graph.
        
        Usage: neighbors <entity> [depth]
        Example: neighbors Python 2
        """
        if not arg:
            print("Error: Please provide entity name")
            return
        
        parts = arg.split()
        entity = parts[0]
        depth = 1
        
        if len(parts) > 1:
            try:
                depth = int(parts[1])
            except ValueError:
                print("Error: Depth must be a number")
                return
        
        try:
            neighbors = self.brain.knowledge_graph.get_neighbors(entity, depth=depth)
            
            if not neighbors:
                print(f"No neighbors found for entity: {entity}")
                return
            
            print(f"\n🔗 Neighbors of '{entity}' (depth={depth})\n")
            for neighbor in neighbors:
                print(f"  • {neighbor.get('name')} ({neighbor.get('type', 'unknown')})")
            print()
        except Exception as e:
            print(f"✗ Error getting neighbors: {e}")
    
    # Command: exit
    def do_exit(self, arg):
        """Exit the CLI."""
        print("\nClosing Second Brain system...")
        if self.brain:
            self.brain.close()
        if self.vector_db:
            self.vector_db.close()
        print("Goodbye! 👋\n")
        return True
    
    # Command: quit (alias for exit)
    def do_quit(self, arg):
        """Exit the CLI."""
        return self.do_exit(arg)
    
    # Handle EOF (Ctrl+D)
    def do_EOF(self, arg):
        """Handle EOF."""
        print()
        return self.do_exit(arg)
    
    # Handle empty line
    def emptyline(self):
        """Do nothing on empty line."""
        pass
    
    # Handle unknown commands
    def default(self, line):
        """Handle unknown commands."""
        print(f"Unknown command: {line}")
        print("Type 'help' for available commands")


def main():
    """Main entry point for CLI."""
    try:
        cli = SecondBrainCLI()
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
