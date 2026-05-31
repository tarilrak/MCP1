#!/usr/bin/env python3
"""
MCP Server for Knowledge Base RAG
Exposes ChromaDB query functionality as an MCP server
"""

import chromadb
import re
import hashlib
import numpy as np
import json
from typing import Any

class Embedder:
    """Same embedding logic as ingest.py and qurey.py"""
    def embed(self, texts, dim=256):
        out = []
        for t in texts:
            v = np.zeros(dim)
            for w in re.findall(r'\b\w+\b', t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
            n = np.linalg.norm(v)
            out.append((v / n if n else v).tolist())
        return out


class KnowledgeBaseServer:
    """MCP Server for querying the knowledge base"""
    
    def __init__(self):
        self.db = chromadb.PersistentClient(path="./chroma_db")
        self.col = self.db.get_collection("knowledge_base")
        self.emb = Embedder()
    
    def query_knowledge_base(self, question: str, topic: str = None, n_results: int = 4) -> dict:
        """
        Query the knowledge base with a question
        
        Args:
            question: The question to ask
            topic: Optional topic filter (e.g., "hr", "product")
            n_results: Number of results to return
            
        Returns:
            Dictionary with context and metadata
        """
        try:
            results = self.col.query(
                query_embeddings=self.emb.embed([question]),
                n_results=n_results,
                where={"topic": topic} if topic else None,
                include=["documents", "metadatas", "distances"]
            )
            
            context = "\n\n".join(results["documents"][0])
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            return {
                "success": True,
                "context": context,
                "metadatas": metadatas,
                "distances": distances,
                "num_results": len(results["documents"][0])
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_topics(self) -> dict:
        """Get list of available topics in the knowledge base"""
        try:
            # Get all documents with metadata
            results = self.col.get(include=["metadatas"])
            topics = set()
            
            for metadata in results["metadatas"]:
                if "topic" in metadata:
                    topics.add(metadata["topic"])
            
            return {
                "success": True,
                "topics": sorted(list(topics)),
                "total_documents": len(results["metadatas"])
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_document_count(self) -> dict:
        """Get total number of documents in the knowledge base"""
        try:
            results = self.col.get(include=[])
            return {
                "success": True,
                "document_count": len(results["ids"])
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# MCP Tool Definitions (for Claude to use)
MCP_TOOLS = [
    {
        "name": "query_knowledge_base",
        "description": "Query the knowledge base to find relevant information about HR policies and product FAQs",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the knowledge base"
                },
                "topic": {
                    "type": "string",
                    "enum": ["hr", "product"],
                    "description": "Optional topic to filter results (hr or product)"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 4)",
                    "default": 4
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "list_topics",
        "description": "List all available topics in the knowledge base",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_document_count",
        "description": "Get the total number of documents in the knowledge base",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]


if __name__ == "__main__":
    # Example usage
    server = KnowledgeBaseServer()
    
    print("MCP Server initialized successfully!")
    print("\nAvailable topics:")
    topics_result = server.list_topics()
    if topics_result["success"]:
        print(f"  Topics: {', '.join(topics_result['topics'])}")
        print(f"  Total documents: {topics_result['total_documents']}")
    
    print("\nTools available:")
    for tool in MCP_TOOLS:
        print(f"  - {tool['name']}: {tool['description']}")
