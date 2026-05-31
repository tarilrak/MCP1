#!/usr/bin/env python3
"""
MCP Server for Knowledge Base RAG
Implements the Model Context Protocol (MCP) with stdio transport
"""

import json
import sys
import chromadb
import re
import hashlib
import numpy as np
from typing import Any, Dict, List


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


class MCPKnowledgeBaseServer:
    """MCP Server implementation for ChromaDB knowledge base"""
    
    def __init__(self):
        self.db = chromadb.PersistentClient(path="./chroma_db")
        self.col = self.db.get_collection("knowledge_base")
        self.emb = Embedder()
        self.request_id = 0
    
    def query_knowledge_base(self, question: str, topic: str = None, n_results: int = 4) -> Dict[str, Any]:
        """Query the knowledge base"""
        try:
            results = self.col.query(
                query_embeddings=self.emb.embed([question]),
                n_results=n_results,
                where={"topic": topic} if topic else None,
                include=["documents", "metadatas", "distances"]
            )
            
            context = "\n\n".join(results["documents"][0])
            
            return {
                "context": context,
                "num_results": len(results["documents"][0]),
                "metadatas": results["metadatas"][0],
                "distances": [float(d) for d in results["distances"][0]]
            }
        except Exception as e:
            raise Exception(f"Query failed: {str(e)}")
    
    def list_topics(self) -> Dict[str, Any]:
        """Get available topics"""
        try:
            results = self.col.get(include=["metadatas"])
            topics = set()
            
            for metadata in results["metadatas"]:
                if "topic" in metadata:
                    topics.add(metadata["topic"])
            
            return {
                "topics": sorted(list(topics)),
                "total_documents": len(results["metadatas"])
            }
        except Exception as e:
            raise Exception(f"Failed to list topics: {str(e)}")
    
    def get_document_count(self) -> Dict[str, Any]:
        """Get document count"""
        try:
            results = self.col.get(include=[])
            return {
                "document_count": len(results["ids"])
            }
        except Exception as e:
            raise Exception(f"Failed to get document count: {str(e)}")
    
    def handle_tool_call(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """Handle tool calls from Claude"""
        if tool_name == "query_knowledge_base":
            return self.query_knowledge_base(
                question=tool_input.get("question"),
                topic=tool_input.get("topic"),
                n_results=tool_input.get("n_results", 4)
            )
        elif tool_name == "list_topics":
            return self.list_topics()
        elif tool_name == "get_document_count":
            return self.get_document_count()
        else:
            raise Exception(f"Unknown tool: {tool_name}")
    
    def send_message(self, message: Dict[str, Any]):
        """Send JSON message to stdout"""
        print(json.dumps(message), file=sys.stdout, flush=True)
    
    def run(self):
        """Main server loop"""
        # Send initialization message
        self.send_message({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": True
                    }
                },
                "clientInfo": {
                    "name": "Claude",
                    "version": "1.0"
                }
            }
        })
        
        # Read and process messages
        for line in sys.stdin:
            try:
                message = json.loads(line)
                self.process_message(message)
            except json.JSONDecodeError:
                continue
            except Exception as e:
                self.send_message({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    }
                })
    
    def process_message(self, message: Dict[str, Any]):
        """Process incoming JSON-RPC messages"""
        method = message.get("method")
        params = message.get("params", {})
        msg_id = message.get("id")
        
        if method == "tools/list":
            self.send_message({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [
                        {
                            "name": "query_knowledge_base",
                            "description": "Query the knowledge base to find relevant information",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "The question to ask"
                                    },
                                    "topic": {
                                        "type": "string",
                                        "enum": ["hr", "product"],
                                        "description": "Optional topic filter"
                                    },
                                    "n_results": {
                                        "type": "integer",
                                        "description": "Number of results (default: 4)"
                                    }
                                },
                                "required": ["question"]
                            }
                        },
                        {
                            "name": "list_topics",
                            "description": "List all available topics",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        },
                        {
                            "name": "get_document_count",
                            "description": "Get total document count",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        }
                    ]
                }
            })
        
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_input = params.get("arguments", {})
            
            try:
                result = self.handle_tool_call(tool_name, tool_input)
                self.send_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result)
                            }
                        ]
                    }
                })
            except Exception as e:
                self.send_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    }
                })


if __name__ == "__main__":
    server = MCPKnowledgeBaseServer()
    server.run()
