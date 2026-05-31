# Knowledge Base MCP Server Setup

Your project is now converted to use **Model Context Protocol (MCP)** for seamless integration with Claude.

## Architecture

```
Documents (MD/PDF)
    ↓
ingest.py (Create embeddings)
    ↓
ChromaDB (Vector database: chroma_db/)
    ↓
MCP Server (mcp_server_stdio.py)
    ↓
Claude (Automatic tool integration)
```

## Files Created

### 1. **mcp_server.py** 
- Standalone MCP server implementation
- Provides the knowledge base query interface
- Can be used independently or integrated with Claude

### 2. **mcp_server_stdio.py**
- Full MCP protocol implementation with stdio transport
- Works with Claude through the MCP protocol
- Handles JSON-RPC messages from Claude

### 3. **mcp_config.json**
- Configuration file for connecting the MCP server to Claude

## Usage

### Step 1: Ensure Database is Built
```bash
python ingest.py
```
This reads documents from `docs/` and builds the ChromaDB database.

### Step 2: Run the MCP Server (for integration)
```bash
python mcp_server.py
```
This validates and shows available tools.

### Step 3: Use with Claude
Claude will automatically have access to these tools:
- `query_knowledge_base` - Search your knowledge base
- `list_topics` - See available topics
- `get_document_count` - Check database size

## How It Works Now

**Before (Manual):**
1. Run `ingest.py` → Creates database
2. Run `qurey.py` → Gets answer

**After (MCP - Automatic):**
1. Run `ingest.py` → Creates database (once)
2. Claude automatically queries database when asked questions

## Available Tools

### query_knowledge_base
Query the knowledge base to find information
```
Parameters:
  - question (required): The question to ask
  - topic (optional): Filter by "hr" or "product"
  - n_results (optional): Number of results (default: 4)
```

### list_topics
Get available topics in the knowledge base
```
Returns: List of topics and total document count
```

### get_document_count
Get total number of documents
```
Returns: Document count in the knowledge base
```

## Updating Knowledge Base

If you add new documents:
```bash
# 1. Place new files in docs/ folder
# 2. Run ingest.py to rebuild database
python ingest.py

# 3. Claude automatically uses updated knowledge base
```

## Key Benefits of MCP

✅ **Seamless Integration** - Claude automatically accesses your knowledge base  
✅ **No Manual Queries** - No need to run separate scripts  
✅ **Scalable** - Easy to add more tools and data sources  
✅ **Protocol Standard** - Works with any MCP-compatible client  

## Next Steps

- Add more documents to `docs/` folder
- Run `ingest.py` when adding new documents
- Ask Claude questions - it will automatically search your knowledge base!
