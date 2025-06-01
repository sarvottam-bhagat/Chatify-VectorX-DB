# VectorX Database MCP Server

A FastMCP server that provides Model Context Protocol (MCP) access to your VectorX vector database containing web content from the Crawl4AI RAG system.

## 🎯 Purpose

This MCP server allows AI assistants and LLM clients to interact with your VectorX database through standardized MCP tools. You can:

- Search documents using semantic similarity
- Query specific URLs or sources
- Get database statistics and information
- Ask questions with AI-powered responses
- Perform text-based searches

## 🛠️ Installation

### 1. Install FastMCP

```bash
pip install fastmcp
```

Or install from the provided requirements file:

```bash
pip install -r mcp_requirements.txt
```

### 2. Set up Environment Variables

Make sure your `.env` file contains:

```env
OPENAI_API_KEY=your_openai_api_key_here
VECTORX_API_TOKEN=your_vectorx_api_token_here
VECTORX_ENCRYPTION_KEY=your_vectorx_encryption_key_here
LLM_MODEL=gpt-4o-mini  # Optional, defaults to gpt-4o-mini
```

## 🚀 Usage

### Running the Server Standalone

```bash
# From the project root directory
python vectorx_mcp_server.py
```

### Running with FastMCP CLI

```bash
# Install the server
fastmcp install vectorx_mcp_server.py --name "VectorX Database"

# Run the server
fastmcp run vectorx_mcp_server.py
```

### Testing with MCP Inspector

```bash
# Test the server with MCP Inspector
npx -y @modelcontextprotocol/inspector python vectorx_mcp_server.py
```

## 🔧 Client Configuration

### Claude Desktop

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "vectorx-database": {
      "command": "python",
      "args": ["C:\\path\\to\\your\\crawl4ai-RAG\\vectorx_mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your-key-here",
        "VECTORX_API_TOKEN": "your-token-here",
        "VECTORX_ENCRYPTION_KEY": "your-key-here"
      }
    }
  }
}
```

### Cursor

Add to your `mcp.json` file:

```json
{
  "mcpServers": {
    "vectorx-database": {
      "command": "python",
      "args": ["./vectorx_mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your-key-here",
        "VECTORX_API_TOKEN": "your-token-here", 
        "VECTORX_ENCRYPTION_KEY": "your-key-here"
      }
    }
  }
}
```

### VS Code (with MCP extension)

Add to your VS Code settings:

```json
{
  "servers": {
    "VectorX Database": {
      "type": "stdio",
      "command": "python",
      "args": ["./vectorx_mcp_server.py"]
    }
  }
}
```

## 🧰 Available Tools

### 1. `search_documents`
Semantic search through the vector database using OpenAI embeddings.

**Parameters:**
- `query` (string): Search query text
- `max_results` (int, optional): Maximum results to return (default: 5)

**Example:**
```
Search for documents about "web crawling techniques"
```

### 2. `get_database_stats`
Get comprehensive statistics about the database.

**Returns:**
- Total documents count
- Unique URLs and sources
- Last update timestamp
- Sample URLs

### 3. `list_all_urls`
List all URLs stored in the database, grouped by source domain.

**Returns:**
- URLs organized by source
- Total URL count
- Source count

### 4. `get_documents_by_url`
Retrieve all document chunks for a specific URL.

**Parameters:**
- `url` (string): The URL to retrieve documents for

**Returns:**
- All chunks for the URL, sorted by chunk number
- Metadata and content for each chunk

### 5. `get_documents_by_source`
Get documents from a specific source domain.

**Parameters:**
- `source` (string): Source domain (e.g., 'docs.example.com')
- `limit` (int, optional): Maximum documents to return (default: 20)

### 6. `ask_question_about_content`
Ask a question about the content with AI-powered responses.

**Parameters:**
- `question` (string): Question about the stored content
- `max_context_chunks` (int, optional): Maximum context chunks (default: 3)

**Returns:**
- AI-generated answer based on relevant content
- Supporting context chunks
- Source URLs

### 7. `search_by_title_or_summary`
Text-based search in document titles and summaries.

**Parameters:**
- `search_term` (string): Term to search for
- `limit` (int, optional): Maximum results (default: 10)

## 📊 Resources

### `vectorx://server/info`
Get information about the server and current database status.

## 🔍 Example Interactions

### Basic Database Query
```
"What is the current state of the database?"
→ Uses get_database_stats tool
```

### Semantic Search
```
"Find documents about API authentication"
→ Uses search_documents tool with semantic search
```

### Specific URL Lookup
```
"Show me all content from https://docs.example.com/api/auth"
→ Uses get_documents_by_url tool
```

### AI-Powered Q&A
```
"How do I implement user authentication based on the documentation?"
→ Uses ask_question_about_content tool with LLM response
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'gen_rag_crawl'
   ```
   **Solution:** Run the server from the project root directory where the `gen-rag-crawl` folder is located.

2. **VectorX Connection Issues**
   ```
   ValueError: VECTORX_API_TOKEN environment variable is required
   ```
   **Solution:** Ensure all required environment variables are set in your `.env` file.

3. **OpenAI API Errors**
   ```
   Error getting embedding: API key not found
   ```
   **Solution:** Verify your OpenAI API key is correct and has sufficient credits.

### Debug Mode

Set the environment variable for verbose output:
```bash
export FASTMCP_DEBUG=true
python vectorx_mcp_server.py
```

### Testing the Server

1. **Test with MCP Inspector:**
   ```bash
   npx -y @modelcontextprotocol/inspector python vectorx_mcp_server.py
   ```

2. **Test individual tools:**
   ```python
   from vectorx_mcp_server import get_database_stats
   stats = get_database_stats()
   print(stats)
   ```

## 🔐 Security

- All VectorX data is encrypted using your encryption key
- API keys are passed through environment variables
- No sensitive data is logged or exposed

## 📁 File Structure

```
crawl4ai-RAG/
├── vectorx_mcp_server.py      # Main MCP server
├── mcp_requirements.txt       # MCP-specific requirements
├── mcp_config.json           # Example configuration
├── MCP_SERVER_README.md      # This file
└── gen-rag-crawl/
    ├── db.py                 # VectorX database wrapper
    └── ...                   # Other RAG system files
```

## 🤝 Integration with Your RAG System

This MCP server works seamlessly with your existing Crawl4AI RAG system:

1. **Data Source:** Uses the same VectorX database as your Streamlit UI
2. **Embeddings:** Uses the same OpenAI embedding model
3. **No Conflicts:** Read-only operations don't interfere with your main system
4. **Shared Configuration:** Uses the same environment variables

## 📈 Performance Tips

- Use specific search terms for better semantic search results
- Limit result counts for faster responses
- Use `get_database_stats` to understand your data before querying
- Use `search_by_title_or_summary` for exact term matching

## 🛠️ Customization

You can modify the server by:

1. **Adding new tools:** Add functions with `@mcp.tool()` decorator
2. **Modifying search logic:** Update the search algorithms in existing tools
3. **Adding resources:** Use `@mcp.resource()` for static information
4. **Changing models:** Update the LLM model in environment variables

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your environment variables are correctly set
3. Ensure your VectorX database has data (run your main RAG system first)
4. Test with the MCP Inspector tool

---

**Ready to explore your knowledge base through MCP!** 🚀💫
