# Crawl4AI RAG System

A powerful Retrieval-Augmented Generation (RAG) system that automatically crawls websites, extracts content, and creates an intelligent chatbot interface for querying the collected information.

## 🚀 Features

- **Automated Web Crawling**: Uses Crawl4AI to extract content from websites and sitemaps
- **Smart Content Processing**: Automatically chunks text while preserving code blocks and paragraphs
- **AI-Powered Analysis**: Generates titles and summaries for content chunks using OpenAI GPT models
- **Vector Search**: Stores content in VectorX for semantic similarity search
- **Interactive Chat Interface**: Streamlit-based UI for querying the knowledge base
- **Contextual Questions**: Automatically generates relevant questions based on crawled content
- **Multiple LLM Support**: Compatible with various OpenAI models

## 🏗️ Architecture

The system consists of four main components:

1. **Crawler (`crawler.py`)**: Web scraping and content extraction
2. **Database (`db.py`)**: Vector database management with VectorX
3. **AI Agent (`pydantic_ai_agent.py`)**: Intelligent query processing
4. **UI (`ui.py`)**: Streamlit web interface

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- VectorX API token and encryption key
- Windows/Linux/macOS

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd crawl4ai-RAG
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   VECTORX_API_TOKEN=your_vectorx_api_token_here
   VECTORX_ENCRYPTION_KEY=your_vectorx_encryption_key_here
   LLM_MODEL=gpt-4-0125-preview  # Optional, defaults to gpt-4-0125-preview
   ```

   **Note**: If you don't have a `VECTORX_ENCRYPTION_KEY`, the system will generate one for you on first run. Make sure to save it securely!

## 🚀 Usage

### Running the Web Interface

1. **Start the Streamlit application**:
   ```bash
   cd gen-rag-crawl
   streamlit run ui.py
   ```

2. **Access the interface**:
   Open your browser and navigate to `http://localhost:8501`

### Using the System

1. **Add Content**:
   - Enter a website URL in the input field
   - The system will automatically try to find and process the sitemap
   - If no sitemap is found, it will process the single URL

2. **Ask Questions**:
   - Once content is processed, use the chat interface to ask questions
   - The system will search through the knowledge base and provide relevant answers
   - Use suggested questions for inspiration

3. **Manage Data**:
   - View database statistics in the sidebar
   - Clear the database to start fresh
   - See processed URLs and sources

### Command Line Usage

You can also run individual components directly:

```bash
# Test the crawler
python crawler.py

# Initialize the database
python db.py
```

## 🔧 Configuration

### Browser Configuration

The crawler is optimized for Windows with the following settings:
- Headless browser mode
- Reduced concurrency for stability
- Windows-specific browser arguments
- Retry logic for failed requests

### Chunking Strategy

Content is intelligently chunked to:
- Preserve code blocks (respects \`\`\` boundaries)
- Maintain paragraph structure
- Ensure readable chunk sizes (default: 5000 characters)
- Avoid breaking sentences

### Vector Database

- Uses VectorX for encrypted vector storage
- OpenAI `text-embedding-3-small` embeddings (1536 dimensions)
- Cosine similarity for search
- Automatic index management

## 📊 Supported Content Types

- **Web Pages**: Any accessible webpage
- **Documentation Sites**: Technical documentation, guides
- **Blogs**: Article content and posts
- **Sitemaps**: Automatic discovery and processing
- **Code Repositories**: GitHub pages and documentation

## 🛡️ Security Features

- Encrypted vector storage with VectorX
- Secure API key management through environment variables
- No data persistence without encryption keys
- Configurable access controls

## 🔍 AI Agent Capabilities

The system includes three main tools:

1. **Document Retrieval**: Semantic search through the knowledge base
2. **Page Listing**: Get all available documentation pages
3. **Full Content Access**: Retrieve complete page content

## 📈 Performance Optimization

- **Parallel Processing**: Concurrent crawling and embedding generation
- **Windows Optimization**: Special handling for Windows asyncio limitations
- **Memory Management**: Efficient chunking and processing
- **Error Handling**: Robust retry mechanisms

## 🐛 Troubleshooting

### Common Issues

1. **Crawler Fails on Windows**:
   - The system automatically sets Windows-specific configurations
   - Reduced concurrency limits prevent memory issues

2. **VectorX Connection Issues**:
   - Ensure your API token is valid
   - Check if the encryption key matches your existing data

3. **OpenAI API Errors**:
   - Verify your API key has sufficient credits
   - Check rate limits if processing many URLs

4. **Memory Issues**:
   - Reduce `max_concurrent` in crawler settings
   - Process smaller batches of URLs

### Debug Mode

Set environment variable for verbose logging:
```bash
export CRAWL4AI_DEBUG=true
```

## 📁 Project Structure

```
crawl4ai-RAG/
├── gen-rag-crawl/
│   ├── crawler.py          # Web crawling and content extraction
│   ├── db.py              # Vector database management
│   ├── pydantic_ai_agent.py # AI agent with tools
│   └── ui.py              # Streamlit web interface
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore patterns
└── README.md              # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

## 🙏 Acknowledgments

- **Crawl4AI**: Web crawling framework
- **VectorX**: Vector database solution
- **OpenAI**: Language models and embeddings
- **Pydantic AI**: AI agent framework
- **Streamlit**: Web interface framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Open an issue on the repository

---

**Happy crawling and chatting!** 🕷️💬
