#!/usr/bin/env python3
"""
VectorX Database MCP Server

A FastMCP server that provides tools to interact with the VectorX database
used in the Crawl4AI RAG system. Allows querying, adding, and managing
vector data through the Model Context Protocol.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import asyncio

from fastmcp import FastMCP
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Import your existing database module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'gen-rag-crawl'))
from db import VectorXCollection, init_collection

# Load environment variables
load_dotenv()

# Initialize the MCP server
mcp = FastMCP(
    name="VectorX Database Server",
    instructions="""
    This server provides access to a VectorX vector database containing web content
    and documentation from the Crawl4AI RAG system. You can:
    
    1. Query documents using semantic search
    2. Get information about stored documents
    3. List all available documents and sources
    4. Get statistics about the database
    5. Search for specific content by URL or metadata
    
    The database contains chunked web content with embeddings, metadata, and
    full text content that can be searched semantically.
    """
)

# Initialize OpenAI client for embeddings
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize VectorX collection
vectorx_collection = None

def get_vectorx_collection():
    """Get or initialize the VectorX collection."""
    global vectorx_collection
    if vectorx_collection is None:
        vectorx_collection = init_collection()
    return vectorx_collection


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", 
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return zero vector as fallback
        return [0.0] * 1536


@mcp.tool()
async def search_documents(
    query: str, 
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Search documents in the VectorX database using semantic similarity.
    
    Args:
        query: The search query text
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Dictionary containing search results with documents, metadata, and relevance scores
    """
    try:
        collection = get_vectorx_collection()
        
        # Get query embedding
        query_embedding = await get_embedding(query)
        
        # Search the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
            include=["documents", "metadatas"]
        )
        
        if not results["documents"][0]:
            return {
                "status": "success",
                "query": query,
                "results_count": 0,
                "message": "No documents found matching the query",
                "results": []
            }
        
        # Format results
        formatted_results = []
        for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            result = {
                "rank": i + 1,
                "content": doc,
                "metadata": metadata,
                "url": metadata.get("url", "Unknown"),
                "title": metadata.get("title", "Untitled"),
                "source": metadata.get("source", "Unknown"),
                "chunk_number": metadata.get("chunk_number", 0),
                "summary": metadata.get("summary", "No summary available")
            }
            formatted_results.append(result)
        
        return {
            "status": "success",
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "error": str(e),
            "results": []
        }


@mcp.tool()
def get_database_stats() -> Dict[str, Any]:
    """
    Get comprehensive statistics about the VectorX database.
    
    Returns:
        Dictionary containing database statistics including document counts,
        sources, URLs, and last update information
    """
    try:
        collection = get_vectorx_collection()
        
        # Get all documents and metadata
        results = collection.get(include=["metadatas"])
        
        if not results["metadatas"]:
            return {
                "status": "success",
                "message": "Database is empty",
                "total_documents": 0,
                "unique_urls": 0,
                "unique_sources": 0,
                "last_updated": None
            }
        
        # Analyze metadata
        urls = set()
        sources = set()
        crawl_times = []
        
        for meta in results["metadatas"]:
            if meta and isinstance(meta, dict):
                # Collect URLs
                url = meta.get("url")
                if url:
                    urls.add(url)
                
                # Collect sources
                source = meta.get("source")
                if source:
                    sources.add(source)
                
                # Collect crawl times
                crawled_at = meta.get("crawled_at")
                if crawled_at:
                    crawl_times.append(crawled_at)
        
        # Find latest crawl time
        last_updated = None
        if crawl_times:
            last_updated = max(crawl_times)
        
        return {
            "status": "success",
            "total_documents": len(results["metadatas"]),
            "unique_urls": len(urls),
            "unique_sources": len(sources),
            "sources": sorted(list(sources)),
            "sample_urls": sorted(list(urls))[:10],  # First 10 URLs
            "total_urls": len(urls),
            "last_updated": last_updated,
            "database_ready": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "total_documents": 0
        }


@mcp.tool()
def list_all_urls() -> Dict[str, Any]:
    """
    List all unique URLs stored in the VectorX database.
    
    Returns:
        Dictionary containing all URLs grouped by source domain
    """
    try:
        collection = get_vectorx_collection()
        
        # Get all metadata
        results = collection.get(include=["metadatas"])
        
        if not results["metadatas"]:
            return {
                "status": "success",
                "message": "No URLs found - database is empty",
                "urls_by_source": {},
                "total_urls": 0
            }
        
        # Group URLs by source
        urls_by_source = {}
        all_urls = set()
        
        for meta in results["metadatas"]:
            if meta and isinstance(meta, dict):
                url = meta.get("url")
                source = meta.get("source", "Unknown")
                
                if url:
                    all_urls.add(url)
                    if source not in urls_by_source:
                        urls_by_source[source] = set()
                    urls_by_source[source].add(url)
        
        # Convert sets to sorted lists
        for source in urls_by_source:
            urls_by_source[source] = sorted(list(urls_by_source[source]))
        
        return {
            "status": "success",
            "urls_by_source": urls_by_source,
            "total_urls": len(all_urls),
            "total_sources": len(urls_by_source)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "urls_by_source": {}
        }


@mcp.tool()
def get_documents_by_url(url: str) -> Dict[str, Any]:
    """
    Retrieve all document chunks for a specific URL.
    
    Args:
        url: The URL to retrieve documents for
    
    Returns:
        Dictionary containing all chunks for the specified URL
    """
    try:
        collection = get_vectorx_collection()
        
        # Query by URL using where clause
        results = collection.get(
            where={"url": url},
            include=["documents", "metadatas"]
        )
        
        if not results["documents"]:
            return {
                "status": "success",
                "url": url,
                "message": f"No documents found for URL: {url}",
                "chunks": [],
                "total_chunks": 0
            }
        
        # Sort by chunk number and format results
        chunks_data = list(zip(results["documents"], results["metadatas"]))
        chunks_data.sort(key=lambda x: x[1].get("chunk_number", 0))
        
        formatted_chunks = []
        for doc, meta in chunks_data:
            chunk_info = {
                "chunk_number": meta.get("chunk_number", 0),
                "title": meta.get("title", "Untitled"),
                "summary": meta.get("summary", "No summary"),
                "content": doc,
                "chunk_size": len(doc),
                "crawled_at": meta.get("crawled_at"),
                "metadata": meta
            }
            formatted_chunks.append(chunk_info)
        
        return {
            "status": "success",
            "url": url,
            "total_chunks": len(formatted_chunks),
            "chunks": formatted_chunks
        }
        
    except Exception as e:
        return {
            "status": "error",
            "url": url,
            "error": str(e),
            "chunks": []
        }


@mcp.tool()
def get_documents_by_source(source: str, limit: int = 20) -> Dict[str, Any]:
    """
    Retrieve documents from a specific source domain.
    
    Args:
        source: The source domain (e.g., 'docs.example.com')
        limit: Maximum number of documents to return (default: 20)
    
    Returns:
        Dictionary containing documents from the specified source
    """
    try:
        collection = get_vectorx_collection()
        
        # Get all documents and filter by source
        results = collection.get(include=["documents", "metadatas"])
        
        if not results["metadatas"]:
            return {
                "status": "success",
                "source": source,
                "message": "Database is empty",
                "documents": [],
                "total_found": 0
            }
        
        # Filter by source
        filtered_docs = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            if meta and meta.get("source") == source:
                doc_info = {
                    "url": meta.get("url", "Unknown"),
                    "title": meta.get("title", "Untitled"),
                    "summary": meta.get("summary", "No summary"),
                    "chunk_number": meta.get("chunk_number", 0),
                    "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                    "full_content": doc,
                    "crawled_at": meta.get("crawled_at"),
                    "metadata": meta
                }
                filtered_docs.append(doc_info)
        
        # Sort by URL and chunk number
        filtered_docs.sort(key=lambda x: (x["url"], x["chunk_number"]))
        
        # Apply limit
        limited_docs = filtered_docs[:limit]
        
        return {
            "status": "success",
            "source": source,
            "total_found": len(filtered_docs),
            "returned_count": len(limited_docs),
            "documents": limited_docs,
            "truncated": len(filtered_docs) > limit
        }
        
    except Exception as e:
        return {
            "status": "error",
            "source": source,
            "error": str(e),
            "documents": []
        }


@mcp.tool()
async def ask_question_about_content(question: str, max_context_chunks: int = 3) -> Dict[str, Any]:
    """
    Ask a question about the content in the database and get a comprehensive answer
    with relevant context chunks.
    
    Args:
        question: The question to ask about the stored content
        max_context_chunks: Maximum number of context chunks to include (default: 3)
    
    Returns:
        Dictionary containing the answer and supporting context
    """
    try:
        collection = get_vectorx_collection()
        
        # Search for relevant content
        query_embedding = await get_embedding(question)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max_context_chunks,
            include=["documents", "metadatas"]
        )
        
        if not results["documents"][0]:
            return {
                "status": "success",
                "question": question,
                "answer": "I couldn't find any relevant content in the database to answer your question.",
                "context_chunks": [],
                "sources": []
            }
        
        # Format context chunks
        context_chunks = []
        sources = set()
        
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunk = {
                "content": doc,
                "title": meta.get("title", "Untitled"),
                "url": meta.get("url", "Unknown"),
                "source": meta.get("source", "Unknown"),
                "summary": meta.get("summary", "No summary")
            }
            context_chunks.append(chunk)
            sources.add(meta.get("url", "Unknown"))
        
        # Create a comprehensive context
        context_text = "\n\n".join([
            f"Title: {chunk['title']}\nContent: {chunk['content']}\nSource: {chunk['url']}"
            for chunk in context_chunks
        ])
        
        # Generate answer using OpenAI
        try:
            response = await openai_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful assistant that answers questions based on provided context. 
                        Use only the information from the context to answer questions. If the context doesn't 
                        contain enough information to answer the question, say so clearly. Cite specific sources 
                        when possible."""
                    },
                    {
                        "role": "user", 
                        "content": f"Question: {question}\n\nContext:\n{context_text}"
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
        except Exception as e:
            answer = f"Error generating answer: {str(e)}. However, I found relevant context that might help answer your question."
        
        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "context_chunks": context_chunks,
            "sources": sorted(list(sources)),
            "total_context_chunks": len(context_chunks)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "question": question,
            "error": str(e),
            "answer": "Error processing question",
            "context_chunks": []
        }


@mcp.tool()
def search_by_title_or_summary(search_term: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search documents by title or summary content (text-based search).
    
    Args:
        search_term: Term to search for in titles and summaries
        limit: Maximum number of results to return (default: 10)
    
    Returns:
        Dictionary containing matching documents
    """
    try:
        collection = get_vectorx_collection()
        
        # Get all documents
        results = collection.get(include=["documents", "metadatas"])
        
        if not results["metadatas"]:
            return {
                "status": "success",
                "search_term": search_term,
                "message": "Database is empty",
                "matches": [],
                "total_matches": 0
            }
        
        # Search in titles and summaries
        matches = []
        search_lower = search_term.lower()
        
        for doc, meta in zip(results["documents"], results["metadatas"]):
            if meta:
                title = meta.get("title", "").lower()
                summary = meta.get("summary", "").lower()
                
                if search_lower in title or search_lower in summary:
                    match = {
                        "url": meta.get("url", "Unknown"),
                        "title": meta.get("title", "Untitled"),
                        "summary": meta.get("summary", "No summary"),
                        "chunk_number": meta.get("chunk_number", 0),
                        "source": meta.get("source", "Unknown"),
                        "content_preview": doc[:300] + "..." if len(doc) > 300 else doc,
                        "crawled_at": meta.get("crawled_at"),
                        "match_in_title": search_lower in title,
                        "match_in_summary": search_lower in summary
                    }
                    matches.append(match)
        
        # Sort by relevance (title matches first, then by URL)
        matches.sort(key=lambda x: (not x["match_in_title"], x["url"], x["chunk_number"]))
        
        # Apply limit
        limited_matches = matches[:limit]
        
        return {
            "status": "success",
            "search_term": search_term,
            "total_matches": len(matches),
            "returned_count": len(limited_matches),
            "matches": limited_matches,
            "truncated": len(matches) > limit
        }
        
    except Exception as e:
        return {
            "status": "error",
            "search_term": search_term,
            "error": str(e),
            "matches": []
        }


# Add a resource for server information
@mcp.resource("vectorx://server/info")
def server_info() -> str:
    """Get information about the VectorX MCP server."""
    stats = get_database_stats()
    
    info = f"""
# VectorX Database MCP Server

This server provides access to a VectorX vector database containing web content
from the Crawl4AI RAG system.

## Current Database Status
- Total Documents: {stats.get('total_documents', 0)}
- Unique URLs: {stats.get('unique_urls', 0)}
- Unique Sources: {stats.get('unique_sources', 0)}
- Last Updated: {stats.get('last_updated', 'Unknown')}

## Available Tools
1. **search_documents** - Semantic search through documents
2. **get_database_stats** - Get database statistics
3. **list_all_urls** - List all stored URLs
4. **get_documents_by_url** - Get all chunks for a specific URL
5. **get_documents_by_source** - Get documents from a specific source
6. **ask_question_about_content** - Ask questions with AI-powered answers
7. **search_by_title_or_summary** - Text-based search in titles/summaries

## Sources Available
{', '.join(stats.get('sources', [])) if stats.get('sources') else 'No sources available'}

Ready to help you explore and query your knowledge base!
"""
    return info


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
