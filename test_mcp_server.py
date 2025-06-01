#!/usr/bin/env python3
"""
Test script for the VectorX MCP Server

This script demonstrates how to test the MCP server's functionality
without running it in MCP mode.
"""

import sys
import os
import asyncio

# Add the current directory to Python path for imports
sys.path.insert(0, '.')

from vectorx_mcp_server import (
    get_database_stats,
    list_all_urls,
    search_documents,
    get_documents_by_url,
    ask_question_about_content,
    search_by_title_or_summary
)

async def test_mcp_server():
    """Test all MCP server functions."""
    print("=" * 60)
    print("🧪 Testing VectorX MCP Server Functions")
    print("=" * 60)
    
    # Test 1: Database Stats
    print("\n1️⃣ Testing get_database_stats()...")
    try:
        stats = get_database_stats()
        print(f"✅ Database Status: {stats['status']}")
        print(f"📊 Total Documents: {stats['total_documents']}")
        print(f"🌐 Unique URLs: {stats['unique_urls']}")
        print(f"📚 Sources: {', '.join(stats['sources'])}")
        print(f"🕒 Last Updated: {stats['last_updated']}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: List URLs
    print("\n2️⃣ Testing list_all_urls()...")
    try:
        urls_result = list_all_urls()
        print(f"✅ Status: {urls_result['status']}")
        print(f"🔗 Total URLs: {urls_result['total_urls']}")
        print(f"📁 Sources: {urls_result['total_sources']}")
        for source, urls in list(urls_result['urls_by_source'].items())[:2]:
            print(f"   📂 {source}: {len(urls)} URLs")
            for url in urls[:3]:  # Show first 3 URLs
                print(f"      - {url}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Semantic Search
    print("\n3️⃣ Testing search_documents()...")
    try:
        search_query = "API authentication and security"
        search_results = await search_documents(search_query, max_results=3)
        print(f"✅ Search Status: {search_results['status']}")
        print(f"🔍 Query: '{search_query}'")
        print(f"📄 Results Found: {search_results['results_count']}")
        
        for i, result in enumerate(search_results['results'][:2], 1):
            print(f"   🎯 Result {i}:")
            print(f"      📝 Title: {result['title']}")
            print(f"      🌐 URL: {result['url']}")
            print(f"      📋 Summary: {result['summary'][:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Get specific URL content
    print("\n4️⃣ Testing get_documents_by_url()...")
    try:
        # Get a URL from the database first
        urls_result = list_all_urls()
        if urls_result['urls_by_source']:
            # Get first URL from first source
            first_source = list(urls_result['urls_by_source'].keys())[0]
            test_url = urls_result['urls_by_source'][first_source][0]
            
            url_docs = get_documents_by_url(test_url)
            print(f"✅ Status: {url_docs['status']}")
            print(f"🔗 URL: {test_url}")
            print(f"📑 Total Chunks: {url_docs['total_chunks']}")
            
            if url_docs['chunks']:
                chunk = url_docs['chunks'][0]
                print(f"   📝 First Chunk Title: {chunk['title']}")
                print(f"   📊 Chunk Size: {chunk['chunk_size']} characters")
        else:
            print("⚠️ No URLs found in database")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: AI-powered Q&A
    print("\n5️⃣ Testing ask_question_about_content()...")
    try:
        question = "What are the main features and capabilities mentioned in the documentation?"
        qa_result = await ask_question_about_content(question, max_context_chunks=2)
        print(f"✅ Status: {qa_result['status']}")
        print(f"❓ Question: {question}")
        print(f"🤖 Answer: {qa_result['answer'][:200]}...")
        print(f"📚 Context Chunks Used: {qa_result['total_context_chunks']}")
        print(f"🔗 Sources: {', '.join(qa_result['sources'][:3])}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 6: Text-based search
    print("\n6️⃣ Testing search_by_title_or_summary()...")
    try:
        search_term = "API"
        text_search = search_by_title_or_summary(search_term, limit=3)
        print(f"✅ Status: {text_search['status']}")
        print(f"🔍 Search Term: '{search_term}'")
        print(f"📄 Matches Found: {text_search['total_matches']}")
        
        for i, match in enumerate(text_search['matches'][:2], 1):
            print(f"   🎯 Match {i}:")
            print(f"      📝 Title: {match['title']}")
            print(f"      🌐 URL: {match['url']}")
            print(f"      ✅ In Title: {match['match_in_title']}")
            print(f"      ✅ In Summary: {match['match_in_summary']}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 MCP Server Testing Complete!")
    print("=" * 60)
    print("\n💡 Next Steps:")
    print("1. Run 'python vectorx_mcp_server.py' to start the MCP server")
    print("2. Use MCP Inspector: 'npx @modelcontextprotocol/inspector python vectorx_mcp_server.py'")
    print("3. Configure Claude Desktop or other MCP clients with your server")
    print("4. Test with your actual AI assistant!")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
