# crawler.py
import os
import json
import asyncio
import sys
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

# Fix for Windows asyncio subprocess issues
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError:
        pass

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI

from db import init_collection

load_dotenv()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize VectorX collection lazily to avoid issues during import
vectorx_collection = None

def get_vectorx_collection():
    global vectorx_collection
    if vectorx_collection is None:
        vectorx_collection = init_collection()
    return vectorx_collection


@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        code_block = chunk.rfind("```")
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif "\n\n" in chunk:
            last_break = chunk.rfind("\n\n")
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif ". " in chunk:
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end)

    return chunks


async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from web content chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4-0125-preview"),
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}...",
                },
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {
            "title": "Error processing title",
            "summary": "Error processing summary",
        }


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536


async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    extracted = await get_title_and_summary(chunk, url)
    embedding = await get_embedding(chunk)

    metadata = {
        "source": urlparse(url).netloc,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path,
    }

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted["title"],
        summary=extracted["summary"],
        content=chunk,
        metadata=metadata,
        embedding=embedding,
    )


async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into VectorX."""
    try:
        collection = get_vectorx_collection()
        collection.add(
            documents=[chunk.content],
            embeddings=[chunk.embedding],
            metadatas=[
                {
                    "url": chunk.url,
                    "chunk_number": chunk.chunk_number,
                    "title": chunk.title,
                    "summary": chunk.summary,
                    **chunk.metadata,
                }
            ],
            ids=[f"{chunk.url}_{chunk.chunk_number}"],
        )
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
    except Exception as e:
        print(f"Error inserting chunk: {e}")


async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    chunks = chunk_text(markdown)
    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    insert_tasks = [insert_chunk(chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_tasks)


async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    # Print number of URLs found
    print(f"Found {len(urls)} URLs to crawl")

    # Reduced concurrency for Windows stability
    if max_concurrent > 3:
        max_concurrent = 3
        print("Reduced max_concurrent to 3 for Windows stability")

    # Windows-optimized browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        browser_type="chromium",
        extra_args=[
            "--no-sandbox",
            "--disable-dev-shm-usage", 
            "--disable-gpu",
            "--disable-extensions",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-features=TranslateUI",
            "--disable-ipc-flooding-protection",
            "--max_old_space_size=4096"
        ],
    )
    
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        process_iframes=False,
        remove_overlay_elements=True,
        page_timeout=30000,  # 30 second timeout
        delay_before_return_html=2.0  # Wait 2 seconds for page to stabilize
    )

    # Add retry logic with individual crawler instances
    async def crawl_with_retry(url: str, max_retries: int = 3):
        """Crawl a single URL with retry logic using individual crawler instances."""
        for attempt in range(max_retries):
            crawler = None
            try:
                # Create a new crawler instance for each attempt
                crawler = AsyncWebCrawler(config=browser_config)
                await crawler.start()
                
                # Use unique session ID for each URL to avoid conflicts
                session_id = f"session_{hash(url) % 10000}"
                
                result = await crawler.arun(
                    url=url, 
                    config=crawl_config, 
                    session_id=session_id
                )
                
                if result.success and result.markdown and result.markdown.raw_markdown:
                    return result.markdown.raw_markdown, None
                else:
                    error_msg = result.error_message or "No content extracted"
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed for {url}: {error_msg}, retrying...")
                        await asyncio.sleep(1)  # Brief delay before retry
                    else:
                        return None, error_msg
                        
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for {url}: {error_msg}, retrying...")
                    await asyncio.sleep(1)
                else:
                    return None, error_msg
            finally:
                if crawler:
                    try:
                        await crawler.close()
                    except Exception as e:
                        print(f"Error closing crawler: {e}")
                        
        return None, "All retry attempts failed"

    # Process URLs with controlled concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    total_urls = len(urls)
    processed_urls = 0
    failed_urls = []

    async def process_url(url: str):
        nonlocal processed_urls
        async with semaphore:
            try:
                markdown, error = await crawl_with_retry(url)
                if markdown:
                    processed_urls += 1
                    print(f"✓ Successfully crawled: {url} ({processed_urls}/{total_urls})")
                    await process_and_store_document(url, markdown)
                else:
                    failed_urls.append((url, error))
                    print(f"✗ Failed: {url} - Error: {error}")
            except Exception as e:
                failed_urls.append((url, str(e)))
                print(f"✗ Exception processing {url}: {e}")

    # Process all URLs
    await asyncio.gather(*[process_url(url) for url in urls], return_exceptions=True)
    
    print(f"\nCrawling Summary:")
    print(f"Successfully crawled: {processed_urls}/{total_urls} URLs")
    print(f"Failed: {len(failed_urls)} URLs")
    
    if failed_urls:
        print(f"\nFailed URLs:")
        for url, error in failed_urls[:10]:  # Show first 10 failures
            print(f"  {url}: {error}")
        if len(failed_urls) > 10:
            print(f"  ... and {len(failed_urls) - 10} more failures")


def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """Get URLs from a sitemap."""
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()

        root = ElementTree.fromstring(response.content)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [loc.text for loc in root.findall(".//ns:loc", namespace)]

        print(f"Found {len(urls)} URLs in sitemap: {sitemap_url}")
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []


async def main():
    """Main function for testing the crawler."""
    # Test with a few URLs first
    test_urls = [
        "https://docs.crawl4ai.com/",
        "https://docs.crawl4ai.com/basic-usage/",
        "https://docs.crawl4ai.com/installation/"
    ]
    print("Testing crawler with sample URLs...")
    await crawl_parallel(test_urls, max_concurrent=2)


if __name__ == "__main__":
    asyncio.run(main())
