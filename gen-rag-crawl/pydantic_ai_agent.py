from __future__ import annotations

from dataclasses import dataclass
from dotenv import load_dotenv
from litellm import AsyncOpenAI
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from typing import List
from db import VectorXCollection

load_dotenv()

llm = os.getenv("LLM_MODEL", "gpt-4-0125-preview")
model = OpenAIModel(llm)

logfire.configure(send_to_logfire="if-token-present")



@dataclass
class PydanticAIDeps:
    collection: VectorXCollection
    openai_client: AsyncOpenAI


system_prompt = """
You are an expert assistant with access to a knowledge base of documentation and content.
Your job is to help users understand and work with the content they've provided.

IMPORTANT: You MUST use the retrieve_relevant_documentation tool for EVERY user question to search the knowledge base before answering. Do NOT provide answers from general knowledge without first searching the stored content.

Your workflow for every question:
1. ALWAYS call retrieve_relevant_documentation with the user's question
2. Analyze the retrieved content
3. Provide an answer based ONLY on the retrieved content
4. If no relevant content is found, clearly state that the information is not in the knowledge base

When analyzing the retrieved content:
1. Provide accurate information based ONLY on the stored content
2. Cite specific examples and sources from the retrieved documents
3. Be clear when you're making assumptions or if information is incomplete
4. Never supplement with external knowledge unless explicitly requested
"""

pydantic_ai_agent = Agent(
    model, system_prompt=system_prompt, deps_type=PydanticAIDeps, retries=2
)


async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536


@pydantic_ai_agent.tool
async def retrieve_relevant_documentation(
    ctx: RunContext[PydanticAIDeps], user_query: str
) -> str:
    """
    ALWAYS call this tool first for ANY user question to search the knowledge base.
    This tool retrieves relevant documentation chunks based on the user's query.
    
    Args:
        user_query: The user's question or topic to search for in the knowledge base
        
    Returns:
        Relevant documentation content that should be used to answer the user's question
    """
    try:
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

        results = ctx.deps.collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas"],
        )

        if not results["documents"][0]:
            return "No relevant documentation found."

        formatted_chunks = []
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            title = metadata.get('title', 'Unknown Title')
            url = metadata.get('url', 'Unknown Source')
            chunk_text = f"""
# {title}

{doc}

Source: {url}
"""
            formatted_chunks.append(chunk_text)

        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


@pydantic_ai_agent.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """Retrieve a list of all available documentation pages."""
    try:
        results = ctx.deps.collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return []

        urls = sorted(set(meta.get("url", "Unknown URL") for meta in results["metadatas"] if meta and meta.get("url")))
        return urls

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []


@pydantic_ai_agent.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """Retrieve the full content of a specific documentation page."""
    try:
        results = ctx.deps.collection.get(
            where={"url": url}, include=["documents", "metadatas"]
        )

        if not results["documents"]:
            return f"No content found for URL: {url}"

        sorted_results = sorted(
            zip(results["documents"], results["metadatas"]),
            key=lambda x: x[1].get("chunk_number", 0),
        )

        page_title = sorted_results[0][1].get("title", "Documentation Page").split(" - ")[0]
        formatted_content = [f"# {page_title}\n"]

        for doc, _ in sorted_results:
            formatted_content.append(doc)

        return "\n\n".join(formatted_content)

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
