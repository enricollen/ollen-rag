"""In-memory MCP client tests: tools are registered and delegate to the rag layer."""
import pytest
from fastmcp import Client
from src import mcp_server


@pytest.mark.asyncio
async def test_tools_registered():
    async with Client(mcp_server.mcp) as client:
        tools = {t.name for t in await client.list_tools()}
        assert {"ingest_document", "retrieve", "rag_query", "list_indices"} <= tools


@pytest.mark.asyncio
async def test_rag_query_tool(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "generate_answer",
        lambda query, **kwargs: {"answer": "Risposta [1]", "sources": [{"id": 1, "text": "t", "score": 0.9, "metadata": {}}]},
    )
    async with Client(mcp_server.mcp) as client:
        result = await client.call_tool("rag_query", {"query": "test?"})
        assert result.data["answer"] == "Risposta [1]"


@pytest.mark.asyncio
async def test_retrieve_tool(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "retrieve_nodes",
        lambda query, **kwargs: [],
    )
    async with Client(mcp_server.mcp) as client:
        result = await client.call_tool("retrieve", {"query": "test?", "filters": [{"key": "a", "value": 1}]})
        assert result.data == {"nodes": []}


@pytest.mark.asyncio
async def test_list_indices_tool(monkeypatch):
    class _Backend:
        def list_indices(self):
            return [{"index": "ollen_rag_sentence"}]
    monkeypatch.setattr(mcp_server, "create_backend", lambda settings=None: _Backend())
    async with Client(mcp_server.mcp) as client:
        result = await client.call_tool("list_indices", {})
        assert result.data["indices"][0]["index"] == "ollen_rag_sentence"
