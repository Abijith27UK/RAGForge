import json
import asyncio
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

async def test():
    from mcp import StdioServerParameters
    params = StdioServerParameters(
        command=r"C:\Users\abiji\Documents\ID4100\RAGForge\backend\.venv\Scripts\python.exe",
        args=[r"C:\Users\abiji\Documents\ID4100\RAGForge\backend\scripts\run_mcp_server.py"],
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as client:
            await client.initialize()
            
            # List tools
            tools = await client.list_tools()
            print("=== TOOLS ===")
            for t in tools:
                print(f"  Name: {t.name}")
                print(f"  Title: {t.title}")
                if hasattr(t, 'inputSchema'):
                    print(f"  Input schema: {json.dumps(t.inputSchema, indent=2)}")
                print()
            
            # List resources
            resources = await client.list_resources()
            print("=== RESOURCES ===")
            for r in resources:
                print(f"  URI: {r.uri}")
                print(f"  Name: {r.name}")
                print()
            
            # List prompts
            prompts = await client.list_prompts()
            print("=== PROMPTS ===")
            for p in prompts:
                print(f"  Name: {p.name}")
                print()

if __name__ == "__main__":
    asyncio.run(test())