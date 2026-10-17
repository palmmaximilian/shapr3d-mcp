"""Long tool calls hand back a job id instead of hitting the client timeout.

Runs over the real stdio transport with a tiny JOB_WAIT_S so a script that
sleeps briefly is enough to force the background path.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

WORKDIR = Path(os.environ["SHAPR3D_MCP_WORKDIR"])


def _payload(result) -> dict:
    if result.structuredContent is not None:
        sc = result.structuredContent
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    return json.loads(result.content[0].text)


def test_slow_tool_returns_job_then_wait_job_collects_result():
    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-c", "from shapr3d_mcp import main; main()"],
            env={**os.environ, "SHAPR3D_MCP_WORKDIR": str(WORKDIR),
                 "SHAPR3D_MCP_JOB_WAIT_S": "0.5"},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name: t for t in (await session.list_tools()).tools}
                assert {"wait_job", "run_model_file", "detect_clash"} <= set(tools)
                # heavy tools may answer with a job handle whatever they
                # normally return: no output schema to validate against
                assert tools["open_in_shapr3d"].outputSchema is None
                assert tools["render_preview"].outputSchema is None

                slow = "import time\ntime.sleep(2)\nresult = Box(10, 10, 10)"
                res = await session.call_tool("create_model", {"name": "job_box", "script": slow})
                assert not res.isError
                handle = _payload(res)
                assert handle["status"] == "running"

                for _ in range(20):
                    res = await session.call_tool("wait_job", {"job_id": handle["job_id"]})
                    assert not res.isError
                    out = _payload(res)
                    if out.get("status") != "running":
                        break
                assert out["stats"]["volume_mm3"] == 1000

                # errors inside a backgrounded job surface on wait_job
                bad = "import time\ntime.sleep(1)\nraise ValueError('boom')"
                res = await session.call_tool("create_model", {"name": "job_bad", "script": bad})
                job_id = _payload(res)["job_id"]
                for _ in range(20):
                    res = await session.call_tool("wait_job", {"job_id": job_id})
                    if res.isError or _payload(res).get("status") != "running":
                        break
                assert res.isError and "boom" in res.content[0].text

                # fast calls still answer inline
                res = await session.call_tool(
                    "create_model", {"name": "job_fast", "script": "result = Box(1, 1, 1)"})
                assert _payload(res)["stats"]["volume_mm3"] == 1

    asyncio.run(run())


def test_bloated_worker_is_recycled_when_idle(monkeypatch):
    import time

    from shapr3d_mcp import server

    monkeypatch.setattr(server, "_JOB_POOL", None)
    monkeypatch.setattr(server, "WORKER_MAX_RSS_MB", 0)  # every worker is "bloated"
    first = server._submit_job(os.getpid).result(timeout=120)
    for _ in range(200):
        if server._JOB_POOL is None:
            break
        time.sleep(0.05)
    assert server._JOB_POOL is None
    second = server._submit_job(os.getpid).result(timeout=120)
    assert second != first

    # under the limit the worker (and its in-memory shape cache) stays
    monkeypatch.setattr(server, "WORKER_MAX_RSS_MB", 1e9)
    pool = server._JOB_POOL
    try:
        assert server._submit_job(os.getpid).result(timeout=120) == second
        time.sleep(0.5)
        assert server._JOB_POOL is pool
        assert server._submit_job(os.getpid).result(timeout=120) == second
    finally:
        pool.shutdown()
