"""CLI entry point for the two-model FreeToken arbiter."""

from __future__ import annotations

import argparse

import httpx
import uvicorn

from .app import ArbiterConfig, build_arbiter_app
from .backends import BackendConfig, BackendController


def build_parser(prog: str = "ft arbiter") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Serve Ornith and Gemma behind one endpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--ornith-url", default=BackendConfig.ornith_url)
    parser.add_argument("--gemma-gpu-url", default=BackendConfig.gemma_gpu_url)
    parser.add_argument("--gemma-cpu-url", default=BackendConfig.gemma_cpu_url)
    parser.add_argument("--daemon-url", default=BackendConfig.daemon_url)
    parser.add_argument("--daemon-token", default=None)
    parser.add_argument("--gemma-model-path", default=BackendConfig.gemma_model_path)
    parser.add_argument("--gemma-cpu-unit", default=BackendConfig.gemma_cpu_unit)
    parser.add_argument("--queue-timeout", type=float, default=ArbiterConfig.queue_timeout_s)
    parser.add_argument("--max-queue-depth", type=int, default=ArbiterConfig.max_queue_depth)
    parser.add_argument("--request-timeout", type=float, default=ArbiterConfig.request_timeout_s)
    return parser


def main(argv: list[str] | None = None, *, prog: str = "ft arbiter") -> int:
    args = build_parser(prog).parse_args(argv)
    backend_config = BackendConfig(
        ornith_url=args.ornith_url,
        gemma_gpu_url=args.gemma_gpu_url,
        gemma_cpu_url=args.gemma_cpu_url,
        daemon_url=args.daemon_url,
        daemon_token=args.daemon_token,
        gemma_model_path=args.gemma_model_path,
        gemma_cpu_unit=args.gemma_cpu_unit,
    )
    arbiter_config = ArbiterConfig(
        queue_timeout_s=args.queue_timeout,
        max_queue_depth=args.max_queue_depth,
        request_timeout_s=args.request_timeout,
    )

    async def serve() -> None:
        timeout = httpx.Timeout(arbiter_config.request_timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
            controller = BackendController(backend_config, client)
            app = build_arbiter_app(arbiter_config, controller, client)
            config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()

    import asyncio

    asyncio.run(serve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
