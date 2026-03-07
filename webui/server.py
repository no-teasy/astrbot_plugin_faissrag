"""
FAISSRAG WebUI Server
Provides web interface for memory management
"""

import asyncio
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


class FAISSRAGWebUIServer:
    """FAISSRAG Web UI Server"""

    def __init__(
        self,
        plugin_instance,
        port: int = 0,
        host: str = "127.0.0.1",
        api_key: str = "",
    ):
        self.plugin = plugin_instance
        self.port = port
        self.host = host
        self.api_key = api_key
        self.server = None
        self.thread: Optional[threading.Thread] = None
        self.url = ""

        self.app = FastAPI(
            title="FAISSRAG Admin Panel",
            description="FAISSRAG Plugin Web Management Panel",
            version="1.2.0",
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Import and setup routes
        from .routes import setup_routes
        setup_routes(self.app, self._get_plugin)

    def _get_plugin(self):
        """Get plugin instance for routes"""
        return self.plugin

    def run_in_thread(self):
        """Run server in background thread"""
        if self.port == 0:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.host, 0))
                actual_port = s.getsockname()[1]
                self.port = actual_port

        self.server = uvicorn.Server(
            uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="warning",
            )
        )

        self.url = f"http://{self.host}:{self.port}"

        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()

    def start(self):
        """Start the server"""
        self.run_in_thread()

    def stop(self):
        """Stop the server"""
        if self.server:
            self.server.should_exit = True
        if self.thread:
            self.thread.join(timeout=5)