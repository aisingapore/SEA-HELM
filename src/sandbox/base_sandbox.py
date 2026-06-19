import json
import socket
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod

import requests

from src.base_logger import get_logger

logger = get_logger(__name__)


class BaseSandbox(ABC):
    def __init__(self, server_name: str):
        self.process: subprocess.Popen | None = None
        self.port: int = self.find_open_port()
        self.timeout_seconds: int = 600
        self._server_name: str = server_name

    def find_open_port(self) -> int:
        """Find an open port on localhost to launch the server on."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def is_sandbox_loaded(self) -> bool:
        """Check whether the server is running and healthy.

        Returns:
            bool: True if the /v1/ping endpoint returns HTTP 200, False otherwise.
        """
        try:
            response = requests.get(f"http://localhost:{self.port}/v1/ping", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _pipe_stream(self, stream, dest) -> None:
        """Copy lines from *stream* to *dest* (sys.stdout or sys.stderr).

        Intended to run in a daemon thread so it does not block the main
        process.

        Args:
            stream: A readable binary stream (e.g. ``subprocess.Popen.stdout``).
            dest: A writable text stream to write to (``sys.stdout`` or ``sys.stderr``).
        """
        try:
            for line in stream:
                dest.write(line.decode(errors="replace"))
                dest.flush()
        except ValueError:
            pass  # stream closed

    @abstractmethod
    def start_process(self):
        """Start the sandbox server process. Must set self.process to the Popen object."""
        raise NotImplementedError(
            "Subclasses must implement start_process() to launch the sandbox server process"
        )

    def start_sandbox(self):
        if self.process is not None:
            logger.info(
                "%s server is already running on port %d", self._server_name, self.port
            )
            return

        self.start_process()
        for stream, dest in (
            (self.process.stdout, sys.stdout),
            (self.process.stderr, sys.stderr),
        ):
            t = threading.Thread(
                target=self._pipe_stream, args=(stream, dest), daemon=True
            )
            t.start()

        start_time = time.time()
        while not self.is_sandbox_loaded():
            if time.time() - start_time > self.timeout_seconds:
                # self.clean_up()
                raise TimeoutError(
                    f"{self._server_name} server did not become healthy within "
                    f"{self.timeout_seconds} seconds"
                )
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"{self._server_name} server process exited unexpectedly "
                    f"(see logs above for details)"
                )
            time.sleep(5)

    def run_code(
        self,
        code: str,
        language: str = "python",
        compile_timeout: int = 10,
        run_timeout: int = 10,
    ) -> str:
        """Run code in the sandbox and return the output.

        Args:
            code: The code to run in the sandbox.
            language: The programming language of the code.

        Returns:
            The output from running the code in the sandbox.
        """
        if self.process is None:
            raise RuntimeError(f"{self._server_name} server is not running")

        response = requests.post(
            f"http://localhost:{self.port}/run_code",
            json={
                "code": code,
                "language": language,
                "compile_timeout": compile_timeout,
                "run_timeout": run_timeout,
            },
        )
        response.raise_for_status()
        return json.loads(response.text)
