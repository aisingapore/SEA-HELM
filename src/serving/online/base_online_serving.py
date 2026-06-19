import socket
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod

import requests
from openai import AsyncOpenAI

from src.base_logger import get_logger

logger = get_logger(__name__)


class BaseOnlineServing(ABC):
    """Abstract base class for online (server-based) model serving.

    Subclasses must implement `_server_name` and `_build_server_command` to
    define the backend-specific server name and launch command respectively.
    All common lifecycle logic (port discovery, health polling, shutdown) is
    provided here.
    """

    def __init__(
        self,
        model_name: str,
        is_base_model: bool = False,
        api_key: str = "",
        max_workers: int = 1024,
        timeout: int = 3600,
        loading_timeout: int = 3600,
        **kwargs,
    ):
        self.model_name = model_name
        self.is_base_model = is_base_model
        self.api_key = api_key
        self.server_kwargs = kwargs
        self.additional_generation_kwargs = {}
        self.process: subprocess.Popen | None = None
        self.port = self.find_open_port()
        self.base_url = f"http://localhost:{self.port}/v1"
        self.timeout = timeout
        self.loading_timeout = loading_timeout
        self.max_workers = max_workers

    @property
    @abstractmethod
    def _server_name(self) -> str:
        """Human-readable name of the backend server (e.g. 'vLLM', 'SGLang')."""

    @abstractmethod
    def _build_server_command(self, merged_kwargs: dict) -> list[str]:
        """Build the shell command list used to launch the server subprocess.

        Args:
            merged_kwargs: Combined kwargs from construction time and load_model call.

        Returns:
            list[str]: The argv list passed to subprocess.Popen.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

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

    def load_model(self) -> None:
        """Start the server subprocess and block until it is healthy.

        Args:
            **kwargs: Additional CLI arguments that override those supplied at
                construction time.

        Raises:
            TimeoutError: If the server does not become healthy within
                ``self.loading_timeout`` seconds.
            RuntimeError: If the server process exits before becoming healthy.
        """
        if self.process is not None:
            logger.info(
                "%s server is already running on port %d", self._server_name, self.port
            )
            return

        cmd = self._build_server_command(self.server_kwargs)
        if cmd is not None:
            logger.info("Starting %s server: %s", self._server_name, " ".join(cmd))
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            for stream, dest in (
                (self.process.stdout, sys.stdout),
                (self.process.stderr, sys.stderr),
            ):
                t = threading.Thread(
                    target=self._pipe_stream, args=(stream, dest), daemon=True
                )
                t.start()

            start_time = time.time()
            while not self.is_model_loaded():
                if time.time() - start_time > self.loading_timeout:
                    self.clean_up()
                    raise TimeoutError(
                        f"{self._server_name} server did not become healthy within "
                        f"{self.loading_timeout} seconds"
                    )
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "%s server process exited unexpectedly (see logs above for details)",
                        self._server_name,
                    )
                time.sleep(5)

            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,  # disable retries in favour of custom retry logic in inference strategy
            )
            logger.info("%s server is ready on port %d", self._server_name, self.port)

    def is_model_loaded(self) -> bool:
        """Check whether the server is running and healthy.

        Returns:
            bool: True if the /health endpoint returns HTTP 200, False otherwise.
        """
        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def find_open_port(self) -> int:
        """Find an open port on the local machine.

        Returns:
            int: An available TCP port number.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def empty_output_dict(self, custom_id: str | None = None) -> dict:
        """Return an empty output dict with the same structure as parse_output.

        Args:
            custom_id (str, optional): The custom ID to include in the output dict. Defaults to None.

        Returns:
            dict: An empty output dict.
        """
        return {
            "finish_reasons": None,
            "responses": None,
            "reasoning_contents": None,
            "custom_ids": custom_id,
            "token_usages": None,
            "function_calls": None,
            "tool_calls": None,
            "logprobs": None,
            "errors": "No response generated due to skipped inference.",
        }

    def cleanup(self) -> None:
        """Shut down the server subprocess and release resources."""
        if self.process is not None:
            logger.info("Shutting down %s server...", self._server_name)
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "%s server did not terminate gracefully; sending SIGKILL...",
                    self._server_name,
                )
                self.process.kill()
                self.process.wait()
            finally:
                self.process = None
            logger.info("%s server has been shut down.", self._server_name)
