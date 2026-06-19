import importlib_metadata

from src.base_logger import get_logger
from src.serving.online.base_online_serving import BaseOnlineServing
from src.serving.online.local_openai_serving import LocalOpenAIServing

logger = get_logger(__name__)


class OnlineSGLangServing(BaseOnlineServing, LocalOpenAIServing):
    """Online serving backend that launches an SGLang OpenAI-compatible server."""

    @property
    def _server_name(self) -> str:
        return "SGLang"

    def handle_kwargs(self, kwargs) -> dict:
        """Pre-process and normalise kwargs before building the server command.

        Handles chat-template related keys (``enable_thinking``, ``thinking``),
        injects the base-model chat template when required, and resolves
        ``tp='auto'`` to the actual CUDA device count.

        Args:
            kwargs (dict): Raw keyword arguments to process.

        Returns:
            dict: Processed keyword arguments ready for CLI conversion.
        """
        # handle chat templates
        if "enable_thinking" in kwargs:
            self.additional_generation_kwargs["chat_template_kwargs"] = {
                "enable_thinking": kwargs.pop("enable_thinking")
            }
        elif "thinking" in kwargs:
            self.additional_generation_kwargs["chat_template_kwargs"] = {
                "thinking": kwargs.pop("thinking")
            }

        if self.is_base_model and "chat_template" not in kwargs:
            kwargs["chat_template"] = "chat_templates/base_model.jinja"

        # handle auto tp
        if "tp" in kwargs:
            if kwargs["tp"] == "auto":
                import torch

                kwargs["tp"] = str(torch.cuda.device_count())

        return kwargs

    def _build_server_command(self, kwargs: dict) -> list[str]:
        """Build the shell command used to launch the SGLang server.

        Args:
            kwargs (dict): CLI arguments to append to the base command.
                Keys are converted from snake_case to kebab-case flags.

        Returns:
            list[str]: The argv list passed to ``subprocess.Popen``.
        """
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_name,
            "--port",
            str(self.port),
        ]

        kwargs = self.handle_kwargs(kwargs)
        for key, value in kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        return cmd

    def get_run_env(self):
        """Get the runtime environment information.

        Returns:
            dict: Dictionary containing the ``transformers`` and ``sglang`` package versions.
        """
        return {
            "transformers_version": importlib_metadata.version("transformers"),
            "sglang_version": importlib_metadata.version("sglang"),
        }
