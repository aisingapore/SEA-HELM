import importlib_metadata

from src.base_logger import get_logger
from src.serving.online.base_online_serving import BaseOnlineServing
from src.serving.online.local_openai_serving import LocalOpenAIServing

logger = get_logger(__name__)


class OnlineVLLMServing(BaseOnlineServing, LocalOpenAIServing):
    """Online serving backend that launches a vLLM OpenAI-compatible server."""

    def __init__(self, model_name: str, **kwargs):
        """Initialize OnlineVLLMServing.

        Args:
            model_name (str): Model identifier or path passed to ``vllm serve``.
            **kwargs: Additional keyword arguments forwarded to
                :class:`BaseOnlineServing` and ultimately converted to CLI flags
                for the vLLM server process.
        """
        super().__init__(model_name=model_name, **kwargs)
        self.special_kwargs = ["enable_prefix_caching"]

    def handle_kwargs(self, kwargs) -> dict:
        """Pre-process and normalise kwargs before building the server command.

        Handles thinking-related keys (``enable_thinking``, ``thinking``,
        ``thinking_mode``), injects the base-model chat template when required,
        and resolves ``tensor_parallel_size='auto'`` to the actual CUDA device
        count.

        Args:
            kwargs (dict): Raw keyword arguments to process.

        Returns:
            dict: Processed keyword arguments ready for CLI conversion.
        """
        for thinking_kwarg in ["enable_thinking", "thinking", "thinking_mode"]:
            if thinking_kwarg in kwargs:
                self.additional_generation_kwargs["chat_template_kwargs"] = {
                    thinking_kwarg: kwargs.pop(thinking_kwarg)
                }
                break

        if self.is_base_model and "chat_template" not in kwargs:
            with open("chat_templates/base_model.jinja") as f:
                chat_template = f.read()
            kwargs["chat_template"] = chat_template

        # handle auto tp
        if "tensor_parallel_size" in kwargs:
            if kwargs["tensor_parallel_size"] == "auto":
                import torch

                kwargs["tensor_parallel_size"] = str(torch.cuda.device_count())

        return kwargs

    @property
    def _server_name(self) -> str:
        return "vLLM"

    def _build_server_command(self, kwargs: dict) -> list[str]:
        """Build the shell command used to launch the vLLM server.

        Boolean flags listed in ``self.special_kwargs`` are emitted as bare
        flags (no value argument); all other keys are converted from
        snake_case to ``--kebab-case <value>`` pairs.

        Args:
            kwargs (dict): CLI arguments to append to the base command.

        Returns:
            list[str]: The argv list passed to ``subprocess.Popen``.
        """
        cmd = ["vllm", "serve", self.model_name, "--port", str(self.port)]

        kwargs = self.handle_kwargs(kwargs)
        for key, value in kwargs.items():
            if key in self.special_kwargs:
                cmd.append(f"--{key.replace('_', '-')}")
            else:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        return cmd

    def get_run_env(self) -> dict:
        """Get the runtime environment information.

        Returns:
            dict: Dictionary containing the ``transformers`` and ``vllm`` package versions.
        """
        return {
            "transformers_version": importlib_metadata.version("transformers"),
            "vllm_version": importlib_metadata.version("vllm"),
        }
