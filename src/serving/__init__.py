from src.base_logger import get_logger
from src.serving.batch.anthropic_serving import ANTHROPIC_MODELS, AnthropicServing
from src.serving.batch.base_batch_serving import BaseBatchServing as BaseBatchServing
from src.serving.batch.openai_serving import (
    OpenAIServing,
    is_openai_model_name_supported,
)
from src.serving.batch.vertexai_serving import VERTEXAI_MODELS, VertexAIServing
from src.serving.offline.base_offline_serving import (
    BaseOfflineServing as BaseOfflineServing,
)
from src.serving.offline.metricx_serving import MetricXServing
from src.serving.offline.openclip_serving import OpenClipServing
from src.serving.offline.vllm_serving import VLLMServing
from src.serving.online.litellm_serving import LiteLLMServing
from src.serving.online.local_openai_serving import LocalOpenAIServing
from src.serving.online.online_sglang_serving import OnlineSGLangServing
from src.serving.online.online_vllm_serving import OnlineVLLMServing

logger = get_logger(__name__)

# model_types
MODEL_TYPE_SERVING_MAP = {
    "vllm": "local_serving",
    "online_vllm": "local_serving",
    "online_sglang": "local_serving",
    "metricx": "local_serving",
    "openclip": "local_serving",
    "local_openai": "remote_serving",
    "openai": "remote_serving",
    "anthropic": "remote_serving",
    "vertexai": "remote_serving",
    "litellm": "remote_serving",
}


def get_serving_type(model_type: str) -> str:
    """Get the serving type based on the model type.

    Args:
        model_type (str): The type of the model.

    Returns:
        str: The serving type ("local" or "batch").
    """
    assert model_type.lower() in MODEL_TYPE_SERVING_MAP, (
        f"""model_type should be one of {list(MODEL_TYPE_SERVING_MAP.keys())}. Received {model_type} instead."""
    )
    return MODEL_TYPE_SERVING_MAP[model_type.lower()]


def is_serving_type_remote(model_type: str) -> bool:
    """Check if the serving type is remote based on the model type.

    Args:
        model_type (str): The type of the model.

    Returns:
        bool: True if the serving type is remote, False otherwise.
    """
    serving_type = get_serving_type(model_type)
    return serving_type == "remote_serving"


def get_serving_class(
    model_name: str,
    model_type: str,
    is_base_model: bool = False,
    seed: int = 42,
    **model_args,
):
    """Get the appropriate serving class based on the model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        BaseServing: The appropriate serving class.
    """
    _model_type = model_type.lower()
    model_types = list(MODEL_TYPE_SERVING_MAP.keys()) + ["none"]
    assert _model_type in model_types, (
        f"model_type should be one of {model_types}. Received {model_type} instead."
    )
    if _model_type == "litellm":
        # typical model args: "api_provider=openai,base_url=http://localhost:8000/v1,api_key=token-abc123"
        logger.info(
            "Initializing model %s using %s...",
            model_name,
            model_args["api_provider"].upper(),
        )
        api_provider = model_args.pop("api_provider")
        serving_class = LiteLLMServing(
            model_name=f"{api_provider}/{model_name}",
            **model_args,
        )
    elif _model_type == "local_openai":
        serving_class = LocalOpenAIServing(
            model_name=model_name,
            is_base_model=is_base_model,
            **model_args,
        )
    elif _model_type == "openai":
        assert is_openai_model_name_supported(model_name), (
            f"Unsupported OpenAI model: {model_name}"
        )
        serving_class = OpenAIServing(
            model_name=model_name,
            is_base_model=is_base_model,
            **model_args,
        )
    elif _model_type == "vertexai":
        assert model_name in VERTEXAI_MODELS, (
            f"Unsupported Vertex AI model: {model_name}"
        )
        serving_class = VertexAIServing(
            model_name=model_name,
            is_base_model=is_base_model,
            **model_args,
        )
    elif _model_type == "anthropic":
        assert model_name in ANTHROPIC_MODELS, (
            f"Unsupported Anthropic model: {model_name}"
        )
        serving_class = AnthropicServing(
            model_name=model_name,
            is_base_model=is_base_model,
            **model_args,
        )
    elif _model_type == "vllm":
        # typical model args: "tensor_parallel_size=1,reasoning_parser=qwen3,enable_thinking=True"
        logger.info("Initializing model %s using vLLMs...", model_name)
        if model_name.startswith("mistralai"):
            model_args["tokenizer_mode"] = model_args.get("tokenizer_mode", "mistral")
            model_args["load_format"] = model_args.get("load_format", "mistral")
            model_args["config_format"] = model_args.get("config_format", "mistral")
        serving_class = VLLMServing(
            model_name=model_name,
            is_base_model=is_base_model,
            seed=seed,
            **model_args,
        )
    elif _model_type == "online_vllm":
        # typical model args: "tensor_parallel_size=1,reasoning_parser=qwen3,enable_thinking=True"
        logger.info("Initializing model %s using online vLLM serving...", model_name)
        if model_name.startswith("mistralai"):
            model_args["tokenizer_mode"] = model_args.get("tokenizer_mode", "mistral")
            model_args["load_format"] = model_args.get("load_format", "mistral")
            model_args["config_format"] = model_args.get("config_format", "mistral")
        serving_class = OnlineVLLMServing(
            model_name=model_name,
            is_base_model=is_base_model,
            seed=seed,
            **model_args,
        )
    elif _model_type == "online_sglang":
        # typical model args: "tp=1,mem_fraction_static=0.9"
        logger.info("Initializing model %s using online SGLang serving...", model_name)
        serving_class = OnlineSGLangServing(
            model_name=model_name,
            is_base_model=is_base_model,
            seed=seed,
            **model_args,
        )
    elif _model_type == "metricx":
        logger.info("Initializing MetricX model using Transformers...")
        serving_class = MetricXServing(model_name=model_name, **model_args)
    elif _model_type == "openclip":
        logger.info("Initializing model %s using OpenCLIP...", model_name)
        serving_class = OpenClipServing(model_name=model_name, **model_args)
    elif _model_type == "none":
        logger.info(
            "Model type is set to None. Please ensure that the model inferences are in the correct folder and format."
        )
        serving_class = None

    return serving_class
