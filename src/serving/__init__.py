from src.base_logger import get_logger
from src.serving.batch.anthropic_serving import ANTHROPIC_MODELS, AnthropicServing
from src.serving.batch.base_batch_serving import BaseBatchServing as BaseBatchServing
from src.serving.batch.openai_serving import OPENAI_MODELS, OpenAIServing
from src.serving.batch.vertexai_serving import VERTEXAI_MODELS, VertexAIServing
from src.serving.local.base_serving import BaseServing as BaseServing
from src.serving.local.litellm_serving import LiteLLMServing
from src.serving.local.local_openai_serving import LocalOpenAIServing
from src.serving.local.metricx_serving import MetricXServing
from src.serving.local.openclip_serving import OpenClipServing
from src.serving.local.vllm_serving import VLLMServing

logger = get_logger(__name__)

# model_types
MODEL_TYPE_SERVING_MAP = {
    "vllm": "local_serving",
    "metricx": "local_serving",
    "openclip": "local_serving",
    "local_openai": "local_serving",
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
    batch_api_calls: bool = False,
    **model_args,
):
    """Get the appropriate serving class based on the model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        BaseServing: The appropriate serving class.
    """
    assert model_type.lower() in [
        "litellm",
        "vllm",
        "local_openai",
        "openai",
        "vertexai",
        "anthropic",
        "metricx",
        "openclip",
        "none",
    ], (
        f"""model_type should be one of ["litellm", "vllm", "local_openai", "openai", "vertexai", "anthropic", "metricx", "openclip"]. Received {model_type} instead."""
    )
    if model_type.lower() == "litellm":
        # typical model args: "api_provider=openai,base_url=http://localhost:8000/v1,api_key=token-abc123"
        logger.info(
            f"Initializing model {model_name} using {model_args['api_provider'].upper()}..."
        )
        api_provider = model_args.pop("api_provider")
        llm = LiteLLMServing(
            model_name=f"{api_provider}/{model_name}",
            **model_args,
        )
    elif model_type.lower() == "local_openai":
        llm = LocalOpenAIServing(
            model_name=model_name,
            is_base_model=is_base_model,
            **model_args,
        )
    elif model_type.lower() == "openai":
        assert model_name in OPENAI_MODELS, f"Unsupported OpenAI model: {model_name}"
        assert batch_api_calls, "Only batch API calls are supported for OpenAI models."
        llm = OpenAIServing(
            model_name=model_name,
            is_base_model=is_base_model,
            **model_args,
        )
    elif model_type.lower() == "vertexai":
        assert model_name in VERTEXAI_MODELS, (
            f"Unsupported Vertex AI model: {model_name}"
        )
        assert batch_api_calls, (
            "Only batch API calls are supported for Vertex AI models."
        )
        llm = VertexAIServing(
            model_name=model_name,
            is_base_model=is_base_model,
            **model_args,
        )
    elif model_type.lower() == "anthropic":
        assert model_name in ANTHROPIC_MODELS, (
            f"Unsupported Anthropic model: {model_name}"
        )
        assert batch_api_calls, (
            "Only batch API calls are supported for Anthropic models."
        )
        llm = AnthropicServing(
            model_name=model_name,
            is_base_model=is_base_model,
            **model_args,
        )
    elif model_type.lower() == "vllm":
        # typical model args: "dtype=bfloat16,enable_prefix_caching=True,gpu_memory_utilization=0.95,tensor_parallel_size=1"
        logger.info(f"Initializing model {model_name} using vLLMs...")
        llm = VLLMServing(
            model_name=model_name,
            is_base_model=is_base_model,
            seed=seed,
            tokenizer_mode="mistral" if model_name.startswith("mistralai") else "auto",
            load_format="mistral" if model_name.startswith("mistralai") else "auto",
            config_format="mistral" if model_name.startswith("mistralai") else "auto",
            **model_args,
        )
    elif model_type.lower() == "metricx":
        logger.info("Initializing MetricX model using Transformers...")
        llm = MetricXServing(model_name=model_name, **model_args)
    elif model_type.lower() == "openclip":
        logger.info(f"Initializing model {model_name} using OpenCLIP...")
        llm = OpenClipServing(model_name=model_name, **model_args)
    elif model_type.lower() == "none":
        logger.info(
            "Model type is set to None. Please ensure that the model inferences are in the correct folder and format."
        )
        llm = None

    return llm
