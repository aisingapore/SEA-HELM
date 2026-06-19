from src.inference_strategy.base_model_inference_strategy import (
    BaseModelInferenceStrategy,
)
from src.inference_strategy.batched_inference_strategy import BatchedInferenceStrategy
from src.inference_strategy.default_inference_strategy import DefaultInferenceStrategy
from src.inference_strategy.logprobs_inference_strategy import LogprobsInferenceStrategy
from src.serving.batch.base_batch_serving import BaseBatchServing as BaseBatchServing
from src.serving.offline.base_offline_serving import (
    BaseOfflineServing as BaseOfflineServing,
)

INFERENCE_SERVING_MAP = {
    "base_model": BaseModelInferenceStrategy,
    "batched": BatchedInferenceStrategy,
    "default": DefaultInferenceStrategy,
    "logprobs": LogprobsInferenceStrategy,
}


def get_inference_strategy_class(
    model,
    strategy_name: str | None = None,
    is_base_model: bool = False,
    is_logprobs: bool = False,
):
    """Get the inference strategy class based on the strategy name.

    Args:
        strategy_name (str): The name of the inference strategy.

    Returns:
        The inference strategy class corresponding to the given strategy name.
    """
    if strategy_name is None or strategy_name.lower() == "auto":
        if isinstance(model, BaseBatchServing):
            return BatchedInferenceStrategy
        elif is_logprobs:
            return LogprobsInferenceStrategy
        elif is_base_model:
            return BaseModelInferenceStrategy
        else:
            return DefaultInferenceStrategy

    strategy_name = strategy_name.lower()
    if strategy_name in INFERENCE_SERVING_MAP:
        return INFERENCE_SERVING_MAP[strategy_name]
    else:
        raise ValueError(
            f"""Inference strategy {strategy_name} not recognized. Supported strategies are: {list(INFERENCE_SERVING_MAP.keys())}."""
        )
