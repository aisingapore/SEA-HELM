from typing import Any

import torch
from transformers import AutoTokenizer

from src.base_logger import get_logger
from src.serving.local.base_serving import BaseServing
from src.serving.local.metricx_models import MT5ForRegression

logger = get_logger(__name__)


class MetricXServing(BaseServing):
    """
    A serving class that uses vLLM for language model completions.

    This class provides methods for generating responses from language models using the vLLM API.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
    ) -> None:
        self.batch_size = batch_size
        self.model_name = model_name
        self.is_model_loaded = False

    def half_batch_size(self) -> None:
        """Halve the batch size for generation."""
        batch_size_before = self.batch_size
        self.batch_size = max(1, self.batch_size // 2)
        logger.info(f"Batch size halved to {self.batch_size}")

        # return whtether the batch size changed
        return self.batch_size != batch_size_before

    def load_model(self) -> None:
        """Load MetricX tokenizer and model onto CUDA and set eval mode.

        The model is large; this should be called right before scoring and the
        allocated memory should be freed promptly after use.
        """
        if self.is_model_loaded:
            # no op as model is loaded
            return
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "google/mt5-xl", use_fast=False
            )
            self.model = MT5ForRegression.from_pretrained(
                self.model_name, torch_dtype="auto"
            )

            self.model = self.model.to("cuda")
            self.model.eval()
            self.is_model_loaded = True

    def generate(
        self, messages: list, logprobs: bool = False, **generation_kwargs
    ) -> Any:
        """Generate a response for a given message.

        Args:
            messages (list): The messages to generate a response for.
            logprobs (bool, optional): Whether to return log probabilities. Defaults to False.
            **generation_kwargs: Additional generation kwargs.

        Returns:
            Any: The generated response.
        """
        # only look at the first message in the conversation
        message = [messages[0]["content"]]

        tokens = self.tokenizer(
            message,
            truncation=True,
            padding=True,
            max_length=1536,
            return_tensors="pt",
        )

        # remove eos token
        tokens["input_ids"] = tokens["input_ids"][:, :-1]
        tokens["attention_mask"] = tokens["attention_mask"][:, :-1]

        # move tokens to cuda device
        tokens["input_ids"] = tokens["input_ids"].to("cuda")
        tokens["attention_mask"] = tokens["attention_mask"].to("cuda")

        with torch.no_grad():
            outputs = self.model(**tokens)

        _scores = outputs.predictions.cpu().tolist()
        return _scores

    def batch_generate(
        self, batch_messages: list[list], logprobs: bool = False, **generation_kwargs
    ) -> list[Any]:
        """Generate responses for a given batch of messages.

        Args:
            batch_messages (list[list]): The batch of messages to generate responses for.
            logprobs (bool, optional): Whether to return log probabilities. Defaults to False.
            **generation_kwargs: Additional generation kwargs.

        Returns:
            list[Any]: The generated responses.
        """
        responses = []
        for i in range(0, len(batch_messages), self.batch_size):
            _batch = batch_messages[i : i + self.batch_size]
            _prompts = []
            for message in _batch:
                # only look at the first message in the conversation
                _prompts.append(message[0]["content"])

            tokens = self.tokenizer(
                _prompts,
                truncation=True,
                padding=True,
                max_length=1536,
                return_tensors="pt",
            )

            # remove eos token
            tokens["input_ids"] = tokens["input_ids"][:, :-1]
            tokens["attention_mask"] = tokens["attention_mask"][:, :-1]

            # move tokens to cuda device
            tokens["input_ids"] = tokens["input_ids"].to("cuda")
            tokens["attention_mask"] = tokens["attention_mask"].to("cuda")

            with torch.no_grad():
                outputs = self.model(**tokens)

            _scores = outputs.predictions.cpu().tolist()

            for _score, _message, _prompt in zip(
                _scores, _batch, _prompts, strict=True
            ):
                responses.append(
                    {
                        "score": _score,
                        "message": _message,
                        "prompt": _prompt,
                    }
                )

        return responses

    def get_response(self, output: dict) -> str:
        """Get the response from the output.

        Args:
            output (dict): The output to get the response from.

        Returns:
            str: The response from the output.
        """
        # no op here, just return the output as is
        return output["score"]

    def cleanup(self) -> None:
        """Cleanup any resources used by the serving class."""
        if not self.is_model_loaded:
            logger.info("Model is not loaded; no cleanup necessary.")
            return

        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        torch.cuda.empty_cache()
