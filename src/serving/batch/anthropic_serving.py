import asyncio
import os

import anthropic
import importlib_metadata
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from src.base_logger import get_logger
from src.serving.batch.base_batch_serving import BaseBatchServing

logger = get_logger(__name__)

env_vars = {"ANTHROPIC_API_KEY": False}

for key in env_vars:
    if os.environ.get(key):
        logger.info(f"Using {key} provided.")
        env_vars[key] = True
    else:
        logger.warning(
            f"{key} not provided. Please set your {key} environment variable."
        )

if env_vars["ANTHROPIC_API_KEY"]:
    ANTHROPIC_MODELS = [model.id for model in anthropic.Anthropic().models.list()]
    logger.warning(
        "Available Anthropic models: %s",
        ", ".join(ANTHROPIC_MODELS) if ANTHROPIC_MODELS else "None",
    )
else:
    ANTHROPIC_MODELS = []
    logger.warning(
        "No Anthropic models found. Please check your ANTHROPIC_API_KEY environment variable."
    )


class AnthropicServing(BaseBatchServing):
    """
    A serving class that uses Anthropic's API for language model completions.

    This class extends BaseServing to provide Anthropic-specific functionality,
    including batch processing capabilities and response handling for Anthropic models.

    Attributes:
        client (anthropic.Anthropic): The Anthropic API client.
        kwargs_map (List[str]): List of allowed generation parameters.
        processing_states (List[str]): List of valid processing states for batch jobs.
        result_types (List[str]): List of valid result types for batch responses.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        is_base_model: bool = False,
        num_retries: int = 5,
    ) -> None:
        """
        Initialize the AnthropicServing instance.

        Args:
            model (str): The Anthropic model identifier to use for completions.
            base_url (str, optional): The base URL for the API endpoint. Defaults to None.
            api_key (str, optional): The API key for authentication. Defaults to None.
            is_base_model (bool, optional): Whether this is a base model that requires special
                chat template handling. Defaults to False.
            num_retries (int, optional): Number of retries for failed requests. Defaults to 5.

        Raises:
            AssertionError: If the specified model is not available in the Anthropic API.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.is_base_model = is_base_model
        self.num_retries = num_retries

        self.client = (
            anthropic.Anthropic()
        )  # API key is loaded from environment variable ANTHROPIC_API_KEY

        assert self.model_name in ANTHROPIC_MODELS, (
            f"Model {self.model_name} is not available in Anthropic API. Available models: {ANTHROPIC_MODELS}"
        )

        self.kwargs_map = ["temperature", "max_tokens", "top_p", "top_k"]
        self.processing_states = ["in_progress", "canceling", "ended"]
        self.result_types = ["succeeded", "errored", "cancelled", "expired"]

    def load_model(self) -> None:
        """No-op for Anthropic serving as model is hosted externally."""
        pass

    def get_run_env(self) -> dict:
        """
        Get the runtime environment information.

        Returns:
            dict: Dictionary containing the Anthropic API version.
        """
        return {"anthropic_version": importlib_metadata.version("anthropic")}

    def _convert_content_to_anthropic_format(self, content):
        """
        Convert content from OpenAI format to Anthropic format for multimodal support.

        Args:
            content: Either a string (text-only) or list of content parts (multimodal)

        Returns:
            Properly formatted content for Anthropic's API
        """
        if isinstance(content, str):
            # Text-only content - wrap in Anthropic's text format
            return [{"type": "text", "text": content}]
        elif isinstance(content, list):
            # Multimodal content - convert each part
            parts = []
            for content_part in content:
                if content_part.get("type") == "text":
                    parts.append({"type": "text", "text": content_part["text"]})
                elif content_part.get("type") == "image_url":
                    # Convert from OpenAI format to Anthropic format
                    image_url = content_part["image_url"]["url"]

                    if image_url.startswith("data:"):
                        # Parse data URL: data:image/jpeg;base64,<data>
                        prefix, base64_data = image_url.split(";base64,")
                        media_type = prefix.split("data:")[1]  # Remove "data:" prefix
                        parts.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_data,
                                },
                            }
                        )
                    else:
                        # Handle regular URLs
                        parts.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": image_url},
                            }
                        )
                elif content_part.get("type") == "audio_url":
                    raise ValueError(
                        "Anthropic API does not support audio content types."
                    )
                else:
                    raise ValueError(
                        f"Invalid content type: {content_part.get('type')}"
                    )
            return parts
        else:
            # Fallback for unexpected content types
            raise ValueError(f"Invalid content type: {type(content)}")

    def prepare_llm_batches(
        self,
        llm_batch_file_path: str,
        conversations: list,
        custom_ids: list | None = None,
        **generation_kwargs,
    ) -> None:
        """
        Prepare batch requests for Anthropic's batch API and save to file.

        Args:
            llm_batch_file_path (str): Path where the batch request file will be saved.
            conversations (list): List of conversations, where each conversation is a list
                of message dictionaries with 'role' and 'content' keys.
            custom_ids (list, optional): List of custom identifiers for each request.
                If None, uses sequential integer strings. Defaults to None.
            **generation_kwargs: Additional generation parameters to include in requests.
        """
        requests = []
        idx = os.path.splitext(os.path.split(llm_batch_file_path)[-1])[0]

        # Anthropic API only allows certain generation kwargs
        selected_generation_kwargs = {}
        for k, v in generation_kwargs.items():
            if k in self.kwargs_map:
                selected_generation_kwargs[k] = v
            else:
                logger.warning(f"Unsupported generation kwarg for Anthropic API: {k}")

        for ix, conversation in enumerate(conversations):
            messages = []
            system_message = None
            for message in conversation:
                if message["role"] == "system":
                    system_message = message["content"]
                    continue

                messages.append(
                    {
                        "role": message["role"],
                        "content": self._convert_content_to_anthropic_format(
                            message["content"]
                        ),
                    }
                )

            if system_message is not None:
                selected_generation_kwargs["system"] = system_message

            request = Request(
                custom_id=custom_ids[ix] if custom_ids else f"{idx}_{ix}",
                params=MessageCreateParamsNonStreaming(
                    model=self.model_name,
                    messages=messages,
                    **selected_generation_kwargs,
                ),
            )
            requests.append({"request": request})

        df = pd.DataFrame(requests)
        df.to_json(llm_batch_file_path, orient="records", lines=True, force_ascii=False)

    async def abatch_generate(
        self,
        file_path: str,
        output_file_path: str,
        sleep_time: int = 10,
    ) -> list:
        """
        Generate batch responses using Anthropic's batch API.

        Args:
            file_path (str): The path to the file containing the batch requests.
            output_file_path (str): The path to the file where the batch responses will be saved.
            sleep_time (int, optional): Time to wait between status checks in seconds.
                Defaults to 10.

        Returns:
            list: List of dictionaries containing custom_id and response data.
        """
        requests_df = pd.read_json(file_path, lines=True)

        message_batch = self.client.messages.batches.create(
            requests=requests_df["request"].tolist(),
        )

        await asyncio.sleep(sleep_time)

        # Refresh job status until it is completed
        logger.info("Waiting for Anthropic batch job to complete...")
        counter = 1

        while message_batch.processing_status != "ended":
            await asyncio.sleep(sleep_time)
            logger.info(
                "Still waiting (%ds has elapsed)...",
                counter * sleep_time,
            )
            counter += 1
            message_batch = self.client.messages.batches.retrieve(
                message_batch.id,
            )

        logger.info("Anthropic batch job completed.")

        responses = self.client.messages.batches.results(
            message_batch.id,
        )
        predictions = [
            {"custom_id": response.custom_id, "response": response.result}
            for response in responses
        ]
        predictions = pd.DataFrame(predictions)
        predictions["custom_id"] = predictions["custom_id"].astype(int)
        predictions = predictions.sort_values("custom_id").reset_index(drop=True)
        predictions.to_json(
            output_file_path, orient="records", lines=True, force_ascii=False
        )
        logger.info("Batch responses saved to %s", output_file_path)
        predictions = predictions.to_dict("records")

        return predictions

    def get_response(self, output: dict) -> str:
        """
        Extract the response text from Anthropic's output format.

        Args:
            output (dict): The output dictionary from Anthropic's API response.

        Returns:
            str: The extracted response text, or empty string if extraction fails.
        """
        try:
            return output["response"]["message"]["content"][0]["text"]
        except Exception as e:
            logger.exception("Error while extracting response: %s", e)
            return ""

    def get_ids_from_batch(self, batch: dict) -> str:
        """
        Extract custom IDs from batch outputs.

        Args:
            batch (dict): The batch output dictionary.
        Returns:
            string: Comma-separated string of custom IDs.
        """
        return batch["request"]["custom_id"]

    def batch_tokenize(self, messages: list[list]) -> list[dict]:
        """
        Tokenize multiple messages in batch.

        Args:
            messages (list[list]): List of messages to tokenize.

        Returns:
            list[dict]: List of None responses.
        """
        # Anthropic does not provide a tokenizer API
        logger.warning(
            "Anthropic does not provide a tokenizer API. Returning None for all tokenizations."
        )
        batch_response = [None for _ in messages]

        return batch_response


if __name__ == "__main__":
    client = anthropic.Anthropic()

    message_batch = client.messages.batches.create(
        requests=[
            Request(
                custom_id="my-first-request",
                params=MessageCreateParamsNonStreaming(
                    model="claude-opus-4-20250514",
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": "Hello, world",
                        }
                    ],
                ),
            ),
            Request(
                custom_id="my-second-request",
                params=MessageCreateParamsNonStreaming(
                    model="claude-opus-4-20250514",
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": "Hi again, friend",
                        }
                    ],
                ),
            ),
        ]
    )
