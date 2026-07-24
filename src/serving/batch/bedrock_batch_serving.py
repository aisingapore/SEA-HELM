import io
import os
import re
import time
from datetime import datetime
from importlib.metadata import version

import boto3
import pandas as pd

from src.base_logger import get_logger
from src.serving.batch.base_batch_serving import BaseBatchServing

logger = get_logger(__name__)

# Generation kwargs that map directly onto the Converse ``inferenceConfig`` block.
# Converse normalises sampling parameters across providers, so unlike the legacy
# ``InvokeModel`` batch path there is no need to dispatch on the model provider.
# Mirrors ``_INFERENCE_CONFIG_MAP`` in the online (real-time) BedrockServing.
_INFERENCE_CONFIG_MAP = {
    "max_tokens": "maxTokens",
    "max_completion_tokens": "maxTokens",
    "temperature": "temperature",
    "top_p": "topP",
    "stop": "stopSequences",
    "stop_sequences": "stopSequences",
}


class BedrockBatchServing(BaseBatchServing):
    """
    A batch serving class that uses Amazon Bedrock for language model completions.

    This class extends :class:`BaseBatchServing` to provide Bedrock-specific
    functionality using the Bedrock batch inference API
    (``CreateModelInvocationJob``) with the **Converse** invocation type. It
    mirrors the other batch serving classes so that the methods used by the
    inference strategies (``prepare_batches``/``create_request``/
    ``batch_generate``/``get_response``/``get_ids_from_batch``/
    ``get_valid_ids_from_batch``/``parse_output``) are interchangeable.

    Bedrock batch inference accepts the provider-agnostic Converse request/
    response format (``modelInvocationType="Converse"``), which normalises the
    schema across providers (Anthropic Claude, Amazon Nova, Meta Llama, Mistral,
    ...). Each batch record's ``modelInput`` is therefore a Converse request body
    (``messages``/``system``/``inferenceConfig``/``additionalModelRequestFields``,
    without ``modelId`` which is set at the job level), and each ``modelOutput``
    is a Converse response body. This is the same request format used by the
    online (real-time) :class:`BedrockServing`, so prompt construction is shared
    between the two paths.

    Constraints:
        - **Model ID**: Newer Claude 4.x models require a cross-region inference
          profile (e.g. ``global.anthropic.claude-haiku-4-5-20251001-v1:0``).
          Bare foundation model IDs are rejected by ``CreateModelInvocationJob``.
        - **Batch size**: Bedrock enforces a minimum of 100 records per batch job.
          Submitting fewer records raises a ``ValidationException``.
        - **Region**: The model's inference profile region and the S3 bucket must
          match. ``global.*`` profiles are compatible with ``ap-southeast-1`` buckets.
        - **S3 cleanup**: Input and output artifacts are deleted from S3 after each
          job via :meth:`delete_s3_outputs`, including on failure (``try/finally``).

    Class Attributes:
        _available_models (list[str] | None): Cached list of Bedrock model and
            inference profile ids, shared across instances (and with the online
            ``BedrockServing``) so repeated instantiations do not each trigger a
            control-plane network call. ``None`` until first fetched.
    """

    _available_models: list[str] | None = None

    @classmethod
    def _get_available_models(cls, aws_profile: str | None = None) -> list[str]:
        """Fetch (and cache) the list of available Bedrock model and inference profile ids.

        The listing is best-effort: inference profile ids are not always returned
        by the listing APIs, so callers should warn rather than block on a miss.

        Args:
            aws_profile (str, optional): Named AWS profile for the boto3 session.
                Falls back to ``AWS_PROFILE``. Only consulted on the first call,
                as the result is cached at the class level.

        Returns:
            list[str]: The available model and inference profile ids, or an empty
                list if the listing failed.
        """
        if cls._available_models is None:
            try:
                session = boto3.Session(
                    profile_name=aws_profile or os.environ.get("AWS_PROFILE")
                )
                bedrock_client = session.client("bedrock")
                foundation_models = [
                    model["modelId"]
                    for model in bedrock_client.list_foundation_models().get(
                        "modelSummaries", []
                    )
                ]
                try:
                    inference_profiles = [
                        profile["inferenceProfileId"]
                        for profile in bedrock_client.list_inference_profiles().get(
                            "inferenceProfileSummaries", []
                        )
                    ]
                except Exception:
                    inference_profiles = []
                cls._available_models = sorted(
                    set(foundation_models) | set(inference_profiles)
                )
                logger.info(
                    "Available Bedrock models: %s",
                    ", ".join(cls._available_models)
                    if cls._available_models
                    else "None",
                )
            except Exception as e:
                cls._available_models = []
                logger.warning(
                    "Unable to list Bedrock models (%s). Please check your AWS credentials and region.",
                    e,
                )
        return cls._available_models

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,  # unused; present for interface compatibility
        api_key: str | None = None,  # unused; AWS credentials come from the environment
        is_base_model: bool = False,
        region: str | None = None,
        aws_profile: str | None = None,
        default_reasoning_effort: str | None = "medium",
        additional_model_request_fields: dict | None = None,
        **kwargs,
    ):
        """
        Initialize the BedrockBatchServing instance.

        S3 input/output location and the IAM execution role required by the
        batch API are read from environment variables, mirroring how
        ``VertexAIServing`` reads ``GCS_BUCKET_NAME``:

        - ``BEDROCK_S3_BUCKET``: bucket used for batch input/output files.
        - ``BEDROCK_BATCH_ROLE_ARN``: IAM role ARN assumed by the batch job.
        - ``AWS_REGION`` / ``AWS_DEFAULT_REGION``: region for the Bedrock client.
        - ``BEDROCK_S3_BUCKET_OWNER`` (optional): expected S3 bucket owner account id.

        Args:
            model_name (str): The Bedrock model id or cross-region inference profile
                id to use (e.g. ``global.anthropic.claude-haiku-4-5-20251001-v1:0``
                or ``anthropic.claude-3-5-sonnet-20240620-v1:0``). Newer Claude 4.x
                models require a cross-region inference profile (``global.*`` or
                ``apac.*``). The model location and S3 bucket must be in the same
                region; ``global.*`` profiles work with ``ap-southeast-1`` buckets.
                Batch inference requires a minimum of 100 records per job.
            base_url (str, optional): Unused. Present for interface compatibility.
            api_key (str, optional): Unused. AWS credentials are resolved by boto3.
            is_base_model (bool, optional): Whether this is a base model. Stored
                for interface compatibility. Defaults to False.
            region (str, optional): AWS region for the Bedrock control-plane and S3
                clients. Falls back to ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` when
                not set. Defaults to None.
            aws_profile (str, optional): Named AWS profile to use. Falls back to
                ``AWS_PROFILE`` when not set. Defaults to None.
            default_reasoning_effort (str, optional): Injects adaptive extended
                thinking into every batch record by setting
                ``additionalModelRequestFields`` to
                ``{"thinking": {"type": "adaptive"}, "output_config": {"effort": <value>}}``.
                Valid values: ``"low"``, ``"medium"``, ``"high"``, ``"max"``
                (``"max"`` is Opus 4.8 only). Only supported by adaptive-thinking
                models (Claude Sonnet 4.6, Opus 4.8, Fable 5). Set to ``None``
                for non-thinking models (e.g. Haiku 4.5) to avoid a
                ``ValidationException``. Defaults to ``"medium"``.
            additional_model_request_fields (dict, optional): Model-specific fields
                passed verbatim as ``additionalModelRequestFields`` in every batch
                record. For adaptive-thinking models (Sonnet 4.6 / Opus 4.8 / Fable 5)
                use ``{"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}``.
                For older thinking models (Claude 3.5 / 4.5 Sonnet) use
                ``{"thinking": {"type": "enabled", "budget_tokens": 10000}}``;
                ``max_tokens`` must exceed ``budget_tokens``. Defaults to None.
            **kwargs: Additional keyword arguments accepted for interface
                compatibility and ignored.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.is_base_model = is_base_model
        self.default_reasoning_effort = default_reasoning_effort
        self.additional_model_request_fields = additional_model_request_fields or {}
        if self.default_reasoning_effort is not None:
            if self.additional_model_request_fields.get("thinking") is None:
                self.additional_model_request_fields["thinking"] = {"type": "adaptive"}
            self.additional_model_request_fields["output_config"] = {
                "effort": self.default_reasoning_effort
            }

        self.friendly_name = "Bedrock"

        # AWS / S3 configuration — must be assigned before _get_available_models()
        # so that self.aws_profile is available for the boto3 session.
        self.bucket_name = os.environ.get("BEDROCK_S3_BUCKET", "")
        self.role_arn = os.environ.get("BEDROCK_BATCH_ROLE_ARN", "")
        self.s3_bucket_owner = os.environ.get("BEDROCK_S3_BUCKET_OWNER")
        self.region = (
            region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        self.aws_profile = aws_profile or os.environ.get("AWS_PROFILE")

        # Inference profile ids are not always returned by the listing APIs, so
        # warn rather than block to avoid false negatives. The left operand
        # primes the class-level cache with this instance's AWS profile, so the
        # inherited (profile-agnostic) is_model_name_supported reads the same list.
        if self._get_available_models(
            self.aws_profile
        ) and not self.is_model_name_supported(model_name):
            logger.warning(
                "Model %s not found in the listed Bedrock models. Proceeding anyway; "
                "ensure it is a valid model id or inference profile id.",
                model_name,
            )

        # Converse normalises sampling parameters across providers, so a single
        # mapping replaces the per-provider dispatch of the InvokeModel path.
        self.kwargs_map = dict(_INFERENCE_CONFIG_MAP)

        if not self.bucket_name:
            logger.warning(
                "BEDROCK_S3_BUCKET is not set. Batch inference will fail without an S3 bucket."
            )
        if not self.role_arn:
            logger.warning(
                "BEDROCK_BATCH_ROLE_ARN is not set. Batch inference will fail without an IAM role."
            )

        client_kwargs = {"region_name": self.region} if self.region else {}
        session = boto3.Session(profile_name=self.aws_profile)
        self.client = session.client("bedrock", **client_kwargs)
        self.s3 = session.client("s3", **client_kwargs)

        # Terminal states for a Bedrock model invocation job.
        self.terminal_states = {
            "Completed",
            "Failed",
            "Stopped",
            "PartiallyCompleted",
            "Expired",
        }

    def load_model(self) -> None:
        """No-op for Bedrock serving as the model is hosted externally."""
        pass

    def get_run_env(self) -> dict:
        """
        Get the runtime environment information.

        Returns:
            dict: Dictionary containing the boto3 version.
        """
        return {"boto3_version": version("boto3")}

    # ------------------------------------------------------------------ #
    # Request building (Converse format)
    # ------------------------------------------------------------------ #
    def _convert_content(self, content: str | list[dict]) -> list[dict]:
        """Convert OpenAI-style message content into Converse content blocks.

        Unlike the online (real-time) :class:`BedrockServing`, which decodes
        image data to raw ``bytes`` for the boto3 ``converse`` call, the batch
        ``modelInput`` is serialised to a JSONL file, so image data is kept as a
        base64 string (the JSON wire representation of the Converse ``bytes`` blob).

        Args:
            content (str | list[dict]): Either a string (text-only) or a list of
                OpenAI-style content parts (multimodal).

        Returns:
            list[dict]: Converse content blocks (``{"text": ...}`` / ``{"image": ...}``).

        Raises:
            ValueError: If a content part is of an unsupported type, or an image
                is provided as a non-base64 URL.
        """
        if isinstance(content, str):
            return [{"text": content}]
        if isinstance(content, list):
            blocks = []
            for part in content:
                part_type = part.get("type")
                if part_type == "text":
                    blocks.append({"text": part["text"]})
                elif part_type == "image_url":
                    image_url = part["image_url"]["url"]
                    if not image_url.startswith("data:"):
                        raise ValueError(
                            "Bedrock Converse batch requires base64-encoded image "
                            f"data URLs, got a non-data URL: {image_url[:32]}..."
                        )
                    prefix, base64_data = image_url.split(";base64,", 1)
                    media_type = prefix.split("data:", 1)[1]
                    image_format = media_type.split("/")[-1].lower()
                    if image_format == "jpg":
                        image_format = "jpeg"
                    blocks.append(
                        {
                            "image": {
                                "format": image_format,
                                "source": {"bytes": base64_data},
                            }
                        }
                    )
                else:
                    raise ValueError(
                        f"Unsupported content type for Bedrock Converse: {part_type}"
                    )
            return blocks
        raise ValueError(f"Invalid content type: {type(content)}")

    def create_request(
        self, conversation: list[dict], custom_id: str, generation_kwargs: dict
    ) -> dict:
        """Build a single Bedrock batch request record in the Converse format.

        Produces a ``{"recordId": ..., "modelInput": ...}`` record where
        ``modelInput`` is a Converse request body (``messages``/``system``/
        ``inferenceConfig``/``additionalModelRequestFields``). ``modelId`` is not
        included as it is set once at the job level.

        Args:
            conversation (list[dict]): List of message dicts with ``role`` and
                ``content`` keys.
            custom_id (str): Unique identifier stored as the record ``recordId``,
                used to correlate responses.
            generation_kwargs (dict): Generation parameters already mapped through
                ``kwargs_map`` (to Converse ``inferenceConfig`` names) by
                ``prepare_batches``.

        Returns:
            dict: A record with ``recordId`` and ``modelInput`` keys.
        """
        system_blocks = []
        messages = []
        for message in conversation:
            role = message["role"]
            content = message["content"]
            if role == "system":
                if isinstance(content, str):
                    system_blocks.append({"text": content})
                else:
                    system_blocks.extend(self._convert_content(content))
                continue
            messages.append({"role": role, "content": self._convert_content(content)})

        # ``generation_kwargs`` is already mapped to inferenceConfig names; only
        # stopSequences needs normalising to a list.
        inference_config = {k: v for k, v in generation_kwargs.items() if v is not None}
        stop = inference_config.get("stopSequences")
        if isinstance(stop, str):
            inference_config["stopSequences"] = [stop]

        model_input = {"messages": messages}
        if system_blocks:
            model_input["system"] = system_blocks
        if inference_config:
            model_input["inferenceConfig"] = inference_config
        if self.additional_model_request_fields:
            model_input["additionalModelRequestFields"] = (
                self.additional_model_request_fields
            )

        return {"recordId": custom_id, "modelInput": model_input}

    # ------------------------------------------------------------------ #
    # Batch generation
    # ------------------------------------------------------------------ #
    def batch_generate(
        self, file_path: str, output_file_path: str, sleep_time: int = 10
    ) -> list:
        """Generate batch responses using the Bedrock batch inference API.

        Steps:
        1. Upload the batch file to the S3 bucket.
        2. Create a model invocation job and wait for completion.
        3. Download the batch predictions from S3.
        4. Delete the batch artifacts from S3.

        Args:
            file_path (str): The path to the file containing the batch requests.
            output_file_path (str): The path where the batch responses are saved.
            sleep_time (int, optional): Time to wait between status checks in
                seconds. Defaults to 10.

        Returns:
            list: List of dictionaries containing the generated outputs.
        """
        input_s3_uri, output_s3_uri, job_prefix = self.upload_batch_file(file_path)

        try:
            job_arn = self.create_batch(job_prefix, input_s3_uri, output_s3_uri)
            time.sleep(sleep_time)

            logger.info("Waiting for Bedrock batch job to complete...")
            counter = 1
            job = self.client.get_model_invocation_job(jobIdentifier=job_arn)
            while job["status"] not in self.terminal_states:
                time.sleep(sleep_time)
                logger.info("Still waiting (%ds has elapsed)...", counter * sleep_time)
                counter += 1
                job = self.client.get_model_invocation_job(jobIdentifier=job_arn)

            if job["status"] == "Completed":
                logger.info("Bedrock batch job completed.")
            else:
                logger.warning(
                    "Bedrock batch job ended with status %s: %s",
                    job["status"],
                    job.get("message", ""),
                )

            predictions = self.download_batch_predictions(
                job_arn, output_s3_uri, output_file_path
            )
        finally:
            self.delete_s3_outputs(job_prefix)

        return predictions

    def upload_batch_file(self, file_path: str) -> tuple[str, str, str]:
        """Upload the batch file to the S3 bucket.

        Args:
            file_path (str): The path to the local batch file.

        Returns:
            tuple[str, str, str]: The input S3 URI, the output S3 URI prefix, and
                the job prefix (used as the top-level S3 key prefix for cleanup).
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        sanitized_model = re.sub(r"[^a-zA-Z0-9-]", "-", self.model_name)
        job_prefix = f"{sanitized_model}-{timestamp}"
        basename = os.path.basename(file_path)
        filename, ext = os.path.splitext(basename)
        sanitized_basename = re.sub(r"[^a-zA-Z0-9-]", "-", filename) + ext

        input_key = f"seahelm-batches/{job_prefix}/input/{sanitized_basename}"
        self.s3.upload_file(file_path, self.bucket_name, input_key)
        input_s3_uri = f"s3://{self.bucket_name}/{input_key}"
        output_s3_uri = f"s3://{self.bucket_name}/seahelm-batches/{job_prefix}/output/"

        logger.info("Batch file %s uploaded to %s", sanitized_basename, input_s3_uri)
        return input_s3_uri, output_s3_uri, job_prefix

    def create_batch(
        self, job_prefix: str, input_s3_uri: str, output_s3_uri: str
    ) -> str:
        """Create a Bedrock model invocation (batch) job using the Converse format.

        Args:
            job_prefix (str): Unique prefix used to derive the job name.
            input_s3_uri (str): The S3 URI of the uploaded batch file.
            output_s3_uri (str): The S3 URI prefix where outputs are written.

        Returns:
            str: The ARN of the created job.
        """
        # Job names must be alphanumeric/hyphen and at most 63 characters.
        job_name = job_prefix[:63].strip("-")

        input_config = {
            "s3InputDataConfig": {
                "s3InputFormat": "JSONL",
                "s3Uri": input_s3_uri,
            }
        }
        output_config = {"s3OutputDataConfig": {"s3Uri": output_s3_uri}}
        if self.s3_bucket_owner:
            input_config["s3InputDataConfig"]["s3BucketOwner"] = self.s3_bucket_owner
            output_config["s3OutputDataConfig"]["s3BucketOwner"] = self.s3_bucket_owner

        response = self.client.create_model_invocation_job(
            jobName=job_name,
            roleArn=self.role_arn,
            modelId=self.model_name,
            modelInvocationType="Converse",
            inputDataConfig=input_config,
            outputDataConfig=output_config,
        )
        job_arn = response["jobArn"]
        logger.info("Batch job sent via Bedrock batch API: %s", job_arn)
        return job_arn

    def download_batch_predictions(
        self,
        job_arn: str,
        output_s3_uri: str,
        output_file_path: str,
    ) -> list[dict]:
        """Download batch predictions from S3.

        Bedrock writes outputs to ``{output_s3_uri}/{job_id}/{input_name}.out``.

        Args:
            job_arn (str): The ARN of the completed job.
            output_s3_uri (str): The S3 URI prefix where outputs were written.
            output_file_path (str): The local path where predictions are saved.

        Returns:
            list[dict]: The generated outputs.
        """
        job_id = job_arn.split("/")[-1]
        out_bucket, out_prefix = self._parse_s3_uri(output_s3_uri)
        result_prefix = f"{out_prefix}{job_id}/"
        output_key = self._find_output_key(out_bucket, result_prefix)

        obj = self.s3.get_object(Bucket=out_bucket, Key=output_key)
        body = obj["Body"].read().decode("utf-8")
        df = pd.read_json(io.StringIO(body), lines=True)

        df = df.sort_values(by="recordId").reset_index(drop=True)
        df.to_json(output_file_path, orient="records", lines=True, force_ascii=False)
        logger.info("Batch predictions downloaded to %s", output_file_path)

        return df.to_dict("records")

    def _find_output_key(self, bucket: str, prefix: str) -> str:
        """Find the predictions output object key under a job prefix.

        Args:
            bucket (str): The S3 bucket.
            prefix (str): The S3 key prefix to search under.

        Returns:
            str: The key of the predictions ``.out`` file.

        Raises:
            FileNotFoundError: If no predictions file is found.
        """
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith(".out") and not key.endswith("manifest.json.out"):
                    return key
        raise FileNotFoundError(
            f"No predictions output found under s3://{bucket}/{prefix}"
        )

    def delete_s3_outputs(self, job_prefix: str) -> None:
        """Delete all batch artifacts (input and output) from the S3 bucket.

        Args:
            job_prefix (str): The top-level S3 key prefix for the job.
        """
        paginator = self.s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(
            Bucket=self.bucket_name, Prefix=f"seahelm-batches/{job_prefix}/"
        ):
            keys.extend(item["Key"] for item in page.get("Contents", []))

        # delete_objects accepts at most 1000 keys per call.
        for start in range(0, len(keys), 1000):
            batch = keys[start : start + 1000]
            self.s3.delete_objects(
                Bucket=self.bucket_name,
                Delete={"Objects": [{"Key": key} for key in batch]},
            )

        logger.info("All objects under 'seahelm-batches/%s/' deleted", job_prefix)

    @staticmethod
    def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
        """Split an ``s3://bucket/key`` URI into its bucket and key.

        Args:
            s3_uri (str): The S3 URI.

        Returns:
            tuple[str, str]: The bucket name and the key (prefix).
        """
        path = s3_uri.replace("s3://", "", 1)
        bucket, _, key = path.partition("/")
        return bucket, key

    # ------------------------------------------------------------------ #
    # Response parsing (Converse format)
    # ------------------------------------------------------------------ #
    def get_response(self, output: dict) -> str:
        """
        Extract the response text from a Bedrock Converse batch output record.

        Args:
            output (dict): A record from the batch output file, containing
                ``recordId``, ``modelInput`` and ``modelOutput`` (the Converse
                response body). Failed records carry an ``error`` object instead.

        Returns:
            str: The extracted response text, or empty string if extraction fails.
        """
        custom_id = output.get("recordId", "")
        try:
            content_blocks = output["modelOutput"]["output"]["message"]["content"]
            # The text lives in a ``{"text": ...}`` block; skip any reasoning or
            # tool-use blocks that may precede it.
            return next(block["text"] for block in content_blocks if "text" in block)
        except Exception as e:
            logger.warning("No response for %s: %s", custom_id, e)
            return ""

    def get_ids_from_batch(self, batch: dict) -> str:
        """
        Extract the custom id from a batch request or output record.

        Args:
            batch (dict): A batch request or output dictionary.

        Returns:
            str: The record id.
        """
        return batch["recordId"]

    def get_valid_ids_from_batch(self, batch: dict) -> str | None:
        """
        Extract the custom id from a batch output record if it succeeded.

        A successful Converse record carries a ``modelOutput`` object; a failed
        record carries an ``error`` object instead.

        Args:
            batch (dict): A batch output dictionary.

        Returns:
            str | None: The record id if the record has a model output,
                None otherwise.
        """
        if isinstance(batch.get("modelOutput"), dict):
            return self.get_ids_from_batch(batch)
        return None

    def batch_tokenize(self, messages: list[list]) -> list:
        """
        Tokenize multiple messages in batch.

        Args:
            messages (list[list]): List of messages to tokenize.

        Returns:
            list: List of None responses (Bedrock has no tokenizer API).
        """
        # Bedrock does not provide a tokenizer API.
        logger.warning(
            "Bedrock does not provide a tokenizer API. Returning None for all tokenizations."
        )
        return [None for _ in messages]
