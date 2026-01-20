import asyncio
import os
from datetime import datetime

import fsspec
import importlib_metadata
import pandas as pd
from google import genai
from google.cloud import storage
from google.genai.types import BatchJob, CreateBatchJobConfig, JobState

from src.base_logger import get_logger
from src.serving.batch.base_batch_serving import BaseBatchServing

logger = get_logger(__name__)

env_vars = [
    "GCS_BUCKET_NAME",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
]

# Ensure GOOGLE_GENAI_USE_VERTEXAI is set to true
if os.getenv("GOOGLE_GENAI_USE_VERTEXAI") != "true":
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

has_google_credentials = False
for key in env_vars:
    if os.environ.get(key):
        logger.info(f"Using {key} provided.")
        has_google_credentials = True
    else:
        logger.warning(
            f"{key} not provided. Please set your {key} environment variable."
        )

if has_google_credentials:
    try:
        client = genai.Client(vertexai=True)
        VERTEXAI_MODELS = [
            m.name.split("/")[-1] for m in client.models.list() if "gemini" in m.name
        ]
        if not VERTEXAI_MODELS:
            logger.warning("No Gemini models found.")
        else:
            logger.warning("Gemini models found from Vertex AI.")
    except Exception as e:
        logger.exception(e)
        logger.warning(
            "Unable to get list of Gemini models from Vertex AI. Please check your Google credentials."
        )
        VERTEXAI_MODELS = []
else:
    logger.warning(
        "Google credentials not found. Please set either one of the environment variables: "
        + ", ".join(env_vars)
    )
    VERTEXAI_MODELS = []


class VertexAIServing(BaseBatchServing):
    """
    A serving class that uses Vertex AI for language model completions.

    This class provides methods for generating responses from language models using the Vertex AI API.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        is_base_model: bool = False,
        num_retries: int = 5,
    ) -> None:
        """Initialize the VertexAIServing instance.

        Args:
            model (str): The model identifier to use for completions.
            base_url (str, optional): The base URL for the API endpoint. Defaults to None.
            api_key (str, optional): The API key for authentication. Defaults to None.
            is_base_model (bool, optional): Whether this is a base model that requires special
                chat template handling. Defaults to False.
            num_retries (int, optional): Number of retries for failed requests. Defaults to 5.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.is_base_model = is_base_model

        assert model_name in VERTEXAI_MODELS, f"Invalid Gemini model name: {model_name}"
        self.num_retries = num_retries

        self.vertex_project = os.environ.get("VERTEXAI_PROJECT", "")
        self.vertex_location = os.environ.get("VERTEXAI_LOCATION", "us-central1")
        self.client = genai.Client(
            vertexai=True,
            project=self.vertex_project,
            location=self.vertex_location,
        )

        self.bucket_name = os.environ.get("GCS_BUCKET_NAME", "")
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

        self.kwargs_map = {
            "max_tokens": "maxOutputTokens",
            "temperature": "temperature",
            "seed": "seed",
            "top_p": "topP",
            "top_k": "topK",
        }

        self.batch_job_states = [
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_PAUSED,
        ]

    def load_model(self) -> None:
        """No-op for Vertex AI serving as model is hosted externally."""
        pass

    def get_run_env(self) -> dict:
        """
        Get the runtime environment information.

        Returns:
            dict: Dictionary containing the Google GEN AI version.
        """
        return {"google-genai": importlib_metadata.version("google-genai")}

    def prepare_llm_batches(
        self,
        llm_batch_file_path: str,
        conversations: list,
        custom_ids: list | None = None,
        **generation_kwargs,
    ) -> None:
        """Prepare the batch file for Vertex AI

        Args:
            llm_batch_file_path (str): The path to the batch file.
            conversations (list): The conversations to prepare.
            custom_ids (list, optional): The custom ids to use. Defaults to None.
            **generation_kwargs: The generation kwargs to use.
        """
        batches = []
        id = os.path.splitext(os.path.split(llm_batch_file_path)[-1])[0]

        for i, convo in enumerate(conversations):
            contents = []
            system_instructions = None
            for turns in convo:
                if turns["role"] == "system":
                    system_instructions = turns["content"]
                    continue

                # Handle multimodal content (list) vs text-only content (string)
                if isinstance(turns["content"], list):
                    # Multimodal content - convert from OpenAI format to Vertex AI format
                    parts = []
                    for content_part in turns["content"]:
                        if content_part.get("type") == "text":
                            parts.append({"text": content_part["text"]})
                        elif content_part.get("type") == "image_url":
                            # Extract base64 data and mime type from data URL
                            image_url = content_part["image_url"]["url"]
                            if image_url.startswith("data:"):
                                # Parse data URL: data:image/jpeg;base64,<data>
                                mime_type, base64_data = image_url.split(";base64,")
                                mime_type = mime_type.split("data:")[
                                    1
                                ]  # Remove "data:" prefix
                                parts.append(
                                    {
                                        "inlineData": {
                                            "data": base64_data,
                                            "mimeType": mime_type,
                                        }
                                    }
                                )
                            else:
                                raise ValueError(f"Invalid image URL: {image_url}")
                        else:
                            raise ValueError(
                                f"Invalid content type: {content_part.get('type')}"
                            )
                    contents.append(
                        {
                            "role": turns["role"],
                            "parts": parts,
                        }
                    )
                else:
                    # Text-only content - use existing format
                    contents.append(
                        {
                            "role": turns["role"],
                            "parts": [
                                {
                                    "text": turns["content"],
                                }
                            ],
                        }
                    )
            # Vertex AI batch API requires a specific format for the generation kwargs
            kwargs = {
                self.kwargs_map[k]: v
                for k, v in generation_kwargs.items()
                if k in self.kwargs_map
            }

            custom_id = f"{id}_{i}" if custom_ids is None else custom_ids[i]
            request = {
                "request": {
                    "contents": contents,
                    "generationConfig": kwargs,
                    "labels": {
                        "custom_id": custom_id,
                    },
                },
            }

            if system_instructions:
                request["request"]["system_instruction"] = {
                    "parts": [{"text": system_instructions}]
                }

            batches.append(request)
        df = pd.DataFrame(batches)
        df.to_json(llm_batch_file_path, orient="records", lines=True, force_ascii=False)

    async def abatch_generate(
        self,
        file_path: str,
        output_file_path: str,
        sleep_time: int = 10,
    ) -> list:
        """Generate batch responses using Vertex AI batch API

        Steps:
        1. Upload batch file to GCS bucket
        2. Create batch prediction job, wait for completion
        3. Download the batch predictions
        4. Delete the batch predictions from GCS bucket

        Args:
            file_path (str): The path to the file containing the batch requests.
            output_file_path (str): The path to the file where the batch responses will be saved.
            sleep_time (int, optional): The time to wait between status checks in seconds. Defaults to 10.

        Returns:
            list: The generated outputs.
        """
        # Upload batch file
        batch_outputs_path, batch_file_uri, batch_outputs_dir = self.upload_batch_file(
            file_path,
        )

        # Create batch prediction job
        batch_job = self.create_batch(batch_outputs_path, batch_file_uri)
        await asyncio.sleep(sleep_time)

        # Refresh the batch job until complete
        logger.info("Waiting for VertexAI batch to complete...")
        counter = 1
        while batch_job.state not in self.batch_job_states:
            await asyncio.sleep(sleep_time)
            logger.info(
                "Still waiting (%ds has elapsed)...",
                counter * sleep_time,
            )
            counter += 1
            batch_job = self.client.batches.get(name=batch_job.name)

        # Check if the job succeeds
        if batch_job.state == JobState.JOB_STATE_SUCCEEDED:
            logger.info("VertexAI batch is completed")
        else:
            logger.warning(f"Job failed: {batch_job.error}")

        # Download batch predictions
        predictions = self.download_batch_predictions(
            batch_job,
            output_file_path,
        )

        # Delete batch predictions
        self.delete_gcs_outputs(batch_outputs_dir)

        return predictions

    def upload_batch_file(self, file_path: str) -> tuple[str, str, str]:
        """Upload the batch file to GCS bucket

        Args:
            file_path (str): The path to the batch file.

        Returns:
            str: The GCS path to the batch outputs directory.
            str: The GCS URI of the uploaded batch file.
            str: The batch outputs directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        batch_outputs_dir = f"{self.model_name}-{timestamp}"
        batch_outputs_path = f"gs://{self.bucket_name}/{batch_outputs_dir}"
        batch_file_name = os.path.basename(file_path)

        blob_path = f"{batch_outputs_dir}/{batch_file_name}"
        blob = self.bucket.blob(blob_path)
        blob.upload_from_filename(file_path, if_generation_match=0)
        logger.info(f"Batch file {batch_file_name} uploaded to {blob_path}")
        batch_file_uri = f"gs://{self.bucket_name}/{blob_path}"
        return batch_outputs_path, batch_file_uri, batch_outputs_dir

    def create_batch(
        self,
        batch_outputs_path: str,
        batch_file_uri: str,
    ) -> BatchJob:
        """
        Create batch prediction job

        Args:
            batch_outputs_path (str): The path to the batch output file.
            batch_file_uri (str): The URI of the batch file.

        Returns:
            BatchJob: The batch job.
        """
        batch_job = self.client.batches.create(
            model=self.model_name,
            src=batch_file_uri,
            config=CreateBatchJobConfig(
                dest=batch_outputs_path,
            ),
        )
        logger.info("Batch file sent via VertexAI batch API")

        return batch_job

    def download_batch_predictions(
        self,
        batch_job: str,
        batch_output_filepath: str,
    ) -> list[dict]:
        """
        Download batch predictions

        Args:
            batch_job (str): The batch job.
            batch_output_filepath (str): The path to the batch output file.

        Returns:
            list[dict]: The generated outputs.
        """
        fs = fsspec.filesystem("gcs")
        file_paths = fs.glob(f"{batch_job.dest.gcs_uri}/*/predictions.jsonl")
        blob_path = f"gs://{file_paths[0]}"

        df = pd.read_json(blob_path, lines=True)
        df["id"] = df.apply(lambda x: self.get_ids_from_batch(x), axis=1)
        df = df.sort_values(by="id")
        df = df.reset_index(drop=True)
        df.to_json(
            batch_output_filepath, orient="records", lines=True, force_ascii=False
        )
        logger.info(f"Batch predictions downloaded to {batch_output_filepath}")

        predictions = df.to_dict("records")
        return predictions

    def delete_gcs_outputs(
        self,
        batch_outputs_dir: str,
    ) -> None:
        """
        Deletes batch predictions from GCS bucket

        Args:
            batch_outputs_dir (str): The directory to delete the batch predictions from.
        """
        blobs = self.bucket.list_blobs(prefix=batch_outputs_dir)

        # Delete each object
        for blob in blobs:
            blob.delete()

        logger.info(f"All objects in '{batch_outputs_dir}' deleted")

    def get_response(self, output: dict) -> str:
        """
        Gets the response from the output

        Args:
            output (dict): The output to get the response from.

        Returns:
            str: The response from the output.
        """
        custom_id = output["request"]["labels"].get("custom_id", "")
        try:
            if output["response"].get("candidates") is not None:
                if output["response"]["candidates"][0].get("content") is not None:
                    response = output["response"]["candidates"][0]["content"]["parts"][
                        0
                    ]["text"]
                else:
                    finishReason = output["response"]["candidates"][0].get(
                        "finishReason", ""
                    )
                    logger.warning(
                        f"No response for {custom_id} because of finishReason: {finishReason}"
                    )
                    response = ""
            elif output["response"].get("promptFeedback") is not None:
                blockReason = output["response"]["promptFeedback"].get(
                    "blockReason", ""
                )
                logger.warning(
                    f"No response for {custom_id} because of blockReason: {blockReason}"
                )
                response = ""
            else:
                logger.warning(f"No response for {custom_id}")
                response = ""
        except Exception:
            logger.warning(f"No response for {custom_id}")
            response = ""
        return response

    def get_ids_from_batch(self, batch: dict) -> str:
        """
        Extract custom IDs from batch outputs.

        Args:
            batch (dict): The batch output dictionary.
        Returns:
            string: Comma-separated string of custom IDs.
        """
        return batch["request"]["labels"]["custom_id"]

    def batch_tokenize(self, messages: list[list]) -> list[dict]:
        """
        Tokenize multiple messages in batch.

        Args:
            messages (list[list]): List of messages to tokenize.

        Returns:
            list[dict]: List of None responses.
        """
        # VertexAI does not provide a tokenizer API
        logger.warning(
            "VertexAI does not provide a tokenizer API. Returning None for all tokenizations."
        )
        batch_response = [None for _ in messages]

        return batch_response


if __name__ == "__main__":
    vertexai_model = VertexAIServing("gemini-2.0-flash-001")
    messages = [
        {
            "role": "system",
            "content": """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. The language used by the AI assistants should also be the same as that used in the user question. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.""",
        },
        {
            "role": "user",
            "content": """[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]""".format(
                question="Bayangkan Anda sedang menulis postingan blog untuk membandingkan dua model smartphone populer. Buatlah kerangka untuk postingan blog tersebut, termasuk poin-poin kunci dan subjudul untuk membandingkan dan membedakan fitur, kinerja, dan pengalaman pengguna dari kedua model tersebut. Harap jawab dalam kurang dari 200 kata.",
                answer_a="##  [Nama Smartphone 1] vs [Nama Smartphone 2]: Pertempuran Smartphone Kelas Atas\n\n**Pendahuluan:**\n\n*  Perkenalan singkat tentang kedua smartphone dan target pasar mereka.\n*  Menjelaskan tujuan postingan blog: membandingkan dan membedakan kedua model.\n\n**Desain dan Layar:**\n\n*  Membandingkan ukuran, berat, material, dan estetika desain.\n*  Membahas jenis layar (AMOLED, LCD), resolusi, refresh rate, dan kualitas visual.\n\n**Performa dan Kamera:**\n\n*  Membandingkan prosesor, RAM, penyimpanan, dan benchmark kinerja.\n*  Membahas spesifikasi kamera (resolusi, aperture, fitur), kualitas foto dan video di berbagai kondisi.\n\n**Fitur dan Software:**\n\n*  Membandingkan sistem operasi, antarmuka pengguna, fitur unik (misalnya, NFC, pengenalan wajah), dan dukungan software.\n\n**Baterai dan Ketahanan:**\n\n*  Membandingkan kapasitas baterai, pengisian daya cepat, dan waktu penggunaan.\n*  Membahas ketahanan air dan debu (jika ada).\n\n**Harga dan Kesimpulan:**\n\n*  Membandingkan harga kedua smartphone.\n*  Menyimpulkan smartphone mana yang lebih unggul berdasarkan kebutuhan dan preferensi pengguna.",
                answer_b="Judul: Perbandingan Antara Smartphone Model A dan Model B\n\nI. Pendahuluan\n- Pengenalan tentang kedua model smartphone yang akan dibandingkan\n- Tujuan dari perbandingan ini\n\nII. Desain\n- Material dan desain fisik dari masing-masing model\n- Ukuran layar dan resolusi yang dimiliki\n- Bobot dan ketebalan smartphone\n\nIII. Fitur\n- Spesifikasi kamera, termasuk resolusi dan fitur tambahan\n- Kapasitas baterai dan teknologi pengisian daya\n- Keamanan dan privasi, seperti sensor sidik jari atau pengenalan wajah\n\nIV. Kinerja\n- Prosesor dan RAM yang digunakan\n- Kapasitas penyimpanan internal dan kemampuan ekspansi\n- Performa dalam penggunaan sehari-hari dan multitasking\n\nV. Pengalaman Pengguna\n- Antarmuka pengguna yang digunakan\n- Kualitas suara dan fitur multimedia\n- Ketersediaan update sistem operasi dan dukungan purna jual\n\nVI. Kesimpulan\n- Ringkasan perbandingan antara kedua model smartphone\n- Rekomendasi untuk konsumen berdasarkan kebutuhan dan preferensi mereka",
            ),
        },
    ]

    response = vertexai_model.generate(messages)
    print(response.choices[0].message.content)
