import asyncio
import os

import importlib_metadata
import pandas as pd
import tiktoken
from openai import OpenAI

from src.base_logger import get_logger
from src.serving.batch.base_batch_serving import BaseBatchServing

logger = get_logger(__name__)

special_token_map = {
    "gpt-3.5": {
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    },
    "gpt-4": {
        "<|im_start|>": 200264,
        "<|im_sep|>": 200266,
        "<|im_end|>": 200265,
    },
}

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    models = client.models.list()
    OPENAI_GPT_MODELS = {x.id for x in models.data if "gpt" in x.id}

    OPENAI_O1_MODELS = {x.id for x in models.data if "o1" in x.id}
    OPENAI_MODELS = OPENAI_GPT_MODELS.union(OPENAI_O1_MODELS)

    OPENAI_O3_MODELS = {x.id for x in models.data if "o3" in x.id}
    OPENAI_MODELS = OPENAI_MODELS.union(OPENAI_O3_MODELS)
except Exception:
    OPENAI_MODELS = set()
    logger.warning(
        "Unable to get list of OpenAI models. Please check your OpenAI API key."
    )


class OpenAIServing(BaseBatchServing):
    """
    A serving class that uses OpenAI for language model completions.

    This class provides methods for generating responses from language models using the OpenAI API.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        is_base_model: bool = False,
        num_retries: int = 5,
    ) -> None:
        """Initialize the OpenAIServing instance.

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
        self.is_base_model = is_base_model

        assert model_name in OPENAI_MODELS, f"Invalid OpenAI model name: {model_name}"
        self.num_retries = num_retries
        if api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        else:
            self.api_key = api_key

        self.tokenizer = tiktoken.encoding_for_model(self.model_name)

        self.kwargs_map = {
            "max_tokens": "max_completion_tokens",
        }
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def load_model(self):
        """No-op for OpenAI serving as model is hosted externally."""
        pass

    def get_run_env(self) -> dict:
        """
        Get the runtime environment information.

        Returns:
            dict: Dictionary containing the OpenAI API version.
        """
        return {"openai_version": importlib_metadata.version("openai")}

    def apply_chat_template(
        self, message: list, add_generation_prompt: bool = True
    ) -> list:
        """Apply the chat template to a message.

        Args:
            message (list): The message to apply the chat template to.
            add_generation_prompt (bool, optional): Whether to add the generation prompt. Defaults to True.

        Returns:
            list: The message with the chat template applied.
        """
        output = []
        if self.model_name.lower().startswith("gpt-4"):
            # for gpt-4 and gpt-4o
            for m in message:
                output.extend(
                    [
                        "<|im_start|>",
                        m["role"],
                        "<|im_sep|>",
                        m["content"],
                        "<|im_end|>",
                    ]
                )

            if add_generation_prompt:
                output.extend(["<|im_start|>", "assistant", "<|im_sep|>"])
        elif self.model_name.lower().startswith("gpt-3.5"):
            for m in message:
                output.extend(
                    ["<|im_start|>", m["role"] + "\n", m["content"], "<|im_end|>"]
                )

            if add_generation_prompt:
                output.extend(["<|im_start|>", "assistant\n"])
        return output

    def tokenize(self, message: list, add_generation_prompt: bool = True) -> list:
        """Tokenize a message.

        Args:
            message (list): The message to tokenize.
            add_generation_prompt (bool, optional): Whether to add the generation prompt. Defaults to True.

        Note:
            This function skips the tokenization of image/audio data.

        Returns:
            list: The tokenized message.
        """
        chat_input = self.apply_chat_template(
            message, add_generation_prompt=add_generation_prompt
        )

        # hack to solve token
        tokens = []
        for k, v in special_token_map.items():
            if k in self.model_name.lower():
                token_map = v
                break

        for input in chat_input:
            if isinstance(input, list):
                for content in input:
                    if content["type"] == "text":
                        tokens.extend(self.tokenizer.encode(content["text"]))
                    # TODO: handle image/audio tokens
            elif input in token_map.keys():
                tokens.append(token_map[input])
            else:
                tokens.extend(self.tokenizer.encode(input))

        return tokens

    def batch_tokenize(self, messages: list[list]) -> list[dict]:
        """
        Tokenize multiple messages in batch.

        Args:
            messages (list[list]): List of messages to tokenize.

        Returns:
            list[dict]: List of tokenization responses.
        """
        # TODO handle cases when encode does not work
        batch_response = [self.tokenize(message) for message in messages]

        return batch_response

    def prepare_llm_batches(
        self,
        llm_batch_file_path: str,
        conversations: list,
        custom_ids: list | None = None,
        **generation_kwargs,
    ) -> None:
        """Prepare LLM batches.

        Args:
            llm_batch_file_path (str): The path to the LLM batch file.
            conversations (list): List of conversations, where each conversation is a list
                of message dictionaries with 'role' and 'content' keys.
            custom_ids (list, optional): List of custom identifiers for each request.
                If None, uses sequential integer strings. Defaults to None.
            **generation_kwargs: Additional generation parameters to include in requests.
        """
        batches = []
        id = os.path.splitext(os.path.split(llm_batch_file_path)[-1])[0]

        kwargs = {
            (self.kwargs_map[k] if k in self.kwargs_map else k): v
            for k, v in generation_kwargs.items()
        }

        for i, convo in enumerate(conversations):
            # prepare body of batch to send to OpenAI API
            body = kwargs.copy()  # must copy to ensure that the values don't change
            body["model"] = self.model_name
            body["messages"] = convo

            custom_id = f"{id}_{i}" if custom_ids is None else custom_ids[i]

            output = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            batches.append(output)
        df = pd.DataFrame(batches)
        df.to_json(llm_batch_file_path, orient="records", lines=True, force_ascii=False)

    async def abatch_generate(
        self,
        file_path: str,
        output_file_path: str,
        sleep_time: int = 10,
    ) -> None:
        """Generate batch responses asynchronously.

        Args:
            file_path (str): The path to the file containing the batch requests.
            output_file_path (str): The path to the file where the batch responses will be saved.
            sleep_time (int, optional): The time to wait between status checks in seconds. Defaults to 10.
        """
        file_obj = self.client.files.create(
            file=open(file_path, "rb"),
            purpose="batch",
        )

        await asyncio.sleep(sleep_time)
        batch_input_file_id = file_obj.id
        assert batch_input_file_id is not None, (
            "Failed to create input file, expected a non null file_id but got {batch_input_file_id}"
        )

        create_batch_response = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        logger.info("Prompts sent via OpenAI batch API")

        logger.info("Waiting for OpenAI batch to complete...")
        counter = 1
        while True:
            await asyncio.sleep(sleep_time)

            retrieved_batch = self.client.batches.retrieve(
                batch_id=create_batch_response.id,
            )

            if retrieved_batch.output_file_id is not None:
                break
            else:
                logger.info(
                    "Still waiting (%ds has elapsed)...",
                    counter * sleep_time,
                )
                counter += 1

        logger.info("OpenAI batch is completed")

        file_content = self.client.files.content(file_id=retrieved_batch.output_file_id)
        file_content.write_to_file(output_file_path)
        # convert file_content to unicode
        pd.read_json(output_file_path, lines=True).to_json(
            output_file_path, orient="records", lines=True, force_ascii=False
        )

        if retrieved_batch.error_file_id is not None:
            error_file_content = self.client.files.content(
                file_id=retrieved_batch.error_file_id
            )
            logger.warning(
                "Errors occurred during batch generation. Error details:\n%s",
                error_file_content.read(),
            )

        return file_content.read()

    def get_response(self, output: dict) -> str:
        """Get the response from the output.

        Args:
            output (dict): The output to get the response from.

        Returns:
            str: The response from the output.
        """
        return output["response"]["body"]["choices"][0]["message"]["content"]

    def get_ids_from_batch(self, batch: dict) -> str:
        """
        Extract custom IDs from batch outputs.

        Args:
            batch (dict): The batch output dictionary.
        Returns:
            string: Comma-separated string of custom IDs.
        """
        return batch["custom_id"]


if __name__ == "__main__":
    openai_model = OpenAIServing("gpt-4-0613")
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

    response = openai_model.generate(messages)
    print(response.choices[0].message.content)
