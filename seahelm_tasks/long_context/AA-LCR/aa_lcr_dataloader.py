import os
import unicodedata
import zipfile

from datasets import load_dataset
from huggingface_hub import snapshot_download

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader
from src.dataloaders.judges.judge_dataloader import JudgeDataloader

logger = get_logger(__name__)


class AALCRDataloader(HuggingFaceDataloader, JudgeDataloader):
    """Dataloader for the AA-LCR dataset.

    AA-LCR is a dataset for evaluating language models on long context reasoning tasks.
    This dataloader handles loading data and preparing it for language model inference.
    """

    def load_dataset(self, limit: int = None):
        """Load the AA-LCR dataset from HuggingFace datasets.

        Args:
            limit: Optional limit on the number of instances to load.

        Note:
            The dataset is stored in self.dataset after loading and image decoding.
        """
        if self.dataset:
            logger.info("Dataset already loaded, skipping loading process")
            pass
        else:
            filepath = self.specific_task_config["filepath"]
            logger.info("Drawing and preparing data from %s", filepath)
            cache_path = snapshot_download(
                repo_id=filepath,
                repo_type="dataset",
            )

            # extract contents of the zip file to the same directory
            with zipfile.ZipFile(
                os.path.join(cache_path, "extracted_text", "AA-LCR_extracted-text.zip"),
                "r",
                metadata_encoding="utf-8",
            ) as zip_ref:
                zip_ref.extractall(cache_path)

            dataset = load_dataset(filepath, split="test")

            if limit is not None:
                dataset = dataset.select(range(limit))

            # preload documents into memory
            filenames = set()
            for row in dataset:
                files = [
                    f"{row['document_category']}/{row['document_set_id']}/{filename}"
                    for filename in row["data_source_filenames"].split(";")
                ]
                filenames.update(files)

            # Build NFC-normalized path map from actual files on disk.
            # ZIP entries may use NFD normalization (e.g. decomposed ş) while
            # dataset metadata uses NFC, so we normalize both sides for lookup.
            lcr_dir = os.path.join(cache_path, "lcr")
            nfc_to_path = {}
            for dirpath, _, disk_files in os.walk(lcr_dir):
                for fname in disk_files:
                    full_path = os.path.join(dirpath, fname)
                    rel_path = os.path.relpath(full_path, lcr_dir)
                    nfc_to_path[unicodedata.normalize("NFC", rel_path)] = full_path

            documents = {}
            for filename in filenames:
                disk_path = nfc_to_path[unicodedata.normalize("NFC", filename)]
                with open(disk_path, "r") as f:
                    documents[filename] = f.read()

            def map_columns(row, index):
                filenames = row["data_source_filenames"].split(";")
                docs = []
                for filename in filenames:
                    docs.append(
                        documents[
                            f"{row['document_category']}/{row['document_set_id']}/{filename}"
                        ]
                    )

                documents_text = "\n\n".join(
                    f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}"
                    for i, doc in enumerate(docs)
                )

                return {
                    "id": row["question_id"],
                    "prompts": [
                        {"documents_text": documents_text, "question": row["question"]}
                    ],
                    "label": row["answer"],
                    "metadata": {
                        "language": "en",
                        "document_category": row["document_category"],
                        "document_set_id": row["document_set_id"],
                        "data_source_filenames": row["data_source_filenames"],
                        "input_tokens": row["input_tokens"],
                    },
                }

            dataset = dataset.map(
                map_columns,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,
                with_indices=True,
            )

            self.dataset = dataset

    def get_judge_prompt_formatter(self, metric):
        judge_prompt = self.task_config.config.judge["judge_prompts"]

        def _prompt_formatter(row, idx):
            responses = row["responses"]

            conversations = []
            custom_ids = []

            # Generate judgements
            messages = [
                {
                    "role": "user",
                    "content": judge_prompt["criteria_prompt"].format(
                        question=row["prompts"][0]["question"],
                        official_answer=row["label"],
                        candidate_answer=responses[0],
                    ),
                },
            ]
            # Add messages and custom IDs to lists
            conversations.append(messages)
            custom_ids.append(f"{row['id']}")
            return {"conversations": conversations, "custom_ids": custom_ids}

        return _prompt_formatter
