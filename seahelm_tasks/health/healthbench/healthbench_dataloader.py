import os
from typing import Any

from datasets import load_dataset
from huggingface_hub import snapshot_download

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader
from src.dataloaders.judges.judge_dataloader import JudgeDataloader

logger = get_logger(__name__)


class HealthBenchDataloader(HuggingFaceDataloader, JudgeDataloader):
    """Dataloader for the HealthBench dataset.

    HealthBench is a dataset for evaluating language models on health-related tasks.
    This dataloader handles loading data and preparing it for language model inference.
    """

    def load_dataset(self, limit: int = None):
        """Load the HealthBench dataset from HuggingFace datasets.

        Args:
            limit: Optional limit on the number of instances to load.

        Note:
            The dataset is stored in self.dataset after loading and image decoding.
        """
        filenames = {
            "healthbench": "2025-05-07-06-14-12_oss_eval.jsonl",
            "healthbench_consensus": "consensus_2025-05-09-20-00-46.jsonl",
            "healthbench_hard": "hard_2025-05-08-21-00-10.jsonl",
        }
        if self.dataset:
            logger.info("Dataset already loaded, skipping loading process")
            pass
        else:
            filepath = self.specific_task_config["filepath"]
            logger.info("Drawing and preparing data from %s", filepath)
            cache_path = snapshot_download(filepath, repo_type="dataset")
            _path = os.path.join(cache_path, filenames[self.task_name])

            dataset = load_dataset("json", split="train", data_files=_path)

            if limit is not None:
                dataset = dataset.select(range(limit))

            # Map dataset fields to standard fields used in SEA-HELM
            def map_columns(row, index):
                return {
                    "id": index,
                    "conversations": row["prompt"],
                    "metadata": {
                        "language": "en",
                        "example_tags": row["example_tags"],
                    },
                    "criteria": row["rubrics"],
                }

            dataset = dataset.map(
                map_columns,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,
                with_indices=True,
            )

            self.dataset = dataset

    def prepare_conversations_for_inference(
        self, turn: int, fewshot_as_multiturn: bool = False
    ) -> list[Any]:
        """Prepare the conversations for inference by formatting prompts for the given turn.

        The formatted conversations should be in chat format compatible with the
        `apply_chat_template` method for prompt tokenization.

        Args:
            turn (int): Which turn to prepare the prompts for (0-indexed).
            fewshot_as_multiturn (bool, optional): Whether to treat few-shot examples as multi-turn dialogues. Defaults to False.

        Returns:
            list[Any]: The formatted conversations.
        """
        logger.info(
            "Performing inference for task '%s', turn %d with %d examples",
            self.task_name.upper(),
            turn,
            self.fewshot_num_examples,
        )

        return self.dataset["conversations"]

    def get_num_turns(self) -> int:
        """Get the number of turns in the dataset.

        Returns:
            int: Number of conversational turns in the task.
        """
        return 1

    def load_example_dataset(self, limit: int | None = None):
        """Load the example dataset from a data source as a datasets.Dataset object.

        Returns:
            datasets.Dataset: The loaded example dataset.
        """
        raise NotImplementedError(
            "HealthBench does not have a separate example dataset, so this method is not implemented."
        )

    def get_judge_prompt_formatter(self, metric):
        judge_prompt = self.task_config.config.judge["judge_prompts"]

        def _prompt_formatter(row, idx):
            responses = row["responses"]
            criteria = row["criteria"]

            conversations = []
            custom_ids = []
            # Generate judgements for each turn and baseline position
            conversation_list = [
                f"{m['role']}: {m['content']}" for m in row["conversations"]
            ]
            conversation_list.append("Assistant: " + responses[0].strip())

            for i, criterion in enumerate(criteria):
                messages = [
                    {
                        "role": "system",
                        "content": judge_prompt["system_prompt"],
                    },
                    {
                        "role": "user",
                        "content": judge_prompt["criteria_prompt"].format(
                            conversation="\n\n".join(conversation_list),
                            rubric_item=f"[{criterion['points']}] {criterion['criterion']}",
                        ),
                    },
                ]
                # Add messages and custom IDs to lists
                conversations.append(messages)
                custom_ids.append(f"{row['id']}_criteria{i}")
            return {"conversations": conversations, "custom_ids": custom_ids}

        return _prompt_formatter
