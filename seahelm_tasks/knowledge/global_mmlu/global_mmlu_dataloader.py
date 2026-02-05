import random

from datasets import load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader
from src.task_config import TaskConfig

logger = get_logger(__name__)


class GlobalMMLUDataloader(HuggingFaceDataloader):
    """Dataloader for the Global MMLU dataset.

    Global MMLU is a dataset for evaluating language models on the multilingual version of MMLU.
    This dataloader handles loading data and preparing it for language model inference.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        default_num_in_context_examples: int,
        is_base_model: bool = False,
        model_name: str = "",
        run_base_path: str = "",
        inference_file_type: str = "jsonl",
    ):
        """Initialize the GlobalMMLUDataloader.

        Args:
            task_config: TaskConfig object containing task-specific settings.
            default_num_in_context_examples: Default number of few-shot examples to use.
            is_base_model: Whether this is a base model (vs instruction-tuned).
            model_name: Name/path of the model being evaluated.
            run_base_path: Base path for storing inference results.
            inference_file_type: File format for inference results ('jsonl' or 'csv').
        """
        super().__init__(
            task_config,
            default_num_in_context_examples,
            is_base_model=is_base_model,
            model_name=model_name,
            run_base_path=run_base_path,
            inference_file_type=inference_file_type,
        )
        self.seed = 976843

    def load_dataset(self, limit: int = None):
        """Load the Global MMLU dataset from HuggingFace datasets.

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

            dataset = load_dataset(filepath, self.lang, split="test")

            category = self.task_config.task_name.removeprefix(
                "global_mmlu_"
            ).removesuffix("-logprobs")
            dataset = dataset.filter(lambda x: x["subject"] == category)

            if limit is not None:
                dataset = dataset.select(range(limit))

            multiplier = (
                (len(dataset) // 4 + 1) if len(dataset) % 4 != 0 else len(dataset) // 4
            )
            options = list("ABCD" * multiplier)
            seeds = random.Random(self.seed).sample(range(2**31 - 1), len(dataset))
            random.Random(self.seed).shuffle(options)

            option_map = {"A": 0, "B": 1, "C": 2, "D": 3}

            # Map dataset fields to standard fields used in SEA-HELM
            def map_columns(row, index):
                rng = random.Random(seeds[index])

                choices = [
                    row["option_a"].strip(),
                    row["option_b"].strip(),
                    row["option_c"].strip(),
                    row["option_d"].strip(),
                ]
                correct_answer = choices.pop(option_map[row["answer"]])
                rng.shuffle(choices)
                choices.insert(option_map[options[index]], correct_answer)

                choices_kwargs = {
                    f"option_{letter.lower()}": choice.strip()
                    for letter, choice in zip("ABCD", choices, strict=True)
                }

                return {
                    "id": row["sample_id"],
                    "label": options[index],
                    "prompts": [{"question": row["question"], **choices_kwargs}],
                    "metadata": {
                        "language": "en",
                        "subject": row["subject"],
                        "subject_category": row["subject_category"],
                        "required_knowledge": row["required_knowledge"],
                        "time_sensitive": row["time_sensitive"],
                        "reference": row["reference"],
                        "culture": row["culture"],
                        "region": row["region"],
                        "country": row["country"],
                        "cultural_sensitivity_label": row["cultural_sensitivity_label"],
                        "is_annotated": row["is_annotated"],
                    },
                }

            dataset = dataset.map(
                map_columns,
                num_proc=16,
                remove_columns=dataset.column_names,
                with_indices=True,
            )

            self.dataset = dataset

    def load_example_dataset(self, limit: int | None = None):
        """Load the example dataset from a data source as a datasets.Dataset object.

        Returns:
            datasets.Dataset: The loaded example dataset.
        """
        if self.example_dataset:
            logger.info("Example dataset already loaded, skipping loading process")
            pass
        else:
            example_filepath = self.specific_task_config["example_filepath"]
            logger.info("Drawing and preparing examples from %s", example_filepath)

            dataset = load_dataset(example_filepath, self.lang, split="dev")

            category = self.task_config.task_name.removeprefix(
                "global_mmlu_"
            ).removesuffix("-logprobs")
            self.example_dataset = dataset.filter(lambda x: x["subject"] == category)

            if limit is not None:
                if len(self.example_dataset) < limit:
                    logger.warning(
                        "Not enough examples! Expected %d examples but only received %d.",
                        limit,
                        len(self.example_dataset),
                    )
                    limit = len(self.example_dataset)
                self.example_dataset = self.example_dataset.select(range(limit))

            multiplier = (
                (len(self.example_dataset) // 4 + 1)
                if len(self.example_dataset) % 4 != 0
                else len(self.example_dataset) // 4
            )
            options = list("ABCD" * multiplier)
            seeds = random.Random(self.seed).sample(
                range(2**31 - 1), len(self.example_dataset)
            )
            random.Random(self.seed).shuffle(options)

            option_map = {"A": 0, "B": 1, "C": 2, "D": 3}

            # Map dataset fields to standard fields used in SEA-HELM
            def map_columns(row, index):
                rng = random.Random(seeds[index])

                choices = [
                    row["option_a"].strip(),
                    row["option_b"].strip(),
                    row["option_c"].strip(),
                    row["option_d"].strip(),
                ]
                correct_answer = choices.pop(option_map[row["answer"]])
                rng.shuffle(choices)
                choices.insert(option_map[options[index]], correct_answer)

                choices_kwargs = {
                    f"option_{letter.lower()}": choice.strip()
                    for letter, choice in zip("ABCD", choices, strict=True)
                }

                return {
                    "id": row["sample_id"],
                    "label": options[index],
                    "prompts": [{"question": row["question"], **choices_kwargs}],
                    "metadata": {
                        "language": "en",
                        "subject": row["subject"],
                        "subject_category": row["subject_category"],
                        "required_knowledge": row["required_knowledge"],
                        "time_sensitive": row["time_sensitive"],
                        "reference": row["reference"],
                        "culture": row["culture"],
                        "region": row["region"],
                        "country": row["country"],
                        "cultural_sensitivity_label": row["cultural_sensitivity_label"],
                        "is_annotated": row["is_annotated"],
                    },
                }

            self.example_dataset = self.example_dataset.map(
                map_columns,
                num_proc=16,
                remove_columns=self.example_dataset.column_names,
                with_indices=True,
            )
