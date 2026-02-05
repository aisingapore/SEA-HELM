from datasets import load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader
from src.task_config import TaskConfig

logger = get_logger(__name__)


class IFEvalDataloader(HuggingFaceDataloader):
    """Dataloader for the IFEval dataset.

    IFEval is a dataset for evaluating language models on their instruction-following capabilities.
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
        """Initialize the IFEvalDataloader.

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

    def load_dataset(self, limit: int = None):
        """Load the IFEval dataset from HuggingFace datasets.

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

            dataset = load_dataset(filepath, split="train")

            # perform edit for id 1174
            def correct_data(row):
                if row["key"] == 1174:
                    old_kwargs = row["kwargs"].copy()
                    old_kwargs[0]["let_relation"] = "at least"
                    return {"kwargs": old_kwargs}
                else:
                    return {"kwargs": row["kwargs"]}

            dataset = dataset.map(correct_data, num_proc=16)

            if limit is not None:
                dataset = dataset.select(range(limit))

            # Map dataset fields to standard fields used in SEA-HELM
            def map_columns(row):
                return {
                    "id": row["key"],
                    "prompts": [{"text": row["prompt"]}],
                    "metadata": {
                        "language": "en",
                        "instruction_id_list": row["instruction_id_list"],
                    },
                    "kwargs": row["kwargs"],
                }

            dataset = dataset.map(
                map_columns,
                num_proc=16,
                remove_columns=dataset.column_names,
            )

            self.dataset = dataset
