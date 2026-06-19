import ast

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader

logger = get_logger(__name__)


class LiveCodeBenchDataloader(HuggingFaceDataloader):
    """Dataloader for the LiveCodeBench dataset.

    LiveCodeBench is a dataset for evaluating code generation models on various programming tasks.
    This dataloader handles loading data and preparing it for model inference.
    """

    def load_dataset(self, limit: int = None):
        """Load the LiveCodeBench dataset from HuggingFace datasets.

        Args:
            limit: Optional limit on the number of instances to load.

        Note:
            The dataset is stored in self.dataset after loading.
        """
        if self.dataset:
            logger.info("Dataset already loaded, skipping loading process")
            pass
        else:
            filename_map = {
                "v1": ["test.jsonl"],
                "v2": ["test.jsonl", "test2.jsonl"],
                "v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
                "v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
                "v5": [
                    "test.jsonl",
                    "test2.jsonl",
                    "test3.jsonl",
                    "test4.jsonl",
                    "test5.jsonl",
                ],
                "v6": [
                    "test.jsonl",
                    "test2.jsonl",
                    "test3.jsonl",
                    "test4.jsonl",
                    "test5.jsonl",
                    "test6.jsonl",
                ],
            }
            filepath = self.specific_task_config["filepath"]
            logger.info("Drawing and preparing data from %s", filepath)

            split = self.task_name.removeprefix("livecodebench_")
            files = []
            for filename in filename_map[split]:
                file = hf_hub_download(
                    repo_id=filepath, repo_type="dataset", filename=filename
                )
                files.append(file)
            dataset = load_dataset("json", data_files=files, split="train")

            if limit is not None:
                dataset = dataset.select(range(limit))

            def map_columns(row, index):
                metadata = ast.literal_eval(row["metadata"])
                if "func_name" in metadata:
                    func_name = metadata["func_name"]
                else:
                    func_name = None

                if row.get("starter_code", "") == "":
                    format_prompt = "### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n```python\n# YOUR CODE HERE\n```"
                else:
                    format_prompt = f"### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n```python\n{row['starter_code']}\n```"
                return {
                    "id": row["question_id"],
                    "prompts": [
                        {
                            "question": row["question_content"],
                            "format_prompt": format_prompt,
                        }
                    ],
                    "test_cases": {
                        "public": ast.literal_eval(row["public_test_cases"]),
                        "private": row["private_test_cases"],
                    },
                    "metadata": {
                        "language": "en",
                        "platform": row["platform"],
                        "contest_date": row["contest_date"],
                        "difficulty": row["difficulty"],
                        "func_name": func_name,
                    },
                }

            dataset = dataset.map(
                map_columns,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,
                with_indices=True,
            )
            self.dataset = dataset
