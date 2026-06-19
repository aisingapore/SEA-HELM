import datasets

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader

logger = get_logger(__name__)


class GlobalMMLULiteDataloader(HuggingFaceDataloader):
    """Dataloader for SEA-HELM local datasets."""

    def load_dataset(self, limit: int | None = None) -> None:
        """Load the dataset from a data source as a datasets.Dataset object.

        Args:
            limit (int, optional): Optional limit on the number of instances to load. Defaults to None.
        """
        if self.dataset:
            logger.info("Dataset already loaded, skipping loading process")
            pass
        else:
            filepath = self.specific_task_config["filepath"]

            logger.info("Drawing and preparing instances from %s", filepath)

            self.dataset = datasets.load_dataset(
                "json", split="train", data_files=filepath
            )
            if limit is not None:
                self.dataset = self.dataset.select(range(limit))

    def load_example_dataset(self, limit: int | None = None):
        """Load the example dataset from a data source as a datasets.Dataset object.

        Args:
            limit (int, optional): Optional limit on the number of instances to load. Defaults to None.
        """
        if self.example_dataset:
            logger.info("Example dataset already loaded, skipping loading process")
            pass
        else:
            example_filepath = self.specific_task_config["example_filepath"]
            logger.info("Drawing and preparing examples from %s", example_filepath)

            self.example_dataset = datasets.load_dataset(
                "json", split="train", data_files=example_filepath
            )

            if limit is not None:
                example_df = self.example_dataset.to_pandas()
                example_df["subject"] = example_df["metadata"].apply(
                    lambda x: x.get("subject", "").lower()
                )
                subject_counts = example_df.groupby("subject").size()
                if subject_counts.min() < limit:
                    logger.warning(
                        "Not enough examples for some subjects! Expected %d examples but only received %d.",
                        limit,
                        subject_counts.min(),
                    )
                elif subject_counts.min() >= limit:
                    # Select the first few examples up to limit for each subject
                    selected_examples = []
                    for subject, _ in subject_counts.items():
                        subject_examples = example_df.query(f'subject == "{subject}"')
                        selected_examples.extend(
                            subject_examples.index.tolist()[:limit]
                        )
                    self.example_dataset = self.example_dataset.select(
                        selected_examples
                    )

            # check if label is of type list and convert it to string
            if isinstance(self.example_dataset.features["label"], datasets.Sequence):
                self.example_dataset = self.example_dataset.map(
                    lambda x: {"label": x["label"][0]}, num_proc=self.num_workers
                )

    def generate_formatted_conversation(
        self,
        specific_task_config: dict,
        values: dict,
        fewshot_as_multiturn: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Generate a formatted conversation with few-shot examples and task prompt.

        Creates a structured conversation format with roles and contents,
        including in-context examples and the main task prompt.

        Args:
            specific_task_config (dict): Language-specific task configuration containing
                templates and example file paths.
            values (dict): Dictionary of values to format into the task template.
            fewshot_as_multiturn (bool): Whether to format examples as multi-turn dialogue.

        Returns:
            tuple[list[str], list[str]]: (roles, contents) where roles is a list of speaker roles
                and contents is a list of corresponding message contents.
        """
        roles = []
        contents = []
        task_prompt_template = specific_task_config["prompt_template"]

        # insert multiturn examples to contents
        if hasattr(self, "example_dataset") and self.example_dataset is not None:
            subject = values.get("subject", "")
            in_context_examples = self.example_dataset.to_pandas()
            in_context_examples = in_context_examples[
                (
                    in_context_examples["metadata"].apply(
                        lambda x: x.get("subject", "").lower()
                    )
                    == subject.lower()
                )
            ]

            if not fewshot_as_multiturn:
                # TODO: add support for ICL as a single turn for chat models
                raise NotImplementedError(
                    "ICL as a single turn for chat models is not supported"
                )
            else:
                for _, row in in_context_examples.iterrows():
                    roles.append("user")
                    contents.append(
                        task_prompt_template["task_template"].format(
                            **row["prompts"][0],
                            answer_tag=task_prompt_template["answer_tag"],
                        )
                    )
                    roles.append("assistant")
                    contents.append(
                        task_prompt_template["answer_template"].format(
                            **row,
                            **row["prompts"][0],
                            answer_tag=task_prompt_template["answer_tag"],
                        )
                    )

        roles.append("user")
        contents.append(task_prompt_template["task_template"].format(**values))

        # append preamble to first user prompt
        contents[0] = (
            task_prompt_template["preamble"].format(
                answer_tag=task_prompt_template["answer_tag"]
            )
            + "\n\n"
            + contents[0]
        ).strip()

        return roles, contents
