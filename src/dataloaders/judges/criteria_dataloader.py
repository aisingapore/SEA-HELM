from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader
from src.dataloaders.judges.judge_dataloader import JudgeDataloader
from src.dataloaders.seahelm_local_dataloader import SeaHelmLocalDataloader


class CriteriaDataloader(JudgeDataloader):
    def get_judge_prompt_formatter(self, metric):
        """Return a formatter that builds per-criterion judge conversations for each dataset row.

        For every criterion attached to a row, constructs a judge conversation that
        includes the original prompt, the model response, and the criterion text.
        If a criterion has ``include_reference`` set, the reference answer is also
        appended to the judge prompt.

        Args:
            metric: The metric object (unused; kept for interface compatibility).

        Returns:
            Callable[[dict, int], dict]: A function that accepts a dataset row and its
                index and returns a dict with keys ``conversations`` (list of message
                lists) and ``custom_ids`` (list of str identifiers).
        """
        judge_prompt = self.task_config.config.judge["judge_prompts"]

        def _prompt_formatter(row, idx):
            responses = row["responses"]
            criteria = row["criteria"]

            conversations = []
            custom_ids = []
            # Generate judgements for each turn and baseline position

            for i, criterion in enumerate(criteria):
                if criterion.get("include_reference", False):
                    reference_text_block = (
                        "\n"
                        + judge_prompt["reference_prompt"].format(
                            reference=row["label"]
                        )
                        + "\n"
                    )
                else:
                    reference_text_block = ""

                messages = [
                    {
                        "role": "system",
                        "content": judge_prompt["system_prompt"],
                    },
                    {
                        "role": "user",
                        "content": judge_prompt["criteria_prompt"].format(
                            prompt=self.task_config.config["languages"][self.lang][
                                "prompt_template"
                            ]["task_template"].format(**row["prompts"][0]),
                            response=responses[0],
                            keyPointText=criterion["criterion"].format(
                                **row["prompts"][0]
                            ),
                            reference_text_block=reference_text_block,
                        ),
                    },
                ]
                # Add messages and custom IDs to lists
                conversations.append(messages)
                custom_ids.append(f"{row['id']}_criteria{i}")
            return {"conversations": conversations, "custom_ids": custom_ids}

        return _prompt_formatter


class SeaHelmLocalCriteriaDataloader(SeaHelmLocalDataloader, CriteriaDataloader): ...


class HuggingFaceCriteriaDataloader(HuggingFaceDataloader, CriteriaDataloader): ...
