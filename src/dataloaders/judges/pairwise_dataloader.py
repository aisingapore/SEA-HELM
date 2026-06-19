from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader
from src.dataloaders.judges.judge_dataloader import JudgeDataloader
from src.dataloaders.seahelm_local_dataloader import SeaHelmLocalDataloader


class PairwiseDataloader(JudgeDataloader):
    def get_judge_prompt_formatter(self, metric):
        """Return a formatter that builds pairwise judge conversations for each dataset row.

        For every row, generates judge message pairs with the model response and baseline
        answer in both orderings (baseline-before and baseline-after) across all turns.
        Rows belonging to categories that require reference answers automatically include
        those references in the judge prompt.

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
            baselines = row["baselines"][self.task_config.config.baseline_model]
            questions = row["prompts"]

            # Determine if this category requires reference answers
            is_with_ref = (
                row["metadata"]["category"]
                in self.task_config.config.categories_with_reference
            )

            # Select appropriate prompts based on reference requirement
            if is_with_ref:
                references = row["references"]
                prompts = judge_prompt["with-reference"]
            else:
                prompts = judge_prompt["without-reference"]

            conversations = []
            custom_ids = []
            # Generate judgements for each turn and baseline position
            for turn in range(len(responses)):
                for baseline_position in ["baseline-before", "baseline-after"]:
                    info = {}

                    # Build conversation context up to current turn
                    for i in range(turn + 1):
                        info[f"question_{i + 1}"] = questions[i]["text"]

                        # Assign answers based on baseline position
                        info[f"answer_a_{i + 1}"] = (
                            responses[i]
                            if baseline_position == "baseline-after"
                            else baselines[i]
                        )

                        info[f"answer_b_{i + 1}"] = (
                            baselines[i]
                            if baseline_position == "baseline-after"
                            else responses[i]
                        )

                        # Add reference answer if required
                        if is_with_ref:
                            info[f"ref_answer_{i + 1}"] = references[i]

                    # Construct judge prompt messages
                    messages = [
                        {
                            "role": "system",
                            "content": prompts[turn]["system_prompt"].format(
                                **self.task_config.config.judgement_labels
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompts[turn]["prompt_template"].format(**info),
                        },
                    ]

                    # Add messages and custom IDs to lists
                    conversations.append(messages)
                    custom_ids.append(
                        f"{row['question_id']}_turn{i + 1}_{baseline_position}"
                    )
            return {"conversations": conversations, "custom_ids": custom_ids}

        return _prompt_formatter


class SeaHelmLocalPairwiseDataloader(SeaHelmLocalDataloader, PairwiseDataloader): ...


class HuggingFacePairwiseDataloader(HuggingFaceDataloader, PairwiseDataloader): ...
