from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader
from src.dataloaders.judges.judge_dataloader import JudgeDataloader

logger = get_logger(__name__)


class MTBenchJudgeDataloader(HuggingFaceDataloader, JudgeDataloader):
    """Dataloader for MTBench Judge datasets."""

    def get_judge_prompt_formatter(self, metric):
        judge_prompt = self.task_config.config.judge["judge_prompts"]

        def _prompt_formatter(row, idx):
            responses = row["responses"]
            criteria = row["criteria"]

            conversations = []
            custom_ids = []
            # Generate judgements for each turn and baseline position

            for i, criterion in enumerate(criteria):
                try:
                    turn = criterion["turn"]
                    if turn == 0:
                        criteria_prompt = judge_prompt["criteria_prompt"][
                            f"turn_{turn + 1}"
                        ]
                        transcript = """[User]\n""" + self.task_config.config[
                            "languages"
                        ][self.lang]["prompt_template"]["task_template"].format(
                            **row["prompts"][0]
                        )
                        response = responses[0]
                    else:
                        criteria_prompt = judge_prompt["criteria_prompt"][
                            f"turn_{turn + 1}"
                        ]
                        transcript = (
                            """[User]\n"""
                            + self.task_config.config["languages"][self.lang][
                                "prompt_template"
                            ]["task_template"].format(**row["prompts"][0])
                            + "\n\n[Assistant (fixed)]\n"
                            + responses[0]
                            + "\n\n[User]\n"
                            + self.task_config.config["languages"][self.lang][
                                "prompt_template"
                            ]["task_template"].format(**row["prompts"][1])
                        )
                        response = responses[1]

                    messages = [
                        {
                            "role": "system",
                            "content": judge_prompt["system_prompt"],
                        },
                        {
                            "role": "user",
                            "content": criteria_prompt.format(
                                transcript=transcript,
                                response=response,
                                keyPointText=criterion["criterion"].format(
                                    **row["prompts"][0]
                                ),
                            ),
                        },
                    ]
                    # Add messages and custom IDs to lists
                    conversations.append(messages)
                    custom_ids.append(f"{row['id']}_turn{turn}_criteria{i}")
                except TypeError as e:
                    # TypeError: can only concatenate str (not "NoneType") to str
                    logger.error(
                        "Error in formatting judge prompt for row %s, turn %d, criterion %d: %s",
                        row["id"],
                        turn,
                        i,
                        e,
                    )

            return {"conversations": conversations, "custom_ids": custom_ids}

        return _prompt_formatter
