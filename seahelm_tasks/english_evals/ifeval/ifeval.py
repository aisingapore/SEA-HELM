from seahelm_tasks.english_evals.ifeval.utils import (
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)
from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

# TODO validate error caused by "Resource [93mpunkt_tab[0m not found." during nltk import


logger = get_logger(__name__)


class IFEvalMetric(SeaHelmMetric):
    def __init__(self, dataloader: AbstractDataloader, task_config: TaskConfig):
        super().__init__(dataloader=dataloader, task_config=task_config)

    def inst_level_acc(self, items):
        # Source: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/ifeval/utils.py
        # Modified from agg_inst_level_acc()
        inst_level_acc = sum(items) / len(items)
        return inst_level_acc

    def postprocess_responses(self):
        self.dataloader.inference_df[self.postprocessed_response_column] = (
            self.dataloader.inference_df[self.response_column].map(lambda x: x[0])
        )

    def calculate_metrics(self):
        prompt_level_strict_acc = []
        inst_level_strict_acc = []
        prompt_level_loose_acc = []
        inst_level_loose_acc = []

        for _, row in self.dataloader.inference_df.iterrows():
            inp = InputExample(
                key=row["id"],
                instruction_id_list=row["metadata"]["instruction_id_list"],
                prompt=row["prompts"][0]["text"],
                kwargs=row["kwargs"],
            )
            response = row[self.postprocessed_response_column]

            out_strict = test_instruction_following_strict(inp, response)
            out_loose = test_instruction_following_loose(inp, response)

            prompt_level_strict_acc.append(int(out_strict.follow_all_instructions))
            inst_level_strict_acc.append(
                self.inst_level_acc(out_strict.follow_instruction_list)
            )
            prompt_level_loose_acc.append(int(out_loose.follow_all_instructions))
            inst_level_loose_acc.append(
                self.inst_level_acc(out_loose.follow_instruction_list)
            )

        metric_dict = {
            "prompt_level_strict_acc": sum(prompt_level_strict_acc)
            / len(prompt_level_strict_acc),
            "inst_level_strict_acc": sum(inst_level_strict_acc)
            / len(inst_level_strict_acc),
            "prompt_level_loose_acc": sum(prompt_level_loose_acc)
            / len(prompt_level_loose_acc),
            "inst_level_loose_acc": sum(inst_level_loose_acc)
            / len(inst_level_loose_acc),
        }

        normalized_prompt_level_strict_acc = [
            self.normalize_score(x, 0, 1) for x in prompt_level_strict_acc
        ]
        self.dataloader.inference_df["individual_scores"] = [
            {"prompt_level_strict_acc": x} for x in normalized_prompt_level_strict_acc
        ]

        metric_dict["normalized_prompt_level_strict_acc"] = (
            100
            * sum(normalized_prompt_level_strict_acc)
            / len(normalized_prompt_level_strict_acc)
        )

        logger.info(
            f"normalized_prompt_level_strict_acc: {metric_dict['normalized_prompt_level_strict_acc']:.2f}"
        )
        return metric_dict
