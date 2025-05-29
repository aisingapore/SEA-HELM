import asyncio
import json
import os
from collections import Counter
from enum import Enum

import pandas as pd

from base_logger import get_logger
from seahelm_tasks.multi_turn.mt_bench.mt_bench_prompts import (
    CATEGORIES_WITH_REFERENCE,
    JUDGE_PROMPTS,
)
from seahelm_tasks.seahelm_metric import SeaHelmMetric
from serving.openai_serving import OPENAI_MODELS, OpenAIServing, LiteLLMServing
from utils import get_error_count

logger = get_logger(__name__)


def evaluate_mt_bench_task(
    inference_df, task_config, task_name, lang, inference_time_taken, is_cached, Metric
):
    try:
        logger.info(
            "--------- Evaluation | Lang: %s | Task: %s ----------",
            lang.upper(),
            task_name.upper(),
        )
        logger.info("Evaluating %s using %s", task_name.upper(), Metric.__name__)
        evaluation_metric = Metric(
            inference_df=inference_df,
            task_config=task_config,
            task=task_name,
            lang=lang,
        )
        metric_json, inference_df = evaluation_metric.evaluate_responses()
        metric_json[task_name]["errors"] = get_error_count(inference_df["errors"])
        metric_json[task_name]["inference_time_taken"] = inference_time_taken
        metric_json[task_name]["is_cached"] = is_cached
    except Exception as e:
        logger.error(
            "Failed to run evaluation for task %s and lang %s", task_name, lang
        )
        logger.exception(e)
        logger.warning(
            "Setting metric %s to 0 for task %s", task_config["metric"], task_name
        )
        metric_json = {
            task_name: {
                task_config["metric"]: 0,
                "error": "Failed to run evaluation for task",
            },
        }
    return metric_json, inference_df, task_config["competency"], task_name, lang


class JudgmentOutcome(Enum):
    WIN = 1
    LOSE = 2
    TIE = 3
    ERROR = 4


class MTBenchMetric(SeaHelmMetric):
    def __init__(
        self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):
        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )
        self.judge_model = task_config["judge_model"]
        self.batch_openai_calls = task_config.get("batch_openai_calls", True)
        if self.judge_model in OPENAI_MODELS:
            if self.batch_openai_calls:
                self.judge = OpenAIServing(self.judge_model)
            else:
                model_args = {}
                for key in task_config:
                    if "litellm_" in key:
                        model_args[key.replace("litellm_", "")] = task_config[key]
                self.judge = LiteLLMServing(self.judge_model, **model_args)
        else:
            raise ValueError(f"Unsupported judge model: {self.judge_model}")

        self.baseline_model = task_config["baseline_model"]
        self.model_name = task_config["model_name"]
        self.output_dir = task_config["output_dir"]

        self.use_cached_results = task_config["use_cached_results"]
        self.judge_seed = task_config["judge_seed"]
        self.judge_temperature = task_config["judge_temperature"]
        self.judge_max_tokens = task_config["judge_max_tokens"]

    def postprocess_responses(self):
        self.inference_df["category"] = [
            x["category"] for x in self.inference_df["metadata"]
        ]

    def get_llm_judgements(self):
        llm_judgement_file_path = os.path.join(
            self.output_dir,
            os.path.basename(self.model_name),
            "inference",
            f"{os.path.basename(self.model_name)}_{self.task}_{self.lang}_{self.baseline_model}_{self.judge_model}_judgement.jsonl",
        )

        if self.use_cached_results and os.path.exists(llm_judgement_file_path):
            judgement_df = pd.read_json(llm_judgement_file_path, lines=True)
        else:
            messages = []
            ids = []
            for i, row in self.inference_df.iterrows():
                responses = row[self.response_column]
                baselines = row["baselines"][self.baseline_model]
                questions = row["prompts"]

                is_with_ref = row["category"] in CATEGORIES_WITH_REFERENCE
                if is_with_ref:
                    references = row["references"]
                    prompts = JUDGE_PROMPTS["with-reference"]
                else:
                    prompts = JUDGE_PROMPTS["without-reference"]

                for turn in range(len(responses)):
                    for baseline_position in ["baseline-before", "baseline-after"]:
                        info = {}
                        for i in range(turn + 1):
                            info[f"question_{i+1}"] = questions[i]["text"]
                            info[f"answer_a_{i+1}"] = (
                                responses[i]
                                if baseline_position == "baseline-after"
                                else baselines[i]
                            )
                            info[f"answer_b_{i+1}"] = (
                                baselines[i]
                                if baseline_position == "baseline-after"
                                else responses[i]
                            )

                            if is_with_ref:
                                info[f"ref_answer_{i+1}"] = references[i]

                        messages.append(
                            [
                                {
                                    "role": "system",
                                    "content": prompts[turn]["system_prompt"],
                                },
                                {
                                    "role": "user",
                                    "content": prompts[turn]["prompt_template"].format(
                                        **info
                                    ),
                                },
                            ]
                        )
                        ids.append(
                            f"{row['question_id']}_turn{i+1}_{baseline_position}",
                        )

            batch_responses = self.judge.batch_generate(messages)
            batch_responses = [x.json() for x in batch_responses]
            judgement_df = pd.DataFrame(batch_responses)
            judgement_df["custom_id"] = ids
            judgement_df.to_json(
                llm_judgement_file_path,
                orient="records",
                lines=True,
                force_ascii=False,
            )
        return judgement_df

    def get_batched_llm_judgments(self):
        llm_judgement_file_path = os.path.join(
            self.output_dir,
            os.path.basename(self.model_name),
            "inference",
            f"{os.path.basename(self.model_name)}_{self.task}_{self.lang}_{self.baseline_model}_{self.judge_model}_judgement.jsonl",
        )

        llm_batch_file_path = os.path.join(
            self.output_dir,
            os.path.basename(self.model_name),
            "inference",
            f"{os.path.basename(self.model_name)}_{self.task}_{self.lang}_{self.baseline_model}_{self.judge_model}_batch.jsonl",
        )

        if self.use_cached_results and os.path.exists(llm_batch_file_path):
            pass
        else:
            self.prepare_llm_batches(llm_batch_file_path)

        if self.use_cached_results and os.path.exists(llm_judgement_file_path):
            judgement_df = pd.read_json(llm_judgement_file_path, lines=True)
        else:
            judgement_df = pd.DataFrame()

        with open(llm_batch_file_path, "r", encoding="utf-8") as f:
            expected_batches = [json.loads(line) for line in f]
        expected_ids = set(batch["custom_id"] for batch in expected_batches)

        retries = 0
        max_retries = 3

        while True:
            existing_ids = (
                set(judgement_df["custom_id"]) if not judgement_df.empty else set()
            )
            missing_ids = expected_ids - existing_ids

            if not missing_ids:
                logger.info("All judgments have been obtained.")
                break

            if retries == 0 and judgement_df.empty:
                logger.info("First run: processing all judgments.")
                batch_file_path_to_use = llm_batch_file_path
            else:
                logger.info(f"Missing {len(missing_ids)} judgments. Retrying...")
                missing_batches = [
                    batch
                    for batch in expected_batches
                    if batch["custom_id"] in missing_ids
                ]
                missing_batch_file_path = llm_batch_file_path.replace(
                    ".jsonl", f"_missing_retry{retries}.jsonl"
                )
                with open(missing_batch_file_path, "w", encoding="utf-8") as f:
                    for batch in missing_batches:
                        f.write(json.dumps(batch, ensure_ascii=False) + "\n")
                batch_file_path_to_use = missing_batch_file_path

            temp_output_file_path = llm_judgement_file_path.replace(
                ".jsonl", f"_temp_retry{retries}.jsonl"
            )

            asyncio.run(
                self.judge.abatch_generate(
                    file_path=batch_file_path_to_use,
                    output_file_path=temp_output_file_path,
                )
            )

            new_judgement_df = pd.read_json(temp_output_file_path, lines=True)

            judgement_df = pd.concat(
                [judgement_df, new_judgement_df], ignore_index=True
            )

            judgement_df = judgement_df.drop_duplicates(subset="custom_id", keep="last")

            judgement_df.to_json(
                llm_judgement_file_path, orient="records", lines=True, force_ascii=False
            )

            os.remove(batch_file_path_to_use)
            os.remove(temp_output_file_path)

            retries += 1
            if retries >= max_retries:
                logger.warning(
                    f"Reached maximum retries ({max_retries}). Some judgments may still be missing."
                )
                break

        existing_ids = (
            set(judgement_df["custom_id"]) if not judgement_df.empty else set()
        )
        missing_ids = expected_ids - existing_ids

        if missing_ids:
            logger.error(
                f"Failed to obtain all judgments after {retries} retries. Missing judgments for IDs: {missing_ids}"
            )
        else:
            logger.info("Successfully obtained all judgments.\n")

        return judgement_df

    def prepare_llm_batches(self, llm_batch_file_path):
        batches = []
        for i, row in self.inference_df.iterrows():
            responses = row[self.response_column]
            baselines = row["baselines"][self.baseline_model]
            questions = row["prompts"]

            is_with_ref = row["category"] in CATEGORIES_WITH_REFERENCE
            if is_with_ref:
                references = row["references"]
                prompts = JUDGE_PROMPTS["with-reference"]
            else:
                prompts = JUDGE_PROMPTS["without-reference"]

            for turn in range(len(responses)):
                for baseline_position in ["baseline-before", "baseline-after"]:
                    info = {}
                    for i in range(turn + 1):
                        info[f"question_{i+1}"] = questions[i]["text"]
                        info[f"answer_a_{i+1}"] = (
                            responses[i]
                            if baseline_position == "baseline-after"
                            else baselines[i]
                        )
                        info[f"answer_b_{i+1}"] = (
                            baselines[i]
                            if baseline_position == "baseline-after"
                            else responses[i]
                        )

                        if is_with_ref:
                            info[f"ref_answer_{i+1}"] = references[i]

                    messages = [
                        {"role": "system", "content": prompts[turn]["system_prompt"]},
                        {
                            "role": "user",
                            "content": prompts[turn]["prompt_template"].format(**info),
                        },
                    ]
                    output = {
                        "custom_id": f"{row['question_id']}_turn{i+1}_{baseline_position}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.judge_model,
                            "messages": messages,
                            "max_tokens": self.judge_max_tokens,
                            "temperature": self.judge_temperature,
                            "seed": self.judge_seed,
                        },
                    }
                    batches.append(output)
        df = pd.DataFrame(batches)
        df.to_json(llm_batch_file_path, orient="records", lines=True, force_ascii=False)

    def calculate_metrics(self):
        if self.batch_openai_calls:
            judgement_df = self.get_batched_llm_judgments()
        else:
            judgement_df = self.get_llm_judgements()

        judgement_meta_df = pd.DataFrame(
            judgement_df["custom_id"].map(lambda x: x.split("_")).to_list(),
            columns=["question_id", "turn", "order"],
        )
        judgement_df = pd.concat([judgement_df, judgement_meta_df], axis=1)

        if self.batch_openai_calls:
            judgement_df["verdict"] = judgement_df.apply(
                lambda row: self.parse_judgment(
                    row["response"]["body"]["choices"][0]["message"]["content"],
                    row["order"] == "baseline-before",
                ),
                axis=1,
            )
        else:
            judgement_df["verdict"] = judgement_df.apply(
                lambda row: self.parse_judgment(
                    row["choices"][0]["message"]["content"],
                    row["order"] == "baseline-before",
                ),
                axis=1,
            )

        all_judgments = []
        for _, row in self.inference_df.iterrows():
            question_id = str(row["question_id"])
            judgements = judgement_df[judgement_df["question_id"] == question_id]

            parsed_judgments = {}

            for order in ["baseline-after", "baseline-before"]:
                judgements_order = judgements[judgements["order"] == order]
                parsed_judgments[order] = judgements_order["verdict"].to_list()

            final_judgement = []
            for turn in judgements["turn"].unique():
                _judgements = judgements[judgements["turn"] == turn]
                _judgements = _judgements["verdict"].tolist()
                verdict = self.get_final_judgment(*_judgements)
                final_judgement.append(verdict)

            parsed_judgments["final"] = final_judgement

            all_judgments.append(parsed_judgments)

        all_judgements_df = pd.DataFrame(all_judgments)
        for col in all_judgements_df.columns:
            self.inference_df[col] = all_judgements_df[col]

        metrics = self.get_mt_bench_metrics(
            self.inference_df["final"], self.inference_df["category"]
        )
        self.inference_df["individual_scores"] = [
            {
                "weighted_win_rate": [self.get_score(x[0]), self.get_score(x[1])],
            }
            for x in self.inference_df["final"]
        ]
        return metrics, self.inference_df

    def get_score(self, judgement):
        if judgement == JudgmentOutcome.WIN:
            return 1
        elif judgement == JudgmentOutcome.TIE:
            return 0.5
        else:
            return 0

    def get_win_rate(self, judgment_list: list):
        total_count = len(judgment_list)
        counts = Counter(judgment_list)
        win_rate = (
            counts[JudgmentOutcome.WIN] + (counts[JudgmentOutcome.TIE] * 0.5)
        ) / total_count
        return win_rate

    def get_mt_bench_metrics(self, judgment_list: list, category_list: list):
        metric_dict = {"categories": {}}
        df = pd.DataFrame()
        df["judgment"] = judgment_list
        df["category"] = category_list

        for category in df["category"].value_counts().keys():
            subset = df[df["category"] == category]
            subset_judgments = subset["judgment"]
            subset_judgments = [
                j for judgment_pair in subset_judgments for j in judgment_pair
            ]
            subset_win_rate = self.get_win_rate(subset_judgments)
            metric_dict["categories"].update({category: subset_win_rate})
            logger.info(f"Win rate for category <{category}>: {subset_win_rate}")

        overall_win_rate = self.get_win_rate(
            [j for judgment_pair in judgment_list for j in judgment_pair]
        )
        metric_dict["win_rate"] = overall_win_rate
        logger.info(f"Overall win rate: {overall_win_rate}")

        weighted_win_rate = sum(list(metric_dict["categories"].values())) / len(
            metric_dict["categories"]
        )
        metric_dict["weighted_win_rate"] = weighted_win_rate * 100
        logger.info(f"Weighted win rate: {weighted_win_rate}")

        return metric_dict

    def parse_judgment(self, judgment: str, reverse: bool) -> JudgmentOutcome:
        """
        Parse LLM's judgment on a pairwise comparison to determine if the model being evaluated
        wins/loses/ties against a baseline model.

        The default order in which answers are presented to the judge is as follows:
        A: Model being evaluated
        B: Baseline model

        Therefore, a win is when the model outputs [[A]] as its judgment.
        However, when controlling for position bias, the judgment is repeated with the answers
        presented in reverse. (reverse=True)

        """
        win = "[[A]]" in judgment
        lose = "[[B]]" in judgment
        tie = "[[C]]" in judgment

        if (win + lose + tie) != 1:
            # Judge responded with conflicting judgments (Sum > 1)
            # Or judge did not provide judgment in correct format (Sum = 0)
            return JudgmentOutcome.ERROR
        elif tie:
            return JudgmentOutcome.TIE
        else:
            if reverse:
                if win:
                    return JudgmentOutcome.LOSE
                elif lose:
                    return JudgmentOutcome.WIN
            else:
                if win:
                    return JudgmentOutcome.WIN
                elif lose:
                    return JudgmentOutcome.LOSE

    def get_final_judgment(
        self, judgment_1: JudgmentOutcome, judgment_2: JudgmentOutcome
    ):
        """
        Compare LLM judgment when pairs of answers are presented in both normal and reverse order.
        (This is to ensure that LLM judgment is consistent and not affected by position bias.)

        (1) If both judgments are different, the result is a TIE.
        (2) If the first judgment is a tie, the result is a TIE.
            (If the second judgment is also a tie, then overall the result is a tie.)
            (Even if the second judgment is not a tie, due to rule (1) above, the result will still be a tie.)
        (3) Otherwise, both judgments should agree on WIN or LOSE (so we can take either one).
        """
        if judgment_1 == JudgmentOutcome.TIE or judgment_1 != judgment_2:
            return JudgmentOutcome.TIE
        else:
            return judgment_1
