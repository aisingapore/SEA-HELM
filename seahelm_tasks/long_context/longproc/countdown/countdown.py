import re
from typing import Any

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


def evaluate_countdown_final_solution(
    nums: list[int], target: int, solution: str
) -> bool:
    """Validate a final Countdown solution sequence.

    The function expects a solution consisting of exactly three lines, each
    of the form "a OP b = c" where OP is one of +, -, *, /. It applies each
    operation in order, consuming inputs from `nums` and appending results,
    and finally checks whether the single remaining number equals `target`.

    Args:
        nums (list[int]): Multiset of available numbers (will be mutated).
        target (int): The target number to reach at the end.
        solution (str): Three-line string describing the operations.

    Returns:
        bool: True if the solution is well-formed, uses valid numbers in the
            correct order, and produces `target`; False otherwise.
    """

    # parse a ? b = c into a, b, c, op
    def _parse_line(
        line: str,
    ) -> tuple[bool, int | None, int | None, int | None, str | None]:
        line = line.strip()
        if len(line.split("=")) != 2:
            return False, None, None, None, None
        lhs, rhs = line.split("=")
        lhs_result = eval(lhs)
        if "+" in lhs:
            op = "+"
        elif "-" in lhs:
            op = "-"
        elif "*" in lhs:
            op = "*"
        elif "/" in lhs:
            op = "/"
        else:
            return (
                False,
                None,
                None,
                None,
                None,
            )
        a, b = lhs.split(op)
        return (
            lhs_result == int(rhs),
            int(a),
            int(b),
            int(rhs),
            op,
        )

    # parse solution into equations
    lines = solution.split("\n")
    if len(lines) != 3:
        return False
    # check if the solution is correct
    for line in lines:
        try:
            correct, a, b, c, op = _parse_line(line)
        except (ValueError, SyntaxError):
            return False
        if not correct:
            return False
        if a not in nums:
            return False
        nums.remove(a)
        if b not in nums:
            return False
        nums.remove(b)
        nums.append(c)
    final_result = list(nums)[0]
    return final_result == target


def evaluate_countdown_search_procedure(
    procedure: str, gt_procedure: str
) -> tuple[float, dict[str, Any] | str]:
    """Compute partial accuracy of the predicted search procedure.

    This compares a predicted search procedure against the ground truth by
    aligning step-by-step textual actions. The first line (initialization)
    must match exactly. Subsequent steps are validated by structure:
    - "Pick two numbers" lines must match exactly and in order
    - "|- Try" lines must operate on the same LHS expression and both either
      drop the branch or not

    Args:
        procedure (str): Predicted search procedure, possibly wrapped in
            <Search Procedure> tags.
        gt_procedure (str): Ground-truth procedure string with the same format.

    Returns:
        tuple[float, dict[str, Any] | str]: A tuple of
            (partial_accuracy, error_report). `partial_accuracy` is the ratio
            of correctly matched lines (excluding the initialization line).
            `error_report` is a small dict or descriptive string indicating the
            first mismatch; if no mismatch occurs, it may be an empty dict.
    """
    # remove some unnecessary information
    gt_procedure = gt_procedure.split("<Search Procedure>")[1]
    gt_procedure = gt_procedure.split("</Search Procedure>")[0]
    if "<Search Procedure>" in procedure:
        procedure = procedure.split("<Search Procedure>")[1]
    if "</Search Procedure>" in procedure:
        procedure = procedure.split("</Search Procedure>")[0]

    # return partial accuracy as a float, and return error report
    # we focus on evaluating the "actions" in the procedure
    pred_lines = procedure.strip().split("\n")
    gt_lines = gt_procedure.strip().split("\n")

    # initalization statement should be the same
    if pred_lines[0] != gt_lines[0]:
        return 0.0, {
            "line_number": 0,
            "prediction": pred_lines[0],
            "ground_truth": gt_lines[0],
        }
    print(pred_lines)
    print(gt_lines)
    pred_lines = pred_lines[1:]
    gt_lines = gt_lines[1:]

    idx = -1
    error_report = {}
    for pred_l, gt_l in zip(pred_lines, gt_lines, strict=True):
        idx += 1
        # fast forward with the same lines
        if pred_l == gt_l:
            continue
        # categorize the gt lines
        if "Pick two numbers" in gt_l:  # pick numbers, it should follow the same order
            if pred_l != gt_l:
                error_report = f"""line: {idx} | gt: {pred_l} | pr: {gt_l}"""
                break
        elif "|- Try" in gt_l:  # try operation
            # everything up to the = should be the same, should be operating on the same numbers
            pred_eq = pred_l.split("=")[0]
            gt_eq = gt_l.split("=")[0]
            if pred_eq != gt_eq:
                error_report = f"""line: {idx} | gt: {pred_l} | pr: {gt_l}"""
                break
            # action should be the same
            dropping_in_gt = "drop this branch" in gt_l
            dropping_in_pred = "drop this branch" in pred_l
            if dropping_in_gt != dropping_in_pred:
                error_report = f"""line: {idx} | gt: {pred_l} | pr: {gt_l}"""
                break
            continue
        else:
            raise ValueError(f"Unknown line: {gt_l}")

    return idx / len(gt_lines), error_report


class CountdownMetric(SeaHelmMetric):
    """Metric implementation for the Countdown task.

    Extracts the final solution from model responses and evaluates both the
    exact solution correctness and a partial-accuracy score for the search
    procedure, aggregating into a metrics dictionary.
    """

    def __init__(self, dataloader: AbstractDataloader, task_config: TaskConfig):
        """Initialize the Countdown metric.

        Args:
            dataloader (AbstractDataloader): The task dataloader instance.
            task_config (TaskConfig): The task configuration.
        """
        super().__init__(dataloader=dataloader, task_config=task_config)
        self.regex_string = {"en": r"(?<=<Solution>)([\s\S]*)(?=</Solution>)"}[
            self.lang
        ]

    def extract_response(
        self,
        response: list[str],
        flags: re.RegexFlag = 0,
        return_original_response_on_failure: bool = False,
    ) -> dict[str, str | None]:
        """Extract the search procedure and the solution span from a response.

        The solution is extracted from within <Solution>...</Solution> tags.
        When extraction fails, `solution` is set to None (so extraction rate can
        be measured downstream).

        Args:
            response (list[str]): The list of turn responses. The first element
                is expected to contain the search procedure and solution.
            flags (re.RegexFlag, optional): Regex flags to use when extracting the answer. Defaults to 0.
            return_original_response_on_failure (bool, optional): Whether to
                fall back to the raw response text if extraction fails.
                Defaults to False for this task, to allow explicit tracking of
                extraction failures.

        Returns:
            dict[str, str | None]: A dict containing the raw search procedure
            and the extracted solution string (or None).
        """
        output = super().extract_response(
            response,
            flags=flags,
            return_original_response_on_failure=return_original_response_on_failure,
        )

        return {"search_procedure": response[0], "solution": output}

    def calculate_metrics(self) -> dict[str, float]:
        """Compute Countdown task metrics.

        Uses the postprocessed responses (containing both the search procedure
        and extracted solution) and ground-truth labels to compute:
        - accuracy: percentage of examples with a fully correct final solution
        - partial_accuracy: average fraction of correctly matched procedure lines
        - extraction_rate: percentage of examples where a solution span was extracted
        - normalized_accuracy: accuracy normalized to [0, 100]

        Side effects:
            - Updates `dataloader.dataframe` with `error_reports` and
              `individual_scores` (normalized partial accuracy per example).

        Returns:
            dict[str, float]: Aggregated metric scores.
        """
        predictions = self.dataloader.dataframe[self.postprocessed_response_column]
        references = self.dataloader.dataframe[self.label_column]

        accuracy_scores, partial_accuracy_scores, extraction_rate_scores = [], [], []
        error_reports = []
        for pred, ref in zip(predictions, references, strict=True):
            if pred["solution"] is None:
                extraction_rate_scores.append(0)
                accuracy_scores.append(0)
            else:
                extraction_rate_scores.append(1)
                is_correct = evaluate_countdown_final_solution(
                    list(ref["nums"]), ref["target"], pred["solution"]
                )

                if is_correct:
                    accuracy_scores.append(1)
                    partial_accuracy_scores.append(1)
                    error_reports.append(None)
                    continue
                else:
                    accuracy_scores.append(0)

            partial_accuracy, error_report = evaluate_countdown_search_procedure(
                pred["search_procedure"], ref["search_procedure"]
            )
            partial_accuracy_scores.append(partial_accuracy)
            error_reports.append(error_report)

        self.dataloader.dataframe["error_reports"] = error_reports
        self.dataloader.update_individual_scores(
            [
                {"normalized_partial_accuracy_score": self.normalize_score(x, 0, 1)}
                for x in partial_accuracy_scores
            ]
        )

        metric_dict = {
            "accuracy": 100 * sum(accuracy_scores) / len(accuracy_scores),
            "partial_accuracy": 100
            * sum(partial_accuracy_scores)
            / len(partial_accuracy_scores),
            "extraction_rate": 100
            * sum(extraction_rate_scores)
            / len(extraction_rate_scores),
            "normalized_accuracy": 100
            * self.normalize_score(sum(accuracy_scores) / len(accuracy_scores), 0, 1),
        }
        return metric_dict
