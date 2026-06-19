import re
import string

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


# code from natural plan benchmark
def _parse_response(response: str) -> list[tuple[str, int]]:
    """Parse the model response into a structured itinerary.
    Returns a parsed plan in a list of `(city, stay_days)` tuples.

    Args:
        response (str): Raw response from the model.

    Returns:
        list[tuple[str, int]]: Structured plan after parsing. Returns an empty
            list if parsing fails or the required elements are not found.
    """
    pattern_visit = r"\d+-\d+"
    pattern_flight = r".*Day (\d+).*from (\w+) to (\w+)"
    pattern_days = r"European cities for (\d+) days"

    days, flights, flight_days = [], [], []
    total_days = None
    for piece in response.split("\n"):
        days_match = re.findall(pattern_days, piece)
        if days_match:
            total_days = int(days_match[0])

        visit_match = re.findall(pattern_visit, piece)
        if visit_match:
            days.append(visit_match[0])
            end_day = int(visit_match[0].split("-")[1])
            # Reach the end of the plan, stop to avoid parsing alternative plans.
            if end_day == total_days:
                break
        flight_match = re.findall(pattern_flight, piece)
        if flight_match:
            flights.append(flight_match[0])

    visit_cities, parsed_plan = [], []
    for flight_day, begin_city, end_city in flights:
        flight_days.append(int(flight_day))
        if not visit_cities:
            visit_cities.append(begin_city)
            visit_cities.append(end_city)
        else:
            visit_cities.append(end_city)

    if not days or not flights or not visit_cities:
        return []
    last_day = int(days[-1].split("-")[1])
    flight_days = [1] + flight_days + [last_day]
    for i, visit_city in enumerate(visit_cities):
        city_stay = flight_days[i + 1] - flight_days[i] + 1
        parsed_plan.append((visit_city, city_stay))

    return parsed_plan


def evaluate_travel_plan_solution(cities: str, durations: str, response: str) -> float:
    """Compute the example-level exact-match accuracy (0 or 1).

    This compares the parsed plan from `response` against the ground-truth
    cities and durations. If the predicted plan exactly matches the ground-truth
    sequence and stays, the score is 1.0; otherwise 0.0.

    Args:
        cities (str): Cities in the format "city1**city2**city3".
        durations (str): Durations in the format "1**2**3" corresponding to
            each city in `cities`.
        response (str): Raw model response containing the plan.

    Returns:
        float: Exact-match accuracy of 0.0 (mismatched) or 1.0 (matched).
    """

    stays = [x for x in cities.split("**") if x]
    days = [int(x) for x in durations.split("**") if x]
    parsed_plan = _parse_response(response)
    num_stays = min(len(stays), len(parsed_plan))
    num_match = 0
    for i in range(num_stays):
        if stays[i] == parsed_plan[i][0] and days[i] == parsed_plan[i][1]:
            num_match += 1
        else:
            break
    hard_score = 0.0 if num_match / len(stays) < 1.0 else 1.0
    return hard_score


_ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _normalize_line(line: str):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return _ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(line))))


def evaluate_travel_plan_search_procedure(
    output: str, gt_procedure: str
) -> tuple[float, str]:
    """Evaluate the match between predicted and ground-truth solving procedures.

    The procedure sections are compared line-by-line after normalization.
    Returns a partial-accuracy score based on the index of the first mismatch
    divided by the total number of ground-truth lines, as well as a short
    error report string describing the first mismatch.

    Args:
        output (str): Model output containing the solving procedure section.
        gt_procedure (str): Ground-truth procedure containing the reference
            steps to compare against.

    Returns:
        tuple[float, str]:
            - partial_accuracy (float): Ratio of correct leading lines in order.
            - error_report (str): Description of the first mismatch, or
              "empty_output" if nothing is present, or an empty string when
              there is no mismatch.
    """
    # remove some unnecessary information
    gt_procedure = gt_procedure.replace("Output the plan in the required format:", "")
    output = output.replace("Output the plan in the required format:", "")
    gt_procedure = "<Solving Procedure>" + gt_procedure.split("<Solving Procedure>")[1]
    if "<Solving Procedure>" in output:
        output = "<Solving Procedure>" + output.split("<Solving Procedure>")[1]

    pred_lines = output.strip().split("\n")
    gt_lines = gt_procedure.strip().split("\n")

    pred_lines = [line.rstrip() for line in pred_lines if line.strip()]
    gt_lines = [line.rstrip() for line in gt_lines if line.strip()]

    idx = -1
    error_report: str = ""
    for pred_l, gt_l in zip(pred_lines, gt_lines, strict=True):
        idx += 1
        # fast forward with the same lines
        _pred_l = _normalize_line(pred_l)
        _gt_l = _normalize_line(gt_l)
        if _gt_l in _pred_l:
            # print(gt_l)
            continue
        else:
            error_report = f"line: {idx} | gt: {pred_l} | pr: {gt_l}"
            break
    if idx < 0:
        idx = 0
        error_report = "empty_output"

    return idx / len(gt_lines), error_report


class TravelPlanningMetric(SeaHelmMetric):
    """Metric for evaluating travel planning task outputs.

    This metric extracts the `<Plan>...</Plan>` section from model responses,
    evaluates exact-match accuracy against ground-truth plans, and computes
    a partial accuracy score based on the solving procedure similarity.
    """

    def __init__(self, dataloader: AbstractDataloader, task_config: TaskConfig):
        """Initialize the metric.

        Args:
            dataloader (AbstractDataloader): The dataloader containing the
                inferences and labels.
            task_config (TaskConfig): The task configuration.
        """
        super().__init__(dataloader=dataloader, task_config=task_config)
        self.regex_string: str = {"en": r"(?<=<Plan>)([\s\S]*)(?=</Plan>)"}[self.lang]

    def extract_response(self, response: list) -> dict[str, str | None]:
        """Extract and structure the plan and the original response.

        Args:
            response (list): The raw response list where the first element is the
                text output from the model for this example/turn.

        Returns:
            dict[str, str | None]: A dictionary with the following keys:
                - "original": The original response string.
                - "plan": The extracted plan between `<Plan>...</Plan>` or
                  `None` if extraction fails.
        """
        output = super().extract_response(
            response, return_original_response_on_failure=False
        )
        return {"original": response[0], "plan": output}

    def calculate_metrics(self) -> dict[str, float]:
        """Compute accuracy, partial accuracy, extraction rate, and normalized accuracy.

        Returns:
            dict[str, float]: A dictionary with percentage metrics, including:
                - "accuracy": Exact-match accuracy (0-100).
                - "partial_accuracy": Line-by-line partial accuracy of procedure (0-100).
                - "extraction_rate": Share of responses with successfully extracted plans (0-100).
                - "normalized_accuracy": Normalized accuracy in percentage (0-100).
        """
        predictions = self.dataloader.dataframe[self.postprocessed_response_column]
        references = self.dataloader.dataframe[self.label_column]

        accuracy_scores, partial_accuracy_scores, extraction_rate_scores = [], [], []
        error_reports = []
        for pred, ref in zip(predictions, references, strict=True):
            if pred["plan"] is None:
                extraction_rate_scores.append(0)
                accuracy_scores.append(0)
            else:
                extraction_rate_scores.append(1)

                accuracy = evaluate_travel_plan_solution(
                    ref["ground_truth_cities"],
                    ref["ground_truth_durations"],
                    pred["plan"],
                )

                if accuracy == 1.0:
                    accuracy_scores.append(1)
                    partial_accuracy_scores.append(1)
                    continue
                else:
                    accuracy_scores.append(0)

            partial_accuracy, error_report = evaluate_travel_plan_search_procedure(
                pred["original"], ref["solving_procedure"]
            )
            partial_accuracy_scores.append(partial_accuracy)
            error_reports.append(error_report)

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
