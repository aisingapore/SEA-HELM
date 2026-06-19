import base64
import json
import pickle
import re
import zlib
from concurrent.futures import ThreadPoolExecutor

from seahelm_tasks.code_generation.livecodebench.test_codes import (
    FUNCTION_TEST_CODE,
    STDIN_TEST_CODE,
)
from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.sandbox import get_sandbox
from src.task_config import TaskConfig

logger = get_logger(__name__)


class LiveCodeBenchMetric(SeaHelmMetric):
    """Metric class for calculating pass@1."""

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
        response_column: str = "responses",
        postprocessed_response_column: str = "cleaned_responses",
        label_column: str = "label",
    ):
        """Initialize the F1AccMetric.

        Args:
            dataloader (AbstractDataloader): The dataloader to use.
            task_config (TaskConfig): The task configuration.
            response_column (str, optional): The column name for model responses. Defaults to "responses".
            postprocessed_response_column (str, optional): The column name for postprocessed model responses. Defaults to "cleaned_responses".
            label_column (str, optional): The column name for labels. Defaults to "label".
        """
        super().__init__(
            dataloader=dataloader,
            task_config=task_config,
            response_column=response_column,
            postprocessed_response_column=postprocessed_response_column,
            label_column=label_column,
        )
        self.sandbox = get_sandbox(self.task_config.sandbox_type)

    def extract_response(
        self,
        response: list,
        flags: re.RegexFlag = re.IGNORECASE,
        return_original_response_on_failure: bool = True,
    ) -> str | int:
        """Extract the output answer from the model's response.

        Args:
            response (list): The model's response.
            flags (re.RegexFlag, optional): Regex flags to use when extracting the answer. Defaults to 0.
            return_original_response_on_failure (bool, optional): Whether to return the original response on failure. Defaults to True.

        Returns:
            str | int: The extracted output answer.
        """
        try:
            pattern = r"```(?:python)?\n((?:[^`]|`(?!``))*?)\n```"
            matches = re.findall(pattern, response[0], flags)
            output = matches[-1].strip()
        except Exception:
            if return_original_response_on_failure:
                output = response[0]
            else:
                return ""

        return output

    def run_test_cases(self, row, idx) -> tuple[bool, int]:
        pred = row[self.postprocessed_response_column]

        test_cases = row["test_cases"]
        public_test_cases = test_cases["public"]
        private_test_cases = json.loads(
            pickle.loads(
                zlib.decompress(base64.b64decode(test_cases["private"].encode("utf-8")))
            )
        )

        _test_case = list(public_test_cases) + private_test_cases

        passed_count = 0
        all_passed = True
        for i, test in enumerate(_test_case):
            if row["metadata"]["func_name"]:
                test_code = FUNCTION_TEST_CODE.format(
                    test_input=test["input"],
                    expected_output=test["output"].strip(),
                    code=pred,
                    fn_name=row["metadata"]["func_name"],
                )
                result = self.sandbox.run_code(test_code, run_timeout=6)
                if result.get("status").lower() == "success":
                    output = result["run_result"]["stdout"]
                    if "TEST_PASSED" in output:
                        passed_count += 1
                    elif "TEST_FAILED:" in output:
                        # Extract failure details from output
                        for line in output.split("\n"):
                            if line.startswith("TEST_FAILED:"):
                                logger.info("Row %d Test case %d failed", idx, i)
                                break
                        all_passed = False
                        break
                    elif "EXECUTION_ERROR:" in output:
                        # Extract error details
                        for line in output.split("\n"):
                            if line.startswith("EXECUTION_ERROR:"):
                                logger.info(
                                    "Row %d Test case %d execution error", idx, i
                                )
                                break
                        all_passed = False
                        break
                    else:
                        logger.error(
                            "Row %d Test case %d: Unknown error in output.",
                            idx,
                            i,
                        )
                        all_passed = False
                        break
                else:
                    logger.error(
                        "Row %d Test case %d: Sandbox execution failed.",
                        idx,
                        i,
                    )
                    all_passed = False
                    break
            else:
                test_code = STDIN_TEST_CODE.format(
                    test_input=test["input"],
                    code=pred,
                )
                result = self.sandbox.run_code(test_code, run_timeout=6)
                if result.get("status").lower() != "success":
                    logger.error(
                        "Row %d Test case %d: Sandbox execution failed.",
                        idx,
                        i,
                    )
                    all_passed = False
                    break

                # Compare output
                actual_output = (
                    result["run_result"]["stdout"].strip()
                    if "run_result" in result
                    else ""
                )
                expected_output = test["output"].strip()

                if str(actual_output) == expected_output:
                    passed_count += 1
                else:
                    logger.info("Row %d Test case %d failed.", idx, i)
                    all_passed = False
                    break

        return all_passed, passed_count

    def calculate_metrics(self) -> dict:
        """Calculate the pass@1 metric.

        Returns:
            dict: A dictionary containing the pass@1 metric.
        """
        self.sandbox.start_sandbox()

        pairs = list(self.dataloader.dataframe.iterrows())
        indices = [idx for idx, _ in pairs]
        rows = [row for _, row in pairs]

        with ThreadPoolExecutor(max_workers=self.dataloader.num_workers) as executor:
            results = list(executor.map(self.run_test_cases, rows, indices))

        all_passes = [r[0] for r in results]

        self.dataloader.update_individual_scores(
            [{"pass@1": int(x)} for x in all_passes]
        )

        # remove private test set due to size
        self.dataloader.dataframe["test_cases"] = self.dataloader.dataframe[
            "test_cases"
        ].apply(
            lambda x: {
                "public": x["public"],
                "private": "PRIVATE_TEST_SET",
            }
        )

        pass_1 = sum(all_passes) / len(self.dataloader.dataframe)
        logger.info("Pass@1 = %.2f", pass_1 * 100)
        metric_dict = {
            "pass@1": 100 * pass_1,
        }
        return metric_dict
