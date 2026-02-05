import logging
import math
import re
import statistics
import string
from abc import abstractmethod

import pandas as pd

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.task_config import TaskConfig

logger = get_logger(__name__)


class SeaHelmMetric:
    """Base class for SEA-HELM metrics."""

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
        response_column: str = "responses",
        postprocessed_response_column: str = "cleaned_response",
        label_column: str = "label",
    ):
        """Initialize the SeaHelmMetric.

        Args:
            dataloader (AbstractDataloader): The dataloader to use.
            task_config (TaskConfig): The task configuration.
            response_column (str, optional): The column name for the responses. Defaults to "responses".
            postprocessed_response_column (str, optional): The column name for the postprocessed responses. Defaults to "cleaned_response".
            label_column (str, optional): The column name for the labels. Defaults to "label".
        """
        self.task_config = task_config
        self.task = task_config.task_name
        self.lang = task_config.lang
        self.dataloader = dataloader
        self.response_column = response_column
        self.postprocessed_response_column = postprocessed_response_column
        self.label_column = label_column

    def get_response(self, row: pd.Series) -> list[str]:
        """Get the response from the row.

        Args:
            row (Series): The row to get the response from.

        Returns:
            list[str]: The response from the row.
        """
        return row[self.response_column]

    def get_response_counts(self) -> dict:
        """Get the response counts from the dataloader.

        Returns:
            dict: The response counts.
        """
        logger.debug("Response Value Counts:")
        counts = self.dataloader.inference_df.apply(
            self.get_response, axis=1
        ).value_counts()
        logger.debug(counts)
        return counts.to_json()

    def drop_error_responses(self) -> None:
        """Drop the error responses from the dataloader."""
        should_drop = []
        for response_list in self.dataloader.inference_df[self.response_column]:
            drop = False
            for response in response_list:
                if response is None:
                    drop = True
                    break
            should_drop.append(drop)

        self.dataloader.inference_df = self.dataloader.inference_df[
            ~pd.Series(should_drop)
        ].copy()

    def replace_error_responses(self, replacement: str = "") -> None:
        """Replace the error responses with the replacement string.

        Args:
            replacement (str, optional): The replacement string. Defaults to "".
        """
        self.dataloader.inference_df[self.response_column] = (
            self.dataloader.inference_df[self.response_column].map(
                lambda response: [
                    replacement if item is None else item for item in response
                ]
            )
        )

    def normalize_answer(self, s: str) -> str:
        """Lower text and remove punctuation and extra whitespace.

        Args:
            s (str): The string to normalize.

        Returns:
            str: The normalized string.
        """

        #########################################################################
        # TODO:
        # 1. Support stripping punctuation of non ASCII characters
        # 2. Consider whether need to strip punctuation like ')', etc.
        # 3. Better to use regex to rewrite this function
        #########################################################################
        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            # Remove punctuations + trailing whitespace/newline
            return text.strip(string.punctuation + " " + "\n")

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))

    @abstractmethod
    def calculate_metrics(self) -> dict:
        """
        Calculates the metric score. Needs to be implemented by subclasses.

        Args:
            self (SeaHelmMetric): The SeaHelmMetric instance.

        Returns:
            dict: A dictionary with metric scores

            Eg:
            (
                metric_dict = {
                    "normalisedscore": 100 * score,
                }
            )
        """
        raise NotImplementedError("Subclasses must implement this method")

    def calculate_stderr(self, values: list) -> float:
        """Calculate the standard error of the mean.

        Args:
            values (list): The values to calculate the standard error of the mean.

        Returns:
            float: The standard error of the mean.
        """
        if len(values) <= 1:
            return 0.0
        return math.sqrt(statistics.variance(values) / len(values))

    def extract_response(
        self,
        response: list,
        flags: re.RegexFlag = 0,
        return_original_response_on_failure: bool = True,
    ) -> str:
        """Extract the output from the model's response.

        Args:
            response (list): The model's response.
            flags (re.RegexFlag, optional): Regex flags to use when extracting the answer. Defaults to 0.
            return_original_response_on_failure (bool, optional): Whether to return the original response on failure. Defaults to True.

        Returns:
            str: The extracted output.
        """
        # try to extract the answer from the response using regex else return the response as it is
        try:
            output = re.search(self.regex_string, response[0], flags=flags).group(1)
            assert output is not None
        except Exception:
            if return_original_response_on_failure:
                output = response[0]
            else:
                return None

        return output.strip()

    def evaluate_responses(
        self, drop_error_response: bool = False
    ) -> dict[str, dict[str, float]]:
        """Evaluate the responses.

        Args:
            drop_error_response (bool, optional): Whether to drop error responses. Defaults to False.

        Returns:
            dict[str, dict[str, float]]: A dictionary containing the metric scores.
                - The outer key is the task name, the inner key is the metric name, and the value is the metric score.
        """
        if drop_error_response:
            logger.info("Dropping error responses")
            self.drop_error_responses()
        else:
            logger.info('Replacing error responses with ""')
            self.replace_error_responses()

        logger.info("Post processing responses...")
        self.postprocess_responses()
        logger.info("Post processing of responses completed!")

        if logger.isEnabledFor(logging.DEBUG):
            self.get_response_counts()
        logger.info("Calculating metrics...")
        output_json = self.calculate_metrics()
        logger.info("Metrics calculation completed!")
        metric = {self.task: output_json}
        return metric

    def postprocess_responses(self) -> None:
        """Postprocess the responses."""
        self.dataloader.inference_df[self.postprocessed_response_column] = (
            self.dataloader.inference_df[self.response_column].map(
                self.extract_response
            )
        )

    def normalize_score(
        self, score: float, min_score: float, max_score: float
    ) -> float:
        """Normalize the score.

        Args:
            score (float): The score to normalize.
            min_score (float): The minimum score.
            max_score (float): The maximum score.

        Returns:
            float: The normalized score.
        """
        if min_score == max_score:
            return 0.0

        normalized_score = max((score - min_score) / (max_score - min_score), 0)
        return normalized_score
