from abc import abstractmethod
from typing import Any

from src.base_logger import get_logger

logger = get_logger(__name__)


class JudgeDataloader:
    """This class contains the logic for preparing conversations for judge models and generating file paths for storing judge batches and responses.

    This should not be used directly; instead, specific judge dataloaders (e.g. SeaHelmLocalCriteriaDataloader) should inherit from this class and implement the get_judge_prompt_formatter method to format conversations according to the requirements of different judge models.
    """

    def get_judge_batch_filepath(self, turn: int = 1, file_type: str = "jsonl") -> str:
        """Generate the file path for storing judge batches for a specific turn.

        Args:
            turn (int): The conversational turn number (1-indexed).
            file_type (str): File format ('jsonl' or 'csv').

        Returns:
            str: Full file path for storing judge batches for the specified turn.
        """
        return self.get_filepath_creator("judge_batch")(turn, file_type)

    def get_judge_batch_response_filepath(
        self, turn: int = 1, file_type: str = "jsonl"
    ) -> str:
        """Generate the file path for storing judge batch responses for a specific turn.

        Args:
            turn (int): The conversational turn number (1-indexed).
            file_type (str): File format ('jsonl' or 'csv').

        Returns:
            str: Full file path for storing judge batch responses for the specified turn.
        """
        return self.get_filepath_creator("judge_batch_response")(turn, file_type)

    def prepare_conversations_for_judgements(
        self,
        metric: Any,
    ):
        """Prepare judge-formatted conversations for all examples in the dataset.

        Applies the judge prompt formatter to each row in the dataset and collects
        the resulting conversations and custom IDs for submission to a judge model.

        Args:
            metric (Any): The metric object that provides the judge prompt formatter.

        Returns:
            tuple[list, list]: (conversations, custom_ids) where conversations is a list
                of judge-formatted conversation turns and custom_ids is a list of
                corresponding unique identifiers.
        """
        logger.info(
            "Preparing judgement conversations for task '%s'",
            self.task_name.upper(),
        )

        _judge_formatter = self.get_judge_prompt_formatter(metric)
        conversations = []
        custom_ids = []
        for row in self.dataset:
            formatted = _judge_formatter(row, 0)
            conversations.extend(formatted["conversations"])
            custom_ids.extend(formatted["custom_ids"])

        return conversations, custom_ids

    @abstractmethod
    def get_judge_prompt_formatter(self, metric) -> Any:
        """Return a formatter function that builds judge conversations for each dataset row.

        This method must be implemented by subclasses to define how to construct judge
        conversations based on the dataset structure and the requirements of the judge
        model.

        Args:
            metric: The metric object that may provide necessary information for formatting.

        Returns:
            A function that accepts a dataset row and its index and returns a dict with keys
            'conversations' (list of message lists) and 'custom_ids' (list of str identifiers).
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Please implement get_judge_prompt_formatter method."
        )
