import functools
from multiprocessing import Pool
from typing import Any

from datasets import Image, load_dataset
from tqdm import tqdm

from src.base_logger import get_logger
from src.dataloaders.huggingface_image_dataloader import HuggingFaceImageDataloader
from src.dataloaders.judges.judge_dataloader import JudgeDataloader

logger = get_logger(__name__)


class XM3600Dataloader(HuggingFaceImageDataloader, JudgeDataloader):
    """Dataloader for the XM3600 multilingual vision-language dataset.

    XM3600 is a multilingual image captioning dataset that extends the original
    Crossmodal-3600 dataset with additional languages. This dataloader handles
    loading images and preparing them for vision-language model inference.
    """

    def load_dataset(self, limit: int = None):
        """Load the XM3600 dataset from HuggingFace datasets.

        Loads the dataset for the specified language, optionally limiting the number
        of samples. Images are decoded from their stored format to PIL Images.

        Args:
            limit: Optional limit on the number of instances to load.

        Note:
            The dataset is stored in self.dataset after loading and image decoding.
        """
        if self.dataset:
            logger.info("Dataset already loaded, skipping loading process")
            pass
        else:
            lang = self.lang
            if lang == "tl":
                lang = "fil"

            dataset = load_dataset(self.specific_task_config["filepath"], split=lang)
            dataset = dataset.cast_column("image", Image(decode=False))
            if limit is not None:
                dataset = dataset.select(range(limit))

            # Decode images
            dataset = dataset.map(
                lambda row: {
                    "id": row["image_id"],
                    "captions": row["captions"],
                    "prompts": [{}],
                    "images": [row["image"]["bytes"]],
                },
                num_proc=self.num_workers,
                remove_columns=[
                    "image_id",
                    "image_locale",
                    "captions_tokenized",
                    "captions_tokenized_lowercase",
                    "image",
                ],
            )

            self.dataset = dataset

    def prepare_conversations_for_judgements(
        self,
        metric: Any,
    ):
        logger.info(
            "Preparing judgement conversations for task '%s'",
            self.task_name.upper(),
        )

        _formatter = self.get_judge_prompt_formatter(metric=metric)
        images = self.dataset.select_columns(["images"])
        images = images.map(lambda x: {"image": x["images"][0]}, num_proc=16)
        images = images.cast_column("image", Image(decode=True))
        with Pool(self.num_workers) as p:
            outputs = list(
                tqdm(
                    p.starmap(
                        _formatter,
                        zip(
                            self.dataset, images, range(len(self.dataset)), strict=True
                        ),
                    ),
                    total=len(self.dataset),
                )
            )

        conversations = []
        custom_ids = []
        for row in outputs:
            conversations.extend(row["conversations"])
            custom_ids.extend(row["custom_ids"])

        return conversations, custom_ids

    def get_judge_prompt_formatter(self, metric):
        return functools.partial(
            self.judge_prompt_formatter,
            metric=metric,
        )

    @staticmethod
    def judge_prompt_formatter(row, image, idx, *, metric) -> list:
        """Static, picklable row formatter for multiprocessing.

        Returns the updated conversations list for a given dataset row.
        """
        raw_response = row["responses"]
        response = metric.extract_response(raw_response)

        conversation = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": response},
                        {"type": "image", "image": image["image"]},
                    ],
                },
                {"role": "assistant", "content": row["captions"]},
            ]
        ]

        custom_ids = [f"{idx}"]
        return {"conversations": conversation, "custom_ids": custom_ids}
