from datasets import Image, load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_image_dataloader import HuggingFaceImageDataloader
from src.task_config import TaskConfig

logger = get_logger(__name__)


class XM3600Dataloader(HuggingFaceImageDataloader):
    """Dataloader for the XM3600 multilingual vision-language dataset.

    XM3600 is a multilingual image captioning dataset that extends the original
    Crossmodal-3600 dataset with additional languages. This dataloader handles
    loading images and preparing them for vision-language model inference.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        default_num_in_context_examples: int,
        is_base_model: bool = False,
        model_name: str = "",
        run_base_path: str = "",
        inference_file_type: str = "jsonl",
    ):
        """Initialize the XM3600Dataloader.

        Args:
            task_config: TaskConfig object containing task-specific settings.
            default_num_in_context_examples: Default number of few-shot examples to use.
            is_base_model: Whether this is a base model (vs instruction-tuned).
            model_name: Name/path of the model being evaluated.
            run_base_path: Base path for storing inference results.
            inference_file_type: File format for inference results ('jsonl' or 'csv').
        """
        super().__init__(
            task_config,
            default_num_in_context_examples,
            is_base_model=is_base_model,
            model_name=model_name,
            run_base_path=run_base_path,
            inference_file_type=inference_file_type,
        )

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
                num_proc=16,
                remove_columns=[
                    "image_id",
                    "image_locale",
                    "captions_tokenized",
                    "captions_tokenized_lowercase",
                    "image",
                ],
            )

            self.dataset = dataset
