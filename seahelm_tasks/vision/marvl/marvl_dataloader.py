from datasets import Image, load_dataset

from src.base_logger import get_logger
from src.dataloaders.huggingface_image_dataloader import HuggingFaceImageDataloader
from src.task_config import TaskConfig

try:
    import opencc

    opencc_converter = opencc.OpenCC("t2s.json")
except ImportError:
    opencc_converter = None

logger = get_logger(__name__)


# TODO Marvl ZH is in traditional chinese, so there is a need to convert it to simplified chinese.
class MarvlDataloader(HuggingFaceImageDataloader):
    """Dataloader for the MARVL (Multicultural Reasoning over Vision and Language) dataset.

    MARVL is a benchmark for evaluating multicultural visual reasoning capabilities
    of vision-language models. The task involves determining whether a given textual
    hypothesis is true or false based on a pair of images (left and right images).
    This tests the model's ability to understand visual content and perform logical
    reasoning across different cultural contexts.

    The dataset supports multiple languages including English (en), Indonesian (id),
    Tamil (ta), and Chinese (zh). Each sample contains two images and a hypothesis
    statement, with the model required to output True or False.

    Note:
        MARVL ZH uses traditional Chinese, which is automatically converted to
        simplified Chinese during dataset loading for consistency.
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
        """Initialize the MarvlDataloader.

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
        self.label_map = {
            "en": {True: "True", False: "False"},
            "id": {True: "Benar", False: "Salah"},
            "ta": {True: "உண்மை", False: "பொய்"},
            "zh": {True: "正确", False: "错误"},
        }

    def load_dataset(self, limit: int = None):
        """Load the MARVL dataset from HuggingFace datasets.

        Loads the dataset from the configured filepath, optionally limiting the number
        of samples. The dataset contains image pairs (left and right images) with
        corresponding hypotheses that need to be evaluated as True or False.

        For Chinese language samples, automatically converts traditional Chinese text
        to simplified Chinese. Processes images and labels according to the language-specific
        label mappings defined in self.label_map.

        Args:
            limit: Optional limit on the number of instances to load.

        Note:
            The dataset is stored in self.dataset after loading and processing.
        """
        if self.dataset:
            logger.info("Dataset already loaded, skipping loading process")
            pass
        else:
            if self.lang == "en":
                # EN data uses the machine translation of the ID dataset
                lang = "id"
            else:
                lang = self.lang

            dataset = load_dataset(self.specific_task_config["filepath"], split=lang)
            dataset = dataset.cast_column("left_img", Image(decode=False)).cast_column(
                "right_img", Image(decode=False)
            )
            if limit is not None:
                dataset = dataset.select(range(limit))

            if lang == "zh":
                assert opencc_converter is not None, (
                    "OpenCC converter is not available, please check if opencc is installed."
                )
                dataset = dataset.map(
                    lambda sample: {
                        "hypothesis": opencc_converter.convert(sample["hypothesis"]),
                    },
                    num_proc=self.num_workers,
                )

            dataset = dataset.map(
                lambda row: {
                    "id": row["id"],
                    "label_t": self.label_map[lang][row["label"]],
                    "prompts": [
                        {
                            "hypothesis": row["hypothesis"]
                            if self.lang != "en"
                            else row["hypo_en"]
                        }
                    ],
                    "images": [
                        row["left_img"]["bytes"],
                        row["right_img"]["bytes"],
                    ],
                    "metadata": {
                        "chapter": row["chapter"],
                        "concept": row["concept"],
                        "language": row["language"],
                    },
                },
                remove_columns=[
                    "label",
                    "hypo_en",
                    "hypothesis",
                    "left_img_id",
                    "right_img_id",
                    "annotator_info",
                    "left_img",
                    "right_img",
                    "resized_left_img",
                    "resized_right_img",
                    "vertically_stacked_img",
                    "horizontally_stacked_img",
                ],
                num_proc=self.num_workers,
            ).rename_column("label_t", "label")

            self.dataset = dataset
