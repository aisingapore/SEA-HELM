import datasets

from src.base_logger import get_logger
from src.dataloaders.huggingface_dataloader import HuggingFaceDataloader

logger = get_logger(__name__)

LABEL_MAP = {
    "entailment": "A",
    "contradiction": "B",
    "neutral": "C",
}

LANG_LABEL_MAP = {
    "my": {
        "entailment": "က",
        "contradiction": "ခ",
        "neutral": "ဂ",
    },
}

CULTURE_MAP = {
    "id": "indonesian",
    "vi": "vietnamese",
    "th": "thai",
    "ta": "singaporean",
    "tl": "filipino",
    "my": "myanmar",
    "ms": "malaysian",
    "kh": "cambodian",
}


class SEANLIDataloader(HuggingFaceDataloader):
    """Dataloader for SEA-NLI tasks.

    SEA-NLI stores all cultures in a single split. Each language config must specify
    a `culture` key to filter the relevant rows, and a `dataset_subset` key (normal/hard).
    """

    def load_dataset(self, limit: int | None = None) -> datasets.Dataset:
        if self.dataset is not None:
            logger.info("Dataset already loaded, skipping loading process")
            return self.dataset

        filepath = self.specific_task_config["filepath"]
        dataset_subset = self.specific_task_config["dataset_subset"]
        culture = CULTURE_MAP.get(self.lang)
        if culture is None:
            raise ValueError(
                f"No culture mapping for language '{self.lang}'. Update CULTURE_MAP in sea_nli_dataloader.py."
            )

        logger.info(
            "Loading SEA-NLI subset='%s', culture='%s' from %s",
            dataset_subset,
            culture,
            filepath,
        )

        dataset = datasets.load_dataset(filepath, dataset_subset, split="test")
        dataset = dataset.filter(lambda row: row["culture"] == culture)

        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))

        lang_label_map = LANG_LABEL_MAP.get(self.lang, LABEL_MAP)

        def _map_row(row: dict, idx: int) -> dict:
            label = lang_label_map.get(row["true_label"].strip().lower())
            return {
                "id": str(row["entry_id"]),
                "prompts": [
                    {
                        "sentence1": row["premise_native"],
                        "sentence2": row["hypothesis_native"],
                    }
                ],
                "label": label,
                "metadata": {
                    "source": filepath,
                    "subset": dataset_subset,
                    "culture": culture,
                    "concept_title": row.get("concept_title", ""),
                    "concept_category": row.get("concept_category", ""),
                },
            }

        self.dataset = dataset.map(
            _map_row,
            with_indices=True,
            num_proc=self.num_workers,
            remove_columns=dataset.column_names,
        )

        logger.info(
            "Loaded %d SEA-NLI examples (subset=%s, culture=%s)",
            len(self.dataset),
            dataset_subset,
            culture,
        )
        return self.dataset

    def load_example_dataset(self, limit: int | None = None) -> None:
        self.example_dataset = None

    def get_num_turns(self) -> int:
        return 1
