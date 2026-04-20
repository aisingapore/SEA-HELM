from src.dataloaders.seahelm_local_dataloader import SeaHelmLocalDataloader


class CulturalContentGenerationPromptDataLoader(SeaHelmLocalDataloader):
    """Dataloader that maps 'prompt_label' to 'label' for evaluation."""

    def load_dataset(self, limit: int | None = None) -> None:
        """Load dataset and map 'prompt_label' to 'label'.

        Args:
            limit (int, optional): Optional limit on the number of instances to load. Defaults to None.
        """
        super().load_dataset(limit)
        if "prompt_label" in self.dataset.column_names:
            self.dataset = self.dataset.rename_column("prompt_label", "label")


class CulturalContentGenerationResponseDataLoader(SeaHelmLocalDataloader):
    """Dataloader that maps 'response_label' to 'label' for evaluation."""

    def load_dataset(self, limit: int | None = None) -> None:
        """Load dataset and map 'response_label' to 'label'.

        Args:
            limit (int, optional): Optional limit on the number of instances to load. Defaults to None.
        """
        super().load_dataset(limit)
        if "response_label" in self.dataset.column_names:
            self.dataset = self.dataset.rename_column("response_label", "label")
