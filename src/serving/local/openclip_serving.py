import collections
import os
from typing import Any

import numpy as np
import open_clip
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

from src.base_logger import get_logger
from src.serving.local.base_serving import BaseServing

logger = get_logger(__name__)


class CLIPImageDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for processing images with CLIP preprocessing.

    This dataset applies CLIP preprocessing transformations to images for feature extraction.

    Args:
        data: List of image data (PIL Images or image paths)
        preprocess: CLIP preprocessing function for image normalization and resizing
    """

    def __init__(self, data, preprocess):
        self.data = data
        self.preprocess = preprocess

    def __getitem__(self, idx):
        """Get preprocessed image at given index.

        Args:
            idx (int): Index of the image to retrieve

        Returns:
            dict: Dictionary containing preprocessed image tensor with key 'image'
        """
        image = self.data[idx]
        image = self.preprocess(image)
        return {"image": image}

    def __len__(self):
        """Get the total number of images in the dataset.

        Returns:
            int: Number of images in the dataset
        """
        return len(self.data)


class CLIPCapDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for processing text captions with CLIP tokenization.

    This dataset tokenizes text captions for CLIP text encoder processing.

    Args:
        data: List of text captions (strings)
        tokenizer: CLIP tokenizer for text preprocessing
    """

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        """Get tokenized caption at given index.

        Args:
            idx (int): Index of the caption to retrieve

        Returns:
            dict: Dictionary containing tokenized caption tensor with key 'caption'
        """
        c_data = self.data[idx]
        c_data = self.tokenizer(c_data).squeeze()
        return {"caption": c_data}

    def __len__(self):
        """Get the total number of captions in the dataset.

        Returns:
            int: Number of captions in the dataset
        """
        return len(self.data)


class OpenClipServing(BaseServing):
    """
    A serving class that uses OpenClip for image and text embeddings.

    This class provides methods for generating responses from language models using the OpenClip.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        num_workers: int = 8,
        device: str = "cuda",
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.device = device
        self.is_model_loaded = False

    def load_model(self) -> None:
        """Load OpenClip tokenizer and model onto CUDA and set eval mode."""
        if self.is_model_loaded:
            # no op as model is already loaded
            return
        else:
            cache_path = snapshot_download(repo_id=self.model_name)
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "xlm-roberta-large-ViT-H-14",
                pretrained=os.path.join(cache_path, "open_clip_pytorch_model.bin"),
                precision="fp32",  # TODO fix fp16 issue
                device=self.device,
            )
            self.model.eval()

            self.tokenizer = open_clip.get_tokenizer("xlm-roberta-large-ViT-H-14")
            self.is_model_loaded = True

    def generate(
        self, messages: list, logprobs: bool = False, **generation_kwargs
    ) -> Any:
        raise NotImplementedError("Use batch_generate for OpenClipServing.")

    def batch_generate(
        self, batch_messages: list[list], logprobs: bool = False, **generation_kwargs
    ) -> list[Any]:
        """Generate responses for a given batch of messages.

        Args:
            batch_messages (list[list]): The batch of messages to generate responses for.
            logprobs (bool, optional): Whether to return log probabilities. Defaults to False.
            **generation_kwargs: Additional generation kwargs.

        Returns:
            list[Any]: The generated responses.

        Note:
            Format of batch messages:
            [[
                {"role": "user", "content": [{"type": "text", "text": "<prediction>"}, {"type": "image", "image": <PIL Image>}]},
                {"role": "assistant", "content": "<reference caption>"},
            ]]
        """
        images = []
        predictions = []
        references = []

        for messages in batch_messages:
            images.append(messages[0]["content"][1]["image"])
            predictions.append(messages[0]["content"][0]["text"])
            references.append(messages[1]["content"])

        image_features = self.extract_all_images(images)

        logger.info("Calculating CLIPScore...")
        # Calculate CLIPScore (image-text similarity)
        per_instance_image_text, candidate_features = self.get_clip_score(
            image_features, predictions
        )

        logger.info("Calculating RefCLIPScore...")
        # Calculate RefCLIPScore (text-text similarity with references)
        per_instance_text_text = self.get_refonlyclipscore(
            references, candidate_features
        )

        responses = [
            {"score": {"per_instance_image_text": it, "per_instance_text_text": tt}}
            for it, tt in zip(
                per_instance_image_text, per_instance_text_text, strict=True
            )
        ]
        return responses

    def get_response(self, output: dict) -> str:
        """Get the response from the output.

        Args:
            output (dict): The output to get the response from.

        Returns:
            str: The response from the output.
        """
        # no op here, just return the output as is
        return output["score"]

    def cleanup(self) -> None:
        """Cleanup any resources used by the serving class."""
        if not self.is_model_loaded:
            logger.info("Model is not loaded; no cleanup necessary.")
            return

        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        torch.cuda.empty_cache()

    def extract_all_images(self, images):
        """Extract CLIP image features from a list of images.

        Processes images in batches using the CLIP vision encoder to extract dense feature
        representations. Features are normalized and returned as a numpy array.

        Args:
            images: List of images (PIL Images or image paths)

        Returns:
            np.ndarray: Array of shape (num_images, feature_dim) containing image features

        Example:
            >>> features = extract_all_images(images, preprocess)
            >>> print(features.shape)  # (100, 1024) for 100 images
        """
        data = torch.utils.data.DataLoader(
            CLIPImageDataset(images, self.preprocess),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        all_image_features = []
        with torch.no_grad():
            for b in tqdm(data):
                b = b["image"].to(self.device)
                # if device == "cuda":
                #     b = b.to(torch.float16)
                all_image_features.append(self.model.encode_image(b).cpu().numpy())
        all_image_features = np.vstack(all_image_features)
        return all_image_features

    def get_clip_score(self, images, candidates, w=2.5):
        """Calculate CLIPScore between images and candidate captions.

        CLIPScore measures the semantic similarity between images and text using CLIP embeddings.
        The score is computed as the cosine similarity between normalized image and text features,
        scaled by a weight factor and clipped to positive values.

        Args:
            images: Either a list of image paths/PIL Images or precomputed image features matrix
            candidates (list): List of candidate caption strings
            w (float, optional): Weight factor for scaling similarities. Defaults to 2.5.

        Returns:
            tuple: A tuple containing:
                - per (np.ndarray): Per-instance CLIPScores of shape (num_instances,)
                - candidates (np.ndarray): Normalized candidate text features

        Example:
            >>> scores, features = get_clip_score(images, captions)
            >>> print(f"Average CLIPScore: {np.mean(scores):.3f}")
        """
        if isinstance(images, list):
            # need to extract image features
            images = self.extract_all_images(images)

        candidates = self.extract_all_captions(candidates)

        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

        per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
        return per, candidates

    def extract_all_captions(self, captions):
        """Extract CLIP text features from a list of captions.

        Processes text captions in batches using the CLIP text encoder to extract dense feature
        representations. Captions are tokenized and encoded into feature vectors.

        Args:
            captions (list): List of caption strings

        Returns:
            np.ndarray: Array of shape (num_captions, feature_dim) containing text features

        Example:
            >>> captions = ["A cat sitting on a table", "A dog running in the park"]
            >>> features = extract_all_captions(captions)
            >>> print(features.shape)  # (2, 1024) for 2 captions
        """
        data = torch.utils.data.DataLoader(
            CLIPCapDataset(captions, self.tokenizer),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        all_text_features = []
        with torch.no_grad():
            for b in tqdm(data):
                b = b["caption"].to(self.device)
                all_text_features.append(self.model.encode_text(b).cpu().numpy())
        all_text_features = np.vstack(all_text_features)
        return all_text_features

    def get_refonlyclipscore(self, references, candidates):
        """Calculate reference-only CLIPScore between candidates and reference captions.

        This function computes the text-text similarity component of RefCLIPScore by measuring
        how well candidate captions align with reference captions using CLIP text embeddings.
        For each candidate, it finds the maximum similarity with any of its reference captions.

        Args:
            references (list): List of lists, where each inner list contains reference captions
                            for the corresponding candidate (e.g., [["ref1", "ref2"], ["ref3"]])
            candidates: Either list of candidate caption strings or precomputed candidate features

        Returns:
            list: Per-instance reference similarity scores, where each score is the maximum
                cosine similarity between a candidate and its reference captions

        Example:
            >>> references = [["A cat on table", "Cat sitting"], ["Dog in park"]]
            >>> candidates = ["A feline on the table", "A canine running"]
            >>> scores = get_refonlyclipscore(model, references, candidates, 'cuda', tokenizer)
            >>> print(f"Reference similarities: {scores}")

        Note:
            The function handles multiple references per candidate by taking the maximum
            similarity score across all references for each candidate.
        """
        if isinstance(candidates, list):
            candidates = self.extract_all_captions(candidates)

        flattened_refs = []
        flattened_refs_idxs = []
        for idx, refs in enumerate(references):
            flattened_refs.extend(refs)
            flattened_refs_idxs.extend([idx for _ in refs])

        flattened_refs = self.extract_all_captions(flattened_refs)

        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))
        flattened_refs = flattened_refs / np.sqrt(
            np.sum(flattened_refs**2, axis=1, keepdims=True)
        )

        cand_idx2refs = collections.defaultdict(list)
        for ref_feats, cand_idx in zip(
            flattened_refs, flattened_refs_idxs, strict=True
        ):
            cand_idx2refs[cand_idx].append(ref_feats)

        assert len(cand_idx2refs) == len(candidates)

        cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

        per = []
        for c_idx, cand in tqdm(enumerate(candidates)):
            cur_refs = cand_idx2refs[c_idx]
            all_sims = cand.dot(cur_refs.transpose())
            per.append(np.max(all_sims))

        return per
