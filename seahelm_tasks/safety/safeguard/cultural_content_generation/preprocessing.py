import os

import pandas as pd

from seahelm_tasks.safety.safeguard.base_preprocessing import (
    BasePreprocessing,
)


class CulturalContentGenerationPreprocessing(BasePreprocessing):
    def __init__(self, task: str, folder: str, output_dir: str):
        super().__init__(task, folder, output_dir)

    def get_data_files(self):
        suffixes = {
            "en": "EN",
            "id": "IN_EN",
            "ms": "MS_EN",
            "my": "MY_EN",
            "ta": "TA_EN",
            "th": "TH_EN",
            "tl": "TL_EN",
            "vi": "VI_EN",
        }

        return {
            lang: os.path.join(self.folder, f"sea_safeguard_bench-{suf}.arrow")
            for lang, suf in suffixes.items()
        }

    def remove_rows_w_sensitive_label(self, df: pd.DataFrame) -> pd.DataFrame:
        if (
            "Sensitive" in df["response_label"].values
            or "Sensitive" in df["prompt_label"].values
        ):
            df = df[
                (df["response_label"] != "Sensitive")
                & (df["prompt_label"] != "Sensitive")
            ].reset_index(drop=True)
        return df

    def apply_mapping(self, df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        df["prompt_label"] = df["prompt_label"].map(mapping)
        df["response_label"] = df["response_label"].map(mapping)
        return df

    # def split_datasets(self, df: pd.DataFrame, mapping, random_state=42):
    #     examples_combinations = [
    #         (mapping["Safe"], 1),
    #         (mapping["Harmful"], 1),
    #         (mapping["Sensitive"], 1),
    #     ]
    #     examples_list, used_indices = [], []
    #     for response_label, n_rows in examples_combinations:
    #         subset = df[df["response_label"] == response_label]
    #         sampled = subset.sample(n=n_rows, random_state=random_state)
    #         examples_list.append(sampled)
    #         used_indices.extend(sampled.index.tolist())

    #     examples_df = (
    #         pd.concat(examples_list)
    #         .sample(frac=1, random_state=random_state)
    #         .reset_index(drop=True)
    #     )
    #     main_df = df.drop(index=used_indices)
    #     return examples_df, main_df

    def format(self, df: pd.DataFrame, lang: str):
        output = []
        for _, row in df.iterrows():
            output_row = {
                "prompt_label": row["prompt_label"],
                "response_label": row["response_label"],
                "prompts": [
                    {
                        "prompt_text": row["local_prompt"],
                        "response_text": row["local_response"],
                    }
                ],
                "metadata": {
                    "language": lang,
                    "en_prompt": row["en_prompt"],
                    "en_response": row["en_response"],
                },
            }
            output.append(output_row)
        return output


if __name__ == "__main__":
    preprocessing = CulturalContentGenerationPreprocessing(
        task="cultural_content_generation",
        folder="/scratch_aisg/SPEC-SF-AISG/xb.yong/VISTEC-AISG-SharedDataset___sea_safeguard_bench/cultural_content_generation/0.0.0/c7c5843bac969d98afcf32ba5a48ba01a82540fc",
        output_dir="seahelm_tasks/safety/safeguard/cultural_content_generation/data",
    )
    preprocessing.run_preprocessing()
