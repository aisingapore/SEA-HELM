import os

import pandas as pd

from seahelm_tasks.safety.safeguard.base_preprocessing import (
    BasePreprocessing,
)


class GeneralPreprocessing(BasePreprocessing):
    def __init__(self, task: str, folder: str, output_dir: str):
        super().__init__(task, folder, output_dir)

    def get_data_files(self):
        suffixes = {
            "en": "EN",
            "id": "IN",
            "ms": "MS",
            "my": "MY",
            "ta": "TA",
            "th": "TH",
            "tl": "TL",
            "vi": "VI",
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
    #         (mapping["Safe"], mapping["Safe"], 2),
    #         (mapping["Harmful"], mapping["Harmful"], 2),
    #         (mapping["Harmful"], mapping["Safe"], 2),
    #     ]
    #     examples_list, used_indices = [], []

    #     for prompt_label, response_label, n_rows in examples_combinations:
    #         subset = df[
    #             (df["prompt_label"] == prompt_label)
    #             & (df["response_label"] == response_label)
    #         ]
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
                    {"prompt_text": row["prompt"], "response_text": row["response"]}
                ],
                "metadata": {"language": lang, "topic": row["source"]},
            }
            output.append(output_row)
        return output


if __name__ == "__main__":
    preprocessing = GeneralPreprocessing(
        task="general",
        folder="/scratch_aisg/SPEC-SF-AISG/xb.yong/VISTEC-AISG-SharedDataset___sea_safeguard_bench/general/0.0.0/c7c5843bac969d98afcf32ba5a48ba01a82540fc",
        output_dir="seahelm_tasks/safety/safeguard/general/data",
    )
    preprocessing.run_preprocessing()
