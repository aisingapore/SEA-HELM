import os

import pandas as pd

from seahelm_tasks.safety.safeguard.base_preprocessing import (
    BasePreprocessing,
)


class CulturalInTheWildPreprocessing(BasePreprocessing):
    def __init__(self, task: str, folder: str, output_dir: str):
        super().__init__(task, folder, output_dir)

    def get_data_files(self):
        suffixes = {
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
        if "Sensitive" in df["prompt_label"].values:
            df = df[(df["prompt_label"] != "Sensitive")].reset_index(drop=True)
        return df

    def apply_mapping(self, df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        df["prompt_label"] = df["prompt_label"].map(mapping)
        return df

    # def split_datasets(self, df: pd.DataFrame, mapping, random_state=42):
    #     examples_combinations = [(mapping["Safe"], 3), (mapping["Harmful"], 3)]
    #     examples_list, used_indices = [], []

    #     for label, n_rows in examples_combinations:
    #         subset = df[df["prompt_label"] == label]
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
                "label": row["prompt_label"],
                "prompts": [
                    {
                        "local_prompt": row["local_prompt"],
                        "en_prompt": row["en_prompt"],
                    }
                ],
                "metadata": {"language": lang, "topic": row["topic"]},
            }

            output.append(output_row)
        return output


if __name__ == "__main__":
    preprocessing = CulturalInTheWildPreprocessing(
        task="cultural_in_the_wild",
        folder="/scratch_aisg/SPEC-SF-AISG/xb.yong/VISTEC-AISG-SharedDataset___sea_safeguard_bench/cultural_in_the_wild/0.0.0/c7c5843bac969d98afcf32ba5a48ba01a82540fc",
        output_dir="seahelm_tasks/safety/safeguard/cultural_in_the_wild/data",
    )
    preprocessing.run_preprocessing()
