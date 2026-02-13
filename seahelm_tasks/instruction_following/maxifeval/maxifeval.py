# %%
#for prototyping
import sys
import os

# Add the root directory (seahelm/) to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import fast_langdetect
import pandas as pd
from base_logger import get_logger
from instruction_registry import get_instruction_dictionaries
from seahelm_tasks.seahelm_metric import SeaHelmMetric
from prompt_response_utils import Prompt, Response, create_prompt_from_dict

logger = get_logger(__name__)
INSTRUCTION_DICT, INSTRUCTION_CONFLICT_DICT = get_instruction_dictionaries()
NUM_INSTRUCTIONS_PER_BQ = 3

# score is per instruction score, not per prompt
# loose score is any float from 0 to 1
# strict score is only 1 or 0 (if not 1)
# generate detailed statistic by instruction category and subcategory

# inference_df should have whole_prompt, response, instruction_n_cat, instruction_n_subcat, 
# instruction_n_score, cat_combination
# assuming that number of instructions is standardized across all basic questions

class MaxIFEvalMetric(SeaHelmMetric):
    def __init__(
            self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):
        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )
        #response_column defaults to responses
        #postprocessed_response_column defaults to cleaned_response
    
    def calculate_metrics(self):
        self.inference_df = self.inference_df.apply(self.evaluate_response, axis=1)

        instruction_score_cols = [col for col in self.inference_df.columns if col.startswith("instruction_") and col.endswith("_score")] 
        #might have different number of instructions, to implement later

        def average_valid_scores(row):
            scores = [row[col] for col in instruction_score_cols if row[col] != -1]
            if not scores:
                return {"averaged_maxifeval_score": None}
            return {"averaged_loose_maxifeval_score": round(sum(scores) / len(scores), 4)}

        self.inference_df["individual_scores"] = self.inference_df.apply(average_valid_scores, axis=1)

        metric_dict = self.summarize_results(self.inference_df)

        return metric_dict, self.inference_df
    
    def postprocess_responses(self):
        self.inference_df[self.postprocessed_response_column] = self.inference_df[self.response_column].map(lambda x:x[0])

        def extract_instruction_fields(metadata_row, i):
            return {
                f'instruction_{i}_cat': metadata_row.get(f'instruction_{i}_cat'),
                f'instruction_{i}_subcat': metadata_row.get(f'instruction_{i}_subcat'),  
                f'instruction_{i}_score': -1
            }
        
        for i in range(1, NUM_INSTRUCTIONS_PER_BQ + 1):
            extracted = self.inference_df['metadata'].apply(lambda row: extract_instruction_fields(row, i))
            extracted_df = pd.DataFrame(extracted.tolist())
            self.inference_df = pd.concat([self.inference_df, extracted_df], axis=1)

    def evaluate_response(self, row):
        #generate prompt object
        row_dict = row.to_dict()
        num_instructions = row_dict['metadata']['num_instructions']
        prompt: Prompt = create_prompt_from_dict(row_dict)

        #wrap prompt in response object
        response: Response = Response(prompt._id, prompt, row[self.postprocessed_response_column])

        #check and get scores
        scores = response.check() #returns a list of scores, excpetion handling in Response.check() logic 
        assert len(scores) == num_instructions, "evaluate_response() error: number of scores is different from number of instructions!"
        for i in range(num_instructions):
            row[f'instruction_{i+1}_score'] = scores[i]
        
        return row
    
    def summarize_results(self, inference_df: pd.DataFrame) -> dict:
        evaluation_stats = {"total_instructions": -1, "overall_strict": -1, "overall_loose": -1, "total_failed_instructions": -1, "category_breakdown": -1, "subcategory_breakdown": -1} #can just put my eval stuff ehre
        instruction_score_cols = [col for col in inference_df.columns if col.startswith("instruction_") and col.endswith("_score")] 
        num_instructions_per_bq = len(instruction_score_cols)
        
        scores_df = inference_df[instruction_score_cols]
        total_scored = (scores_df != -1).values.sum() #values creates an array of booleans
        total_unscored = (scores_df == -1).values.sum()
        total_instructions = total_scored + total_unscored
        total_strict = (scores_df == 1).values.sum()
        total_loose = (scores_df[scores_df != -1]).sum().sum()

        evaluation_stats['total_instructions'] = total_instructions 
        evaluation_stats['total_failed_instructions'] = round(total_unscored/total_instructions, 4)
        evaluation_stats['overall_strict'] = round(total_strict/total_scored, 4) #percentage of the scored instructions!
        evaluation_stats['overall_loose'] = round(total_loose/total_scored, 4)

        #breakdown by category and subcategory
        cat_breakdown = {}
        subcat_breakdown = {}

        for i in range(1, num_instructions_per_bq + 1):
            cat_col = f'instruction_{i}_cat'
            subcat_col = f'instruction_{i}_subcat'
            score_col = f'instruction_{i}_score'

            cats = inference_df[cat_col]
            subcats = inference_df[subcat_col]
            scores = inference_df[score_col]

            for cat, subcat, score in zip(cats, subcats, scores):

                #update cat breakdown
                if cat not in cat_breakdown:
                    cat_breakdown[cat] = {"strict_score": 0, "loose_score": 0, "total": 0, "total_failed": 0}

                cat_breakdown[cat]["total"] += 1 #total includes all instructions (failed and successful)

                if score == -1:
                    cat_breakdown[cat]["total_failed"] += 1
                    continue
                else:
                    cat_breakdown[cat]["loose_score"] += score
                    if score == 1:
                        cat_breakdown[cat]["strict_score"] += 1

                #update subcat breakdown
                if subcat not in subcat_breakdown:
                    subcat_breakdown[subcat] = {"strict_score": 0, "loose_score": 0, "total": 0, "total_failed": 0}

                subcat_breakdown[subcat]["total"] += 1

                if score == -1:
                    subcat_breakdown[subcat]["total_failed"] += 1
                    continue
                else:
                    subcat_breakdown[subcat]["loose_score"] += score
                    if score == 1:
                        subcat_breakdown[subcat]["strict_score"] += 1


        #convert breakdown into percentages
        for _, stats in cat_breakdown.items():
            total_scored = stats['total'] - stats['total_failed']
            stats['strict_score'] = round(stats['strict_score']/total_scored,4)
            stats['loose_score'] = round(stats['loose_score']/total_scored,4)
            stats['total_failed'] = round(stats['total_failed']/stats['total'],4)
        
        for _, stats in subcat_breakdown.items():
            total_scored = stats['total'] - stats['total_failed']
            stats['strict_score'] = round(stats['strict_score']/total_scored,4)
            stats['loose_score'] = round(stats['loose_score']/total_scored,4)
            stats['total_failed'] = round(stats['total_failed']/stats['total'],4)
        
        evaluation_stats["category_breakdown"] = cat_breakdown
        evaluation_stats["subcategory_breakdown"] = subcat_breakdown

        return evaluation_stats




# %%
