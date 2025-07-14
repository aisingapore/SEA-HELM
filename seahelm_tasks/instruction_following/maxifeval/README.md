MaxIFEval scoring and checkers based on MaxIFEval paper (Waseda and Oppo) (https://arxiv.org/pdf/2506.01776). Deviations from paper are documented in instruction_checkers.py. Each prompt comprises a single basic question (from brainstorming, planning, reasoning, role playing, editing, creative writing categories in Magpie-Align/Magpie-Llama-3.3-Pro-500K-Filtered dataset) randomly grouped with 3 instructions. An \_INSTRUCTION_CONFLICT_DICTIONARY defined instruction_registry.py is used to prevent conflicting instructions from being grouped together (although this list of conflicts is not yet comprehensive).

Datasets

<ul>
<li>all_bq_instructions.jsonl: 52,870 prompts (basic question + 3 instructions) from Magpie dataset</li>
<li>beta_maxifeval_prompts.jsonl: 7,879 prompts, filtered from all_bq_instructions.jsonl (only the brainstorming basic questions)</li>
<li>gemma_responses.jsonl: gemma generated responses (not formatted properly, can just disregard this)</li>
<li>toy_inference_ds.json: inference ds with gemma responses, formatted correctly, can run evaluation on this</li>
</ul>

Files

<ul>
<li>instruction_checkers.py: implementation of 32 rules-based checkers, with some helper functions (split_into_sentences, count_words) being based on Google's IFEval implementation (https://github.com/google-research/google-research/blob/master/instruction_following_eval/instructions_util.py).</li>
<li>instruction_registry.py: definition of _INSTRUCTION_DICIONARY and _INSTRUCTION_CONFLICT_DICTIONARY (accessible in other modules through  get_instruction_dictionaries), includes logic for conflict detection (based on Google's IFEval implementation) and generating random instruction groupings (generation_instruction_groups)</li>
<li>maxifeval.py: implementation of MaxIFEvalMetric</li>
<li>prompt_response_utils.py: implementation of Prompt and Response objects used to store entire prompts (basic question + instructions), responses, and checker logic</li>
