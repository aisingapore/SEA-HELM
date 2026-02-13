# %%
from instruction_registry import get_instruction_dictionaries, generate_instruction_groups
import instruction_checkers
from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset
import json

_INSTRUCTION_DICTIONARY, _INSTRUCTION_CONFLICT_DICTIONARY = get_instruction_dictionaries()
_NUM_INSTRUCTIONS_PER_GRP = 3

#add restrictions on the production of code to some of the formatting things

# %%
"""
Prompt class for prompts
- basic question is the question from magpie
- initialize with _id of the prompt, the _basic_question, and _basic_question_cat
- to randomly generate instructions, call Prompt.generate_instructions()
- to set specific instructions, call Prompt.set_instructions(list[str])
- if instructions already set/generated, then call self.reset_instructions() or self.regenerate_instructions()
- to synthesize the whole prompt, call self.synthesize_prompt(). need to make sure that self._instructions is available first. 
in other words, generate_instructions() first.
"""
@dataclass
class Prompt:
    _id: str
    _basic_question: str
    _basic_question_cat: str 
    _instructions: Optional[dict[str, dict[str, instruction_checkers.Instruction | str]]] = field(default=None, init=False) 
    #should be a dict, key as checker name, value as a dict with attributes checker_instance and description
    _whole_prompt: Optional[str] = field(default=None, init=False)

    def generate_instructions(self, instruction_dict=None, conflict_dict=None, num_instructions_per_grp=_NUM_INSTRUCTIONS_PER_GRP, **kwargs):
        if not instruction_dict: instruction_dict = _INSTRUCTION_DICTIONARY
        if not conflict_dict: conflict_dict = _INSTRUCTION_CONFLICT_DICTIONARY

        #handle CopyRequest
        kwargs["prompt_to_repeat"] = self._basic_question

        #check if the prompt already has instructions or not, if has instructions, then don't do anything
        if hasattr(self, "_instructions") and self._instructions is not None:
            print("Prompt object already has existing instructions. To regenerate instructions, use Prompt.regenerate_instructions.")
            return

        #get instruction groups that don't conflict with each other 
        instructions = generate_instruction_groups(instruction_dict, conflict_dict, num_grps=1, num_instructions_per_grp=num_instructions_per_grp)[0]
        
        #instantiate the checkers, build the descriptions
        for key, value in instructions.items():
            checker_instance = value(self._id)
            checker_description = checker_instance.build_description(**kwargs)
            instructions[key] = {
                "checker_instance": checker_instance,
                "description": checker_description
            }
        
        self._instructions: dict[str, dict[instruction_checkers.Instruction, str]] = instructions
        return instructions

    def regenerate_instructions(self, instruction_dict=None, conflict_dict=None, num_instructions_per_grp=_NUM_INSTRUCTIONS_PER_GRP, **kwargs):
        #just an alternative function to regenerate instructions
        if hasattr(self, "_instructions"):
            self._instructions = None
        return self.generate_instructions(instruction_dict, conflict_dict, num_instructions_per_grp, **kwargs)
    
    def set_instructions(self, instruction_dict=None, conflict_dict=None, instructions: list=[], **kwargs):
        #use this method if i want to set specific instructions
        if hasattr(self, "_instructions") and self._instructions is not None:
            print("Prompt object already has existing instructions. To reset instructions, use Prompt.reset_instructions.")

        #handle CopyRequest
        kwargs["prompt_to_repeat"] = self._basic_question

        if not instruction_dict: instruction_dict = _INSTRUCTION_DICTIONARY
        if not conflict_dict: conflict_dict = _INSTRUCTION_CONFLICT_DICTIONARY       
        if not instructions:
            raise ValueError("Pass in Instructions as a list of the names (str) of the checkers that you want to set the prompt with.")
        
        #ensure all provided instruction names exist
        for instr in instructions:
            if instr not in instruction_dict:
                raise ValueError(f"Instruction '{instr}' not found as key in instruction_dict. Make sure spelling is correct!")

        # Check for pairwise conflicts
        for i, instr1 in enumerate(instructions):
            for instr2 in instructions[i + 1:]:
                if instr2 in conflict_dict.get(instr1, set()):
                    raise ValueError(f"Conflict detected between '{instr1}' and '{instr2}'.")
        
        #if all ok, then create self._instructions
        temp_dict = {}
        for instruction in instructions:
            checker_var = instruction_dict[instruction]
            checker_instance = checker_var(self._id)
            checker_description = checker_instance.build_description(**kwargs)
            temp_dict[instruction] = {
                "checker_instance": checker_instance,
                "description": checker_description
            }
        self._instructions = temp_dict
        return self._instructions
    
    def reset_instructions(self, instruction_dict=None, conflict_dict=None, instructions: list=[], **kwargs):
        if hasattr(self, "_instructions"):
            self._instructions = None
        return self.set_instructions(instruction_dict, conflict_dict, instructions, **kwargs)

    def synthesize_prompt(self):
        if not hasattr(self, "_instructions") or self._instructions is None:
            raise ValueError("Instructions not found. Call `generate_instructions()` first.")
        descriptions = [value["description"] for _, value in self._instructions.items()]
        self._whole_prompt = self._basic_question + " " + " ".join(descriptions)
        return self._whole_prompt
    
    def get_instructions(self):
        if not hasattr(self, "_instructions"):
            raise ValueError("Prompt._instructions has not been set yet!")
        return self._instructions

    def get_whole_prompt(self) -> str:
        if not hasattr(self, "_whole_prompt"):
            raise ValueError("Whole prompt has not been synthesized yet. Call `generate_instructions()` then `synthesize_prompt()` first.")
        return self._whole_prompt
    
    # def to_dict(self) -> dict:
    #     return {
    #         "_id": self._id,
    #         "_basic_question": self._basic_question,
    #         "_basic_question_cat": self._basic_question_cat,
    #         "_instructions": {checker_name: (
    #                           None if value['checker_instance'].get_instruction_args() == {None}
    #                           else value['checker_instance'].get_instruction_args()
    #                           )
    #                           for checker_name, value in self.get_instructions().items()
    #         },
    #         "_whole_prompt": self.get_whole_prompt()
    #    }
    
    def to_dict(self) -> dict:
        
        instructions = self.get_instructions()
        temp_fields = {}
        for idx, k in enumerate(instructions):
            v = instructions[k]
            checker = v['checker_instance']
            args = checker.get_instruction_args()
            temp_fields[f"instruction_{idx+1}_cat"] = checker.get_category()
            temp_fields[f"instruction_{idx+1}_subcat"] = k
            temp_fields[f"instruction_{idx+1}_kwargs"] = [{}] if args == {None} else [args]

        return {
            "_id": self._id,
            "prompts": [{"text": self._whole_prompt}],
            "prompt_templates": ["{text}"],
            "metadata": {
                "basic_question": self._basic_question,
                "basic_question_cat": self._basic_question_cat,
                "num_instructions": len(instructions),
                **temp_fields
            }
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)
    
    def __eq__(self, other):
        if not isinstance(other, Prompt): return NotImplemented
        return self.to_dict() == other.to_dict()
    
    #to directly modify the _instruction private instance variable
    def set_instance_instruction_attr(self, instructions):
        self._instructions = instructions
        return self._instructions

# %% 
"""
Response class is to store a response and to evalute the response.
Don't use Response class to generate a response
"""

# %%
@dataclass
class Response:
    _id: str #id convention should be {model_name}_{prompt_id}
    _prompt: Prompt
    _response: str
    _scores: Optional[list[int]] = field(default=None, init=False)

    def check(self) -> list[float]:
        #make sure _prompt has been initialized with instructions and a prompt
        if not hasattr(self._prompt, "_instructions"):
            raise ValueError("_prompt that this Response object has been initialized with doesn't have _instructions!")
        
        prompt = self._prompt
        instructions = prompt.get_instructions()

        #get checkers
        checkers = [value["checker_instance"] for key, value in instructions.items()]
        scores = []

        #for each checker, check the response
        for checker in checkers:
            try:
                score = checker.check(self._response)
            except Exception as e:
                print(f"Error with {prompt._id}: {prompt.get_whole_prompt()}")
                print(f"Checker name: {checker.__class__}")
                print(f"Exception: {e}")
                print(f"Response:\n{self._response}")
                print("-----")
                score = 0
            scores.append(score)
        
        return scores 
    
    def get_instruction_names_and_categories(self) -> list:
        if not hasattr(self._prompt, "_instructions"):
            raise ValueError("_prompt that this Response object has been initialized with doesn't have _instructions!")
        instruction_names = []
        instruction_categories = []
        prompt = self._prompt
        instructions = prompt.get_instructions()
        for name, value in instructions.items():
            instruction_names.append(name)
            instruction_categories.append(value["checker_instance"].get_category())
        return instruction_names, instruction_categories


def create_prompts_from_ds(basic_questions_dataset: Dataset, id_head: str):
    prompts = []
    for idx, example in enumerate(basic_questions_dataset):
        id = f"{id_head}_{idx}"
        basic_question = example["basic_question"]
        basic_question_cat = example["basic_question_category"]

        prompt = Prompt(
            id,
            basic_question,
            basic_question_cat
        )
        prompt.generate_instructions()
        prompt.synthesize_prompt()
        prompts.append(prompt)

    return prompts

# def create_prompt_from_dict(entry) -> Prompt:
#     #not going to do validation now
#     temp = Prompt(entry['_id'], entry['_basic_question'], entry['_basic_question_cat'])
#     instructions = {
#         name: (
#             (_checker := _INSTRUCTION_DICTIONARY[name](entry['_id'])),
#             {
#                 "checker_instance": _checker,
#                 "description": _checker.build_description(**({**value, "prompt_to_repeat": entry["_basic_question"]}) if value is not None else {})
#             }
#         )[1]  #get only the dict, not the walrus _checker binder
#         for name, value in entry['_instructions'].items()
#     }
#     temp.set_instance_instruction_attr(instructions)
#     temp.synthesize_prompt()
#     return temp

def create_prompt_from_dict(obj) -> Prompt:
    temp = Prompt(obj['_id'], obj['metadata']['basic_question'], obj['metadata']['basic_question_cat'])
    num_instructions = obj['metadata']['num_instructions']
    instructions = {}
    for i in range(1,num_instructions+1):
        name = obj['metadata'][f'instruction_{i}_subcat']
        checker_instance = _INSTRUCTION_DICTIONARY[name](obj['_id'])
        kwargs = obj['metadata'][f'instruction_{i}_kwargs'][0]
        #if kwargs=={} : kwargs = {} 
        description = checker_instance.build_description(**kwargs)
        instructions[name] = {'checker_instance': checker_instance, 'description': description}
    temp.set_instance_instruction_attr(instructions)
    temp.synthesize_prompt()
    return temp

def save_prompts_as_jsonl(prompts: list[Prompt], file_name: str):
    with open(file_name, "w") as f:
        for prompt in prompts:
            json_line = json.dumps(prompt.to_dict())
            f.write(json_line + "\n")

def load_prompts_from_json(file_name: str) -> list[Prompt]:
    prompts = []
    with open(file_name,"r") as f:
        temp_json_str = json.load(f)
    
    for prompt in temp_json_str:
        prompts.append(create_prompt_from_dict(prompt))
    
    return prompts

def load_prompts_from_jsonl(file_name: str) -> list[Prompt]:
    with open(file_name, "r") as f:
        prompts = [create_prompt_from_dict(json.loads(line)) for line in f]
    return prompts

#ensure that the response_values and prompts match up
def create_responses(prompts: list[Prompt], response_values: list[str]) -> list[Response]:
    assert len(prompts) == len(response_values), (
        "Different number of prompts and response_values detected, make sure that the two are aligned and matching!"
    )
    responses = []
    for prompt, response_value in zip(prompts, response_values):
        response = Response(prompt._id, prompt, response_value)
        responses.append(response)
    return responses

# #%%
# # testing_prompt = Prompt




# # %%
# ###THIS IS ALL TESTING CODE FROM HERE ONWARDS!!
# test_prompt = Prompt(
#     "test_prompt",
#     "this is my testing basic question.",
#     "testing...",
# )

# test_prompt.generate_instructions()
# test_prompt.synthesize_prompt()
# test_dict = test_prompt.to_dict()

# #%%
# import json

# json_str = json.dumps(test_dict)
# test_json = json.loads(json_str)
# with open("output.json","w") as f:
#     json.dump(test_json, f, indent=2)
# #%%
# test_json
# test_dict_conversion = create_prompt_from_dict(test_json)
# test_dict_conversion.get_whole_prompt() == test_prompt.get_whole_prompt()

# # %%
# response_example = (
#     """
#     "
#     «this is my book title bruvs i have no idea how many words this is but i guess i'll find out»
#     [0] philip tham [1] my name is philip [0.5] this shouldn't work
#     "
#     """
# )

# # %%
# test_prompt.get_instructions()
# # %%
# testing_response = Response(
#     "philip_test_prompt",
#     test_prompt,
#     response_example
# )
# # %%
# testing_response.check()


# # %%

# %%
