# %%
import instruction_checkers
import re
import random
import itertools

# %%
"""
Category: _KEYWORD
    - frequency: frequency of a specified keyword (word, natural_relation, num_words)
    - together: keywords need to appear together a certain number of times, with word1 appearing more than word2 (word1, word2, num_words)
    - banned: use of words in a provided list are banned (forbidden_words: list)
    - paragraph_end: must contain at least n paras, and a word must appear in last sentence of each para (para_num, word)
    - first_word: first word of response must be a certain word (word)

Category: _LENGTH
    - max_word: response must not exceed certain number of words (num_words)
    - range_words: response length must be within a range of words (min_length, max_length)

Category: _FORMAT
    - postscript_at_end: add a postscript at the end of the response with a specific marker (addition)
    - title_brackets: response must have a title in double angle brackets and the title should not exceed given lenght (max_length)
    - highlight: at least n parts of the response must be highlgihted using double asterisks (num_parts)
    - JSON:  output must be in JSON format (nil)
    - separator: must have two responses, with a given sentence being a separator between the responses (sentence)
    - markdown_title: response must have a #-marked titlenot exceeding a specified length (max_length)
    - ordered_list: response must have an ordered list of specific number of items (num_items)
    - start_bold_italic: all paragraphs must start with *** to indicate bold and italic (nil)

Category: _REPEAT
    - copy_request: response my repeat the whole request and then provide the answer (prompt_to_repeat)
    - before_answer: repeat a given sentence a specified number of times before the response (num_repeats, sentence)
    - first_last_same: first and last sentences must be exactly the same (nil)
    - last_sentence: last sentence must be repeated a specified number of times after a separator of ##### (num_repeats)
    - sentence_n_times: a specified sentence must appear a specified number of times (num_repeats, sentence)
    - all_sentences_twice: all sentences must be repeated twice (nil)

Category: _MARKS
    - wrap_in_quotes: enclose the entire response in double quotes (nil)
    - no_commas: entire response should have no commas (nil)
    - replace_with_exclamation: all commas, periods, and question marks should be replaced by exclamation marks (nil)
    - end_with_semicolon: all sentences should end with semicolons instead of periods (nil)
    - replace_with_asterisks: all punctuation marks replaced with asterisk (nil)

Category: _CITATION
    - square_bracket_citation: response must contain at least a specified number of quotes in [x] format (num_quotes)
    - start_from_zero_citation: response must contain references that must start from 0, like [0], [1], and must be in increasing order (nil)
    - inline_citation: nresponse must contain references directly in parentheses in the double quotes, ie inline citation style (nil)

Category: _EMOJI
    - end_emoji: response must end with a specified number of specified emojis (num_emojis, emoji)
    - emoji_frequency: response must have a specified number fo specified emojis (num_emojis, emoji, natural_relation)
    - banned_emoji: response should have emojis but must not include banned emoji (emoji)
"""

# %%
#create instruction dictionary
_INSTRUCTION_DICTIONARY = {}
for subclass in instruction_checkers.Instruction.__subclasses__():
    category = subclass._CATEGORY.upper() 
    subclass_name = re.sub(r'(?<!^)(?=[A-Z])', '_', subclass.__name__).lower().replace("_checker", "")
    if subclass_name == "j_s_o_n": subclass_name = "JSON"
    key = f"_{category}:{subclass_name}"
    _INSTRUCTION_DICTIONARY[key] = subclass 

# %%
#preliminary conflict dictionary
#note that this is not necessarily symmetrical, use test_conflict_dictionary_symmetry
_INSTRUCTION_CONFLICT_DICTIONARY = {
    # --- _KEYWORD ---
    "_KEYWORD:frequency": {"_KEYWORD:together", "_KEYWORD:banned"},
    "_KEYWORD:together": {"_KEYWORD:frequency", "_KEYWORD:banned"},
    "_KEYWORD:banned": {"_KEYWORD:frequency", "_KEYWORD:together"},
    "_KEYWORD:paragraph_end": set(),
    "_KEYWORD:first_word": set(),

    # --- _LENGTH ---
    "_LENGTH:max_word": {"_LENGTH:range_words"},
    "_LENGTH:range_words": {"_LENGTH:max_word"},

    # --- _FORMAT ---
    "_FORMAT:postscript_at_end": {"_FORMAT:JSON",  "_REPEAT:last_sentence"},
    "_FORMAT:title_brackets": {"_FORMAT:markdown_title"},
    "_FORMAT:highlight": {"_FORMAT:JSON"},
    
    "_FORMAT:JSON": set(_INSTRUCTION_DICTIONARY.keys()).difference({"_KEYWORD:banned", "_KEYWORD:together", "_EMOJI:emoji_frequency","_EMOJI:banned_emoji"}),
     
    "_FORMAT:separator": {"_FORMAT:JSON", "_REPEAT:last_sentence"},
    "_FORMAT:markdown_title": {"_FORMAT:title_brackets"},
    "_FORMAT:ordered_list": {"_FORMAT:JSON"},
    "_FORMAT:start_bold_italic": {"_FORMAT:JSON"},

    # --- _REPEAT ---
    "_REPEAT:copy_request": {"_FORMAT:JSON", "_REPEAT:before_answer", "_REPEAT:all_sentences_twice", "_FORMAT:start_bold_italic", "_KEYWORD:first_word", "_MARKS:wrap_in_quotes", "_MARKS:no_commas", "_MARKS:replace_with_exclamation", "_MARKS:end_with_semicolon", "_MARKS:replace_with_asterisks"},
    "_REPEAT:before_answer": {"_FORMAT:JSON", "_MARKS:wrap_in_quotes"},
    "_REPEAT:first_last_same": {"_REPEAT:last_sentence"},
    "_REPEAT:last_sentence": {"_REPEAT:first_last_same"},
    "_REPEAT:sentence_n_times": set(),
    "_REPEAT:all_sentences_twice": {"_FORMAT:JSON"},

    # --- _MARKS ---
    "_MARKS:wrap_in_quotes": {"_REPEAT:copy_request", "_FORMAT:start_bold_italic"},
    "_MARKS:no_commas": {
        "_MARKS:replace_with_exclamation", "_MARKS:replace_with_asterisks"
    },
    "_MARKS:replace_with_exclamation": {
        "_MARKS:no_commas", "_MARKS:replace_with_asterisks", "_FORMAT:JSON"
    },
    "_MARKS:end_with_semicolon": {"_FORMAT:JSON"},
    "_MARKS:replace_with_asterisks": {
        "_MARKS:no_commas", "_MARKS:replace_with_exclamation", "_FORMAT:JSON"
    },

    # --- _CITATION ---
    "_CITATION:square_bracket_citation": set(_INSTRUCTION_DICTIONARY.keys()), #going to make this conflict with everything first
    "_CITATION:start_from_zero_citation": {"_CITATION:inline_citation"},
    "_CITATION:inline_citation": {
        "_CITATION:square_bracket_citation", "_CITATION:start_from_zero_citation"
    },

    # --- _EMOJI ---
    "_EMOJI:end_emoji": set(),
    "_EMOJI:emoji_frequency": {"_EMOJI:banned_emoji","_EMOJI:end_emoji","_EMOJI:emoji_frequency"},
    "_EMOJI:banned_emoji": {"_EMOJI:emoji_frequency","_EMOJI:end_emoji"},
}



def test_conflict_dictionary_symmetry(conflict_dict, verbose): #CHECK VERBOSITY
    """
    Tests whether the instruction conflict dictionary is symmetric.
    Prints any asymmetric pairs.
    """
    asymmetric_pairs = []

    for instr, conflicts in conflict_dict.items():
        for other in conflicts:
            if other not in conflict_dict:
                if verbose: print(f"⚠️ '{other}' is missing from the conflict dictionary.")
                continue
            if instr not in conflict_dict[other]:
                asymmetric_pairs.append((instr, other))

    if not asymmetric_pairs:
        return True
    else:
        if verbose: print("asymmetries found:")
        for a, b in asymmetric_pairs:
            if verbose: print(f"  - '{a}' conflicts with '{b}', but not vice versa.")
        return False


def conflict_make(conflicts):
  """Makes sure if A conflicts with B, B will conflict with A.

  Args:
    conflicts: Dictionary of potential conflicts where key is instruction id
      and value is set of instruction ids that it conflicts with.

  Returns:
    Revised version of the dictionary. All instructions conflict with
    themselves. If A conflicts with B, B will conflict with A.
    Modifies conflicts dictionary in place.
  """
  for key in conflicts:
    for k in conflicts[key]:
      conflicts[k].add(key)
    conflicts[key].add(key)
  return conflicts

# %%
def is_instruction_set_compatible(instructions, conflict_dict):
    for x, y in itertools.combinations(instructions, 2):
        if y in conflict_dict.get(x, set()):
            return False
    return True

def generate_instruction_groups(instruction_dict: dict, conflict_dict: dict, *, num_grps: int = 5, num_instructions_per_grp: int = 3, verbose=False) -> list:
    #each grp will be a dict with key as the str name, and the value the class variable
    grps = []
    for idx in range(num_grps):
        instruction_list = list(conflict_dict.keys())
        grp = random.sample(instruction_list, k=num_instructions_per_grp)
        is_compatible = False
        while not is_compatible:
            is_compatible = is_instruction_set_compatible(grp, conflict_dict)
            if not is_compatible: 
                if verbose: print("conflicts found, regenerating instruction grouping...")
                grp = random.sample(instruction_list, k=num_instructions_per_grp)
        grp_dict = {instruction: instruction_dict[instruction] for instruction in grp}
        grps.append(grp_dict)
    return grps


def get_instruction_dictionaries(instruction_dict: dict = _INSTRUCTION_DICTIONARY, conflict_dict: dict = _INSTRUCTION_CONFLICT_DICTIONARY, verbose=False) -> tuple[dict, dict]:
    is_symmetrical = test_conflict_dictionary_symmetry(conflict_dict, verbose) #make sure symmetrical
    while not is_symmetrical:
        if verbose: print("conflict dict was not symmetrical, making it symmetrical...")
        conflict_dict = conflict_make(conflict_dict)
        is_symmetrical = test_conflict_dictionary_symmetry(conflict_dict, verbose)
    if verbose: print("symmetrical")
    
    return (instruction_dict, conflict_dict)


if __name__ == "__main__":
    #make sure _INSTRUCTION_CONFLICT_DICTIONARY is symmetrical
    is_symmetrical = test_conflict_dictionary_symmetry(_INSTRUCTION_CONFLICT_DICTIONARY)
    while not is_symmetrical:
        print("conflict dict was not symmetrical, making symmetrical...")
        _INSTRUCTION_CONFLICT_DICTIONARY = conflict_make(_INSTRUCTION_CONFLICT_DICTIONARY)
        is_symmetrical = test_conflict_dictionary_symmetry(_INSTRUCTION_CONFLICT_DICTIONARY)
    print("symmetrical conflict dict")





