# %%
import collections
import json
import random
import re
import string
from typing import Final, Callable
import operator
import nltk


_LEGAL_NATURAL_RELATIONS: Final[dict[str, Callable[[int, int], bool]]] = {
    "exactly": operator.eq,
    "at least": operator.ge,
    "at most": operator.le,
}

_MAX_NUM = 50

_EMOJIS = "😀😃😄😁😆😅😂🤣😊😇🙂🙃😉😌😍🥰😘😗😙😚😋😛😝😜🤪🤨🧐🤓😎🥸🤩🥳😏😒😞😔😟😕🙁☹️"

"""
32 different types of instructions evaluable through rules based eval
Instructions are the constraints that are added on top of basic questions (BQ)


Need to also figure out how to generate instructions + base question = prompt
Should I have Instruction and Checker separately or together in the same object?
In the google IfEval, both the instruction and checker are the same object

https://github.com/google-research/google-research/blob/master/instruction_following_eval/instructions.py

1. Keyword
    - Frequency of a keyword (FrequencyChecker): word, natural_relation, num_words
    - Keywords need to appear together a certain number of times (TogetherChecker): word1, word2, num_words
    - Banned keywords (BannedChecker): forbidden_words (list, comma separated tokens)
    - Need to contain n paragraphs, and end with certain word (ParagraphEndChecker): para_num, word
    - First word (FirstWordChecker): word

2. Length
    - Maximum number of words (MaxWordChecker): max_length
    - Response within a range of words (RangeWordsChecker): min_length, max_length

3. Format
    - Addition of postscript (PostscriptAtEndChecker): addition
    - Title AND title brackets of specified max number of words (TitleBracketsChecker): max_length
    - Markdown highlight (HighlightChecker): num_parts
    - Json output (JSONChecker): nil
    - Separate two responses with a sentence (SeparatorChecker): sentence
    - Markdown title (MarkdownTitleChecker): max_length
    - Ordered list (OrderedListChecker): num_items
    - Markdown bold italic (StartBoldItalicChecker): nil

4. Repeat
    - Copy request (CopyRequestChecker): prompt_to_repeat 
    - Before answer (BeforeAnswerChecker): num_repeats, sentence
    - First and last sentence the same (FirstLastSameChecker): nil
    - Last sentence repeat n times (LastSentenceChecker): num_repeats
    - Sentence repeat n times (SentenceNTimesChecker): num_repeats, sentence
    - All sentences twice (AllSentencesTwiceChecker): nil

5. Marks
    - Wrap in quotes (WrapInQuotesChecker): nil
    - No commas (NoCommasChecker): nil
    - Replace all comms/periods/question marks with exclamation marks (ReplaceWithExclamationChecker): nil
    - End all sentences with semicolons (EndWithSemicolonChecker): nil
    - Replace all punctuation marks with asterisks (ReplaceWithAsterisksChecker): nil

6. Citation (this is quite cool)
    - Citations enclosed in [] (SquareBracketCitationChecker): num_quotes
    - Citations start with 0 (StartFromZeroCitationChecker): nil
    - Citations inline in parentheses (InlineCitationChecker): nil

7. Emoji
    - End with emoji (EndEmojiChecker): num_emojis, emoji
    - Frequency of emoji (EmojiFrequencyChecker): num_emojis, emoji, natural_relation
    - Banned emoji (BannedEmojiChecker): emoji
"""

# %%
#instructions_util.py from the google github

_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"
_WORD_LIST = ["western", "sentence", "signal", "dump", "spot", "opposite", "bottom", "potato", "administration", "working", "welcome", "morning", "good", "agency", "primary", "wish", "responsibility", "press", "problem", "president", "steal", "brush", "read", "type", "beat", "trainer", "growth", "lock", "bone", "case", "equal", "comfortable", "region", "replacement", "performance", "mate", "walk", "medicine", "film", "thing", "rock", "tap", "total", "competition", "ease", "south", "establishment", "gather", "parking", "world", "plenty", "breath", "claim", "alcohol", "trade", "dear", "highlight", "street", "matter", "decision", "mess", "agreement", "studio", "coach", "assist", "brain", "wing", "style", "private", "top", "brown", "leg", "buy", "procedure", "method", "speed", "high", "company", "valuable", "pie", "analyst", "session", "pattern", "district", "pleasure", "dinner", "swimming", "joke", "order", "plate", "department", "motor", "cell", "spend", "cabinet", "difference", "power", "examination", "engine", "horse", "dimension", "pay", "toe", "curve", "literature", "bother", "fire", "possibility", "debate", "activity", "passage", "hello", "cycle", "background", "quiet", "author", "effect", "actor", "page", "bicycle", "error", "throat", "attack", "character", "phone", "tea", "increase", "outcome", "file", "specific", "inspector", "internal", "potential", "staff", "building", "employer", "shoe", "hand", "direction", "garden", "purchase", "interview", "study", "recognition", "member", "spiritual", "oven", "sandwich", "weird", "passenger", "particular", "response", "reaction", "size", "variation", "a", "cancel", "candy", "exit", "guest", "condition", "fly", "price", "weakness", "convert", "hotel", "great", "mouth", "mind", "song", "sugar", "suspect", "telephone", "ear", "roof", "paint", "refrigerator", "organization", "jury", "reward", "engineering", "day", "possession", "crew", "bar", "road", "description", "celebration", "score", "mark", "letter", "shower", "suggestion", "sir", "luck", "national", "progress", "hall", "stroke", "theory", "offer", "story", "tax", "definition", "history", "ride", "medium", "opening", "glass", "elevator", "stomach", "question", "ability", "leading", "village", "computer", "city", "grand", "confidence", "candle", "priest", "recommendation", "point", "necessary", "body", "desk", "secret", "horror", "noise", "culture", "warning", "water", "round", "diet", "flower", "bus", "tough", "permission", "week", "prompt", "connection", "abuse", "height", "save", "corner", "border", "stress", "drive", "stop", "rip", "meal", "listen", "confusion", "girlfriend", "living", "relation", "significance", "plan", "creative", "atmosphere", "blame", "invite", "housing", "paper", "drink", "roll", "silver", "drunk", "age", "damage", "smoke", "environment", "pack", "savings", "influence", "tourist", "rain", "post", "sign", "grandmother", "run", "profit", "push", "clerk", "final", "wine", "swim", "pause", "stuff", "singer", "funeral", "average", "source", "scene", "tradition", "personal", "snow", "nobody", "distance", "sort", "sensitive", "animal", "major", "negotiation", "click", "mood", "period", "arrival", "expression", "holiday", "repeat", "dust", "closet", "gold", "bad", "sail", "combination", "clothes", "emphasis", "duty", "black", "step", "school", "jump", "document", "professional", "lip", "chemical", "front", "wake", "while", "inside", "watch", "row", "subject", "penalty", "balance", "possible", "adult", "aside", "sample", "appeal", "wedding", "depth", "king", "award", "wife", "blow", "site", "camp", "music", "safe", "gift", "fault", "guess", "act", "shame", "drama", "capital", "exam", "stupid", "record", "sound", "swing", "novel", "minimum", "ratio", "machine", "shape", "lead", "operation", "salary", "cloud", "affair", "hit", "chapter", "stage", "quantity", "access", "army", "chain", "traffic", "kick", "analysis", "airport", "time", "vacation", "philosophy", "ball", "chest", "thanks", "place", "mountain", "advertising", "red", "past", "rent", "return", "tour", "house", "construction", "net", "native", "war", "figure", "fee", "spray", "user", "dirt", "shot", "task", "stick", "friend", "software", "promotion", "interaction", "surround", "block", "purpose", "practice", "conflict", "routine", "requirement", "bonus", "hole", "state", "junior", "sweet", "catch", "tear", "fold", "wall", "editor", "life", "position", "pound", "respect", "bathroom", "coat", "script", "job", "teach", "birth", "view", "resolve", "theme", "employee", "doubt", "market", "education", "serve", "recover", "tone", "harm", "miss", "union", "understanding", "cow", "river", "association", "concept", "training", "recipe", "relationship", "reserve", "depression", "proof", "hair", "revenue", "independent", "lift", "assignment", "temporary", "amount", "loss", "edge", "track", "check", "rope", "estimate", "pollution", "stable", "message", "delivery", "perspective", "mirror", "assistant", "representative", "witness", "nature", "judge", "fruit", "tip", "devil", "town", "emergency", "upper", "drop", "stay", "human", "neck", "speaker", "network", "sing", "resist", "league", "trip", "signature", "lawyer", "importance", "gas", "choice", "engineer", "success", "part", "external", "worker", "simple", "quarter", "student", "heart", "pass", "spite", "shift", "rough", "lady", "grass", "community", "garage", "youth", "standard", "skirt", "promise", "blind", "television", "disease", "commission", "positive", "energy", "calm", "presence", "tune", "basis", "preference", "head", "common", "cut", "somewhere", "presentation", "current", "thought", "revolution", "effort", "master", "implement", "republic", "floor", "principle", "stranger", "shoulder", "grade", "button", "tennis", "police", "collection", "account", "register", "glove", "divide", "professor", "chair", "priority", "combine", "peace", "extension", "maybe", "evening", "frame", "sister", "wave", "code", "application", "mouse", "match", "counter", "bottle", "half", "cheek", "resolution", "back", "knowledge", "make", "discussion", "screw", "length", "accident", "battle", "dress", "knee", "log", "package", "it", "turn", "hearing", "newspaper", "layer", "wealth", "profile", "imagination", "answer", "weekend", "teacher", "appearance", "meet", "bike", "rise", "belt", "crash", "bowl", "equivalent", "support", "image", "poem", "risk", "excitement", "remote", "secretary", "public", "produce", "plane", "display", "money", "sand", "situation", "punch", "customer", "title", "shake", "mortgage", "option", "number", "pop", "window", "extent", "nothing", "experience", "opinion", "departure", "dance", "indication", "boy", "material", "band", "leader", "sun", "beautiful", "muscle", "farmer", "variety", "fat", "handle", "director", "opportunity", "calendar", "outside", "pace", "bath", "fish", "consequence", "put", "owner", "go", "doctor", "information", "share", "hurt", "protection", "career", "finance", "force", "golf", "garbage", "aspect", "kid", "food", "boot", "milk", "respond", "objective", "reality", "raw", "ring", "mall", "one", "impact", "area", "news", "international", "series", "impress", "mother", "shelter", "strike", "loan", "month", "seat", "anything", "entertainment", "familiar", "clue", "year", "glad", "supermarket", "natural", "god", "cost", "conversation", "tie", "ruin", "comfort", "earth", "storm", "percentage", "assistance", "budget", "strength", "beginning", "sleep", "other", "young", "unit", "fill", "store", "desire", "hide", "value", "cup", "maintenance", "nurse", "function", "tower", "role", "class", "camera", "database", "panic", "nation", "basket", "ice", "art", "spirit", "chart", "exchange", "feedback", "statement", "reputation", "search", "hunt", "exercise", "nasty", "notice", "male", "yard", "annual", "collar", "date", "platform", "plant", "fortune", "passion", "friendship", "spread", "cancer", "ticket", "attitude", "island", "active", "object", "service", "buyer", "bite", "card", "face", "steak", "proposal", "patient", "heat", "rule", "resident", "broad", "politics", "west", "knife", "expert", "girl", "design", "salt", "baseball", "grab", "inspection", "cousin", "couple", "magazine", "cook", "dependent", "security", "chicken", "version", "currency", "ladder", "scheme", "kitchen", "employment", "local", "attention", "manager", "fact", "cover", "sad", "guard", "relative", "county", "rate", "lunch", "program", "initiative", "gear", "bridge", "breast", "talk", "dish", "guarantee", "beer", "vehicle", "reception", "woman", "substance", "copy", "lecture", "advantage", "park", "cold", "death", "mix", "hold", "scale", "tomorrow", "blood", "request", "green", "cookie", "church", "strip", "forever", "beyond", "debt", "tackle", "wash", "following", "feel", "maximum", "sector", "sea", "property", "economics", "menu", "bench", "try", "language", "start", "call", "solid", "address", "income", "foot", "senior", "honey", "few", "mixture", "cash", "grocery", "link", "map", "form", "factor", "pot", "model", "writer", "farm", "winter", "skill", "anywhere", "birthday", "policy", "release", "husband", "lab", "hurry", "mail", "equipment", "sink", "pair", "driver", "consideration", "leather", "skin", "blue", "boat", "sale", "brick", "two", "feed", "square", "dot", "rush", "dream", "location", "afternoon", "manufacturer", "control", "occasion", "trouble", "introduction", "advice", "bet", "eat", "kill", "category", "manner", "office", "estate", "pride", "awareness", "slip", "crack", "client", "nail", "shoot", "membership", "soft", "anybody", "web", "official", "individual", "pizza", "interest", "bag", "spell", "profession", "queen", "deal", "resource", "ship", "guy", "chocolate", "joint", "formal", "upstairs", "car", "resort", "abroad", "dealer", "associate", "finger", "surgery", "comment", "team", "detail", "crazy", "path", "tale", "initial", "arm", "radio", "demand", "single", "draw", "yellow", "contest", "piece", "quote", "pull", "commercial", "shirt", "contribution", "cream", "channel", "suit", "discipline", "instruction", "concert", "speech", "low", "effective", "hang", "scratch", "industry", "breakfast", "lay", "join", "metal", "bedroom", "minute", "product", "rest", "temperature", "many", "give", "argument", "print", "purple", "laugh", "health", "credit", "investment", "sell", "setting", "lesson", "egg", "middle", "marriage", "level", "evidence", "phrase", "love", "self", "benefit", "guidance", "affect", "you", "dad", "anxiety", "special", "boyfriend", "test", "blank", "payment", "soup", "obligation", "reply", "smile", "deep", "complaint", "addition", "review", "box", "towel", "minor", "fun", "soil", "issue", "cigarette", "internet", "gain", "tell", "entry", "spare", "incident", "family", "refuse", "branch", "can", "pen", "grandfather", "constant", "tank", "uncle", "climate", "ground", "volume", "communication", "kind", "poet", "child", "screen", "mine", "quit", "gene", "lack", "charity", "memory", "tooth", "fear", "mention", "marketing", "reveal", "reason", "court", "season", "freedom", "land", "sport", "audience", "classroom", "law", "hook", "win", "carry", "eye", "smell", "distribution", "research", "country", "dare", "hope", "whereas", "stretch", "library", "if", "delay", "college", "plastic", "book", "present", "use", "worry", "champion", "goal", "economy", "march", "election", "reflection", "midnight", "slide", "inflation", "action", "challenge", "guitar", "coast", "apple", "campaign", "field", "jacket", "sense", "way", "visual", "remove", "weather", "trash", "cable", "regret", "buddy", "beach", "historian", "courage", "sympathy", "truck", "tension", "permit", "nose", "bed", "son", "person", "base", "meat", "usual", "air", "meeting", "worth", "game", "independence", "physical", "brief", "play", "raise", "board", "she", "key", "writing", "pick", "command", "party", "yesterday", "spring", "candidate", "physics", "university", "concern", "development", "change", "string", "target", "instance", "room", "bitter", "bird", "football", "normal", "split", "impression", "wood", "long", "meaning", "stock", "cap", "leadership", "media", "ambition", "fishing", "essay", "salad", "repair", "today", "designer", "night", "bank", "drawing", "inevitable", "phase", "vast", "chip", "anger", "switch", "cry", "twist", "personality", "attempt", "storage", "being", "preparation", "bat", "selection", "white", "technology", "contract", "side", "section", "station", "till", "structure", "tongue", "taste", "truth", "difficulty", "group", "limit", "main", "move", "feeling", "light", "example", "mission", "might", "wait", "wheel", "shop", "host", "classic", "alternative", "cause", "agent", "consist", "table", "airline", "text", "pool", "craft", "range", "fuel", "tool", "partner", "load", "entrance", "deposit", "hate", "article", "video", "summer", "feature", "extreme", "mobile", "hospital", "flight", "fall", "pension", "piano", "fail", "result", "rub", "gap", "system", "report", "suck", "ordinary", "wind", "nerve", "ask", "shine", "note", "line", "mom", "perception", "brother", "reference", "bend", "charge", "treat", "trick", "term", "homework", "bake", "bid", "status", "project", "strategy", "orange", "let", "enthusiasm", "parent", "concentrate", "device", "travel", "poetry", "business", "society", "kiss", "end", "vegetable", "employ", "schedule", "hour", "brave", "focus", "process", "movie", "illegal", "general", "coffee", "ad", "highway", "chemistry", "psychology", "hire", "bell", "conference", "relief", "show", "neat", "funny", "weight", "quality", "club", "daughter", "zone", "touch", "tonight", "shock", "burn", "excuse", "name", "survey", "landscape", "advance", "satisfaction", "bread", "disaster", "item", "hat", "prior", "shopping", "visit", "east", "photo", "home", "idea", "father", "comparison", "cat", "pipe", "winner", "count", "lake", "fight", "prize", "foundation", "dog", "keep", "ideal", "fan", "struggle", "peak", "safety", "solution", "hell", "conclusion", "population", "strain", "alarm", "measurement", "second", "train", "race", "due", "insurance", "boss", "tree", "monitor", "sick", "course", "drag", "appointment", "slice", "still", "care", "patience", "rich", "escape", "emotion", "royal", "female", "childhood", "government", "picture", "will", "sock", "big", "gate", "oil", "cross", "pin", "improvement", "championship", "silly", "help", "sky", "pitch", "man", "diamond", "most", "transition", "work", "science", "committee", "moment", "fix", "teaching", "dig", "specialist", "complex", "guide", "people", "dead", "voice", "original", "break", "topic", "data", "degree", "reading", "recording", "bunch", "reach", "judgment", "lie", "regular", "set", "painting", "mode", "list", "player", "bear", "north", "wonder", "carpet", "heavy", "officer", "negative", "clock", "unique", "baby", "pain", "assumption", "disk", "iron", "bill", "drawer", "look", "double", "mistake", "finish", "future", "brilliant", "contact", "math", "rice", "leave", "restaurant", "discount", "sex", "virus", "bit", "trust", "event", "wear", "juice", "failure", "bug", "context", "mud", "whole", "wrap", "intention", "draft", "pressure", "cake", "dark", "explanation", "space", "angle", "word", "efficiency", "management", "habit", "star", "chance", "finding", "transportation", "stand", "criticism", "flow", "door", "injury", "insect", "surprise", "apartment"]
#taken from IFEVal word list

def split_into_sentences(text):
    """Split the text into sentences.

    Args:
        text: A string that consists of more than or equal to one sentences.

    Returns:
        A list of strings where each string is a sentence.
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(
        _MULTIPLE_DOTS,
        lambda match: "<prd>" * len(match.group(0)) + "<stop>",
        text,
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(
        _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(
        _ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text
    )
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_words(text):
    """Counts the number of words."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"[a-zA-Z']+")
    tokens = tokenizer.tokenize(text)
    num_words = len(tokens)
    return num_words

def split_into_paras(text):
    """Splits a text into paras"""
    return re.split(r'\n\s*\n+', text)

def generate_sentence() -> str:
    #maybe create a generate sentence function in the future (IFEval also doesn't have this yet)
    sentences = ["This is a sentence.", "For now, this is fine.", "Testing testing testing"]
    return random.choice(sentences)

def generate_keywords(k: int = 1) -> list:
    return random.sample(_WORD_LIST, k=k)

# %%
class Instruction:
    #trivial, just to prevent errors
    _CATEGORY = ''
    _LIMITATIONS = []
    _INSTRUCTION_ARGS_KEYS = []

    def __init__(self, instruction_id):
        self._id = instruction_id

    def get_instruction_args_keys(self):
        return self.__class__._INSTRUCTION_ARGS_KEYS

    def get_category(self):
        return self.__class__._CATEGORY

    def get_limitations(self):
        return self.__class__._LIMITATIONS

    def get_description_template(self):
        if hasattr(self.__class__, "_DESCRIPTION_TEMPLATE") and self.__class__._DESCRIPTION_TEMPLATE:
            return self.__class__._DESCRIPTION_TEMPLATE
        else:
            return f"Not templatable.\n Base description is: {self.__class__._DESCRIPTION}"

    def build_description(self, **kwargs):
        raise NotImplementedError("`build_description` not implemented.")

    def get_instruction_args(self):
        raise NotImplementedError("`get_instruction_args` not implemented.")

    def check(self, value):
        raise NotImplementedError("`check_following` not implemented.")

    def get_id(self):
        return self._id
    
    def get_description(self):
        if not hasattr(self, "_description"):
            raise ValueError("Description not found. Call `build_description()` first.")
        return self._description


# %%
def test_basic_methods(dummy):
    print(f".get_category(): {dummy.get_category()}")
    print(f".get_description_template(): {dummy.get_description_template()}")
    print(f".get_instruction_args_keys(): {dummy.get_instruction_args_keys()}")
    print(f".get_limitations(): {dummy.get_limitations()}")
    try:
        print(f".build_description(): {dummy.build_description()}")
    except:
        print("build_description() failed, probably requires arguments...")


# %% [markdown]
# how to use each Instruction child class
# <ul>
# <li>initialize with a unique instruction id</li>
# <li>after initialization, call build_description() even if the instruction is non modifiable</li>
# <li>if instruction is non modifiable, there won't be a _description_template field, only a _description field</li>
# <li>get_instruction_args() returns a dict of args that can be modified, if none, an empty dict is returned</li>
# <li>get_instruction_args_keys() returns a list of input args, if none, an empty list is returned</li>
# <li>check() to score a string</li>
# <li>get_limitations() returns a list of limitations, each limitation in natural language</li>

# %%
#verified checker
class FrequencyChecker(Instruction):
    _CATEGORY = "keyword"
    _DESCRIPTION_TEMPLATE = "In the response, the word or phrase \"{word}\" should appear {natural_relation} {num_words} times."
    _INSTRUCTION_ARGS_KEYS = ["word", "natural_relation", "num_words"]
    _LIMITATIONS = [
        "The basic question must allow for a response to have repeating words.",
        "The basic question must not have restrictions on the number of words in the response."
    ]

    def build_description(self, **kwargs) -> str:
        """
        Building instruction description
        Args:
            word: string
            natural_relation: string, natural relation in LEGAL_NATURAL_RELATIONS
            word_num: int, number of times word should appear

        Returns:
            String representing instruction description

        Raises:
            ValueError if natural_relation is not in _LEGAL_NATURAL_RELATIONS
        """
        word = kwargs.get("word", None)
        natural_relation = kwargs.get("natural_relation", None)
        num_words = kwargs.get("num_words", None)

        if natural_relation is None:
            self._natural_relation = random.choice(list(_LEGAL_NATURAL_RELATIONS.keys()))
        elif natural_relation not in _LEGAL_NATURAL_RELATIONS:
            raise ValueError("The supported relation for comparison must be in "
                       f"{list(_LEGAL_NATURAL_RELATIONS.keys())}, but {natural_relation} is given.")
        else:
            self._natural_relation = natural_relation

        self._word = word if word is not None else generate_keywords()[0]
        self._num_words = random.randint(1,_MAX_NUM) if num_words is None or num_words < 1 else num_words
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(word=self._word, natural_relation=self._natural_relation, num_words=self._num_words)

        return self._description

    def get_instruction_args(self) -> dict:
        """Returns values set by build_description"""
        return {"word": self._word, "natural_relation": self._natural_relation, "num_words": self._num_words}


    def check(self, value: str) -> float:
        """
        Checks a given string to see if instructions are followed. Only individual standalone words are valid.
        Args:
            value: str, input sentence to validate
        Returns:
            float, score between 0 and 1
        Raise:
            ValueError if self._natural_relation is not in _LEGAL_NATURAL_RELATIONS

        Scoring:
        - Perfect score when word appears exactly N times ("exactly"), appears N times or more ("at least"), or N or fewer ("at most")
        - For imperfect matches, score decreases quadratically according to max(0,1-0.1D*D) where D is diff between target and actual freq
        """
        actual_occurrences = len(re.findall(
            rf'(?<!\w){re.escape(self._word)}(?!\w)', value, flags=re.IGNORECASE
        ))

        if self._natural_relation not in list(_LEGAL_NATURAL_RELATIONS.keys()):
            raise ValueError("The supported relation for comparison must be in "
                       f"{list(_LEGAL_NATURAL_RELATIONS.keys())}, but {self._natural_relation} is given.")

        is_success = _LEGAL_NATURAL_RELATIONS[self._natural_relation](actual_occurrences, self._num_words)

        if is_success: return 1.0
        diff = abs(actual_occurrences - self._num_words)

        return round(max(0, 1.0-0.1*(diff*diff)),4)



# %% [markdown]
# <h1> together checker

#verified checker
# %%
class TogetherChecker(Instruction):
    _CATEGORY = "keyword"
    _DESCRIPTION_TEMPLATE = (
        'Your response must contain both "{word1}" and "{word2}" a minimum '
        'of {num_words} times each, with the frequency of "{word1}" exceeding '
        'that of "{word2}".'
    )
    _INSTRUCTION_ARGS_KEYS = ["word1", "word2", "num_words"]
    _LIMITATIONS = [
        "The basic question must allow for a response with different words.",
        "The basic question must allow for repeating words."
    ]

    def build_description(self, **kwargs):
        """
        Args:
            word1: string
            word2: string
            num_words: int, number of times word should appear

        Returns:
            String representing instruction description
         """
        word1 = kwargs.get("word1", None)
        word2 = kwargs.get("word2", None)
        num_words = kwargs.get("num_words", None)

        if word1 is None or word2 is None:
            generated = generate_keywords(2)
            word1 = word1 if word1 is not None else generated[0]
            word2 = word2 if word2 is not None else generated[1]

        self._word1 = word1
        self._word2 = word2

        self._num_words = random.randint(1, _MAX_NUM) if num_words is None or num_words < 1 else num_words
        self._description = self._DESCRIPTION_TEMPLATE.format(word1=self._word1, word2=self._word2, num_words=self._num_words)

        return self._description

    def get_instruction_args(self) -> dict:
        """Returns values set by build_description"""
        return {"word1": self._word1, "word2": self._word2, "num_words": self._num_words}

    def check(self, value: str) -> float:
        """
        Only individual standalone words are valid

        Scoring:
        - Both words appear together: 0.3 points
        - Each word must meet minimum frequency N: 0.15 points each
        - Word1 frequency must be greater than word2 frequency: 0.4 points IFF both meet minimum N
        """
        actual_occurrence_word1 = len(re.findall(
            rf'(?<!\w){re.escape(self._word1)}(?!\w)', value, flags=re.IGNORECASE
            ))
        actual_occurrence_word2 = len(re.findall(
            rf'(?<!\w){re.escape(self._word2)}(?!\w)', value, flags=re.IGNORECASE
            ))
        score = 0

        word1_present = actual_occurrence_word1 > 0
        word2_present = actual_occurrence_word2 > 0
        word1_frequent = actual_occurrence_word1 >= self._num_words
        word2_frequent = actual_occurrence_word2 >= self._num_words

        if word1_present and word2_present:
            score += 0.3
        if word1_frequent:
            score += 0.15
        if word2_frequent:
            score += 0.15
        if word1_frequent and word2_frequent and actual_occurrence_word1 > actual_occurrence_word2:
            score += 0.4

        return score



# %% [markdown]
# <h1>banned checker

# %%
#verified checker
class BannedChecker(Instruction):
  _CATEGORY = "keyword"
  _DESCRIPTION_TEMPLATE = "Your response must NOT contain: {formatted_forbidden_words}."
  _INSTRUCTION_ARGS_KEYS = ["forbidden_words"]
  _LIMITATIONS = [
        "The basic question must not explicitly require the inclusion of specific words."
    ]

  def build_description(self, **kwargs) -> str:
    forbidden_words = kwargs.get("forbidden_words", None)
    """
    Args:
        forbidden_words: a non empty list of strings, is empty or not provided, will generate
    Returns:
        String representing instruction description

    Raises:
        ValueError if list of words is not given
    """
    if forbidden_words is None or len(forbidden_words)==0:
      forbidden_words = generate_keywords(5) #randomly geneate 5 keywords if forbidden_words is none or empty
    elif not all(isinstance(word, str) for word in forbidden_words):
      forbidden_words = [str(word) for word in forbidden_words] #convert all elements to string

    self._forbidden_words = forbidden_words
    self._description = self._DESCRIPTION_TEMPLATE.format(formatted_forbidden_words=", ".join(str(x) for x in forbidden_words))

    return self._description

  def get_instruction_args(self) -> dict:
    """Returns values set by build_description"""
    return {"forbidden_words": self._forbidden_words}

  def check(self, value: str) -> float:
    num_forbidden_words = 0

    """
    my interpretation of the paper: if "cat" and "dog" are forbidden, input "cat cat cat" counts as 3 forbidden words
    alternative interpretation might be that only one forbidden word is present

    Scoring:
    - No forbidden words: 1 point
    - One forbidden word: 0.7 points
    - Two forbidden words: 0.1 point
    - Three or more forbidden words: 0 points
    """
    for word in self._forbidden_words:
      num_forbidden_words += len(
          re.findall(rf'(?<!\w){re.escape(word)}(?!\w)', value, flags=re.IGNORECASE
                     ))

      if num_forbidden_words >= 3: break

    if num_forbidden_words == 0: return 1.0
    elif num_forbidden_words == 1: return 0.7
    elif num_forbidden_words == 2: return 0.1
    return 0.0


# %% [markdown]
# <h1>para end

# %%
#verified checker
class ParagraphEndChecker(Instruction):
  _CATEGORY = "keyword"
  _DESCRIPTION_TEMPLATE = (
      'Your response must contain at least {num_paras} paragraphs,'
      ' and "{word}" must appear in the last sentence of each paragraph.'
  )
  _INSTRUCTION_ARGS_KEYS = ["num_paras", "word"]
  _LIMITATIONS = [
        "The basic question must not have any restrictions on the number of paragraphs.",
        "The basic question must allow for a response with repeating words."
    ]

  def build_description(self, **kwargs) -> str:
    """
    Args:
        num_paras: int, exact number of paras to generate
        word: str word to include in last sentence of each para

    Returns:
        String representing instruction description
    """
    num_paras = kwargs.get("num_paras", None)
    word = kwargs.get("word", None)

    self._num_paras = random.randint(1, _MAX_NUM) if num_paras is None  or num_paras < 1 else num_paras
    self._word = word if word is not None else generate_keywords()[0]
    self._description = self._DESCRIPTION_TEMPLATE.format(num_paras=self._num_paras, word=self._word)

    return self._description

  def get_instruction_args(self) -> dict:
    """Returns values set by build_description"""
    return {"num_paras": self._num_paras, "word": self._word}

  def check(self, value: str) -> float:
    """
    Checks for num of paras, and if keyword appears in last sentence of each paragraph.

    Scoring:
    - If valid paragraph count < N (required para count), score is 0
    - Score = max(0,1-0.2E*E) where E is the number of paras that fail the last sentence requirement
    - Paras assumed to be split by \n\n* (see split_into_paras helper function)
    """
    paragraphs = split_into_paras(value)
    num_paragraphs = len(paragraphs)
    num_invalid_paragraphs = 0

    for paragraph in paragraphs:
      sentences = split_into_sentences(paragraph)
      if not sentences: continue #handle empty paragraphs
      last_sentence = sentences[-1]
      if not re.search(rf'(?<!\w){re.escape(self._word)}(?!\w)', last_sentence, flags=re.IGNORECASE):
        num_invalid_paragraphs += 1

    if num_paragraphs - num_invalid_paragraphs < self._num_paras: return 0.0

    score = round(max(0,1.0-0.2*(num_invalid_paragraphs**2)),4)

    return score

# %% [markdown]
# <h1>first word checker

# %%
#checker verified
class FirstWordChecker(Instruction):
  _CATEGORY = "keyword"
  _DESCRIPTION_TEMPLATE = "The first word of your response must be \"{word}\"."
  _INSTRUCTION_ARGS_KEYS = ["word"]
  _LIMITATIONS = [
        "The basic question must not have any restrictions on the first word of the response."
    ]

  def build_description(self, **kwargs) -> str:
    word = kwargs.get("word", None)
    self._word = word if word is not None else generate_keywords()[0]
    self._description = self._DESCRIPTION_TEMPLATE.format(word=self._word)

    return self._description

  def get_instruction_args(self) -> dict:
     """
     Returns values set by build_description
     """
     return {"word": self._word}

  def check(self, value: str) -> float:
    """
    Binary scoring system, scoring is case-insensitive
    From paper, no idea what "if first section contains #, both first and second sections are checked" means, so i'm skipping this

    Scoring:
    - Binary: 1.0 if first word is correct, 0.0 otherwise
    """
    return 1.0 if re.findall(rf"\A\W*{re.escape(self._word)}", value, flags=re.IGNORECASE) else 0.0 #accounts for punctuation like " too, so this is good


# %% [markdown]
# <h1>max length

# %%
#verified checker (using ifeval implementation)
class MaxWordChecker(Instruction):
    _CATEGORY = "length"
    _DESCRIPTION_TEMPLATE = "Your response word count must not exceed {max_length}."
    _INSTRUCTION_ARGS_KEYS = ["max_length"]
    _LIMITATIONS = [
        "The basic question must not have any restrictions on the word count."
    ]

    def build_description(self, **kwargs):
        max_length = kwargs.get("max_length", None)
        self._max_length = max_length if max_length is not None and max_length > 0 else random.randint(10, _MAX_NUM) #minimum is 10 words
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(max_length=self._max_length)

        return self._description

    def get_instruction_args(self) -> dict:
        return {"max_length": self._max_length}

    def check(self, value: str):
        """
        Word count checking for non English scripts not implemented yet.

        Scoring:
        - When W<= H: Score = 1.0
        - Otherwise: Score = max(0, 1 - 20R × R), where R = |Wordcount-Max|/Max
        """
        word_count = count_words(value)
        if word_count <= self._max_length: return 1.0
        diff_ratio = (word_count - self._max_length)/self._max_length
        return round(max(0, 1.0-20*(diff_ratio**2)), 4)

# %% [markdown]
# <h1>response within a range of lengths

# %%
#verified checker (ifeval implementation of count_words())
class RangeWordsChecker(Instruction):
    _CATEGORY = "length"
    _DESCRIPTION_TEMPLATE = (
        "Your response word count must be between {min_length} and {max_length}."
    )
    _INSTRUCTION_ARGS_KEYS = ["min_length", "max_length"]
    _LIMITATIONS = [
        "The basic question must not have any restrictions on the number of words in the response."
    ]


    def build_description(self, **kwargs):
        min_length = kwargs.get("min_length", None)
        max_length = kwargs.get("max_length", None)

        if max_length is None: max_length = random.randint(11, _MAX_NUM) #smallest max
        if min_length is None: min_length = random.randint(1, max_length-10) #random.randint is inclusive, so possible that max and min lengths are the same
        if max_length - min_length < 10:
            min_length = max(1, max_length - 10) #enforce that there must be at least a range of 10 words
        self._min_length = min_length
        self._max_length = max_length
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(min_length=self._min_length, max_length=self._max_length)

        return self._description

    def get_instruction_args(self) -> dict:
        return {"min_length": self._min_length, "max_length": self._max_length}

    def check(self, value: str) -> float:
        """
        Word count checking for non English scripts not implemented yet.

        Scoring:
        - When L ≤ W ≤ H: Score = 1.0
        - Otherwise: Score = max(0, 1 - 20R × R)
        - If W<L: R = |W-L|/L
        - If W>H: R = |W-H|/H
        """
        word_count = count_words(value)

        if self._min_length <= word_count <= self._max_length: return 1.0

        if word_count < self._min_length:
            diff_ratio = (self._min_length - word_count)/self._min_length
        else:
            diff_ratio = (word_count - self._max_length)/self._max_length
        return round(max(0, 1.0-20*(diff_ratio**2)), 4)


# %% [markdown]
# <h1>postscript at end

# %%
#check verified
class PostscriptAtEnd(Instruction):
    _CATEGORY = "format"
    _DESCRIPTION_TEMPLATE = "Explicitly add a one-sentence postscript beginning with \"{addition}\" at the end of your response."
    _INSTRUCTION_ARGS_KEYS = ["addition"]
    _LIMITATIONS = [
        "The basic question must not have any restrictions on adding a postscript.",
        "The basic question must allow for a response that ends with the specified postscript."
    ]

    def build_description(self, **kwargs) -> str:
        """
        Args:
            addition: str, postscript marker, str should end with ':', but will be added if not already there

        Returns:
            String representing instruction description

        Note:
        - postscript refers to additional remarks at the end of a document. Example: XXX end of doc. PS: Information updated as of 2025
        - assumption that the postscript always starts a line, if postscript is at the end, that means last line is a postscript
        - valid postscripts in the middle also start lines
        - this implementation also makes it such that there must be an exact match with the addition (ie P.S. != P. S. and case sensitive)
        """
        addition = kwargs.get("addition", None)
        if not addition: addition = generate_keywords()[0]
        if not addition.endswith(":"): addition += ":"
        self._addition = addition
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(addition=self._addition)

        return self._description

    def get_instruction_args(self) -> dict:
        return {"addition": self._addition}

    def check(self, value: str) -> float:
        """
        All postscripts must start a line. A perfect postscript is a postscript on the last line.

        Scoring:
        - Presence of required text: 0.5
        - Correct placement at the end: 0.5
        """

        lines = value.strip().split('\n')
        if len(lines[-1])==1: lines = lines[:-1] # guard against possible end in quotation marks on new line
        last_line = lines[-1]

        if re.match(rf"^\s*[^a-zA-Z0-9\s]*{re.escape(self._addition)}", last_line, flags=re.IGNORECASE):
            return 1.0


        #catch if it's anywhere else in the text
        if re.search(rf"(?:^|\n)\s*{re.escape(self._addition)}", value):
            return 0.5

        return 0.0

# %% [markdown]
# <h1>title brackets

# %%
#verified checker
class TitleBracketsChecker(Instruction):
    _CATEGORY = 'format'
    _DESCRIPTION_TEMPLATE = (
        "Your response must include a title enclosed in double angle brackets or book title brackets, like \"«\" and \"»\" " #added example of guillemet
        "and the title should not exceed {max_length} words."
    )
    _INSTRUCTION_ARGS_KEYS = ["max_length"]
    _LIMITATIONS = [
        "The basic question must allow for a response with a title enclosed in double angle brackets or book title brackets.",
        "The basic question must not explicitly require a title exceeding the specified maximum word count."
    ]

    def build_description(self, **kwargs) -> str:
            max_length = kwargs.get("max_length", None)

            self._max_length = max_length if max_length is not None and max_length > 0 else random.randint(1, _MAX_NUM) #could be max_len 1
            self._description = self.__class__._DESCRIPTION_TEMPLATE.format(max_length=self._max_length)
            return self._description

    def get_instruction_args(self) -> dict:
        return {"max_length": self._max_length}

    def check(self, value: str) -> float:
        """
        Scoring:
        - Proper enclosure in « xxx » or "yyy" etc based on language (for eng, "" and guillemets) (0.1 points)
        - Length requirement (0.9 points)
        - Enclosed but length more than self._max_length: Score = 0.1 + max(0, 0.9 - 0.1R × R) where R is the diff ratio

        Note:
        - need more information on what other kinds of title brackets might be accepted
        - considers all titles in the text, but only considers the first illegal title
        """
        valid_pattern = r"\s*«(.*?)»\s*" #can expand this later

        score = 0

        matches = re.findall(valid_pattern, value)
                
        if not matches: return 0.0
        score += 0.1
        for match in matches:
            content_length = count_words(match)
            if content_length > self._max_length:
                diff_ratio = (content_length - self._max_length)/self._max_length
                score += round(max(0, 0.9-0.1*(diff_ratio**2)), 4)
                return score
        return 1.0

# %% [markdown]
# <h1>markdown highlight

# %%
#verified checker
class HighlightChecker(Instruction):
    _CATEGORY = 'format'
    _DESCRIPTION_TEMPLATE = (
        "In your response, highlight at least {num_parts} parts using Markdown, "
        "use double asterisks (**) to mark highlighted text."
    )
    _INSTRUCTION_ARGS_KEYS = ["num_parts"]
    _LIMITATIONS = [
            "The basic question must not have any restrictions on highlighting and the use of double asterisks (**)."
        ]

    def build_description(self, **kwargs) -> str:
        num_parts = kwargs.get("num_parts", None)
        self._num_parts = num_parts if num_parts is not None and num_parts > 0 else random.randint(1, _MAX_NUM) #could be max_len 1
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(num_parts=self._num_parts)
        return self._description

    def get_instruction_args(self) -> dict:
        return {"num_parts": self._num_parts}

    def check(self, value: str) -> float:
        """
        Scoring:
        - When number of highlighted sections > self._num_parts, score is 1.0
        - Otherwise: score = max(0,1-0.1 D*D) where D is diff from requirement
        """
        pattern = r"\s*\*\*(.*?)\*\*\s*"
        highlighted_sections = re.findall(pattern, value)
        num_highlighted_sections = len(highlighted_sections)
        if num_highlighted_sections >= self._num_parts: return 1.0
        diff = self._num_parts - num_highlighted_sections
        return round(max(0,1-0.1*(diff**2)),4)

# %% [markdown]
# <h1>json output

# %%
#verified checker (ifeval implementation)
class JSONChecker(Instruction):
    _CATEGORY = 'format'
    _DESCRIPTION = (
        "Your entire output should be wrapped in JSON format. "
        "Please ensure that the JSON format is valid and can be parsed."
    )
    _INSTRUCTION_ARGS_KEYS = [None] # No modifiable arguments
    _LIMITATIONS = [
        "The basic question must allow the response to be in JSON format."
    ]


    def build_description(self, **kwargs) -> str:
        # No arguments to set, description is fixed
        self._description = self.__class__._DESCRIPTION # Set instance variable for consistency
        return self._description

    def get_instruction_args(self) -> dict:
        return {None} # No modifiable arguments


    def check(self, value) -> float:
        """
        Scoring:
        - Binary scoring: 1 for valid json, else 0
        """
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
        except ValueError as _:
            return 0.0
        return 1.0


# %% [markdown]
# <h1>two answers with separator

# %%
#checker verified
class SeparatorChecker(Instruction):
    _CATEGORY = 'format'
    _DESCRIPTION_TEMPLATE = (
        "You should provide two different responses. "
        "Start with a line break between responses, then separate them with "
        "\"{sentence}\"."
    )
    _INSTRUCTION_ARGS_KEYS = ["sentence"]
    _LIMITATIONS = [
        "The basic question must not have any restrictions on the number of responses.",
        "The basic question must not have any restrictions on the use of separators."
    ]


    def build_description(self, **kwargs) -> str:
        sentence = kwargs.get("sentence", None)
        #random side note: maybe i should make a generate sentence kind of util function later...
        self._sentence = sentence.strip() if sentence is not None else generate_sentence().strip()
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(sentence=self._sentence)
        return self._description

    def get_instruction_args(self) -> dict:
        return {"sentence": self._sentence}

    def check(self, value: str) -> float:
        """
        Scoring:
        - Binary scorer: if separator appears exactly once and in between responses, 1, else 0

        Note:
        - Paper didn't include mention of checking for a response after the separator, but i'll implement that
        bc doesn't make sense to omit this (could just be a last line of the separator and nothing after which
        should be illegal)
        - As per paper, separatopr must appear on its own line between responses, also ignores case
        """
        pattern = rf"^\s*\W*{re.escape(self._sentence)}\W*\s*$\n" #must be followed by a new line
        matches = re.findall(pattern, value, flags=re.MULTILINE | re.IGNORECASE)
        return 1.0 if len(matches)==1 else 0.0

# %% [markdown]
# <h1>markdown title

# %%
#verified checker
class MarkdownTitleChecker(Instruction):
    _CATEGORY = 'format'
    _DESCRIPTION_TEMPLATE = (
        "In your response, a #-marked title, not exceeding {max_length} words is required."
    )
    _INSTRUCTION_ARGS_KEYS = ["max_length"]
    _LIMITATIONS = [
            "The basic question must not have any restrictions on the use of titles.",
            "The basic question must not have any restrictions on the length of titles."
        ]

    def build_description(self, **kwargs) -> str:
        max_length = kwargs.get("max_length", None)
        self._max_length = max_length if max_length is not None and max_length > 0 else random.randint(1, _MAX_NUM) #0 words is allowed, so could just be #\n
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(max_length = self._max_length)
        return self._description

    def get_instruction_args(self) -> dict:
        return {"max_length": self._max_length}

    def check(self, value: str) -> float:
        """
        Scoring:
        - title presence with # marker (0.1 points)
        - title length requirement (0.9 points)
        - title present but length > M: Score = 0.1 + max(0, 0.9 - 0.1D × D), where

        Note:
        - assume that title must be on new line
        - only captures first illegal title, ignores the rest
        """
        pattern = r"^\s*#(.*?)\s*$"
        matches = re.findall(pattern, value, flags=re.MULTILINE)
        if not matches: return 0.0
        for match in matches:
            title_length = count_words(match)
            if title_length > self._max_length:
                diff = title_length - self._max_length
                return round(0.1 + max(0, 0.9 - 0.1*(diff**2)), 4)
        return 1.0


# %% [markdown]
# <h1>ordered list

# %%
#verified checker
class OrderedListChecker(Instruction):
    _CATEGORY = 'format'
    _DESCRIPTION_TEMPLATE = (
        "Your response must include an ordered list with {num_items} items and "
        "each list item should start with a number and a period, such as '1.', "
        "'2.', etc."
    )
    _INSTRUCTION_ARGS_KEYS = ["num_items"]
    _LIMITATIONS = [
            "The basic question must allow for a response with an ordered list.",
            "The basic question must not have any restrictions on the number of items in an ordered list."
        ]

    def build_description(self, **kwargs) -> str:
        num_items = kwargs.get("num_items", None)
        self._num_items = num_items if num_items is not None and num_items >0 else random.randint(1, _MAX_NUM) #could be 1 item
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(num_items=self._num_items)
        return self._description

    def get_instruction_args(self) -> dict:
        return {"num_items": self._num_items}

    def check(self, value: str) -> float:
        """
        Scoring:
        - When number of items >= self._num_items, score = 1.0
        - otherwise, score = max(0, 1-0.1D*D) where D is the difference

        Note:
        - Each ordered list item must start on a new line and being with 1. 2. etc
        - assuming that each list item number will NOT be bolded (as per LLM list generating convention)
        """
        pattern = r"^\s*(\d+)\.(.*?)\s*$"
        num_ordered_items = len(re.findall(pattern, value, flags=re.MULTILINE))
        if num_ordered_items >= self._num_items: return 1.0
        diff = self._num_items - num_ordered_items
        return round(max(0,1-0.1*(diff**2)),4)


# %% [markdown]
# <h1>Markdown bold italic

# %%
#verified checker
class StartBoldItalicChecker(Instruction):
    _CATEGORY = 'format'
    _DESCRIPTION = (
        "In your response, all paragraphs must start with Markdown's \"***\" "
        "to indicate bold and italic."
    )
    _INSTRUCTION_ARGS_KEYS = [None]
    _LIMITATIONS = [
        "The basic question must allow not have restrictions on the use of bold and italic markdown \"***\"."
    ]


    def build_description(self, **kwargs):
        self._description = self.__class__._DESCRIPTION
        return self._description

    def get_instruction_args(self) -> dict:
        return {None}

    def check(self, value: str) -> float:
        """
        Scoring:
        - When all paragraphs start with *** (1.0 point)
        - Else, score = max(0,1-0.1I*I) where I is the num paras not starting with ***

        Note:
        - Paper only checks if the para starts with ***
        - assumption that each para is split according to how split_into_paras is configured
        - initially thought that there should be closing ***, but apparently bold and italic only requires *** in the opening
        """
        pattern = r"^\s*\*\*\*"
        paragraphs = split_into_paras(value)
        num_illegal_paras = 0
        for para in paragraphs:
            if not re.match(pattern, para, re.MULTILINE):
                num_illegal_paras += 1
        if num_illegal_paras == 0: return 1.0
        return round(max(0,1-0.1*(num_illegal_paras**2)),4)


# %% [markdown]
# <h1>copy request

# %%
#SPECIAL, THIS IS THE ONLY CHECKER THAT MUST TAKE THE BQ INTO ACCOUNT
#i want to do some logic here such that this is ALWAYS the first checker
#verified checker
class CopyRequestChecker(Instruction):
    _CATEGORY = 'repeat'
    _DESCRIPTION_TEMPLATE = "Repeat the basic question (without the three constraints at the end of the prompt) without any changes and then provide the answer."
    _INSTRUCTION_ARGS_KEYS = ["prompt_to_repeat"]
    _LIMITATIONS = [
        "The basic question must not have any restrictions on repeating the request."
    ]

    def build_description(self, **kwargs) -> str:
        self._prompt_to_repeat = kwargs.get("prompt_to_repeat", "This is a placeholder basic question. Can you generate a basic response?") #not adding the Instructions here, because otherwise i need to take the other two questions into account too
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format()
        return self._description

    def get_instruction_args(self) -> dict:
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def check(self, value: str) -> float:
        """
        Scoring:
        - Binary, response starts with exact request (1.0 points), otherwise 0

        Notes:
        - Case insensitive
        """
        prompt_len = len(self._prompt_to_repeat)
        if value.strip()[:prompt_len].lower() == self._prompt_to_repeat.lower(): #strip value for safety
            return 1.0
        return 0.0

# %% [markdown]
# <h1>before answer

# %%
#verified checker, made it looser to allow for punctuation
class BeforeAnswerChecker(Instruction):
    _CATEGORY = 'repeat'
    _DESCRIPTION_TEMPLATE = (
        "Repeat {num_repeats} times \"{sentence}\" in the first line before response." #added "in the first line" here, was not in the original paper
    )
    _INSTRUCTION_ARGS_KEYS = ["num_repeats", "sentence"]
    _LIMITATIONS = [
        "The basic question must allow for repetitions."
    ]

    def build_description(self, **kwargs) -> str:
        num_repeats = kwargs.get("num_repeats", None)
        sentence = kwargs.get("sentence", None)
        self._num_repeats = num_repeats if num_repeats is not None and num_repeats > 0 else random.randint(1, _MAX_NUM)
        self._sentence = sentence.strip() if sentence is not None else generate_sentence().strip()
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(num_repeats=self._num_repeats, sentence=self._sentence)
        return self._description

    def get_instruction_args(self) -> dict:
        return {"num_repeats": self._num_repeats, "sentence": self._sentence}

    def check(self, value: str) -> float:
        """
        Scoring
        - Where C is the repetition count, if C==0, score is 0
        - When C>0, score = max(0,1-0.2*diff*diff), where diff is |self._num_repeats - C|
        - Case insensitive
        - Valid repeats are either separated by spaces, or by a single new line, or no separation
        (testtesttest or test\ntest\ntest\n or test test test)

        Notes:
        - might this be too strict? i'm not making this
        """
        actual_repeats = 0
        punct = re.escape(string.punctuation) #allow for punctuation before repetitions
        pattern = rf"^\s*[{punct}]*{re.escape(self._sentence)}[\s{punct}]*"

        while re.match(pattern, value, flags=re.IGNORECASE):
            actual_repeats += 1
            value = re.sub(pattern, '', value, count=1, flags=re.IGNORECASE)

        diff = abs(actual_repeats - self._num_repeats)
        if actual_repeats == 0 and self._num_repeats != 0: return 0.0
        return round(max(0,1-0.2*(diff**2)),4)


# %% [markdown]
# <h1>first and last sentence the same

# %%
#verified checker
class FirstLastSameChecker(Instruction):
    _CATEGORY = 'repeat'
    _DESCRIPTION = "The first sentence of your response should be exactly the same as the last sentence."
    _INSTRUCTION_ARGS_KEYS = [None]
    _LIMITATIONS = [
        "The basic question must allow for repetitions.",
        "The basic question must allow for responses with more than one sentence."
    ]

    def build_description(self, **kwargs) -> str:
        self._description = self.__class__._DESCRIPTION
        return self._description

    def get_instruction_args(self) -> dict:
        return {None}

    def check(self, value: str) -> float:
        """
        Scoring:
        - when first and last are the same (ignoring case), score = 1.0
        - else: 0.0

        Note:
        - only compares specifically the first and last SENTENCES. does not work if there is no punctuation
        - paper apparently "ignores punctuation" but this implementation cares about exact match, i think punctuation matching is
        important
        - Added check for at least two sentences, doesn't make sense if only one sentence in the response
        - punctuation insensitive when it comes to leading punctuation (because of potential conflict with things like wrap in double quotes)
        """
        sentences = split_into_sentences(value)
        if len(sentences) < 2: return 0.0 #added check for at least two sentences
        if len(sentences[-1])==1: sentences = sentences[:-1] #guard against ending with "
        sent_1 = re.sub(rf"^[{re.escape(string.punctuation)}\s]+", "", sentences[0]) #might have a " in front
        sent_2 = re.sub(rf"^[{re.escape(string.punctuation)}\s]+", "", sentences[-1])
        return 1.0 if sent_1.lower().strip() == sent_2.lower().strip() else 0.0


# %% [markdown]
# <h1>last sentence repeat n times

# %%
#verified checker, but this one is a tricky task, especially when combined with other instructions
class LastSentenceChecker(Instruction):
    _CATEGORY = 'repeat'
    _DESCRIPTION_TEMPLATE = (
        "At the end of your response, output the last sentence once, then output a line of #####, then on new lines, repeat the last sentence {num_repeats} times."
        #"Separate the repetitions from the last line using #####."
    )
    _INSTRUCTION_ARGS_KEYS = ["num_repeats"]
    _LIMITATIONS = [
        "The basic question must allow for repetitions.",
        "The basic question must allow for the use of '#####' as a separator.",
        "The basic question must not require the last line of the response to not be a sentence (like a letter etc)."
    ]

    def build_description(self, **kwargs) -> str:
        """
        Note:
        - Original paper description was just "At the end of your response, repeat the last sentence {num_repeats} times."
        - This is quite troublesome because you don't know exactly what the last sentence to repeat is, and where the repetitions begin.
        - If just take the last sentence of the entire output and count how many reps, this is naive because last sentence might be malformed
        and there might be valid repetitions above.
        - Introduced the separator ##### for clarity
        - In the check, if there is no separator, the check defaults to the naive approach
        """
        num_repeats = kwargs.get("num_repeats", None)
        self._num_repeats = num_repeats if num_repeats is not None and num_repeats > 0 else random.randint(1, _MAX_NUM)
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(num_repeats=self._num_repeats)
        return self._description

    def get_instruction_args(self) -> dict:
        return {"num_repeats": self._num_repeats}

    def check(self, value: str) -> float:
        """
        Scoring:
        - No repetitions found: 0.0 points
        - At least 1 repetition: score = max(0,1-0.2D*D), where D = |required reps - actual reps|
        - If no delimiter, max score is 0.5

        Notes:
        - Has to be a well formed sentence ending with .
        - If no delimiter found, naively looks at the last sentence and then counts how many reps, breaks at first mismatch
        - If delimiter found, counts the number of repetitions after the delimiter, does not break early if mismatch found
        - Ignores leading punctuation like ###
        - "Repeat 4 times" means the sentence has to repeat 4 times (including the first time that it was mentioned)
        """
        # value = value.strip()
        # sentences = split_into_sentences(value)
        # last_sentence = sentences[-1].strip().lower()
        # sentences = sentences[:-1][::-1] #reverse sentences too
        # actual_reps = 0
        # for sentence in sentences:
        #     sentence = sentence.lower().strip()
        #     if sentence != last_sentence:
        #         break
        #     actual_reps += 1
        # diff = abs(self._num_repeats - actual_reps)


        text_segments = re.split(r"^\s*#+\s*$", value, flags=re.MULTILINE)

        if len(text_segments) == 1: #no delimiter found
            sentences = split_into_sentences(value)
            if len(sentences[-1])==1: sentences = sentences[:-1] #guard against possible end with double quotes
            last_sentence = sentences[-1].lower().strip()
            last_sentence = re.sub(rf"^[{re.escape(string.punctuation)}\s]+", "", last_sentence)
            trimmed_sentences = sentences[:-1][::-1]
            actual_reps = 0
            for sentence in trimmed_sentences:
                sentence = re.sub(rf"^[{re.escape(string.punctuation)}\s]+", "", sentence) #remove leading punctuation here too
                if sentence.lower().strip() == last_sentence:
                    actual_reps += 1

            if actual_reps == 0 and self._num_repeats != 0: return 0.0
            diff = abs(self._num_repeats - actual_reps)
            return round(min(max(0,1-0.2*(diff**2)), 0.5),4) #max score can only be 0.5 bc no delimiter

        #delimiter found
        #find last sentence of the response
        for i in range(2, len(text_segments)+1):
            text_segment = text_segments[-i].strip()
            if text_segment: break
        else:
            return 0.0 #no valid text_segment before ### found
        sentences = split_into_sentences(text_segment) #don't need to guard against possible ending with " because there's a delimiter
        last_sentence = sentences[-1].lower().strip()
        last_sentence = re.sub(rf"^[{re.escape(string.punctuation)}\s]+", "", last_sentence) #remove all leading punctuation like ###
        repetitions = text_segments[1]
        actual_reps = 0
        for sentence in split_into_sentences(repetitions): #works even if empty after the separator
            sentence = re.sub(rf"^[{re.escape(string.punctuation)}\s]+", "", sentence) #remove leading punctuation here too
            if sentence.lower().strip() == last_sentence.lower().strip():
                actual_reps += 1

        if actual_reps == 0 and self._num_repeats != 0: return 0.0
        diff = abs(self._num_repeats - actual_reps)
        return round(max(0,1-0.2*(diff**2)),4)


# %% [markdown]
# <h1>sentence repeat n times

# %%
#verified checker
class SentenceNTimesChecker(Instruction):
    _CATEGORY = 'repeat'
    _DESCRIPTION_TEMPLATE = (
        "In your response, \"{sentence}\" must appear {num_repeats} times."
    )
    _INSTRUCTION_ARGS_KEYS = ["num_repeats", "sentence"]
    _LIMITATIONS = [
        "The basic question must allow for repetitions.",
        "The basic question must not have any restrictions on the number of sentences in the response."
    ]

    def build_description(self, **kwargs) -> str:
        num_repeats = kwargs.get("num_repeats", None)
        sentence = kwargs.get("sentence", None)
        self._num_repeats = num_repeats if num_repeats is not None and num_repeats > 0 else random.randint(1, _MAX_NUM)
        if sentence is None:
            sentence = generate_sentence().strip()
        self._sentence = sentence.strip() if sentence[-1] in ['.', '!', '?'] else sentence.strip() + '.'
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(num_repeats=self._num_repeats, sentence=self._sentence)
        return self._description

    def get_instruction_args(self) -> dict:
        return {"num_repeats": self._num_repeats, "sentence": self._sentence}

    def check(self, value: str) -> float:
        """
        Scoring:
        - when sentence never appears: score = 0.0
        - when sentence appears at least once: score = max(0, 1-0.2D*D), where D is |required_occurrences - actual|

        Note:
        - case insensitive
        - ignores leading punctuation (like ###)
        """
        sentences = split_into_sentences(value)
        actual_occurrences = 0
        self._sentence = re.sub(rf"^[{re.escape(string.punctuation)}\s]+", "", self._sentence) #strip target sentence of leading punct
        for sentence in sentences:
            sentence = re.sub(rf"^[{re.escape(string.punctuation)}\s]+", "", sentence) #strip search sentence of leading punct
            if sentence.lower().strip() == self._sentence.lower().strip():
                actual_occurrences += 1

        if actual_occurrences == 0 and self._num_repeats != 0: return 0.0
        diff = abs(self._num_repeats - actual_occurrences)
        return round(max(0,1-0.2*(diff**2)),4)


# %% [markdown]
# <h1>all sentences twice

# %%
#verified checker
class AllSentencesTwiceChecker(Instruction):
    _CATEGORY = 'repeat'
    _DESCRIPTION = "All sentences in your response must be repeated twice."
    _INSTRUCTION_ARGS_KEYS = [None]
    _LIMITATIONS = [
        "The basic question must allow for repetitions.",
        "The basic question must not have any restrictions on the number of sentences in the responsd."
    ]

    def build_description(self, **kwargs) -> str:
        self._description = self.__class__._DESCRIPTION
        return self._description

    def get_instruction_args(self) -> dict:
        return {None}

    def check(self, value: str) -> float:
        """
        Scoring:
        - Odd number of sentences: score = 0.0
        - Even number: score = max(0, 1-0.2I*I) where I is the number of invalid pairs

        Note:
        - Must be well formed sentence, otherwise will not work properly
        - Case insensitive
        - Strips away leading punctuation 
        """
        sentences = split_into_sentences(value)
        #guard against possible last sentence as quotation mark
        if len(sentences[-1])==1: sentences = sentences[:-1]
        if len(sentences) % 2 != 0: return 0.0
        num_invalid_pairs = 0
        for i in range(0, len(sentences), 2):
            sent_1 = re.sub(rf"^[{re.escape(string.punctuation)}\s]+", "", sentences[i])
            sent_2 = re.sub(rf"^[{re.escape(string.punctuation)}\s]+", "", sentences[i+1])
            if sent_1.lower().strip() != sent_2.lower().strip():
                num_invalid_pairs += 1

        return round(max(0,1-0.2*(num_invalid_pairs**2)),4)


# %% [markdown]
# <h1>marks

# %%
#verified checker (tricky in combination with other constraints)
class WrapInQuotesChecker(Instruction):
    _CATEGORY = 'marks'
    _DESCRIPTION = "Enclose your entire response in double quotes. The opening and closing quotes must be on new lines." #added the second instruction so that this conflicts with fewer instructions (eg repeat all sentences twice)
    _INSTRUCTION_ARGS_KEYS = [None]
    _LIMITATIONS = [
        "The basic question must not have restrictions on the use of double quotes."
    ]

    def build_description(self, **kwargs) -> str:
        self._description = self.__class__._DESCRIPTION
        return self._description

    def get_instruction_args(self) -> dict:
        return {None}

    def check(self, value: str) -> float:
        """
        Scoring:
        - When response starts and ends with valid quotes, score is 1.0
        - else 0.0
        """
        value = value.strip()
        return 1.0 if value.startswith('"') and value.endswith('"') else 0.0


# %% [markdown]
# <h1> no commas

# %%
#verified checker (pretty simple)
class NoCommasChecker(Instruction):
    _CATEGORY = 'marks'
    _DESCRIPTION = "Avoid using any commas throughout your response."
    _INSTRUCTION_ARGS_KEYS = [None]
    _LIMITATIONS = [
        "The basic question must not require the use of commas.",
        "The basic question must allow for responses without commas."
    ]


    def build_description(self, **kwargs) -> str:
        self._description = self.__class__._DESCRIPTION
        return self._description

    def get_instruction_args(self) -> dict:
        return {None}


    def check(self, value: str) -> float:
        """
        Scoring:
        - C is the total number of commas found
        - score = max(0, 1-0.03C*C)
        """
        num_commas = len(re.findall(r",", value))
        return round(max(0,1-0.03*(num_commas**2)),4)


# %% [markdown]
# <h1>replace all punc with !

# %%
#verified checker
class ReplaceWithExclamationChecker(Instruction):
    _CATEGORY = 'marks'
    _DESCRIPTION = "Replace all commas, periods, and question marks in your response with an exclamation mark."
    """
    Note:
    - original paper prompt was "replace all commas, periods, and question marks in your response into exclamation marks."
    """
    _INSTRUCTION_ARGS_KEYS = [None]
    _LIMITATIONS = [
        "The basic question must not have restrictions on the use of exclamation marks.",
        "The basic question must not have restrictions on the use of commas, periods, and question marks."
    ]

    def build_description(self, **kwargs) -> str:
        self._description = self.__class__._DESCRIPTION
        return self._description

    def get_instruction_args(self) -> dict:
        return {None}

    def check(self, value: str) -> float:
        """
        Scoring:
        - No exclamation marks found: 0.0
        - otherwise, score = max(0,1-0.03W*W) where W is presence of remaining wrong punctuation marks
        - penalizes for every , . ? that hasn't been replaced with !
        """
        num_exclamation_marks = len(re.findall(r"!", value))
        if num_exclamation_marks == 0: return 0.0

        num_illegal_marks = len(re.findall(r"[.,?]", value))
        return round(max(0,1-0.03*(num_illegal_marks**2)),4)


# %% [markdown]
# <h1>end all sentences with semicolon

# %%
#verified checker
class EndWithSemicolonChecker(Instruction):
    _CATEGORY = 'marks'
    _DESCRIPTION = "All sentences in your response must end with a semicolon instead of a period, question, or exclamation mark."
    """
    Note:
    - og description was "All sentences in your response must end with a semicolon instead of a period."
    """
    _INSTRUCTION_ARGS_KEYS = [None]
    _LIMITATIONS = [
        "The basic question must not have restrictions on the use of semicolons.",
        "The basic question must not have restrictions on the use of punctuations."
    ]

    def build_description(self, **kwargs) -> str:
        self._description = self.__class__._DESCRIPTION
        return self._description

    def get_instruction_args(self) -> dict:
        return {None}

    def check(self, value: str) -> float:
        """
        Scoring:
        - W: number of sentences not ending with semicolon (ie ending with . ! or ?)
        - Score = max(0,1-0.03W*W)
        - If no ; found, score is 0.0

        Note:
        - og paper did not penalize if no ; found, but i'm going to penalize in this implementation
        - even if there's only one sentence, it should be well formed sentence with a ; at the end.
        - if there're few sentences and they're all illegally ending with . ! ? then should be 0 bc there are no semicolons
        """
        num_semicolons = len(re.findall(r";", value))
        if num_semicolons == 0: return 0.0
        num_illegal_marks = len(re.findall(r"[.!?]", value))
        return round(max(0,1-0.03*(num_illegal_marks**2)),4)



# %% [markdown]
# <h1>replace with asterisks

# %%
#verified checker
class ReplaceWithAsterisksChecker(Instruction):
    _CATEGORY = 'marks'
    _DESCRIPTION = "In your response, all punctuation marks (commas, periods, exclamation marks, brackets, quotation marks, etc.) must be replaced with asterisks *."
    _INSTRUCTION_ARGS_KEYS = [None]
    _LIMITATIONS = [
        "The basic question must have no restrictions on the use of asterisks.",
        "The basic question must not require to use of commas, periods, exclamation marks, or any other punctuation."
    ]

    def build_description(self, **kwargs) -> str:
        self._description = self.__class__._DESCRIPTION
        return self._description

    def get_instruction_args(self) -> dict:
        return {None}

    def check(self, value: str) -> float:
        """
        Scoring:
        - No asterisks found: 0.0
        - Otherwise: score = max(0,1-0.03W*W) where W is the count of remaining punctuation marks
        """
        num_asterisks = len(re.findall(r"\*", value))
        if num_asterisks == 0: return 0.0
        illegal_patterns = f'[{re.escape(string.punctuation.replace("*",""))}]' #emojis etc are not considered illegal, only punctuation (apart from *)
        num_illegal_marks = len(re.findall(illegal_patterns, value))
        return round(max(0,1-0.03*(num_illegal_marks**2)),4)


# %% [markdown]
# <h1>citations
# %%
#i'm not too sure what this does, so i'm going to leave this out of all the checkers for now
"""Not very sure about what this checker is supposed to be looking for, what's a 'quote'? """
class SquareBracketCitationChecker(Instruction):
    _CATEGORY = 'citation'
    _DESCRIPTION_TEMPLATE = (
            "Your response must contain at least {num_quotes} quotes, and the quoted content must be in [x] format."
        )
    _LIMITATIONS = [
            f"The basic question must allow for quotations to be in [x] format.",
            f"The basic question must allow for '[' and ']' to be generated."
        ]
    _INSTRUCTION_ARGS_KEYS = ["num_quotes"]

    def build_description(self, **kwargs) -> str:
        num_quotes = kwargs.get("num_quotes", None)
        self._num_quotes = num_quotes if num_quotes is not None and num_quotes > 0 else random.randint(1, _MAX_NUM)
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(num_quotes=self._num_quotes)
        return self._description

    def get_instruction_args(self) -> dict:
        return {"num_quotes": self._num_quotes}

    def check(self, value: str) -> float:
        """
        Scoring:
        - No quotations found: 0.0
        - Otherwise score = max(0, 1-0.3D*D)-P
        - D is max(0, N-C) where N is the required number of quotes, C is actual number of quotes
        - P = 0.5 if format invalid, 0 if valid
        """
        allowed_patterns = pattern = r'\[(.*?)\]|"(.*?)"' #question: how to define a "quotation"
        quotations = re.findall(allowed_patterns, value)
        print(quotations)
        num_quotations = len(quotations)
        if num_quotations == 0: return 0.0
        invalid_pattern = r'"(.*?)"' #if there are invalid patterns in the text, penalize
        p = 0
        if re.search(invalid_pattern, value):
            p = 0.5
        diff = max(0, self._num_quotes - num_quotations)
        return round(max(0, 1-0.3*(diff**2))-p, 4)


# %% [markdown]
# <h1>citations start with 0

# %%
#check verified
class StartFromZeroCitationChecker(Instruction):
    _CATEGORY = 'citation'
    _DESCRIPTION = "Your response must contain references, and your references should start from number 0. References should look like [0], [1], [2], etc."
    """
    Note:
    - og paper description was "Your response must contain references, and your references should start from number 0." which is too broad imo.
    - og paper had quite a naive definition of starting from 0, should also penalize if the numbers increase in order
    """
    _LIMITATIONS = [
        "The basic question must not have any restrictions on the use of references.",
        "The basic question must not have any restrictions on the use of digits."
    ]
    _INSTRUCTION_ARGS_KEYS = [None]

    def build_description(self, **kwargs) -> str:
        self._description = self.__class__._DESCRIPTION
        return self._description

    def get_instruction_args(self) -> dict:
        return {None}

    def check(self, value: str) -> float:
        """
        Scoring:
        - Format correctness: 0.5 points, starting from 0: 0.3 points, correct order: 0.2
        - When no [x] format found: 0.0
        - When format correct, score = 0.5 + 0.3Z where Z is 1 if starts with 0, 0 otherwise + 0.2X
        where X is 1 if correct order else 0
        """
        pattern = r'\[(.*?)\]'
        citations = re.findall(pattern, value)
        if len(citations) == 0: return 0.0
        z = 1 if citations[0] == '0' else 0

        try:
            numbers = [int(c) for c in citations]
            is_consecutive = all(b == a + 1 for a, b in zip(numbers, numbers[1:]))
        except ValueError:
            is_consecutive = False
        x = 1 if is_consecutive else 0
        return (0.5 + 0.3*z + 0.2*x)

# %% [markdown]
# <h1> citations inline in parentheses

# %%
#verified checker
class InlineCitationChecker(Instruction):
    _CATEGORY = 'citation'
    _DESCRIPTION = (
        "Your response must contain references, and the references should be included inline "
        "directly in parentheses immediately after any quoted content "
        "rather than at the end of the entire response. Author names should have proper capitalization."
        )
    """
    Note:
    - og paper phrasing of this was "Your response must contain references, and the references
    should be included directly in parentheses after the quoted content rather than at the
    end of the response." but this is too vague, because edge cases like (random text) which are
    not citations will still be matched
    - gold example in the paper was "This is a quote (Smith 2020)", ie a bracketed inline
    reference within the quotation marks. this is the regex pattern i choose to capture
    """
    _LIMITATIONS = [
        "The basic question must not have any restrictions on the use of references.",
        "The basic question must not have any restrictions on the use of parentheses."
    ]
    _INSTRUCTION_ARGS_KEYS = [None]

    def build_description(self, **kwargs) -> str:
        self._description = self.__class__._DESCRIPTION
        return self._description

    def get_instruction_args(self) -> dict:
        return {None}

    def check(self, value: str) -> float:
        """
        Scoring:
        - Binary scoring, when citations in parenthesis throughout text, score=1.0, else 0
        - 0 if no citations or if citations only at the end

        Notes:
        - og paper doesn't have a way to check for citations at the end of the paper
        """
        pattern = r"\([^)]+\b\d{4}\)" #pattern matches for (at-least-one-word 4 digit date)
        citations = re.findall(pattern, value)
        if len(citations) == 0: return 0.0
        return 1.0

# %% [markdown]
# <h1> end with emoji

# %%
#verified checker, made to be compatible with wrap in double quotes too
class EndEmojiChecker(Instruction):
    _CATEGORY = 'emoji'
    _DESCRIPTION_TEMPLATE = "Your response must end with {num_emojis} \"{emoji}\"."
    _INSTRUCTION_ARGS_KEYS = ["num_emojis", "emoji"]
    _LIMITATIONS = [
        "The basic question must not have any restrictions on the use of emojis."
    ]

    def build_description(self, **kwargs) -> str:
        num_emojis = kwargs.get("num_emojis", None)
        emoji = kwargs.get("emoji", None)
        self._num_emojis = num_emojis if num_emojis is not None and num_emojis > 0 else random.randint(1, _MAX_NUM)
        self._emoji = emoji if emoji is not None else random.choice(_EMOJIS)
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(num_emojis=self._num_emojis, emoji=self._emoji)
        return self._description

    def get_instruction_args(self) -> dict:
        return {"num_emojis": self._num_emojis, "emoji": self._emoji}

    def check(self, value: str) -> float:
        """
        Scoring: (og paper left this out, I'm going to use same formula from LastSentenceChecker)
        - No repetitions of emoji: 0.0
        - When at least 1 repetition found: score = max(0,1-0.2D*D) where D = |required_number - actual_number|
        """
        actual_occrrences = 0
        value = value.strip()
        if value.endswith('"'): value = value[:-1].strip() #guard against wrap in double quotation conflict
        while value.endswith(self._emoji):
            value = value[:-1].strip()
            actual_occrrences += 1
        if actual_occrrences == 0 and self._num_emojis != 0: return 0.0
        diff = abs(self._num_emojis - actual_occrrences)
        return round(max(0, 1-0.2*(diff**2)),4)

# %%
#verified checker
class EmojiFrequencyChecker(Instruction):
    _CATEGORY = 'emoji'
    _DESCRIPTION_TEMPLATE = (
        "In your response, emoji \"{emoji}\" should appear {natural_relation} {num_emojis} times."
    )
    _LIMITATIONS = [
        "The basic question must not have restrictions on the number of emojis."
    ]
    _INSTRUCTION_ARGS_KEYS = ["emoji", "natural_relation", "num_emojis"]

    def build_description(self, **kwargs) -> str:
        emoji = kwargs.get("emoji", None)
        natural_relation = kwargs.get("natural_relation", None)
        num_emojis = kwargs.get("num_emojis", None)
        self._emoji = emoji if emoji is not None else random.choice(_EMOJIS)
        self._natural_relation = random.choice(list(_LEGAL_NATURAL_RELATIONS.keys())) if natural_relation is None or natural_relation not in _LEGAL_NATURAL_RELATIONS else natural_relation
        self._num_emojis = num_emojis if num_emojis is not None and num_emojis > 0 else random.randint(1, _MAX_NUM)
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(emoji=self._emoji, natural_relation=self._natural_relation, num_emojis=self._num_emojis)
        return self._description

    def get_instruction_args(self):
        return {"emoji": self._emoji, "natural_relation": self._natural_relation, "num_emojis": self._num_emojis}

    def check(self, value: str) -> float:
        """
        Scoring:
        - Score = max(0, 1 -0.1 D*D) where D varies by relation type
        - "Exactly": D = |C-N|
        - "At least": D = max(0, N-C)
        - "At most": D = max(0, C-N)
        - Where C is actual emoji count, N is the target number

        Note:
        - if "at most", a response with no emojis could be a valid answer too
        """
        actual_occurrences = len(re.findall(rf"{self._emoji}", value))
        if self._natural_relation not in list(_LEGAL_NATURAL_RELATIONS.keys()):
            raise ValueError("The supported relation for comparison must be in "
                      f"{list(_LEGAL_NATURAL_RELATIONS.keys())}, but {self._natural_relation} is given.")

        is_success = _LEGAL_NATURAL_RELATIONS[self._natural_relation](actual_occurrences, self._num_emojis)

        if is_success: return 1.0
        diff = abs(actual_occurrences - self._num_emojis)
        return round(max(0, 1-0.1*(diff**2)),4)

# %% [markdown]
# <h1>banned emoji

# %%
#verified checker
class BannedEmojiChecker(Instruction):
    _CATEGORY = 'emoji'
    _DESCRIPTION_TEMPLATE = "Your response should include emoji expressions, but \"{banned_emoji}\" must not appear."
    _INSTRUCTION_ARGS_KEYS = ["banned_emoji"]
    """
    Note:
    - Only one banned emoji, not a list of banned emojis (which was the case for banned words), following og paper implementation
    """
    _LIMITATIONS = [
        "The basic question must not explicitly require the use of certain emojis.",
        "The basic question must not have any restrictions on the use of emojis."
    ]

    def build_description(self, **kwargs) -> str:
        banned_emoji = kwargs.get("banned_emoji", None)
        self._banned_emoji = banned_emoji if banned_emoji is not None else random.choice(_EMOJIS)
        self._description = self.__class__._DESCRIPTION_TEMPLATE.format(banned_emoji=self._banned_emoji)
        return self._description

    def get_instruction_args(self) -> dict:
        return {"banned_emoji": self._banned_emoji}

    def check(self, value: str) -> float:
        """
        og paper scoring:
        - Contains any emoji: 0.1
        - Avoid banned emoji: 0.9
        - When any emoji AND no banned emoji: 1.0
        - When no emoji AND no banned emoji: 0.9
        - When any emoji AND banned emoji: 0.1
        - When no emoji AND has banned emoji (impossible, but in the paper): 0.0
        - imo, gives too much emphasis to banned emoji, there should be a stronger penalty on having no emojis at all

        Scoring implemented:
        - If has banned emoji: 0.0
        - If has no emoji: 0.5
        - If has emoji but no banned emoji: 1.0

        Notes:
        - uses emoji library to check for emoji expressions
        """
        if not re.search(re.escape(self._banned_emoji), value):
            contains_emoji = any(emoji in value for emoji in list(_EMOJIS))
            return 1.0 if contains_emoji else 0.5

        return 0.0