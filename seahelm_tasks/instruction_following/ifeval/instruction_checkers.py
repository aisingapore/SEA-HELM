import json
import re
import string

import fast_langdetect

COMPARISON_RELATIONS = {"exactly", "at least", "less than"}

"""
Instruction Checkers
====================
1. Length Constraints
    - Number of words (NumberOfWordsChecker)
    - Number of sentences (NumberOfSentencesChecker)
    - Number of paragraphs (NumberOfParagraphsChecker)

2. Detectable Content
    - Postscript (PostscriptChecker)
    - Placeholders (PlaceholderChecker)

3. Detectable Format
    - Title (TitleChecker)
    - Bullet lists (BulletListChecker)
    - Sections (SectionChecker)
    - Highlighted sections (HighlightedSectionChecker)
    - JSON (JSONChecker)
    - Constrained options (ConstrainedOptionsChecker)

4. Start-End
    - Start with specific word (StartChecker)
    - End with specific phrase (EndChecker)
    - Wrap with quotation marks (QuotationMarkChecker)

5. Keywords
    - Existence of keywords (WordExistenceChecker)
    - Forbidden keywords (WordAbsenceChecker)
    - Frequency of keyword (WordFrequencyChecker)
    - Frequency of number (NumberFrequencyChecker)

6. Combination
    - Repeat prompt then answer (RepeatPromptChecker)
    - Separate two different responses with marker (TwoResponsesChecker)

7. Language
    - Check answer of response (ResponseLanguageChecker)
"""


class Checker:
    """
    Base Checker class for SEA-IFEval instruction-following evaluation.
    Stores information about the category of instruction and the model's response.
    Classes built on top of Checker will store additional kwargs specific to each category
    as defined in the dataset.
    """

    def __init__(self, category: str, language: str, response: str):
        self.category, self.subcategory = category.split(":", 1)
        self.model_response = response
        self.language = language

    def evaluate_response(self, value):
        raise NotImplementedError("`evaluate_response` not implemented.")


#########################
# 1. Length Constraints #
#########################


class NumberOfWordsChecker(Checker):
    def __init__(
        self, category: str, language: str, response: str, relation: str, num_words: int
    ):
        assert language not in {
            "th",
            "lo",
            "km",
            "my",
        }, f"NumberOfWordsChecker should not be run for language {language}"
        super().__init__(category, language, response)
        self.comparison_relation = relation
        self.num_words = int(num_words)

    def count_words(self, text: str):
        """
        Counts the number of words (defined as space-separated units of text).
        Only applicable to English, Malay, Indonesian, Vietnamese, Filipino, Tamil.
        Thai, Lao, Burmese and Khmer do not use spaces in between words.
        Hyphenated words are counted as two words.
        """
        # Equivalent to IFEval's use of NLTK's RegexpTokenizer
        regexp = re.compile(r"\w+", re.UNICODE | re.MULTILINE | re.DOTALL)

        return len(regexp.findall(text))

    def evaluate_response(self):
        assert self.comparison_relation in COMPARISON_RELATIONS, (
            f"{self.comparison_relation} is not a valid comparison relation."
        )

        num_words = self.count_words(self.model_response)

        if self.comparison_relation == "exactly":
            return num_words == self.num_words

        elif self.comparison_relation == "at least":
            return num_words >= self.num_words

        elif self.comparison_relation == "less than":
            return num_words < self.num_words


class NumberOfSentencesChecker(Checker):
    def __init__(
        self,
        category: str,
        language: str,
        response: str,
        relation: str,
        num_sentences: int,
    ):
        super().__init__(category, language, response)
        self.comparison_relation = relation
        self.num_sentences = int(num_sentences)

    def count_sentences(self, text: str):
        """
        Count the number of sentences in a text (defined as texts separated by
        *** divider. (This is a divider specified in the prompts.))
        If there are texts found between the dividers that are merely spaces or newlines,
        those will be removed and not counted as sentences.

        Differs from the original IFEval's implementation as they did not specify separation
        of sentences by asterisks in their prompts. This was implemented for SEA-IFEval to
        avoid the problem of ambiguity in sentence boundaries for languages like Thai.
        """
        sentences = text.strip("*").split("***")
        sentences = [s for s in sentences if s.strip()]

        return len(sentences)

    def evaluate_response(self):
        assert self.comparison_relation in COMPARISON_RELATIONS, (
            f"{self.comparison_relation} is not a valid comparison relation."
        )

        num_sentences = self.count_sentences(self.model_response)

        if self.comparison_relation == "exactly":
            return num_sentences == self.num_sentences

        elif self.comparison_relation == "at least":
            return num_sentences >= self.num_sentences

        elif self.comparison_relation == "less than":
            return num_sentences < self.num_sentences


class NumberOfParagraphsChecker(Checker):
    def __init__(
        self, category: str, language: str, response: str, num_paragraphs: int
    ):
        super().__init__(category, language, response)
        self.num_paragraphs = int(num_paragraphs)

    def count_paragraphs(self, text: str):
        """
        Counts the number of paragraphs (defined as blocks of text separated by ***).
        The original IFEval implementation fails the model if any block created from
        splitting by *** is empty (except the first and last blocks).
        This means a model that separates paragraphs by flanking the paragraph on both sides
        with *** will likely fail. However, this might be unfair since the prompt is not
        entirely clear on how the *** divider is meant to be used.
        We therefore simply do not count empty blocks as a paragraph, but do not fail the
        model immediately.
        """
        paragraphs = re.split(r"\s?\*\*\*\s?", text)
        paragraphs = [p for p in paragraphs if p.strip()]
        return len(paragraphs)

    def evaluate_response(self):
        """
        Unlike for words and sentences, only checks if the number of paragraphs is
        exactly equivalent to the required number.
        """
        num_paragraphs = self.count_paragraphs(self.model_response)

        return num_paragraphs == self.num_paragraphs


#########################
# 2. Detectable Content #
#########################


class PlaceholderChecker(Checker):
    def __init__(
        self, category: str, language: str, response: str, num_placeholders: str
    ):
        super().__init__(category, language, response)
        self.num_placeholders = int(num_placeholders)

    def count_placeholders(self, text: str):
        """
        Searches for all instances of [...] in the text.
        Differs from the original IFEval by ensuring that placeholders are not empty
        square brackets (in which case they would not actually be placeholders as
        instructed in the prompts).
        """
        placeholders = re.findall(r"\[.*?\]", text)
        # Ensure that placeholders are indeed placeholders and not just empty square brackets
        placeholders = [p for p in placeholders if p.strip("[").strip("]").strip()]
        return len(placeholders)

    def evaluate_response(self):
        """
        Checks if there are at least X number of placeholders (that are not empty).
        """
        num_placeholders = self.count_placeholders(self.model_response)

        return num_placeholders >= self.num_placeholders


class PostscriptChecker(Checker):
    def __init__(
        self, category: str, language: str, response: str, postscript_marker: str
    ):
        super().__init__(category, language, response)
        self.postscript_marker = postscript_marker
        # Original IFEval searched for P.P.S (without the final period)
        self.postscript_patterns = {
            "P.S.": r"\s*p\.\s?s\..*$",
            "P.P.S.": r"\s*p\.\s?p\.\s?s\..*$",
            "ป.ล.": r"\s*ป\.\s?ล\..*$",
            "ป.ป.ล.": r"\s*ป\.\s?ป\.\s?ล\..*$",
            "பி.கு.": r"\s*பி\.\s?கு\..*$",
            "பி.பி.கு.": r"\s*பி\.\s?பி\.\s?கு\..*$",
            "ປ.ລ.": r"\s*ປ\.\s?ລ\..*$",
        }

    def evaluate_response(self):
        text = self.model_response.lower()
        postscript_pattern = self.postscript_patterns[self.postscript_marker]
        postscript_exists = re.findall(postscript_pattern, text, flags=re.MULTILINE)

        return True if postscript_exists else False


########################
# 3. Detectable Format #
########################

CONSTRAINED_OPTIONS = {
    "en": {"My answer is yes.", "My answer is no.", "My answer is maybe."},
    "id": {
        "Jawaban saya adalah ya.",
        "Jawaban saya adalah tidak.",
        "Jawaban saya adalah mungkin.",
    },
    "vi": {
        "Câu trả lời của tôi là có.",
        "Câu trả lời của tôi là không.",
        "Câu trả lời của tôi là có thể.",
    },
    "tl": {"Ang sagot ko ay oo.", "Ang sagot ko ay hindi.", "Ang sagot ko ay siguro."},
    "th": {"คำตอบของฉันคือใช่", "คำตอบของฉันคือไม่", "คำตอบของฉันคืออาจจะ"},
    "jv": {"Jawabanku ya.", "Jawabanku ora.", "Jawabanku mungkin."},
    "su": {"Jawapan abdi enya.", "Jawapan abdi henteu.", "Jawapan abdi meureun."},
    "ta": {"என்னுடைய பதில் ஆம்.", "என்னுடைய பதில் இல்லை.", "என்னுடைய பதில் ஒருவேளை இருக்கலாம்."},
    "ms": {
        "Jawapan saya ialah ya.",
        "Jawapan saya ialah tidak.",
        "Jawapan saya ialah mungkin.",
    },
    "my": {"ကျွန်ုပ်၏အဖြေသည် ဟုတ်ပါတယ်။", "ကျွန်ုပ်၏အဖြေသည် မဟုတ်ပါ။", "ကျွန်ုပ်၏အဖြေသည် တစ်ချို့ဖြစ်နိုင်ပါတယ်။"},
    "lo": {"ຄໍາຕອບຂອງຂ້ອຍແມ່ນຈິງ.", "ຄໍາຕອບຂອງຂ້ອຍແມ່ນບໍ່ຈິງ.", "ຄໍາຕອບຂອງຂ້ອຍແມ່ນອາດຈະຈິງ."},
    "km": {"ខ្ញុំយល់ស្រប។", "ខ្ញុំមិនយល់ស្រប។", "ខ្ញុំមិនច្បាស់ដែរ។"},
}


class HighlightedSectionChecker(Checker):
    def __init__(
        self, category: str, language: str, response: str, num_highlights: int
    ):
        super().__init__(category, language, response)
        self.num_highlights = int(num_highlights)

    def count_highlighted_sections(self, text: str):
        highlights = re.findall(r"\*[^\n\*]*\*", text)
        highlights = [h for h in highlights if h.strip("*").strip()]

        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", text)
        # Equivalent to original IFEval's implementation of removing ** prefix and ** suffix
        double_highlights = [h for h in double_highlights if h.strip("*").strip()]

        return len(highlights + double_highlights)

    def evaluate_response(self):
        num_highlights = self.count_highlighted_sections(self.model_response)
        return num_highlights >= self.num_highlights


class BulletListChecker(Checker):
    def __init__(self, category: str, language: str, response: str, num_bullets: int):
        super().__init__(category, language, response)
        self.num_bullets = int(num_bullets)

    def count_bullet_points(self, text: str):
        asterisk_points = re.findall(
            r"^\s*\* \*?\*?[^\*\n\s].*$", text, flags=re.MULTILINE
        )
        hyphen_points = re.findall(
            r"^\s*- \*?\*?[^\*\n\s].*$", text, flags=re.MULTILINE
        )
        num_bullets = len(asterisk_points) + len(hyphen_points)
        return num_bullets

    def evaluate_response(self):
        """
        Checks if the response contains the correct number of bullet points (* or -).
        Unlike the original IFEval:
          1. There must be a space after the bullet point for it to be counted
             (to avoid cases where the bullet point symbol is used for other markdown purposes)
          2. The character after the space can be an asterisk (likely bolded point)
        """
        num_bullets = self.count_bullet_points(self.model_response)
        return num_bullets == self.num_bullets


class SectionChecker(Checker):
    def __init__(
        self,
        category: str,
        language: str,
        response: str,
        section_title: str,
        num_sections: int,
    ):
        super().__init__(category, language, response)
        self.section_title = section_title
        self.num_sections = int(num_sections)

    def count_sections(self, text: str):
        # Original IFEval used \s? + section_title + \s?d+\s?
        # This misses out on cases where letters are used instead of numbers
        # \w seems to include Thai characters and numbers as well
        # We use \s+ instead of \s? to require at least one space between the section title and the number
        section_title_pattern = r"\b" + self.section_title + r"\s+[\d|\w]+\b"

        # Equivalent to original IFEval's method of splitting by pattern and
        # subtracting 1 from number of split components
        sections = re.findall(section_title_pattern, text)
        return len(sections)

    def evaluate_response(self):
        num_sections = self.count_sections(self.model_response)
        return num_sections == self.num_sections


class JSONChecker(Checker):
    def __init__(self, category: str, language: str, response: str):
        super().__init__(category, language, response)

    def evaluate_response(self):
        """
        Checks if the entire output is wrapped in JSON. If the model says anything outside
        of the JSON block, it will fail this task.
        """
        # Original IFEval code removed prefixes (```json, ```JSON) sequentially, but if a model uses
        # multiple prefixes, it should be wrong, since that is invalid Markdown. We change it to remove
        # prefixes only if present, so that removal only happens once. Ending backticks are also only
        # removed if the prefix is present.
        text = self.model_response.strip()
        if text.startswith("```json"):
            text = text.removeprefix("```json").removesuffix("```").strip()
        elif text.startswith("```Json"):
            text = text.removeprefix("```Json").removesuffix("```").strip()
        elif text.startswith("```JSON"):
            text = text.removeprefix("```JSON").removesuffix("```").strip()
        elif text.startswith("```"):
            text = text.removeprefix("```").removesuffix("```").strip()

        try:
            json.loads(text)
        except ValueError as _:
            return False
        return True


class ConstrainedOptionsChecker(Checker):
    def __init__(self, category: str, language: str, response: str):
        super().__init__(category, language, response)
        self.options = CONSTRAINED_OPTIONS[language]

    def evaluate_response(self):
        """Checks if the output contains any of the provided options."""
        for option in self.options:
            if option in self.model_response.strip():
                return True

        return False


class TitleChecker(Checker):
    def __init__(self, category: str, language: str, response: str):
        super().__init__(category, language, response)

    def evaluate_response(self):
        title_pattern = r"<<[^\n]+>>"
        titles = re.findall(title_pattern, self.model_response)
        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False


################
# 4. Start-End #
################


class StartChecker(Checker):
    def __init__(self, category: str, language: str, response: str, first_word: str):
        super().__init__(category, language, response)
        self.first_word = first_word

    def evaluate_response(self):
        text = self.model_response.strip().lower()
        return text.startswith(self.first_word.lower())


class EndChecker(Checker):
    def __init__(self, category: str, language: str, response: str, end_phrase: str):
        super().__init__(category, language, response)
        self.end_phrase = end_phrase

    def evaluate_response(self):
        text = self.model_response.strip().lower()
        return text.endswith(self.end_phrase.lower())


class QuotationMarkChecker(Checker):
    def __init__(self, category: str, language: str, response: str):
        super().__init__(category, language, response)

    def evaluate_response(self):
        text = self.model_response.strip()
        # Equivalent to original IFEval's check for index 0 and -1
        if text.startswith('"') and text.endswith('"'):
            # Original IFEval checked for length of text with the punctuation included
            # Changed to check for content within quotation marks instead
            if text.strip('"').strip():
                return True

        return False


###############
# 5. Keywords #
###############


class WordExistenceChecker(Checker):
    def __init__(self, category: str, language: str, response: str, keywords: list):
        super().__init__(category, language, response)
        self.keywords = keywords

    def evaluate_response(self):
        text = self.model_response
        for word in self.keywords:
            if self.language in {"en", "id", "vi", "jv", "su", "ms"}:
                # Original IFEval does not include \b around the keyword
                if not re.search(r"\b" + word + r"\b", text, flags=re.IGNORECASE):
                    return False
            elif self.language in {"my", "ta", "tl", "th", "lo", "km"}:
                if not re.search(word, text):
                    return False
        return True


class WordAbsenceChecker(Checker):
    def __init__(
        self, category: str, language: str, response: str, forbidden_words: list
    ):
        super().__init__(category, language, response)
        self.forbidden_words = forbidden_words

    def evaluate_response(self):
        text = self.model_response
        for word in self.forbidden_words:
            if self.language in {"en", "id", "vi", "jv", "su", "ms"}:
                if re.search(r"\b" + word + r"\b", text, flags=re.IGNORECASE):
                    return False
            elif self.language in {"my", "ta", "tl", "th", "lo", "km"}:
                if re.search(word, text):
                    return False
        return True


class WordFrequencyChecker(Checker):
    def __init__(
        self,
        category: str,
        language: str,
        response: str,
        keyword: str,
        relation: str,
        frequency: int,
    ):
        super().__init__(category, language, response)
        self.keyword = keyword
        self.comparison_relation = relation
        self.frequency = int(frequency)

    def count_keyword_occurrences(self, text: str):
        if self.language in {"en", "id", "vi", "jv", "su", "ms"}:
            pattern = r"\b" + self.keyword + r"\b"
        elif self.language in {"my", "ta", "tl", "th", "lo", "km"}:
            pattern = self.keyword

        return len(re.findall(pattern, text, flags=re.IGNORECASE))

    def evaluate_response(self):
        assert self.comparison_relation in COMPARISON_RELATIONS, (
            f"{self.comparison_relation} is not a valid comparison relation."
        )

        num_occurrences = self.count_keyword_occurrences(self.model_response)

        if self.comparison_relation == "exactly":
            return num_occurrences == self.frequency

        elif self.comparison_relation == "at least":
            return num_occurrences >= self.frequency

        elif self.comparison_relation == "less than":
            return num_occurrences < self.frequency


class NumberFrequencyChecker(Checker):
    def __init__(
        self,
        category: str,
        language: str,
        response: str,
        number: str,
        relation: str,
        frequency: int,
    ):
        super().__init__(category, language, response)
        self.number = number
        self.comparison_relation = relation
        self.frequency = int(frequency)

    def count_number_occurrences(self, text: str):
        return len(re.findall(self.number, text))

    def evaluate_response(self):
        assert self.comparison_relation in COMPARISON_RELATIONS, (
            f"{self.comparison_relation} is not a valid comparison relation."
        )

        num_occurrences = self.count_number_occurrences(self.model_response)

        if self.comparison_relation == "exactly":
            return num_occurrences == self.frequency

        elif self.comparison_relation == "at least":
            return num_occurrences >= self.frequency

        elif self.comparison_relation == "less than":
            return num_occurrences < self.frequency


##################
# 6. Combination #
##################


class RepeatPromptChecker(Checker):
    def __init__(
        self, category: str, language: str, response: str, prompt_to_repeat: str
    ):
        super().__init__(category, language, response)
        self.prompt_to_repeat = prompt_to_repeat

    def evaluate_response(self):
        text = self.model_response.strip().lower()
        prompt_is_repeated = text.startswith(self.prompt_to_repeat.strip().lower())
        # Original IFEval only checked whether the prompt is repeated.
        # However, models sometimes just repeat the prompt without answering the question.
        answer_exists = text.removeprefix(self.prompt_to_repeat.strip().lower()).strip()
        if prompt_is_repeated and answer_exists:
            return True
        return False


class TwoResponsesChecker(Checker):
    def __init__(self, category: str, language: str, response: str):
        super().__init__(category, language, response)

    def evaluate_response(self):
        text = self.model_response.strip()
        responses = text.split("******")
        # Added punctuation stripping to ensure that there is indeed content in the split responses
        responses = [r for r in responses if r.strip().strip(string.punctuation)]
        if len(responses) == 2:
            return True
        return False


###############
# 7. Language #
###############


class ResponseLanguageChecker(Checker):
    def __init__(
        self, category: str, language: str, response: str, response_language: str
    ):
        super().__init__(category, language, response)
        self.response_language = response_language

    def evaluate_response(self):
        # FastText cannot work when \n is in the text
        text = self.model_response.replace("\n", " ")

        # Using a version of langdetect that is better and covers more languages
        # Original langdetect used in IFEval did not cover some SEA languages.
        prediction = fast_langdetect.detect(text)
        if self.response_language == "id":
            if prediction["lang"] in {"id", "ms"}:
                return True
        elif prediction["lang"] == self.response_language:
            return True

        return False
