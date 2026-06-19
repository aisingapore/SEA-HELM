def merge_ta_words():
    """Read and merge ta_words.txt and ta_words_2.txt files."""

    words_1 = []
    words_2 = []

    # Read ta_words.txt
    try:
        with open("ta_words.txt", "r", encoding="utf-8") as f:
            words_1 = [line.strip() for line in f if line.strip()]
        print(f"Read {len(words_1)} words from ta_words.txt")
    except FileNotFoundError:
        print("ta_words.txt not found")

    print(len(words_1))
    # Read ta_words_2.txt
    try:
        with open("english-to-tamil-glossary-book.txt", "r", encoding="utf-8") as f:
            words_2 = [line.strip() for line in f if line.strip()]

        for word in words_2:
            _word = word.split(" : ")[1]
            _words = _word.split(" / ")
            for w in _words:
                _word = w.strip()
                if len(_word.split(" ")) <= 2:
                    words_1.append(_word)
        print(f"Read {len(words_2)} words from ta_words_2.txt")
    except FileNotFoundError:
        print("ta_words_2.txt not found")

    print(len(set(words_1)))
    return set(words_1)


def write_merged_words(words, output_file="ta_words_merged.txt"):
    """Write merged words to output file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for word in sorted(words):
                f.write(f"{word}\n")
        print(f"Wrote {len(words)} words to {output_file}")
    except IOError as e:
        print(f"Error writing to {output_file}: {e}")


if __name__ == "__main__":
    words_1 = merge_ta_words()
    write_merged_words(words_1)
