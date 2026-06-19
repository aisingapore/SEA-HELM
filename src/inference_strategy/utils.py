import io


def check_cached_file_integrity(conversations: list, filepath: str) -> bool:
    """Check integrity of cached response file.

    This function checks if the number of lines in the cached response file matches
    the number of conversations.

    Args:
        conversations: List of conversation histories, each a list of
            role/content message dicts.
        filepath: Path to the cached response file.

    Returns:
        bool: True if the cached file is deemed valid, False otherwise.
    """
    BLOCK_SIZE = 8 * 1024 * 1024  # 8 MB
    with open(filepath, "rb") as f:
        fi = io.FileIO(f.fileno())
        fb = io.BufferedReader(fi, BLOCK_SIZE)
        line_count = sum(1 for _ in iter(lambda: fb.readline(), b""))

    return line_count == len(conversations)
