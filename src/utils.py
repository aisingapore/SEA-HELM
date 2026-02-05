import logging
import os
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd


def handle_arg_string(arg: str) -> bool | int | float | str:
    """Handle and convert argument strings to appropriate Python types.

    Converts string arguments to their most appropriate Python type:
    - "true"/"false" (case-insensitive) -> bool
    - Numeric strings -> int
    - Float strings -> float
    - Everything else -> str (unchanged)

    Args:
        arg (str): The argument string to convert

    Returns:
        bool | int | float | str: The converted value in its appropriate type
    """
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def simple_parse_args_string(
    args_string: str,
) -> dict[str, bool | int | float | str]:
    """Parse a simple argument string into a dictionary.

    Parses something like "args1=val1,arg2=val2" into a dictionary with
    automatically converted value types.

    Args:
        args_string (str): Comma-separated key=value pairs to parse

    Returns:
        dict[str, bool | int | float | str]: Dictionary with parsed arguments
            and type-converted values
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        k: handle_arg_string(v) for k, v in [arg.split("=", 1) for arg in arg_list]
    }

    return args_dict


# taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/loggers/utils.py
def get_commit_from_path(repo_path: Path | str) -> str | None:
    """Retrieve Git commit hash from a repository path.

    Extracts the current Git commit hash by reading the .git directory structure.
    Handles both regular .git directories and Git worktrees.

    Args:
        repo_path (Union[Path, str]): Path to the Git repository

    Returns:
        str | None: The Git commit hash if found, None otherwise

    Note:
        This function is adapted from the lm-evaluation-harness project.
    """
    try:
        git_folder = Path(repo_path, ".git")
        if git_folder.is_file():
            git_folder = Path(
                git_folder.parent,
                git_folder.read_text(encoding="utf-8").split("\n")[0].split(" ")[-1],
            )
        if Path(git_folder, "HEAD").exists():
            head_name = (
                Path(git_folder, "HEAD")
                .read_text(encoding="utf-8")
                .split("\n")[0]
                .split(" ")[-1]
            )
            head_ref = Path(git_folder, head_name)
            git_hash = head_ref.read_text(encoding="utf-8").replace("\n", "")
        else:
            git_hash = None
    except Exception as err:
        logging.debug(
            f"Failed to retrieve a Git commit hash from path: {str(repo_path)}. Error: {err}"
        )
        return None
    return git_hash


def get_git_commit_hash() -> str | None:
    """Get the Git commit hash of the current repository.

    Attempts to get the Git commit hash using the 'git describe --always' command.
    If Git is not available or the command fails, falls back to reading the
    .git directory directly.

    Returns:
        str | None: The Git commit hash if found, None otherwise

    Note:
        Source: https://github.com/EleutherAI/gpt-neox/blob/main/megatron/neox_arguments/neox_args.py
    """
    try:
        git_hash = subprocess.check_output(["git", "describe", "--always"]).strip()
        git_hash = git_hash.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # FileNotFoundError occurs when git not installed on system
        git_hash = get_commit_from_path(os.getcwd())  # git hash of repo if exists
    return git_hash


def get_error_count(errors: pd.Series) -> dict[str, int]:
    """Count occurrences of different error types from a pandas Series.

    Processes a Series containing lists of errors and returns a dictionary
    with the count of each unique error type across all entries.

    Args:
        errors (pd.Series): Series where each element is a list of errors

    Returns:
        dict[str, int]: Dictionary mapping error types to their occurrence counts
    """
    error_df = pd.DataFrame(errors.to_list())

    counter = Counter({})
    for _, value in error_df.items():
        counts = Counter(value.value_counts().to_dict())
        counter.update(counts)

    return dict(counter)
