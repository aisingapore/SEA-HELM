import subprocess
from typing import Callable

import torch.utils.collect_env as collect_env
from torch.utils.collect_env import get_pretty_env_info as get_pretty_env_info


def custom_get_pip_packages(
    run_lambda: Callable,
    patterns: list[str] | None = None,
) -> tuple[str, str]:
    """
    This version of get_pip_packages calls 'uv pip list' instead of 'pip list'.
    It returns a tuple of a dummy uv-pip version and the output of the command.

    Args:
        run_lambda (Callable): The lambda function to run.
        patterns (list[str], optional): The patterns to filter the output. Defaults to None.

    Returns:
        tuple[str, str]: A tuple of a dummy uv-pip version and the output of the command.

    Raises:
        Exception: If an error occurs while running the command.
    """
    try:
        # Run "uv pip list" instead of the standard "pip list"
        cmd = "uv pip list"
        completed = subprocess.run(
            cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        filtered_out = completed.stdout.decode("utf-8").strip() or ""
    except Exception:
        # In case of failure, return an empty string so that splitlines() has something to work on.
        filtered_out = ""
    # Optionally, you can get the version of uv pip by running "uv pip --version"
    try:
        cmd_version = "uv pip --version"
        completed = subprocess.run(
            cmd_version.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        uv_pip_version = completed.stdout.decode("utf-8").strip()
    except Exception:
        uv_pip_version = "uv-pip (version unknown)"
    return uv_pip_version, filtered_out


try:
    uv_version = subprocess.run(
        "uv --version".split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    if uv_version.returncode == 0:
        collect_env.get_pip_packages = custom_get_pip_packages
except FileNotFoundError:
    # uv is not installed, but allowed to pass in case conda is being used instead.
    pass
