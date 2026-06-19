import subprocess

from src.base_logger import get_logger
from src.sandbox.base_sandbox import BaseSandbox
from src.sandbox.docker_sandbox import DockerSandbox
from src.sandbox.enroot_sandbox import EnrootSandbox
from src.sandbox.podman_sandbox import PodmanSandbox
from src.sandbox.singularity_sandbox import SingularitySandbox

logger = get_logger(__name__)

sandbox_classes = {
    "singularity": SingularitySandbox,
    "enroot": EnrootSandbox,
    "podman": PodmanSandbox,
    "docker": DockerSandbox,
}


def get_sandbox(default: str | None = None) -> BaseSandbox:
    """Factory method to get a sandbox instance.

    Args:
        default (str | None, optional): The default sandbox to use if available. Defaults to None.

    Returns:
        BaseSandbox: An instance of a sandbox. Currently returns a SingularitySandbox.
    """
    if default and default.lower() in sandbox_classes:
        try:
            result = subprocess.run(
                f"{default.lower()} -v",
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode == 0:
                logger.info(
                    f"{default} is available. Using {sandbox_classes[default.lower()].__name__}."
                )
                return sandbox_classes[default.lower()]()
        except subprocess.CalledProcessError:
            logger.info(f"{default} is not available. Falling back to auto-detection.")

    for sandbox_name, sandbox_class in sandbox_classes.items():
        try:
            result = subprocess.run(
                f"which {sandbox_name}",
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode == 0:
                logger.info(
                    f"{sandbox_name} is available. Using {sandbox_class.__name__}."
                )
                return sandbox_class()
        except subprocess.CalledProcessError:
            logger.info(f"{sandbox_name} is not available. Trying next option.")

    raise EnvironmentError(
        "No compatible sandbox environment found. Please install Enroot, Singularity, Podman, or Docker."
    )
