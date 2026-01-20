from datetime import datetime
from enum import Enum
from io import TextIOBase
from types import TracebackType

from tqdm import tqdm


class COLORS(Enum):
    """ANSI color escape codes for styling terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class EloPrintWrapper:
    """A print helper that timestamps messages and optionally logs to file.

    Args:
        log_file (str | None): Optional file path. When provided, messages
            are also appended to this file without ANSI color codes.
    """

    def __init__(self, log_file: str | None = None) -> None:
        self.log_file: str | None = log_file
        self.file_handle: TextIOBase | None = None
        if self.log_file:
            self.file_handle = open(self.log_file, "a", encoding="utf-8")

    def print(
        self,
        *args: object,
        end: str = "\n",
        color: COLORS | None = None,
        show: bool = True,
    ) -> None:
        """Print a timestamped message to stdout and optionally to a log file.

        Args:
            *args (Any): Message parts to print. They will be joined by spaces.
            end (str): End-of-line string (default: "\n").
            color (COLORS | None): Optional color for terminal printing. When
                provided, the message is wrapped with the corresponding ANSI
                escape codes.
            show (bool): Whether to print to stdout. If False, only logs to
                file when a log file is configured.
        """
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare message with timestamp
        message_args = [f"[{timestamp}]"] + list(args)

        # Print to console if show is True
        if show:
            if color is not None:
                print(color.value, end="")
            print(*message_args, end=end)
            if color is not None:
                print(COLORS.ENDC.value, end="")

        # Write to file if log_file is specified
        if self.file_handle:
            # Convert args to string and write without color codes
            message = " ".join(str(arg) for arg in message_args) + end
            self.file_handle.write(message)
            self.file_handle.flush()  # Ensure immediate write to file

    def close(self) -> None:
        """Close the log file if it's open."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def __enter__(self) -> "EloPrintWrapper":
        """Enter the context manager.

        Returns:
            EloPrintWrapper: The current instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager and close any open file handle.

        Args:
            exc_type (type[BaseException] | None): Exception type if an
                exception occurred, otherwise None.
            exc_val (BaseException | None): Exception instance if an
                exception occurred, otherwise None.
            exc_tb (TracebackType | None): Traceback if an exception
                occurred, otherwise None.
        """
        self.close()


class EloTqdmWrapper:
    """A thin wrapper around ``tqdm`` with a toggle to disable progress bars.

    When ``show_progress`` is False, method calls become no-ops while
    preserving the public API surface.

    Args:
        length (int): Total number of steps for the progress bar.
        show_progress (bool): Whether to display a visual progress bar.
    """

    def __init__(self, length: int, show_progress: bool = True) -> None:
        self.show_progress: bool = show_progress
        self.length: int = length
        if self.show_progress:
            self.pbar = tqdm(total=self.length)

    def set_description(self, message: str) -> None:
        """Set the description text shown to the left of the progress bar.

        Args:
            message (str): Description text.
        """
        if self.show_progress:
            self.pbar.set_description(message)

    def update(self, n: int) -> None:
        """Advance the progress bar by ``n`` steps.

        Args:
            n (int): Number of steps to increment.
        """
        if self.show_progress:
            self.pbar.update(n)

    def close(self) -> None:
        """Close the underlying ``tqdm`` progress bar if it is active."""
        if self.show_progress:
            self.pbar.close()
