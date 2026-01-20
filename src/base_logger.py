import logging


class MultiLineFormatter(logging.Formatter):
    """Multi-line formatter."""

    def get_header_length(self, record: logging.LogRecord) -> int:
        """Get the header length of a given record.

        Args:
            record (logging.LogRecord): The log record to get the header length of.

        Returns:
            int: The header length of the log record.
        """
        return len(
            super().format(
                logging.LogRecord(
                    name=record.name,
                    level=record.levelno,
                    pathname=record.pathname,
                    lineno=record.lineno,
                    msg="",
                    args=(),
                    exc_info=None,
                )
            )
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format a record with added indentation.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log record.
        """
        indent = " " * (self.get_header_length(record) - 2) + "| "
        head, *trailing = super().format(record).split("\n")
        formatted_text = [head] + [indent + line for line in trailing]
        return "\n".join(formatted_text)


def setup_root_logger(filepath: str | None = None) -> logging.Logger:
    """Setup the root logger.

    Args:
        filepath (str, optional): The filepath to the log file. Defaults to None.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = MultiLineFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if filepath is not None:
        # setup file handler
        file_handler = logging.FileHandler(filepath)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The logger.
    """
    return logging.getLogger(name)


if __name__ == "__main__":
    setup_root_logger()
    logger = get_logger(__name__)

    message = """This is a multiline test message

This is line two of the message\n"""

    logger.error(message)
