"""Module for logging utilities."""

import logging
import sys
import os
import dotenv

dotenv.load_dotenv()


def get_log_level() -> int:
    """Get the log level from the environment.

    Returns:
        int: The log level.

    """
    level_str = os.getenv("LOG_LEVEL", "DEBUG").upper()
    return getattr(logging, level_str, logging.DEBUG)


class AnsiColorFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes to log messages if supported."""

    COLORS = {
        "grey": "\033[90m",
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "bold_red": "\033[91m\033[1m",
        "cyan": "\033[96m",
        "reset": "\033[0m",
    }

    LEVEL_COLORS = {
        logging.DEBUG: COLORS["blue"],
        logging.INFO: COLORS["green"],
        logging.WARNING: COLORS["yellow"],
        logging.ERROR: COLORS["red"],
        logging.CRITICAL: COLORS["bold_red"],
    }

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        use_color: bool = True,
    ) -> None:
        """Initialize the AnsiColorFormatter.

        Args:
            fmt (str | None, optional): The format string. Defaults to None.
            datefmt (str | None, optional): The date format. Defaults to None.
            use_color (bool, optional): Whether to use color. Defaults to True.

        """
        super().__init__(fmt, datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format the log message.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message.

        """
        _original_levelname = record.levelname
        _original_msg = record.msg

        if self.use_color:
            color = self.LEVEL_COLORS.get(record.levelno, self.COLORS["reset"])

            record.levelname = f"{color}{record.levelname}{self.COLORS['reset']}"
            asctime = self.formatTime(record, self.datefmt)

            time_str = f"{self.COLORS['grey']}{asctime}{self.COLORS['reset']}"
            name_str = f"{self.COLORS['blue']}({record.name}){self.COLORS['reset']}"
            file_str = (
                f"{self.COLORS['grey']}({record.filename}:"
                f"{record.lineno}){self.COLORS['reset']}"
            )

            formatted_message = (
                f"{time_str} | {record.levelname} | {record.getMessage()} {file_str}"
            )

            return formatted_message

        return super().format(record)


def get_colored_logger(name: str, level: int | None = None) -> logging.Logger:
    """Create and configure a logger.

    Automatically detects if attached to a terminal to decide on coloring.
    """
    if level is None:
        level = get_log_level()

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)

        use_color = False
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            use_color = True

        base_fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s (%(filename)s:%(lineno)d)"

        formatter = AnsiColorFormatter(
            fmt=base_fmt,
            datefmt="%Y-%m-%d %H:%M:%S",
            use_color=use_color,
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    logger.propagate = False

    return logger
