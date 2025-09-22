import sys
from typing import Optional


def error_message_detail(error: Exception, error_detail: Optional[object] = None) -> str:
    """Return a readable error message with filename and line number when possible.

    error_detail is expected to be the sys module (so callers can pass sys),
    but this function will gracefully handle missing or unexpected values.
    """
    if error_detail is None:
        error_detail = sys

    exc_info = getattr(error_detail, "exc_info", lambda: (None, None, None))()
    _, _, exc_tb = exc_info

    filename = getattr(getattr(exc_tb, "tb_frame", None), "f_code", None)
    filename = getattr(filename, "co_filename", "<unknown>")
    lineno = getattr(exc_tb, "tb_lineno", -1)

    error_message = (
        "error occurred in file [{0}] at line [{1}] with error [{2}]".format(
            filename, lineno, str(error)
        )
    )

    return error_message


class SensorException(Exception):
    def __init__(self, error_message: Exception, error_detail: Optional[object] = None) -> None:
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message