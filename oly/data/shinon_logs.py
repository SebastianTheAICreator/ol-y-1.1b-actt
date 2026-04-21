"""
Backward-compatible aliases for the old Shinon log module name.

New code should import from ``oly.data.oly_logs`` and use ``load_oly_logs``.
"""

from oly.data.oly_logs import (  # noqa: F401
    compress_log_directory,
    compress_message,
    compress_session_log,
    extract_samples_from_session_log,
    get_messages,
    get_probe_distribution,
    load_json_log,
    load_oly_logs,
)

load_shinon_logs = load_oly_logs

__all__ = [
    "compress_log_directory",
    "compress_message",
    "compress_session_log",
    "extract_samples_from_session_log",
    "get_messages",
    "get_probe_distribution",
    "load_json_log",
    "load_oly_logs",
    "load_shinon_logs",
]
