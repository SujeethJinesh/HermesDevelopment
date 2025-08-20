"""Network guard module for hermetic execution.

This module provides network blocking functionality that can be imported
by sitecustomize.py in the venv.
"""

import socket
import os
from typing import Any


class HermeticNetworkError(Exception):
    """Raised when network access is attempted in hermetic mode."""
    pass


# Store original socket functions
_original_socket = socket.socket
_original_create_connection = socket.create_connection
_original_getaddrinfo = socket.getaddrinfo


def _blocked_socket(family: int = socket.AF_INET, *args: Any, **kwargs: Any) -> socket.socket:
    """Replacement socket() that blocks network access."""
    # Allow Unix domain sockets
    if family == socket.AF_UNIX:
        return _original_socket(family, *args, **kwargs)
    
    # Block all network sockets
    raise HermeticNetworkError("Network socket creation blocked in hermetic mode")


def _blocked_create_connection(*args: Any, **kwargs: Any) -> socket.socket:
    """Replacement create_connection() that blocks network access."""
    raise HermeticNetworkError("Network connections blocked in hermetic mode")


def _blocked_getaddrinfo(*args: Any, **kwargs: Any) -> list:
    """Replacement getaddrinfo() that blocks DNS lookups."""
    # Allow localhost lookups
    if args and args[0] in ("localhost", "127.0.0.1", "::1", None):
        return _original_getaddrinfo(*args, **kwargs)
    
    raise HermeticNetworkError("DNS lookups blocked in hermetic mode")


def install_network_guard() -> None:
    """Install network blocking by replacing socket module functions."""
    if os.environ.get("HERMES_HERMETIC") != "1":
        return
    
    # Replace socket module functions
    socket.socket = _blocked_socket
    socket.create_connection = _blocked_create_connection
    socket.getaddrinfo = _blocked_getaddrinfo


def uninstall_network_guard() -> None:
    """Restore original socket functions."""
    socket.socket = _original_socket
    socket.create_connection = _original_create_connection
    socket.getaddrinfo = _original_getaddrinfo