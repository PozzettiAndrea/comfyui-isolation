"""IPC (Inter-Process Communication) for comfyui-isolation."""

from .bridge import WorkerBridge
from .worker import BaseWorker, register
from .protocol import Request, Response, encode_object, decode_object

__all__ = [
    "WorkerBridge",
    "BaseWorker",
    "register",
    "Request",
    "Response",
    "encode_object",
    "decode_object",
]
