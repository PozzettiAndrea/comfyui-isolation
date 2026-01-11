"""
IPC Protocol - Message format for bridge-worker communication.

Uses JSON for simplicity and debuggability. Large binary data (images, tensors)
is serialized as base64-encoded strings within the JSON.
"""

import json
import base64
import pickle
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class Request:
    """
    Request message from bridge to worker.

    Attributes:
        id: Unique request ID for matching responses
        method: Method name to call on worker
        args: Keyword arguments for the method
    """
    id: str
    method: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "Request":
        """Deserialize from JSON string."""
        d = json.loads(data)
        return cls(**d)


@dataclass
class Response:
    """
    Response message from worker to bridge.

    Attributes:
        id: Request ID this is responding to
        result: Result value (None if error)
        error: Error message (None if success)
        traceback: Full traceback string (only if error)
    """
    id: str
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if response indicates success."""
        return self.error is None

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "Response":
        """Deserialize from JSON string."""
        d = json.loads(data)
        return cls(**d)


def encode_binary(data: bytes) -> str:
    """Encode binary data as base64 string."""
    return base64.b64encode(data).decode('utf-8')


def decode_binary(encoded: str) -> bytes:
    """Decode base64 string to binary data."""
    return base64.b64decode(encoded)


def encode_object(obj: Any) -> Dict[str, Any]:
    """
    Encode a Python object for JSON serialization.

    Returns a dict with _type and _data keys for special types,
    or the original object if it's JSON-serializable.

    Special handling for ComfyUI types:
    - IMAGE: torch tensor (B, H, W, C) float32 - encoded as "comfyui_image"
    - MASK: torch tensor (B, H, W) or (H, W) float32 - encoded as "comfyui_mask"
    """
    if obj is None:
        return None

    # Handle torch tensors (including ComfyUI IMAGE/MASK)
    if hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
        arr = obj.cpu().numpy()
        shape = arr.shape

        # Detect ComfyUI types by shape
        # IMAGE: (B, H, W, C) where C is typically 3 or 4
        # MASK: (B, H, W) or (H, W)
        obj_type = "tensor"  # Default
        if len(shape) == 4 and shape[-1] in (3, 4):
            obj_type = "comfyui_image"
        elif len(shape) in (2, 3) and arr.dtype in ('float32', 'float64'):
            # Could be a mask - check if values are in [0, 1] range
            if arr.min() >= 0 and arr.max() <= 1:
                obj_type = "comfyui_mask"

        return {
            "_type": obj_type,
            "_dtype": str(arr.dtype),
            "_shape": list(shape),
            "_data": encode_binary(pickle.dumps(arr)),
        }

    # Handle numpy arrays
    if hasattr(obj, '__array__'):
        import numpy as np
        arr = np.asarray(obj)
        return {
            "_type": "numpy",
            "_dtype": str(arr.dtype),
            "_shape": list(arr.shape),
            "_data": encode_binary(pickle.dumps(arr)),
        }

    # Handle PIL Images
    if hasattr(obj, 'save') and hasattr(obj, 'mode'):
        import io
        buffer = io.BytesIO()
        obj.save(buffer, format="PNG")
        return {
            "_type": "image",
            "_format": "PNG",
            "_data": encode_binary(buffer.getvalue()),
        }

    # Handle bytes
    if isinstance(obj, bytes):
        return {
            "_type": "bytes",
            "_data": encode_binary(obj),
        }

    # Handle lists/tuples recursively
    if isinstance(obj, (list, tuple)):
        encoded = [encode_object(item) for item in obj]
        return {
            "_type": "list" if isinstance(obj, list) else "tuple",
            "_data": encoded,
        }

    # Handle dicts recursively
    if isinstance(obj, dict):
        return {k: encode_object(v) for k, v in obj.items()}

    # For simple objects with __dict__, serialize as dict
    # This avoids pickle module path issues across process boundaries
    if hasattr(obj, '__dict__') and not hasattr(obj, '__slots__'):
        return {
            "_type": "object",
            "_class": obj.__class__.__name__,
            "_data": {k: encode_object(v) for k, v in obj.__dict__.items()},
        }

    # For complex objects that can't be JSON serialized, use pickle
    try:
        json.dumps(obj)
        return obj  # JSON-serializable, return as-is
    except (TypeError, ValueError):
        return {
            "_type": "pickle",
            "_data": encode_binary(pickle.dumps(obj)),
        }


def decode_object(obj: Any) -> Any:
    """
    Decode a JSON-deserialized object back to Python types.

    Reverses the encoding done by encode_object.
    """
    if obj is None:
        return None

    if not isinstance(obj, dict):
        return obj

    # Check for special encoded types
    obj_type = obj.get("_type")

    if obj_type == "numpy":
        return pickle.loads(decode_binary(obj["_data"]))

    if obj_type == "tensor":
        import torch
        arr = pickle.loads(decode_binary(obj["_data"]))
        return torch.from_numpy(arr)

    # ComfyUI IMAGE: (B, H, W, C) tensor
    if obj_type == "comfyui_image":
        import torch
        arr = pickle.loads(decode_binary(obj["_data"]))
        return torch.from_numpy(arr)

    # ComfyUI MASK: (B, H, W) or (H, W) tensor
    if obj_type == "comfyui_mask":
        import torch
        arr = pickle.loads(decode_binary(obj["_data"]))
        return torch.from_numpy(arr)

    if obj_type == "image":
        import io
        from PIL import Image
        buffer = io.BytesIO(decode_binary(obj["_data"]))
        return Image.open(buffer)

    if obj_type == "bytes":
        return decode_binary(obj["_data"])

    if obj_type == "pickle":
        return pickle.loads(decode_binary(obj["_data"]))

    # Simple object serialized as dict - restore as SimpleNamespace
    if obj_type == "object":
        from types import SimpleNamespace
        data = {k: decode_object(v) for k, v in obj["_data"].items()}
        ns = SimpleNamespace(**data)
        ns._class_name = obj.get("_class", "unknown")
        return ns

    if obj_type in ("list", "tuple"):
        decoded = [decode_object(item) for item in obj["_data"]]
        return decoded if obj_type == "list" else tuple(decoded)

    # Regular dict - decode values recursively
    return {k: decode_object(v) for k, v in obj.items()}
