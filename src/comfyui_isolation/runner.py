"""
Generic runner for isolated subprocess execution.

This module is the entry point for subprocess execution. The runner handles
requests for ANY @isolated class in the environment, importing classes on demand.

Usage:
    python -m comfyui_isolation.runner \
        --node-dir /path/to/ComfyUI-SAM3DObjects/nodes \
        --comfyui-base /path/to/ComfyUI \
        --import-paths ".,../vendor"

The runner:
1. Sets COMFYUI_ISOLATION_WORKER=1 (so @isolated decorator becomes no-op)
2. Adds paths to sys.path
3. Reads JSON-RPC requests from stdin
4. Dynamically imports classes as needed (cached)
5. Calls methods and writes responses to stdout
"""

import os
import sys
import json
import argparse
import traceback
import warnings
import logging
import importlib
from typing import Any, Dict, Optional

# Suppress warnings that could interfere with JSON IPC
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.disable(logging.WARNING)

# Mark that we're in worker mode - this makes @isolated decorator a no-op
os.environ["COMFYUI_ISOLATION_WORKER"] = "1"


def setup_paths(node_dir: str, comfyui_base: Optional[str], import_paths: Optional[str]):
    """Setup sys.path for imports."""
    from pathlib import Path

    node_path = Path(node_dir)

    # Set COMFYUI_BASE env var for stubs to use
    if comfyui_base:
        os.environ["COMFYUI_BASE"] = comfyui_base

    # Add comfyui-isolation stubs directory (provides folder_paths, etc.)
    stubs_dir = Path(__file__).parent / "stubs"
    sys.path.insert(0, str(stubs_dir))

    # Add import paths
    if import_paths:
        for p in import_paths.split(","):
            p = p.strip()
            if p:
                full_path = node_path / p
                sys.path.insert(0, str(full_path))

    # Add node_dir itself
    sys.path.insert(0, str(node_path))


def serialize_result(obj: Any) -> Any:
    """Serialize result for JSON transport."""
    from comfyui_isolation.ipc.protocol import encode_object
    return encode_object(obj)


def deserialize_arg(obj: Any) -> Any:
    """Deserialize argument from JSON transport."""
    from comfyui_isolation.ipc.protocol import decode_object
    return decode_object(obj)


# Cache for imported classes and instances
_class_cache: Dict[str, type] = {}
_instance_cache: Dict[str, object] = {}


def get_instance(module_name: str, class_name: str) -> object:
    """Get or create an instance of a class."""
    cache_key = f"{module_name}.{class_name}"

    if cache_key not in _instance_cache:
        # Import the class if not cached
        if cache_key not in _class_cache:
            print(f"[Runner] Importing {class_name} from {module_name}...", file=sys.stderr)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            _class_cache[cache_key] = cls

        # Create instance
        cls = _class_cache[cache_key]
        _instance_cache[cache_key] = cls()
        print(f"[Runner] Created instance of {class_name}", file=sys.stderr)

    return _instance_cache[cache_key]


def run_worker(node_dir: str, comfyui_base: Optional[str], import_paths: Optional[str]):
    """Main worker loop - reads requests from stdin, writes responses to stdout."""

    # Setup paths first
    setup_paths(node_dir, comfyui_base, import_paths)

    # Send ready signal
    ready_msg = {"status": "ready"}
    print(json.dumps(ready_msg), flush=True)

    # Main loop - read requests, execute, respond
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        response = {"jsonrpc": "2.0", "id": None}

        try:
            request = json.loads(line)
            response["id"] = request.get("id")

            method_name = request.get("method")
            params = request.get("params", {})

            if method_name == "shutdown":
                # Clean shutdown
                response["result"] = {"status": "shutdown"}
                print(json.dumps(response), flush=True)
                break

            # Get module/class from request
            module_name = request.get("module")
            class_name = request.get("class")

            if not module_name or not class_name:
                response["error"] = {
                    "code": -32602,
                    "message": "Missing 'module' or 'class' in request",
                }
                print(json.dumps(response), flush=True)
                continue

            # Get or create instance
            try:
                instance = get_instance(module_name, class_name)
            except Exception as e:
                response["error"] = {
                    "code": -32000,
                    "message": f"Failed to import {module_name}.{class_name}: {e}",
                    "data": {"traceback": traceback.format_exc()}
                }
                print(json.dumps(response), flush=True)
                continue

            # Get the method
            method = getattr(instance, method_name, None)
            if method is None:
                response["error"] = {
                    "code": -32601,
                    "message": f"Method not found: {method_name}",
                }
                print(json.dumps(response), flush=True)
                continue

            # Deserialize arguments
            deserialized_params = {}
            for key, value in params.items():
                deserialized_params[key] = deserialize_arg(value)

            # Redirect stdout to stderr during method execution
            # This prevents print() in node code from corrupting JSON protocol
            original_stdout = sys.stdout
            sys.stdout = sys.stderr

            # Also redirect at file descriptor level for C libraries that bypass Python
            # (e.g., pymeshfix prints "INFO- Loaded X vertices..." directly to fd 1)
            stdout_fd = original_stdout.fileno()
            stderr_fd = sys.stderr.fileno()
            stdout_fd_copy = os.dup(stdout_fd)
            os.dup2(stderr_fd, stdout_fd)

            # Call the method
            print(f"[Runner] Calling {class_name}.{method_name}...")
            try:
                result = method(**deserialized_params)
            finally:
                # Restore file descriptor first, then Python stdout
                os.dup2(stdout_fd_copy, stdout_fd)
                os.close(stdout_fd_copy)
                sys.stdout = original_stdout

            # Serialize result
            serialized_result = serialize_result(result)
            response["result"] = serialized_result

            print(f"[Runner] {class_name}.{method_name} completed", file=sys.stderr)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[Runner] Error: {e}", file=sys.stderr)
            print(tb, file=sys.stderr)
            response["error"] = {
                "code": -32000,
                "message": str(e),
                "data": {"traceback": tb}
            }

        print(json.dumps(response), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Isolated node runner")
    parser.add_argument("--node-dir", required=True, help="Node package directory")
    parser.add_argument("--comfyui-base", help="ComfyUI base directory")
    parser.add_argument("--import-paths", help="Comma-separated import paths")

    args = parser.parse_args()

    run_worker(
        node_dir=args.node_dir,
        comfyui_base=args.comfyui_base,
        import_paths=args.import_paths,
    )


if __name__ == "__main__":
    main()
