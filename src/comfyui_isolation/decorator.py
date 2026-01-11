"""
Decorator-based API for easy subprocess isolation.

This module provides the @isolated decorator that makes it simple to run
ComfyUI node methods in isolated subprocess environments.

Architecture:
    When a method is called in the HOST process (ComfyUI), the decorator
    intercepts the call, spawns a subprocess using the isolated venv, and
    forwards the call to that subprocess.

    When the same module is imported in the WORKER subprocess, the decorator
    detects COMFYUI_ISOLATION_WORKER=1 env var and becomes a transparent no-op,
    allowing the method to execute normally.

    This means NO CODE GENERATION - the original node file IS the worker.

Example:
    from comfyui_isolation import isolated

    @isolated(env="myenv", requirements=["torch", "heavy-package"])
    class MyNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",)}}

        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "process"
        CATEGORY = "MyNodes"

        def process(self, image):
            # This code runs in isolated subprocess
            import torch
            import heavy_package
            return (heavy_package.run(image),)
"""

import os
import sys
import inspect
import threading
import subprocess
import json
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from .env.config import IsolatedEnv
from .env.config_file import load_env_from_file
from .env.manager import IsolatedEnvManager
from .ipc.protocol import encode_object, decode_object


# Global cache for environment managers and subprocess handles
_env_cache: Dict[str, IsolatedEnvManager] = {}
_process_cache: Dict[str, subprocess.Popen] = {}
_cache_lock = threading.Lock()


def _is_worker_mode() -> bool:
    """Check if we're running inside the worker subprocess."""
    return os.environ.get("COMFYUI_ISOLATION_WORKER") == "1"


def isolated(
    env: str,
    requirements: Optional[List[str]] = None,
    config: Optional[str] = None,
    python: str = "3.10",
    cuda: Optional[str] = "auto",
    timeout: float = 600.0,
    log_callback: Optional[Callable[[str], None]] = None,
    import_paths: Optional[List[str]] = None,
):
    """
    Class decorator that makes node methods run in isolated subprocess.

    The decorated class's FUNCTION method will be intercepted and executed
    in an isolated Python environment with its own dependencies.

    Args:
        env: Name of the isolated environment (used for caching)
        requirements: List of pip requirements (e.g., ["torch", "numpy"])
        config: Path to TOML config file (relative to node directory)
        python: Python version for the isolated env (default: "3.10")
        cuda: CUDA version ("12.4", "12.8", "auto", or None for CPU)
        timeout: Default timeout for calls in seconds (default: 10 minutes)
        log_callback: Optional callback for logging
        import_paths: List of directories to add to sys.path in subprocess
                      (relative to node directory, e.g., [".", "../vendor"])
    """
    def decorator(cls):
        # If we're in worker mode, be a no-op - just return the class unchanged
        if _is_worker_mode():
            return cls

        # --- HOST MODE: Wrap the class to proxy calls to subprocess ---

        # Get the FUNCTION attribute to know which method to intercept
        func_name = getattr(cls, 'FUNCTION', None)
        if not func_name:
            raise ValueError(
                f"Node class {cls.__name__} must have FUNCTION attribute. "
                f"This tells ComfyUI which method to call."
            )

        original_method = getattr(cls, func_name, None)
        if original_method is None:
            raise ValueError(
                f"Node class {cls.__name__} has FUNCTION='{func_name}' but "
                f"no method with that name is defined."
            )

        # Get the source file directory
        source_file = Path(inspect.getfile(cls))
        node_dir = source_file.parent

        # Handle if we're in a nodes/ subdirectory
        if node_dir.name == "nodes":
            node_package_dir = node_dir.parent
        else:
            node_package_dir = node_dir

        # Determine the module path for importing in subprocess
        # Just use the file stem - node_dir is added to sys.path
        module_name = source_file.stem  # e.g., "depth_estimate"

        # Only proxy the FUNCTION method - helper methods are called internally
        # and don't need proxying (they run in the subprocess with the main method)
        proxy = _create_proxy_method(
            env_name=env,
            requirements=requirements,
            config_path=config,
            python_version=python,
            cuda_version=cuda,
            method_name=func_name,
            module_name=module_name,
            class_name=cls.__name__,
            node_dir=node_dir,
            node_package_dir=node_package_dir,
            default_timeout=timeout,
            log_callback=log_callback,
            original_method=original_method,
            import_paths=import_paths,
        )

        setattr(cls, func_name, proxy)

        # Store metadata on class
        cls._isolated_env = env
        cls._isolated_node_dir = node_dir

        return cls

    return decorator


def _create_proxy_method(
    env_name: str,
    requirements: Optional[List[str]],
    config_path: Optional[str],
    python_version: str,
    cuda_version: Optional[str],
    method_name: str,
    module_name: str,
    class_name: str,
    node_dir: Path,
    node_package_dir: Path,
    default_timeout: float,
    log_callback: Optional[Callable],
    original_method: Callable,
    import_paths: Optional[List[str]] = None,
) -> Callable:
    """Create a proxy method that forwards calls to the isolated worker."""

    log = log_callback or (lambda msg: print(f"[{env_name}] {msg}"))

    @wraps(original_method)
    def proxy(self, *args, timeout: Optional[float] = None, **kwargs):
        # Get or create subprocess
        process = _get_or_create_process(
            env_name=env_name,
            requirements=requirements,
            config_path=config_path,
            python_version=python_version,
            cuda_version=cuda_version,
            module_name=module_name,
            class_name=class_name,
            node_dir=node_dir,
            node_package_dir=node_package_dir,
            log_callback=log,
            import_paths=import_paths,
        )

        # Handle positional arguments by binding to signature
        sig = inspect.signature(original_method)
        try:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            call_kwargs = dict(bound.arguments)
            del call_kwargs['self']
        except TypeError:
            call_kwargs = kwargs

        # Serialize arguments
        serialized_params = {}
        for key, value in call_kwargs.items():
            serialized_params[key] = encode_object(value)

        # Build JSON-RPC request
        import random
        request_id = random.randint(1, 1000000)
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "module": module_name,
            "class": class_name,
            "method": method_name,
            "params": serialized_params,
        }

        # Send request
        request_json = json.dumps(request) + "\n"
        process.stdin.write(request_json)
        process.stdin.flush()

        # Read response with timeout
        actual_timeout = timeout if timeout is not None else default_timeout

        import select
        ready, _, _ = select.select([process.stdout], [], [], actual_timeout)

        if not ready:
            # Timeout - kill process and raise
            process.kill()
            _remove_process(env_name, node_package_dir)
            raise TimeoutError(f"Method {method_name} timed out after {actual_timeout}s")

        response_line = process.stdout.readline()
        if not response_line:
            # Process died
            _remove_process(env_name, node_package_dir)
            raise RuntimeError(f"Worker process died unexpectedly")

        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from worker: {repr(response_line[:200])}\nError: {e}")

        # Check for error
        if "error" in response:
            error = response["error"]
            error_msg = error.get("message", "Unknown error")
            error_data = error.get("data", {})
            tb = error_data.get("traceback", "")

            # Re-raise as exception so ComfyUI can display it
            raise RuntimeError(f"{error_msg}\n\nWorker traceback:\n{tb}")

        # Deserialize result
        result = response.get("result")
        return decode_object(result)

    return proxy


def _get_or_create_process(
    env_name: str,
    requirements: Optional[List[str]],
    config_path: Optional[str],
    python_version: str,
    cuda_version: Optional[str],
    module_name: str,
    class_name: str,
    node_dir: Path,
    node_package_dir: Path,
    log_callback: Callable,
    import_paths: Optional[List[str]] = None,
) -> subprocess.Popen:
    """Get or create a subprocess for the given environment."""

    cache_key = f"{env_name}:{node_package_dir}"

    with _cache_lock:
        if cache_key in _process_cache:
            proc = _process_cache[cache_key]
            if proc.poll() is None:  # Still running
                return proc
            else:
                # Process died, remove from cache
                del _process_cache[cache_key]

    # Ensure environment is set up
    env_manager, env_config = _get_or_create_env(
        env_name=env_name,
        requirements=requirements,
        config_path=config_path,
        python_version=python_version,
        cuda_version=cuda_version,
        node_package_dir=node_package_dir,
        log_callback=log_callback,
    )

    # Get Python executable from isolated env
    python_exe = env_manager.get_python(env_config)

    # Determine ComfyUI base directory
    comfyui_base = None
    if node_package_dir.parent.name == "custom_nodes":
        comfyui_base = node_package_dir.parent.parent

    # Build import paths string
    import_paths_str = None
    if import_paths:
        import_paths_str = ",".join(import_paths)

    # Build command to run the runner module
    cmd = [
        str(python_exe),
        "-m", "comfyui_isolation.runner",
        "--node-dir", str(node_dir),
    ]

    if comfyui_base:
        cmd.extend(["--comfyui-base", str(comfyui_base)])

    if import_paths_str:
        cmd.extend(["--import-paths", import_paths_str])

    log_callback(f"Starting worker process...")
    log_callback(f"  Python: {python_exe}")

    # Spawn subprocess
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
        cwd=str(node_dir),
    )

    # Wait for ready signal with timeout - DON'T start stderr thread yet
    # so we can capture startup errors properly
    import select
    startup_timeout = 30  # 30 seconds to start up

    ready, _, _ = select.select([process.stdout], [], [], startup_timeout)

    if not ready:
        # Timeout waiting for ready signal
        process.kill()
        stderr_output = process.stderr.read()
        raise RuntimeError(f"Worker failed to start (timeout after {startup_timeout}s):\n{stderr_output}")

    ready_line = process.stdout.readline()

    if not ready_line:
        # Process exited without sending ready signal
        process.wait()  # Ensure process is done
        stderr_output = process.stderr.read()
        raise RuntimeError(f"Worker process failed to start:\n{stderr_output}")

    try:
        ready_msg = json.loads(ready_line)
    except json.JSONDecodeError:
        process.kill()
        stderr_output = process.stderr.read()
        raise RuntimeError(f"Worker sent invalid ready signal: {ready_line}\nStderr:\n{stderr_output}")

    if ready_msg.get("status") != "ready":
        process.kill()
        stderr_output = process.stderr.read()
        raise RuntimeError(f"Worker sent unexpected ready signal: {ready_msg}\nStderr:\n{stderr_output}")

    log_callback(f"Worker ready")

    # NOW start stderr forwarding thread (after successful startup)
    def forward_stderr():
        for line in process.stderr:
            log_callback(line.rstrip())

    stderr_thread = threading.Thread(target=forward_stderr, daemon=True)
    stderr_thread.start()

    # Cache the process
    with _cache_lock:
        _process_cache[cache_key] = process

    return process


def _remove_process(env_name: str, node_package_dir: Path):
    """Remove a process from cache."""
    cache_key = f"{env_name}:{node_package_dir}"
    with _cache_lock:
        if cache_key in _process_cache:
            del _process_cache[cache_key]


def _get_or_create_env(
    env_name: str,
    requirements: Optional[List[str]],
    config_path: Optional[str],
    python_version: str,
    cuda_version: Optional[str],
    node_package_dir: Path,
    log_callback: Callable,
) -> tuple:
    """Get or create an environment manager and config.

    Returns:
        Tuple of (IsolatedEnvManager, IsolatedEnv)
    """

    cache_key = f"{env_name}:{node_package_dir}"

    with _cache_lock:
        if cache_key in _env_cache:
            return _env_cache[cache_key]

    # Auto-discover config file if not specified
    resolved_config_path = config_path
    if not resolved_config_path:
        candidates = [
            f"comfyui_isolation_reqs.toml",
            f"comfyui_isolation_{env_name}.toml",
            f"{env_name}_isolation.toml",
            f"isolation.toml",
        ]
        for candidate in candidates:
            if (node_package_dir / candidate).exists():
                resolved_config_path = candidate
                break

    if resolved_config_path:
        config_file = node_package_dir / resolved_config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        env_config = load_env_from_file(config_file, node_package_dir)
    else:
        if requirements is None:
            requirements = []

        actual_cuda = cuda_version
        if cuda_version == "auto":
            from .env.detection import detect_cuda_version
            actual_cuda = detect_cuda_version()

        env_config = IsolatedEnv(
            name=env_name,
            python=python_version,
            cuda=actual_cuda,
            requirements=requirements,
        )

    # Create environment manager
    env_manager = IsolatedEnvManager(base_dir=node_package_dir, log_callback=log_callback)

    log_callback("=" * 50)
    log_callback(f"Setting up isolated environment: {env_name}")
    log_callback("=" * 50)

    # Ensure environment is ready
    if env_manager.is_ready(env_config):
        log_callback("Environment already ready, skipping setup")
    else:
        log_callback("Creating isolated environment...")
        env_manager.setup(env_config)

    # Cache the manager and config together
    result = (env_manager, env_config)
    with _cache_lock:
        _env_cache[cache_key] = result

    return result


def shutdown_all_processes():
    """Shutdown all cached processes."""
    with _cache_lock:
        for process in _process_cache.values():
            try:
                # Send shutdown command
                shutdown_req = json.dumps({"jsonrpc": "2.0", "id": 0, "method": "shutdown"}) + "\n"
                process.stdin.write(shutdown_req)
                process.stdin.flush()
                process.wait(timeout=5)
            except Exception:
                process.kill()
        _process_cache.clear()


# Register cleanup on module unload
import atexit
atexit.register(shutdown_all_processes)
