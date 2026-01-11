"""
Decorator-based API for easy subprocess isolation.

This module provides the @isolated decorator that makes it simple to run
ComfyUI node methods in isolated subprocess environments.

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

import inspect
import hashlib
import textwrap
import threading
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

from .env.config import IsolatedEnv
from .env.config_file import load_env_from_file
from .ipc.bridge import WorkerBridge
from .ipc.protocol import encode_object, decode_object


# Global cache for bridges
_bridge_cache: Dict[str, WorkerBridge] = {}
_bridge_cache_lock = threading.Lock()


def isolated(
    env: str,
    requirements: Optional[List[str]] = None,
    config: Optional[str] = None,
    python: str = "3.10",
    cuda: Optional[str] = "auto",
    timeout: float = 300.0,
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
        timeout: Default timeout for calls in seconds (default: 5 minutes)
        log_callback: Optional callback for logging
        import_paths: List of directories to add to sys.path in subprocess
                      (relative to node directory, e.g., ["worker", "vendor"])

    Example with inline requirements:
        @isolated(env="esrgan", requirements=["torch>=2.0", "realesrgan"])
        class ESRGANUpscale:
            FUNCTION = "upscale"

            def upscale(self, image, scale):
                import torch
                from realesrgan import RealESRGANer
                model = RealESRGANer(scale=scale)
                return (model.enhance(image),)

    Example with TOML config and import paths (for complex node packages):
        @isolated(env="sam3d", config="comfyui_isolation_reqs.toml",
                  import_paths=["worker", "vendor"])
        class SAM3DNode:
            FUNCTION = "generate"

            def generate(self, image, mask):
                from worker.stages import run_generate_slat
                result = run_generate_slat(...)
                return result
    """
    def decorator(cls):
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
            node_dir = node_dir.parent

        # Extract source code of the method
        try:
            source = inspect.getsource(original_method)
        except OSError as e:
            raise ValueError(
                f"Could not get source code of {cls.__name__}.{func_name}. "
                f"The @isolated decorator requires the method source to be available. "
                f"Error: {e}"
            )

        # Check for additional isolated methods
        isolated_methods = getattr(cls, 'ISOLATED_METHODS', [func_name])
        if func_name not in isolated_methods:
            isolated_methods = [func_name] + list(isolated_methods)

        # Collect all method sources
        method_sources = {}
        for method_name in isolated_methods:
            method = getattr(cls, method_name, None)
            if method is not None:
                try:
                    method_sources[method_name] = inspect.getsource(method)
                except OSError:
                    pass  # Skip methods without accessible source

        # Create proxy methods for each isolated method
        for method_name in isolated_methods:
            if method_name not in method_sources:
                continue

            method_source = method_sources[method_name]
            original = getattr(cls, method_name)

            # Create the proxy
            proxy = _create_proxy_method(
                env_name=env,
                requirements=requirements,
                config_path=config,
                python_version=python,
                cuda_version=cuda,
                method_name=method_name,
                method_sources=method_sources,
                node_dir=node_dir,
                default_timeout=timeout,
                log_callback=log_callback,
                original_method=original,
                import_paths=import_paths,
            )

            setattr(cls, method_name, proxy)

        # Store metadata on class
        cls._isolated_env = env
        cls._isolated_methods = isolated_methods
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
    method_sources: Dict[str, str],
    node_dir: Path,
    default_timeout: float,
    log_callback: Optional[Callable],
    original_method: Callable,
    import_paths: Optional[List[str]] = None,
) -> Callable:
    """
    Create a proxy method that forwards calls to the isolated worker.
    """
    @wraps(original_method)
    def proxy(self, *args, timeout: Optional[float] = None, **kwargs):
        # Get or create bridge
        bridge = _get_or_create_bridge(
            env_name=env_name,
            requirements=requirements,
            config_path=config_path,
            python_version=python_version,
            cuda_version=cuda_version,
            method_sources=method_sources,
            node_dir=node_dir,
            log_callback=log_callback,
            import_paths=import_paths,
        )

        # Handle positional arguments by binding to signature
        sig = inspect.signature(original_method)
        try:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            call_kwargs = dict(bound.arguments)
            del call_kwargs['self']  # Don't send self
        except TypeError as e:
            # Fall back to just using kwargs
            call_kwargs = kwargs

        # Call the worker
        actual_timeout = timeout if timeout is not None else default_timeout
        result = bridge.call(method_name, timeout=actual_timeout, **call_kwargs)

        return result

    return proxy


def _get_or_create_bridge(
    env_name: str,
    requirements: Optional[List[str]],
    config_path: Optional[str],
    python_version: str,
    cuda_version: Optional[str],
    method_sources: Dict[str, str],
    node_dir: Path,
    log_callback: Optional[Callable],
    import_paths: Optional[List[str]] = None,
) -> WorkerBridge:
    """
    Get or create a WorkerBridge for the given environment.

    Creates the isolated environment and generates worker code on first call.
    """
    # Create cache key
    cache_key = f"{env_name}:{node_dir}"

    with _bridge_cache_lock:
        if cache_key in _bridge_cache:
            return _bridge_cache[cache_key]

    # Generate worker file
    worker_dir = node_dir / f"_generated_{env_name}"
    worker_dir.mkdir(exist_ok=True)
    worker_file = worker_dir / "__main__.py"

    # Generate worker code from method sources
    worker_code = _generate_worker_code(method_sources, node_dir, import_paths)

    # Check if regeneration is needed
    source_hash = hashlib.md5(worker_code.encode()).hexdigest()
    hash_file = worker_dir / ".source_hash"

    needs_regen = True
    if hash_file.exists() and worker_file.exists():
        cached_hash = hash_file.read_text().strip()
        if cached_hash == source_hash:
            needs_regen = False

    if needs_regen:
        worker_file.write_text(worker_code)
        hash_file.write_text(source_hash)
        if log_callback:
            log_callback(f"[{env_name}] Generated worker code")

    # Create environment config
    if config_path:
        # Load from TOML file
        config_file = node_dir / config_path
        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_file}. "
                f"Specify a valid path or use inline requirements."
            )
        env_config = load_env_from_file(config_file, node_dir)
    else:
        # Use inline requirements
        if requirements is None:
            requirements = []

        # Handle 'auto' CUDA
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

    # Override worker location to use generated file
    env_config.worker_script = str(worker_file.relative_to(node_dir))

    # Create bridge
    bridge = WorkerBridge(
        env=env_config,
        worker_script=worker_file,
        base_dir=node_dir,
        log_callback=log_callback or (lambda msg: print(f"[{env_name}] {msg}")),
        auto_start=True,
    )

    # Ensure environment is ready
    bridge.ensure_environment()

    # Cache the bridge
    with _bridge_cache_lock:
        _bridge_cache[cache_key] = bridge

    return bridge


def _generate_worker_code(
    method_sources: Dict[str, str],
    node_dir: Path,
    import_paths: Optional[List[str]] = None,
) -> str:
    """
    Generate worker code from method sources.

    Transforms class methods into standalone worker methods.
    """
    # Build the worker file
    lines = [
        '"""Auto-generated worker for isolated node execution."""',
        '',
        'import sys',
        'import warnings',
        'import logging',
        'import os',
        '',
        '# Suppress output that could interfere with JSON IPC',
        'warnings.filterwarnings("ignore")',
        'os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")',
        'logging.disable(logging.WARNING)',
        '',
    ]

    # Auto-detect ComfyUI location from node_dir
    # Nodes are typically at: ComfyUI/custom_nodes/NodePackage/
    comfyui_base = None
    if node_dir.parent.name == "custom_nodes":
        comfyui_base = node_dir.parent.parent
    elif node_dir.parent.parent.name == "custom_nodes":
        # Handle nested: ComfyUI/custom_nodes/NodePackage/nodes/
        comfyui_base = node_dir.parent.parent.parent

    # Add import paths to sys.path
    lines.append(f'# Add node package paths for imports')
    lines.append(f'_NODE_DIR = "{node_dir}"')

    # Add ComfyUI base first so folder_paths is available
    if comfyui_base and comfyui_base.exists():
        lines.append(f'# ComfyUI base (for folder_paths, etc.)')
        lines.append(f'sys.path.insert(0, "{comfyui_base}")')

    if import_paths:
        for path in import_paths:
            full_path = node_dir / path
            lines.append(f'sys.path.insert(0, "{full_path}")')

    # Also add the node_dir itself so relative imports work
    lines.append(f'sys.path.insert(0, "{node_dir}")')
    lines.append('')

    lines.extend([
        'from comfyui_isolation import BaseWorker, register',
        '',
        '',
        'class GeneratedWorker(BaseWorker):',
        '    """Auto-generated worker with isolated node methods."""',
        '',
    ])

    # Add each method
    for method_name, source in method_sources.items():
        # Parse and transform the method
        transformed = _transform_method(source, method_name)
        lines.append(transformed)
        lines.append('')

    # Add main entry point
    lines.extend([
        '',
        'if __name__ == "__main__":',
        '    GeneratedWorker().run()',
        '',
    ])

    return '\n'.join(lines)


def _transform_method(source: str, method_name: str) -> str:
    """
    Transform a class method into a worker method.

    Changes:
    - Removes 'self' from first parameter (we add our own)
    - Adds @register decorator
    - Handles indentation
    """
    # Dedent the source
    source = textwrap.dedent(source)

    # Split into lines
    lines = source.split('\n')

    # Find the def line
    result_lines = []
    found_def = False

    for line in lines:
        stripped = line.lstrip()

        if not found_def and stripped.startswith('def '):
            # This is the function definition
            found_def = True

            # Parse the signature
            # Pattern: def method_name(self, arg1, arg2, ...):
            import re
            match = re.match(r'def\s+(\w+)\s*\(\s*self\s*,?\s*(.*?)\)\s*:', stripped)

            if match:
                name = match.group(1)
                params = match.group(2).strip()

                # Rebuild with self added back (for BaseWorker methods)
                if params:
                    new_def = f"def {name}(self, {params}):"
                else:
                    new_def = f"def {name}(self):"

                # Add decorator and method
                result_lines.append(f'    @register("{method_name}")')
                result_lines.append(f'    {new_def}')
            else:
                # Fallback: keep original but add decorator
                result_lines.append(f'    @register("{method_name}")')
                result_lines.append(f'    {stripped}')
        else:
            # Regular line - add class indentation
            if line.strip():  # Non-empty
                result_lines.append('    ' + line)
            else:
                result_lines.append('')

    return '\n'.join(result_lines)


def shutdown_all_bridges():
    """Shutdown all cached bridges."""
    with _bridge_cache_lock:
        for bridge in _bridge_cache.values():
            try:
                bridge.stop()
            except Exception:
                pass
        _bridge_cache.clear()


# Register cleanup on module unload
import atexit
atexit.register(shutdown_all_bridges)
