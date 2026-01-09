# comfyui-isolation

Process isolation for ComfyUI custom nodes. Run your node's inference code in a separate Python environment with its own dependencies.

## Why?

ComfyUI custom nodes often require conflicting dependencies:
- Node A needs `torch==2.1.0` with CUDA 11.8
- Node B needs `torch==2.8.0` with CUDA 12.8
- Both want different versions of `numpy`, `transformers`, etc.

This package lets each node run in its own isolated Python environment, completely avoiding dependency conflicts.

## Installation

```bash
pip install comfyui-isolation
```

Or install from source:

```bash
pip install -e .
```

Requires [uv](https://github.com/astral-sh/uv) for fast environment creation:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

### 1. Define your environment

```python
from comfyui_isolation import IsolatedEnv

env = IsolatedEnv(
    name="my-node",
    python="3.10",
    cuda="12.8",  # or None for auto-detect
    requirements=["torch==2.8.0", "nvdiffrast"],
    wheel_sources=["https://my-wheels.github.io/"],
)
```

### 2. Create a worker script

```python
# worker.py
from comfyui_isolation import BaseWorker, register

class MyWorker(BaseWorker):
    def setup(self):
        # Load your model here (runs once)
        import torch
        self.model = load_my_model()

    @register("process")
    def process_image(self, image, params):
        return self.model(image, **params)

if __name__ == "__main__":
    MyWorker().run()
```

### 3. Use in your node

```python
from pathlib import Path
from comfyui_isolation import IsolatedEnv, WorkerBridge

env = IsolatedEnv(name="my-node", python="3.10", cuda="12.8")
bridge = WorkerBridge(env, worker_script=Path("worker.py"))

# First call creates the environment (cached for next time)
result = bridge.call("process", image=my_image, params={"size": 512})
```

## API Reference

### IsolatedEnv

Configuration for an isolated environment:

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Unique name for caching |
| `python` | str | Python version (e.g., "3.10") |
| `cuda` | str | CUDA version (e.g., "12.8") or None for CPU |
| `requirements` | list | Pip requirements |
| `requirements_file` | Path | Path to requirements.txt |
| `wheel_sources` | list | URLs for `--find-links` |
| `index_urls` | list | URLs for `--extra-index-url` |

### WorkerBridge

Bridge for communicating with isolated workers:

```python
bridge = WorkerBridge(env, worker_script)

# Ensure environment exists
bridge.ensure_environment(verify_packages=["torch"])

# Call methods
result = bridge.call("method_name", arg1=value1, arg2=value2)

# Lifecycle
bridge.start()   # Start worker (auto-called on first bridge.call)
bridge.stop()    # Stop worker
bridge.is_running  # Check if running
```

### BaseWorker

Base class for worker scripts:

```python
class MyWorker(BaseWorker):
    def setup(self):
        """Called once on startup - load models here"""
        pass

    def teardown(self):
        """Called on shutdown - cleanup here"""
        pass

    @register("my_method")
    def my_method(self, arg1, arg2):
        """Callable from bridge.call("my_method", ...)"""
        return result
```

### GPU Detection

```python
from comfyui_isolation import detect_cuda_version, get_gpu_summary

# Auto-detect CUDA version (12.8 for Blackwell, 12.4 for others)
cuda = detect_cuda_version()  # "12.8", "12.4", or None

# Get detailed GPU info
print(get_gpu_summary())
# GPU 0: NVIDIA GeForce RTX 5090 (sm_120) [Blackwell - CUDA 12.8]
```

## Performance

**Q: Does subprocess communication slow things down?**

No. The overhead is negligible for AI/ML workloads:
- Worker startup: ~1-2 seconds (one-time)
- Per-call overhead: ~1-5ms for JSON serialization
- Your inference: 10-60+ seconds

The worker stays alive between calls, so startup cost is paid only once.

## How It Works

1. **Environment Creation**: Uses `uv` to create a venv with the specified Python/CUDA versions
2. **IPC Protocol**: JSON over stdin/stdout with base64-encoded binary data (images, tensors)
3. **Worker Lifecycle**: Lazy start on first call, singleton pattern, graceful shutdown

## Examples

See the `examples/` directory:
- `basic_node/` - Simple ComfyUI node with isolation

## License

MIT - see LICENSE file.
