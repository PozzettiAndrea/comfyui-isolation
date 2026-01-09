"""
WorkerBridge - Main IPC class for communicating with isolated workers.

This is the primary interface that ComfyUI node developers use.
"""

import json
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..env.config import IsolatedEnv
from ..env.manager import IsolatedEnvManager
from .protocol import encode_object, decode_object


class WorkerBridge:
    """
    Bridge for communicating with a worker process in an isolated environment.

    This class manages the worker process lifecycle and handles IPC.

    Features:
    - Lazy worker startup (starts on first call)
    - Singleton pattern (one worker per environment)
    - Auto-restart on crash
    - Graceful shutdown
    - Timeout support

    Example:
        from comfyui_isolation import IsolatedEnv, WorkerBridge

        env = IsolatedEnv(
            name="my-node",
            python="3.10",
            cuda="12.8",
            requirements=["torch==2.8.0"],
        )

        bridge = WorkerBridge(env, worker_script=Path("worker.py"))

        # Call methods on the worker
        result = bridge.call("process", image=my_image)
    """

    # Singleton instances by environment hash
    _instances: Dict[str, "WorkerBridge"] = {}
    _instances_lock = threading.Lock()

    def __init__(
        self,
        env: IsolatedEnv,
        worker_script: Path,
        base_dir: Optional[Path] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        auto_start: bool = True,
    ):
        """
        Initialize the bridge.

        Args:
            env: Isolated environment configuration
            worker_script: Path to the worker Python script
            base_dir: Base directory for environments (default: worker_script's parent)
            log_callback: Optional callback for logging (default: print)
            auto_start: Whether to auto-start worker on first call (default: True)
        """
        self.env = env
        self.worker_script = Path(worker_script)
        self.base_dir = base_dir or self.worker_script.parent
        self.log = log_callback or print
        self.auto_start = auto_start

        self._manager = IsolatedEnvManager(self.base_dir, log_callback=log_callback)
        self._process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None

    @classmethod
    def get_instance(
        cls,
        env: IsolatedEnv,
        worker_script: Path,
        base_dir: Optional[Path] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> "WorkerBridge":
        """
        Get or create a singleton bridge instance for an environment.

        Args:
            env: Isolated environment configuration
            worker_script: Path to the worker Python script
            base_dir: Base directory for environments
            log_callback: Optional callback for logging

        Returns:
            WorkerBridge instance (reused if same env hash)
        """
        env_hash = env.get_env_hash()

        with cls._instances_lock:
            if env_hash not in cls._instances:
                cls._instances[env_hash] = cls(
                    env=env,
                    worker_script=worker_script,
                    base_dir=base_dir,
                    log_callback=log_callback,
                )
            return cls._instances[env_hash]

    @property
    def python_exe(self) -> Path:
        """Get the Python executable path for the isolated environment."""
        return self._manager.get_python(self.env)

    @property
    def is_running(self) -> bool:
        """Check if worker process is currently running."""
        with self._process_lock:
            return self._process is not None and self._process.poll() is None

    def ensure_environment(self, verify_packages: Optional[list] = None) -> None:
        """
        Ensure the isolated environment exists and is ready.

        Args:
            verify_packages: Optional list of packages to verify
        """
        self._manager.setup(self.env, verify_packages=verify_packages)

    def start(self) -> None:
        """
        Start the worker process.

        Does nothing if worker is already running.
        """
        with self._process_lock:
            if self._process is not None and self._process.poll() is None:
                return  # Already running

            python_exe = self.python_exe
            if not python_exe.exists():
                raise RuntimeError(
                    f"Python executable not found: {python_exe}\n"
                    f"Run ensure_environment() first or check your env configuration."
                )

            if not self.worker_script.exists():
                raise RuntimeError(f"Worker script not found: {self.worker_script}")

            self.log(f"Starting worker process...")
            self.log(f"  Python: {python_exe}")
            self.log(f"  Script: {self.worker_script}")

            self._process = subprocess.Popen(
                [str(python_exe), str(self.worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Start stderr reader thread
            self._stderr_thread = threading.Thread(
                target=self._read_stderr,
                daemon=True,
                name=f"worker-stderr-{self.env.name}",
            )
            self._stderr_thread.start()

            # Test connection
            try:
                response = self._send_raw({"method": "ping"}, timeout=30.0)
                if response.get("result") != "pong":
                    raise RuntimeError(f"Worker ping failed: {response}")
                self.log("Worker started successfully")
            except Exception as e:
                self.stop()
                raise RuntimeError(f"Worker failed to start: {e}")

    def _read_stderr(self) -> None:
        """Read stderr from worker and forward to log callback."""
        if not self._process or not self._process.stderr:
            return

        for line in self._process.stderr:
            line = line.rstrip()
            if line:
                self.log(line)

    def stop(self) -> None:
        """
        Stop the worker process gracefully.
        """
        with self._process_lock:
            if self._process is None or self._process.poll() is not None:
                return

            self.log("Stopping worker...")

            # Send shutdown command
            try:
                self._send_raw({"method": "shutdown"}, timeout=5.0)
            except Exception:
                pass

            # Wait for graceful shutdown
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.log("Worker didn't stop gracefully, terminating...")
                self._process.terminate()
                try:
                    self._process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()

            self._process = None
            self.log("Worker stopped")

    def _send_raw(self, request: dict, timeout: float = 300.0) -> dict:
        """
        Send a raw request and wait for response.

        Args:
            request: Request dict
            timeout: Timeout in seconds

        Returns:
            Response dict
        """
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Worker process is not running")

        # Add request ID
        if "id" not in request:
            request["id"] = str(uuid.uuid4())[:8]

        # Send request
        request_json = json.dumps(request) + "\n"
        self._process.stdin.write(request_json)
        self._process.stdin.flush()

        # Read response
        import time
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for response after {timeout}s")

            response_line = self._process.stdout.readline()
            if not response_line:
                raise RuntimeError("Worker process closed unexpectedly")

            response_line = response_line.strip()
            if not response_line:
                continue

            # Skip non-JSON lines (library output that escaped)
            if response_line.startswith('[') and not response_line.startswith('[{'):
                continue

            try:
                return json.loads(response_line)
            except json.JSONDecodeError:
                continue

    def call(
        self,
        method: str,
        timeout: float = 300.0,
        **kwargs,
    ) -> Any:
        """
        Call a method on the worker.

        Args:
            method: Method name to call
            timeout: Timeout in seconds (default: 5 minutes)
            **kwargs: Arguments to pass to the method

        Returns:
            The method's return value

        Raises:
            RuntimeError: If worker returns an error
            TimeoutError: If call times out
        """
        # Auto-start if needed
        if self.auto_start and not self.is_running:
            self.start()

        # Encode arguments
        encoded_args = encode_object(kwargs)

        # Send request
        request = {
            "method": method,
            "args": encoded_args,
        }
        response = self._send_raw(request, timeout=timeout)

        # Check for error
        if "error" in response and response["error"]:
            error_msg = response["error"]
            tb = response.get("traceback", "")
            raise RuntimeError(f"Worker error: {error_msg}\n{tb}")

        # Decode and return result
        return decode_object(response.get("result"))

    def __enter__(self) -> "WorkerBridge":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops worker."""
        self.stop()

    def __del__(self) -> None:
        """Destructor - stops worker."""
        try:
            self.stop()
        except Exception:
            pass
