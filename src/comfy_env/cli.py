"""
CLI for comfy-env.

Provides the `comfy-env` command with subcommands:
- install: Install dependencies from config
- info: Show runtime environment information
- resolve: Show resolved wheel URLs
- doctor: Verify installation
- list-packages: Show all packages in the built-in registry

Usage:
    comfy-env install
    comfy-env install --isolated
    comfy-env install --dry-run

    comfy-env info

    comfy-env resolve nvdiffrast==0.4.0
    comfy-env resolve --all

    comfy-env doctor

    comfy-env list-packages
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for comfy-env CLI."""
    parser = argparse.ArgumentParser(
        prog="comfy-env",
        description="Environment management for ComfyUI custom nodes",
    )
    parser.add_argument(
        "--version", action="version", version=f"comfy-env {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # install command
    install_parser = subparsers.add_parser(
        "install",
        help="Install dependencies from config",
        description="Install CUDA wheels and dependencies from comfy-env.toml",
    )
    install_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config file (default: auto-discover)",
    )
    install_parser.add_argument(
        "--isolated",
        action="store_true",
        help="Create isolated venv instead of installing in-place",
    )
    install_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be installed without installing",
    )
    install_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify wheel URLs exist before installing",
    )
    install_parser.add_argument(
        "--dir", "-d",
        type=str,
        help="Directory containing the config (default: current directory)",
    )

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show runtime environment information",
        description="Display detected Python, CUDA, PyTorch, and GPU information",
    )
    info_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # resolve command
    resolve_parser = subparsers.add_parser(
        "resolve",
        help="Resolve wheel URLs for packages",
        description="Show resolved wheel URLs without installing",
    )
    resolve_parser.add_argument(
        "packages",
        nargs="*",
        help="Package specs (e.g., nvdiffrast==0.4.0)",
    )
    resolve_parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Resolve all packages from config",
    )
    resolve_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config file",
    )
    resolve_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify URLs exist (HTTP HEAD check)",
    )

    # doctor command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Verify installation and diagnose issues",
        description="Check if packages are properly installed and importable",
    )
    doctor_parser.add_argument(
        "--package", "-p",
        type=str,
        help="Check specific package",
    )
    doctor_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config file",
    )

    # list-packages command
    list_parser = subparsers.add_parser(
        "list-packages",
        help="Show all packages in the built-in registry",
        description="List CUDA packages that comfy-env knows how to install",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        return 0

    try:
        if parsed.command == "install":
            return cmd_install(parsed)
        elif parsed.command == "info":
            return cmd_info(parsed)
        elif parsed.command == "resolve":
            return cmd_resolve(parsed)
        elif parsed.command == "doctor":
            return cmd_doctor(parsed)
        elif parsed.command == "list-packages":
            return cmd_list_packages(parsed)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_install(args) -> int:
    """Handle install command."""
    from .install import install

    mode = "isolated" if args.isolated else "inplace"
    node_dir = Path(args.dir) if args.dir else Path.cwd()

    try:
        install(
            config=args.config,
            mode=mode,
            node_dir=node_dir,
            dry_run=args.dry_run,
            verify_wheels=args.verify,
        )
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Installation failed: {e}", file=sys.stderr)
        return 1


def cmd_info(args) -> int:
    """Handle info command."""
    from .resolver import RuntimeEnv

    env = RuntimeEnv.detect()

    if args.json:
        import json
        print(json.dumps(env.as_dict(), indent=2))
        return 0

    print("Runtime Environment")
    print("=" * 40)
    print(f"  OS:           {env.os_name}")
    print(f"  Platform:     {env.platform_tag}")
    print(f"  Python:       {env.python_version}")

    if env.cuda_version:
        print(f"  CUDA:         {env.cuda_version}")
    else:
        print("  CUDA:         Not detected")

    if env.torch_version:
        print(f"  PyTorch:      {env.torch_version}")
    else:
        print("  PyTorch:      Not installed")

    if env.gpu_name:
        print(f"  GPU:          {env.gpu_name}")
        if env.gpu_compute:
            print(f"  Compute:      {env.gpu_compute}")

    print()
    return 0


def cmd_resolve(args) -> int:
    """Handle resolve command."""
    from .resolver import RuntimeEnv, WheelResolver, parse_wheel_requirement
    from .registry import PACKAGE_REGISTRY
    from .env.config_file import discover_env_config, load_env_from_file

    env = RuntimeEnv.detect()
    resolver = WheelResolver()

    packages = []

    # Get packages from args or config
    if args.all or (not args.packages and args.config):
        # Load from config
        if args.config:
            config = load_env_from_file(Path(args.config))
        else:
            config = discover_env_config(Path.cwd())

        if config and config.no_deps_requirements:
            packages = config.no_deps_requirements
        else:
            print("No CUDA packages found in config", file=sys.stderr)
            return 1
    elif args.packages:
        packages = args.packages
    else:
        print("Specify packages or use --all with a config file", file=sys.stderr)
        return 1

    print(f"Resolving wheels for: {env}")
    print("=" * 60)

    all_ok = True
    for pkg_spec in packages:
        package, version = parse_wheel_requirement(pkg_spec)
        if version is None:
            print(f"  {package}: No version specified, skipping")
            continue

        pkg_lower = package.lower()
        try:
            # Check if package is in registry with github_release method
            if pkg_lower in PACKAGE_REGISTRY:
                registry_config = PACKAGE_REGISTRY[pkg_lower]
                method = registry_config.get("method")

                if method == "github_release":
                    # Resolve URL from registry sources
                    url = _resolve_github_release_url(package, version, env, registry_config)
                    status = "OK" if args.verify else "resolved"
                    print(f"  {package}=={version}: {status}")
                    print(f"    {url}")
                else:
                    # For other methods, just show what method will be used
                    print(f"  {package}=={version}: uses {method} method")
                    if "index_url" in registry_config:
                        index_url = _substitute_template(registry_config["index_url"], env)
                        print(f"    index: {index_url}")
                    elif "package_template" in registry_config:
                        pkg_name = _substitute_template(registry_config["package_template"], env)
                        print(f"    installs as: {pkg_name}")
            else:
                # Fall back to WheelResolver
                url = resolver.resolve(package, version, env, verify=args.verify)
                status = "OK" if args.verify else "resolved"
                print(f"  {package}=={version}: {status}")
                print(f"    {url}")
        except Exception as e:
            print(f"  {package}=={version}: FAILED")
            _print_wheel_not_found_error(package, version, env, e)
            all_ok = False

    return 0 if all_ok else 1


def _substitute_template(template: str, env) -> str:
    """Substitute environment variables into a URL template."""
    vars_dict = env.as_dict()
    result = template
    for key, value in vars_dict.items():
        if value is not None:
            result = result.replace(f"{{{key}}}", str(value))
    return result


def _resolve_github_release_url(package: str, version: str, env, config: dict) -> str:
    """Resolve URL for github_release method packages."""
    sources = config.get("sources", [])
    if not sources:
        raise ValueError(f"No sources configured for {package}")

    # Build template variables
    vars_dict = env.as_dict()
    vars_dict["version"] = version
    vars_dict["py_tag"] = f"cp{env.python_short}"
    if env.cuda_version:
        vars_dict["cuda_major"] = env.cuda_version.split(".")[0]

    # Filter sources by platform
    current_platform = env.platform_tag
    compatible_sources = [
        s for s in sources
        if current_platform in s.get("platforms", [])
    ]

    if not compatible_sources:
        available = set()
        for s in sources:
            available.update(s.get("platforms", []))
        raise ValueError(
            f"No {package} wheels for platform {current_platform}. "
            f"Available: {', '.join(sorted(available))}"
        )

    # Return URL from first compatible source
    source = compatible_sources[0]
    url_template = source.get("url_template", "")
    url = url_template
    for key, value in vars_dict.items():
        if value is not None:
            url = url.replace(f"{{{key}}}", str(value))

    return url


def _print_wheel_not_found_error(package: str, version: str, env, error: Exception) -> None:
    """Print a formatted error message for wheel not found."""
    from .errors import WheelNotFoundError

    if isinstance(error, WheelNotFoundError):
        print(f"    CUDA wheel not found: {package}=={version}")
        print()
        print("+------------------------------------------------------------------+")
        print("|  CUDA Wheel Not Found                                            |")
        print("+------------------------------------------------------------------+")
        print(f"|  Package:   {package}=={version:<46} |")
        print(f"|  Requested: cu{env.cuda_short}-torch{env.torch_mm}-{env.python_short}-{env.platform_tag:<17} |")
        print("|                                                                  |")
        print(f"|  Reason: {error.reason:<54} |")
        print("|                                                                  |")
        print("|  Suggestions:                                                    |")
        print(f"|    1. Check if wheel exists: comfy-env resolve {package:<15} |")
        print(f"|    2. Build wheel locally: comfy-env build {package:<18} |")
        print("|                                                                  |")
        print("+------------------------------------------------------------------+")
    else:
        print(f"    {error}")


def cmd_doctor(args) -> int:
    """Handle doctor command."""
    from .install import verify_installation
    from .env.config_file import discover_env_config, load_env_from_file

    print("Running diagnostics...")
    print("=" * 40)

    # Check environment
    print("\n1. Environment")
    cmd_info(argparse.Namespace(json=False))

    # Check packages
    print("2. Package Verification")

    packages = []
    if args.package:
        packages = [args.package]
    elif args.config:
        config = load_env_from_file(Path(args.config))
        if config:
            packages = (config.requirements or []) + (config.no_deps_requirements or [])
    else:
        config = discover_env_config(Path.cwd())
        if config:
            packages = (config.requirements or []) + (config.no_deps_requirements or [])

    if packages:
        # Extract package names from specs
        pkg_names = []
        for pkg in packages:
            name = pkg.split("==")[0].split(">=")[0].split("[")[0]
            pkg_names.append(name)

        all_ok = verify_installation(pkg_names)
        if all_ok:
            print("\nAll packages verified!")
            return 0
        else:
            print("\nSome packages failed verification.")
            return 1
    else:
        print("  No packages to verify (no config found)")
        return 0


def cmd_list_packages(args) -> int:
    """Handle list-packages command."""
    from .registry import PACKAGE_REGISTRY, list_packages

    if args.json:
        import json
        result = {}
        for name, config in PACKAGE_REGISTRY.items():
            result[name] = {
                "method": config["method"],
                "description": config.get("description", ""),
            }
            if "index_url" in config:
                result[name]["index_url"] = config["index_url"]
            if "package_template" in config:
                result[name]["package_template"] = config["package_template"]
        print(json.dumps(result, indent=2))
        return 0

    print("Built-in CUDA Package Registry")
    print("=" * 60)
    print()
    print("These packages can be installed without specifying wheel_sources.")
    print("Just add them to your comfy-env.toml:")
    print()
    print("  [cuda]")
    print("  torch-scatter = \"2.1.2\"")
    print("  torch-cluster = \"1.6.3\"")
    print()
    print("-" * 60)

    # Group packages by method
    by_method = {}
    for name, config in PACKAGE_REGISTRY.items():
        method = config["method"]
        if method not in by_method:
            by_method[method] = []
        by_method[method].append((name, config))

    method_labels = {
        "index": "PEP 503 Index (pip --extra-index-url)",
        "github_index": "GitHub Pages (pip --find-links)",
        "pypi_variant": "PyPI with CUDA variant names",
    }

    for method, packages in by_method.items():
        print(f"\n{method_labels.get(method, method)}:")
        for name, config in sorted(packages):
            desc = config.get("description", "")
            print(f"  {name:20} - {desc}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
