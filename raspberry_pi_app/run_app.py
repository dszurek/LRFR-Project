"""Quick launcher for Raspberry Pi app on Windows.

This script provides a convenient way to run the app with proper environment setup.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    """Launch the Raspberry Pi app in the Poetry environment."""
    print("=" * 70)
    print("LRFR Raspberry Pi App - Windows Launcher")
    print("=" * 70)
    print()
    
    # Check if we're in a Poetry project
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.exists():
        print("❌ Error: pyproject.toml not found")
        print(f"   Expected at: {pyproject}")
        sys.exit(1)
    
    print("✅ Poetry project detected")
    print(f"   Project root: {PROJECT_ROOT}")
    print()
    
    # Check if app.py exists
    app_py = PROJECT_ROOT / "raspberry_pi_app" / "app.py"
    if not app_py.exists():
        print("❌ Error: app.py not found")
        print(f"   Expected at: {app_py}")
        sys.exit(1)
    
    print("✅ App file found")
    print()
    
    # Launch with Poetry
    print("Launching app with Poetry...")
    print("-" * 70)
    print()
    
    cmd = ["poetry", "run", "python", "raspberry_pi_app/app.py"]
    
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 70)
        print(f"❌ App exited with error code {e.returncode}")
        print("=" * 70)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("App closed by user (Ctrl+C)")
        print("=" * 70)
        sys.exit(0)
    except FileNotFoundError:
        print()
        print("=" * 70)
        print("❌ Error: Poetry not found in PATH")
        print()
        print("Please install Poetry:")
        print("  https://python-poetry.org/docs/#installation")
        print()
        print("Or run directly:")
        print(f"  cd {PROJECT_ROOT}")
        print("  poetry run python raspberry_pi_app/app.py")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()
