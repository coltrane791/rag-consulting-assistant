#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
from shutil import which

# --- SETTINGS ---
VENV_NAME = ".venv"
REQUIRED_PACKAGES = ["openai", "python-docx", "pymupdf", "faiss-cpu", "python-dotenv", "numpy"]

def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)

def activate_venv_and_rerun():
    """Activate .venv and re-run this script inside it."""
    is_windows = platform.system() == "Windows"
    venv_dir = os.path.join(os.getcwd(), VENV_NAME, "Scripts" if is_windows else "bin")
    activate_script = os.path.join(venv_dir, "activate")

    if not os.path.exists(activate_script):
        print(f"[!] Could not find virtual environment activation script at:\n    {activate_script}")
        print("    Create it with: python -m venv .venv")
        sys.exit(1)

    print(f"[i] Activating virtual environment: {VENV_NAME} ...")
    # Use a shell trick to activate then re-run this script with same interpreter name (python)
    if is_windows:
        # Prefer powershell if available, else cmd
        ps = which("powershell")
        if ps:
            cmd = f'powershell -NoProfile -Command "& {{ & \'{activate_script}\' ; python venv_check.py }}"'
        else:
            cmd = f'{activate_script} && python venv_check.py'
    else:
        cmd = f'source "{activate_script}" && python venv_check.py'

    os.system(cmd)
    sys.exit(0)

def ensure_in_venv():
    # Quick heuristic: Python path should include .venv
    if VENV_NAME not in sys.executable:
        print(f"[!] Python is not running inside '{VENV_NAME}'.")
        activate_venv_and_rerun()
    else:
        print(f"[✓] Using Python inside '{VENV_NAME}': {sys.executable}")

def check_and_install_packages():
    print("[i] Checking required packages...")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        res = run_cmd(f"pip show {pkg}")
        if res.returncode != 0:
            missing.append(pkg)

    if not missing:
        print("[✓] All required packages are installed.")
        return False  # nothing installed

    print(f"[!] Missing packages: {', '.join(missing)}")
    ans = input("    Install them now? [Y/n]: ").strip().lower()
    if ans in ("", "y", "yes"):
        print("[i] Installing missing packages...")
        install_cmd = f"pip install {' '.join(missing)}"
        proc = run_cmd(install_cmd)
        if proc.returncode != 0:
            print("[x] pip install failed. Output:\n", proc.stdout, proc.stderr)
            sys.exit(1)
        print("[✓] Installation complete.")
        return True
    else:
        print("[i] Skipping installation.")
        return False

def refresh_requirements():
    print("[i] Updating requirements.txt ...")
    proc = run_cmd("pip freeze")
    if proc.returncode != 0:
        print("[x] Could not run 'pip freeze'.")
        return
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(proc.stdout)
    print("[✓] requirements.txt updated.")

def main():
    ensure_in_venv()
    installed = check_and_install_packages()
    if installed or not os.path.exists("requirements.txt"):
        refresh_requirements()
    else:
        # Optionally check staleness
        proc = run_cmd("pip freeze")
        current = set(proc.stdout.strip().splitlines())
        try:
            with open("requirements.txt", "r", encoding="utf-8") as f:
                saved = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            saved = set()
        if current != saved:
            print("[!] requirements.txt may be outdated.")
            ans = input("    Update it now with current environment? [Y/n]: ").strip().lower()
            if ans in ("", "y", "yes"):
                refresh_requirements()
            else:
                print("[i] Left requirements.txt unchanged.")
    print("[✓] Venv check complete.")

if __name__ == "__main__":
    main()
