#!/usr/bin/env python3
"""
SESTRAV Repository Status & Health Checker.
Directly checks environment imports, Git status, and pipeline validation freeze state.
"""
import os
import sys
import json
import subprocess

def check_imports() -> bool:
    print("=== Checking Environment Imports ===")
    required_packages = ["torch", "torch_geometric", "pydantic", "mhcflurry", "yaml", "pytest"]
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  [✓] {pkg} is available")
        except ImportError:
            missing.append(pkg)
            print(f"  [✗] {pkg} is NOT available")
    
    if missing:
        print(f"Error: Missing required environment packages: {', '.join(missing)}")
        return False
    return True

def check_git_status() -> bool:
    print("\n=== Checking Git Workspace Status ===")
    try:
        status_out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        if not status_out:
            print("  [✓] Git working directory is perfectly clean")
            return True
        
        print("  [!] Git working directory has unstaged/uncommitted changes:")
        for line in status_out.splitlines():
            print(f"      {line}")
        return False
    except Exception as e:
        print(f"  [✗] Failed to run git command: {e}")
        return False

def check_freeze_status() -> bool:
    print("\n=== Checking Validation Dataset Freeze ===")
    freeze_path = os.path.join("results", "freeze_status.json")
    if not os.path.exists(freeze_path):
        print(f"  [✗] {freeze_path} does not exist. Validation report has not been run.")
        return False
    
    try:
        with open(freeze_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        valid = data.get("valid", False)
        freeze_mode = data.get("freeze_mode", False)
        version = data.get("dataset_version", "unknown")
        
        if valid and freeze_mode:
            print(f"  [✓] Validation status is VALID (Dataset version: {version})")
            print("  [✓] Freeze mode is active (immutability guards locked)")
            return True
        else:
            print("  [✗] Validation is INVALID or freeze mode is disabled:")
            print(f"      valid: {valid}, freeze_mode: {freeze_mode}")
            return False
    except Exception as e:
        print(f"  [✗] Error reading/parsing {freeze_path}: {e}")
        return False

def check_test_suite_status() -> bool:
    print("\n=== Checking Pytest Configuration ===")
    pytest_ini = "pytest.ini"
    if os.path.exists(pytest_ini):
        print("  [✓] pytest.ini config found")
        return True
    else:
        print("  [✗] pytest.ini config not found at project root")
        return False

def main():
    print("==================================================")
    print("       SESTRAV REPOSITORY HEALTH CHECKER         ")
    print("==================================================")
    
    imports_ok = check_imports()
    git_ok = check_git_status()
    freeze_ok = check_freeze_status()
    tests_ok = check_test_suite_status()
    
    print("\n=================== Summary ======================")
    all_ok = True
    for name, status in [("Imports", imports_ok), ("Git Cleanliness", git_ok), ("Dataset Freeze", freeze_ok), ("Tests Config", tests_ok)]:
        icon = "[✓] PASS" if status else "[✗] WARN/FAIL"
        print(f"  {name:<20}: {icon}")
        if not status and name != "Git Cleanliness": # Git dirty can be a warning rather than fatal error
            all_ok = False
            
    if all_ok:
        print("\n[✓] Repository is in a healthy, hardened state.")
        sys.exit(0)
    else:
        print("\n[✗] Repository has outstanding environment or validation failures.")
        sys.exit(1)

if __name__ == "__main__":
    main()
