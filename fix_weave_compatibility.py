#!/usr/bin/env python
"""
Quick fix for Weave compatibility issues.
Run this if you encounter TypeError with weave.init()
"""

import subprocess
import sys

def main():
    print("Fixing Weave compatibility issues...")
    
    # Try to install compatible gql version
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gql==3.4.1"])
        print("✓ Installed compatible gql version")
    except subprocess.CalledProcessError:
        print("Note: Could not install gql==3.4.1, but the notebook will still work")
    
    print("\nThe notebook is configured to handle initialization errors automatically.")
    print("You can now run the notebook even if Weave tracking is not available.")
    print("\nTo run the notebook:")
    print("  jupyter notebook wb-logprobs.ipynb")

if __name__ == "__main__":
    main()
