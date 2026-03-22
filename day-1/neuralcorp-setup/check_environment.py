# check_environment.py — confirms the environment is correctly set up
import sys

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 11   # minimum Python version NeuralCorp requires

current_major = sys.version_info.major
current_minor = sys.version_info.minor

print("=" * 40)
print("NeuralCorp Environment Check")
print("=" * 40)

if current_major == REQUIRED_MAJOR and current_minor >= REQUIRED_MINOR:
    print(f"✅ Python {current_major}.{current_minor} — OK")
else:
    print(f"❌ Python {current_major}.{current_minor} found — need 3.11+")

print(f"\nEnvironment path:\n  {sys.executable}")

# Detect if we're inside a virtual environment (not bare system Python)
in_venv = sys.prefix != sys.base_prefix
if in_venv:
    print("✅ Running inside a virtual environment — good!")
else:
    print("⚠️  Not in a virtual environment. Run: conda activate ml-fundamentals")