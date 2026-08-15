import subprocess
import sys


STEPS = [
    "save_geometric_features.py",
    "combine_features.py",
    "calculate_csi.py"
]


def run_step(script):
    print("\n" + "=" * 60)
    print(f"Running: {script}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, script]
    )

    if result.returncode != 0:
        print(
            f"\nERROR: {script} failed."
        )
        sys.exit(result.returncode)


for script in STEPS:
    run_step(script)


print("\n" + "=" * 60)
print("STAGE 2 PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)