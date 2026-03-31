import argparse
import subprocess
import sys
import os
import glob
import pandas as pd
import numpy as np

# csv_files = glob.glob("moot/optimize/**/*.csv", recursive=True)
# lengths = []
# for csv_file in csv_files:
#     df = pd.read_csv(csv_file)
#     if len(df) > 10000:
#         lengths.append(len(df))
# length = np.array(lengths)
# print(np.min(length), np.max(length), np.mean(length), np.median(length))
# print(len(lengths))
# exit(0)


# Build command to call train.py with environment variables for arguments
cmd = [sys.executable, "pass_1.py"]
for algorithm_id in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
    csv_files = glob.glob("moot/optimize/**/*.csv", recursive=True)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if len(df) <= 10000:
            continue
        env = os.environ.copy()
        env["MOOT_ALGORITHM_ID"] = str(algorithm_id)
        env["MOOT_CSV_PATH"] = csv_file
        result = subprocess.run(cmd, env=env)
        print("--------------------------------")
        print()
