# graph_breaks_runner.py

import os
import subprocess

env = os.environ.copy()
env["TORCH_LOGS"] = "graph_breaks"
env["TORCHDYNAMO_VERBOSE"] = "1"

count_flops=False
torch_profile=True

with open(f"graph_breaks_flp_cnt_{count_flops}_prof_{torch_profile}.txt", "w") as f:
    proc = subprocess.Popen(
        ["python3", "run_with_profile.py", "profile", str(count_flops)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    for line in proc.stdout:
        f.write(line)   # write EVERYTHING, no spam

    proc.wait()

print("Saved full log to graph_breaks.txt")