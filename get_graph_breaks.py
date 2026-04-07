# graph_breaks_runner.py

import os
import subprocess

env = os.environ.copy()
env["TORCH_LOGS"] = "graph_breaks"
env["TORCHDYNAMO_VERBOSE"] = "1"

with open("graph_breaks.txt", "w") as f:
    proc = subprocess.Popen(
        ["python3", "run_script.py", "profile"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    for line in proc.stdout:
        f.write(line)   # write EVERYTHING, no spam

    proc.wait()

print("Saved full log to graph_breaks.txt")