import sys
import json
import argparse
from pathlib import Path
from time import perf_counter
import torch
from torch.profiler import profile, ProfilerActivity, schedule

# Choose whether scoring is enabled or not
SCORE_RESULTS = True
VISUALIZE = False


# 1. SETUP PATHS AND IMPORT MODULES
SRC_DIR = Path.cwd() / "src"
sys.path.insert(0, str(SRC_DIR))

import utils
import train
import build
import evaluate
import flops
print("Modules imported successfully.")

# 2. DEFINE CONFIG
PRESETS = {
    "profile": {
      "epochs": 7,
      "max_augments": 300,
      "checkpoint_epochs": (), 
      "inference_epoch": 0,
    },
    "low": {
        "epochs": 90,
        "max_augments": 80,
        "checkpoint_epochs": (),
        "inference_epoch": 90,
    },
    "medium": {
        "epochs": 240,
        "max_augments": 80,
        "checkpoint_epochs": (),
        "inference_epoch": 240,
    },
    "high": {
        "epochs": 650,
        "max_augments": 300,
        "checkpoint_epochs": (645, 648, 650),
        "inference_epoch": 648,
    },
}

parser = argparse.ArgumentParser(description="Train and evaluate mdlARC.")
parser.add_argument(
    "preset",
    nargs="?",
    choices=tuple(PRESETS),
    default="medium",
    help="Training preset to run.",
)

def str2bool(v):
    return v.lower() in ("true", "1", "yes")

parser.add_argument(
    "count_flops",
    nargs="?",
    type=str2bool,
    default=False
)
cli_args = parser.parse_args()
preset = PRESETS[cli_args.preset]
is_counting_flops = cli_args.count_flops
print(f"Using preset: {cli_args.preset}")

args_dict = {
    "name": "profile_run",
    "data_path": Path("assets/challenges.json"),
    "train_log_file": Path("runs/training_log.txt"),
    "save_path": Path("runs/tiny.pt"),
    "checkpoint_path": None, 
    "checkpoint_epochs": list(preset["checkpoint_epochs"]),
    
    # Hyperparameters
    "epochs": preset["epochs"], 
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "do_validate": False,
    "val_batch_size": 70,

    "enable_aug": True,
    "max_augments": preset["max_augments"],
    "enable_color_aug": True,
    "color_apply_to_test": True,
    "enable_dihedral_aug": True,
    "dihedral_apply_to_test": True,

    "optimizer": "normuon",
    "normuon_lr": 1.66e-3,
    "normuon_momentum": 0.95,
    "normuon_beta2": 0.95,
    "adamw_lr": 3e-4,

    "warmup_pct": 0.02,
    "wsd_decay_start_pct": 0.8,
    "lr_floor": 0.0,

    "weight_decay": 0.1,
    "attention_weight_decay": 0.01,
    "token_embedding_weight_decay": 0.01,
    "task_embedding_weight_decay": 0.01,  # Applies to task/example and dihedral embeddings.

    "grad_clip": 1.0,
    "dropout": 0.1,
    # Let attention dropout follow the shared dropout knob.
    "attention_dropout": None,
    "seed": 42,

    # Architecture
    "d_model": 768,
    "n_heads": 12,
    "d_ff": 3072,
    "n_layers": 8,

    "inference_temperature": None,
    "inference_top_k": None,

    # train logging
    "train_log_mode": "never", # options: never, step, 10_steps, epoch
    "log_location": "none", # options: none, terminal, file, both.
}
cfg = argparse.Namespace(**args_dict) # Convert dictionary to Namespace
Path("runs").mkdir(parents=True, exist_ok=True) # Create runs dir

# 3. BUILD
print("Building model and data...")
model, dataset, dataloader, device, data_path = build.build_model_and_data(cfg)
flops.reset_flops()

# 4. TRAIN (with PyTorch profiler)
print("Starting Training...")
t_start = perf_counter()

PROFILE_OUTPUT = Path(f"runs/profiler_trace_count_flops_{is_counting_flops}.json")
STATS_OUTPUT = Path(f"runs/profile_stats_count_flops_{is_counting_flops}.json")

steps_per_epoch = len(dataloader)
num_epochs = preset["epochs"]

# Schedule: skip into a middle epoch, warmup 2 steps, record 5.
# With 7 epochs, we skip 3 full epochs to land in epoch 4 (0-indexed: epoch 3).
skip_steps = 3 * steps_per_epoch
warmup_steps = 2
active_steps = 5

with profile(
    activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
    schedule=schedule(
        wait=skip_steps,
        warmup=warmup_steps,
        active=active_steps,
        repeat=1,
    ),
    record_shapes = True,
    on_trace_ready=lambda p: p.export_chrome_trace(str(PROFILE_OUTPUT)),
) as prof:
    train.train_model(
        cfg,
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
        data_path=data_path,
        prof=prof,
    )

train_time = perf_counter() - t_start
forward_flops_total = flops.get_flops()
backward_flops_total = 2 * forward_flops_total  # standard 2x estimate
optimizer_flops_total = flops.get_opt_flops()
total_flops = forward_flops_total + backward_flops_total + optimizer_flops_total

forward_flops_per_epoch = forward_flops_total / num_epochs
backward_flops_per_epoch = backward_flops_total / num_epochs
optimizer_flops_per_epoch = optimizer_flops_total / num_epochs
total_flops_per_epoch = forward_flops_per_epoch + backward_flops_per_epoch + optimizer_flops_per_epoch

forward_flops_per_step = forward_flops_total / (num_epochs * steps_per_epoch)
backward_flops_per_step = backward_flops_total / (num_epochs * steps_per_epoch)
optimizer_flops_per_step = optimizer_flops_total / (num_epochs * steps_per_epoch)
total_flops_per_step = forward_flops_per_step + backward_flops_per_step + optimizer_flops_per_step



# Write stats
stats = {
    "model": {
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "d_ff": cfg.d_ff,
        "n_layers": cfg.n_layers,
        "batch_size": cfg.batch_size,
        "optimizer": cfg.optimizer,
    },
    "training": {
        "num_epochs": num_epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": num_epochs * steps_per_epoch,
        "wall_time_s": round(train_time, 2),
        "time_per_step_s": round(train_time / (num_epochs * steps_per_epoch), 4),
    },
    "flop_counts": {
        "fwd_per_step": forward_flops_per_step,
        "bwd_per_step": backward_flops_per_step,
        "opt_per_step": optimizer_flops_per_step,
        "total_per_step": total_flops_per_step,
        "fwd_per_epoch": forward_flops_per_epoch,
        "bwd_per_epoch": backward_flops_per_epoch,
        "opt_per_epoch": optimizer_flops_per_epoch,
        "total_per_epoch": total_flops_per_epoch,
        "fwd_total": forward_flops_total,
        "bwd_total": backward_flops_total,
        "opt_total": optimizer_flops_total,
        "total": total_flops,
        
    },
    "throughput_flop_per_s": {
        "achieved_tflop_per_s": round((total_flops * 1e-12) / train_time, 2),
        "peak_tflop_per_s": 210,
        "mfu": round((total_flops * 1e-12) / (train_time * 210), 4),
        "acheived_tflops_per_s_in_single_step": (total_flops_per_step*1e-12)/(train_time / (num_epochs * steps_per_epoch)),
    },
    "profiler": {
        "trace_file": str(PROFILE_OUTPUT),
        "captured_epoch": 4,
        "skip_steps": skip_steps,
        "warmup_steps": warmup_steps,
        "active_steps": active_steps,
    },
}

with open(STATS_OUTPUT, "w") as f:
    json.dump(stats, f, indent=2)
print(f"Profile stats written to {STATS_OUTPUT}")
print(f"Profiler trace exported to {PROFILE_OUTPUT}")

# 5. EVALUATE / INFERENCE

print("Starting Evaluation...")
utils.cleanup_memory(globals()) # Force garbage collection before eval
inference_checkpoint_path = cfg.save_path
if preset["inference_epoch"] != cfg.epochs:
    inference_checkpoint_path = train._checkpoint_path_for_epoch(
        cfg.save_path,
        preset["inference_epoch"],
        cfg.epochs,
    )

try:
    eval_result = evaluate.run_evaluation(
        cfg,
        run_name="submission_eval",
        max_augments=cfg.max_augments,        
        data_path=cfg.data_path,
        checkpoint_path=inference_checkpoint_path,
        batch_size=100,
        splits=["test"],          
        task_ids=None,
    )
    SUBMISSION_FILE = Path(f"runs/{eval_result[0]}/submission.json")
    print("Evaluation complete. submission.json generated.")

    # 6. RESULTS: score the results (if enabled), then visualise
    if SCORE_RESULTS: # scoring, if enabled
        SOLUTIONS_FILE = Path("assets/solutions.json")
        score = utils.score_arc_submission(SOLUTIONS_FILE, SUBMISSION_FILE)
        if VISUALIZE:
            utils.visualize_submissions(SUBMISSION_FILE, SOLUTIONS_FILE, mode="!")
    else:
        if VISUALIZE:
            utils.visualize_submissions(SUBMISSION_FILE, mode="submission")
except Exception as e:
    print(f"Evaluation skipped due to error (inference_epoch={preset['inference_epoch']}): {e}")
    print("Training profiling was still captured successfully.")