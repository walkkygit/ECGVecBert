"""
Dandelion hyperparameter sweep: submit one SageMaker finetuning job per configuration.

Each job runs vqvae_bert_finetuning.py with its own run_tag, so every run writes its own
checkpoint / metrics files on S3 (no overwriting). Collect the results afterwards with:
    python collect_results.py

Usage (from the SageMaker notebook, after `git pull origin master`):
    python sweep_launcher.py --round 1            # class-weight round
    python sweep_launcher.py --round 2 --class_weight 2.5   # LR round at the winning class weight
    python sweep_launcher.py --configs '[{"class_weight": 2.5, "lr": 3e-5}]'   # any ad-hoc list
    python sweep_launcher.py --round 1 --dry_run  # print what would be submitted

Instance: ml.g5.8xlarge by default (account quota 32 -> jobs run in parallel).
ml.g4dn.8xlarge has quota 1, so it can only run one job at a time.
"""

import argparse
import json
import logging
import time

from vqvae_bert_finetuning_launcher import launch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SEEDS = "42"                       # single seed for the sweep; 5 seeds only for the final winner
INSTANCE_TYPE = "ml.g5.8xlarge"    # quota 32 -> parallel
SUBMIT_GAP_SEC = 5                 # small gap between submissions

# Rounds, one knob at a time. Values already run are not repeated
# (baseline so far: class_weight 1.9, lr 1e-4, lora_r 8, dropout 0.1, lora_dropout 0.1, weight_decay 1e-4, all modules).
ROUNDS = {
    1: [{"class_weight": 2.5}, {"class_weight": 3.5}, {"class_weight": 4.5}],
    2: [{"lr": 3e-5}, {"lr": 1e-5}],
    3: [{"lora_r": 4}, {"target_modules": "W_Q,W_K,W_V"}],
    4: [{"dropout": 0.3, "lora_dropout": 0.3}, {"weight_decay": 1e-3}],
}


def short_tag(cfg: dict) -> str:
    """Console-friendly job tag from the config, e.g. cw2p5-lr3e-05."""
    parts = []
    for k, v in cfg.items():
        key = {"class_weight": "cw", "lr": "lr", "lora_r": "r", "dropout": "do", "lora_dropout": "ld",
               "weight_decay": "wd", "target_modules": "tm"}.get(k, k)
        val = "attn" if k == "target_modules" else (f"{v:.0e}" if isinstance(v, float) and v < 1e-2 else str(v))
        parts.append(f"{key}{val}")
    return "-".join(parts).replace(".", "p")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit a round of Dandelion sweep jobs")
    parser.add_argument("--round", type=int, choices=sorted(ROUNDS), help="which predefined round to submit")
    parser.add_argument("--configs", type=str, default=None, help="JSON list of config dicts (overrides --round)")
    # fixed values applied to every job in the round (e.g. the winning class weight from round 1)
    parser.add_argument("--class_weight", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--target_modules", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=SEEDS)
    parser.add_argument("--instance_type", type=str, default=INSTANCE_TYPE)
    parser.add_argument("--dry_run", action="store_true", help="print the jobs, submit nothing")
    args = parser.parse_args()

    if args.configs:
        configs = json.loads(args.configs)
    elif args.round:
        configs = ROUNDS[args.round]
    else:
        parser.error("give --round N or --configs '[...]'")

    fixed = {k: getattr(args, k) for k in ["class_weight", "lr", "lora_r", "dropout", "lora_dropout",
                                           "weight_decay", "target_modules"] if getattr(args, k) is not None}

    log.info(f"{len(configs)} job(s), fixed={fixed}, seeds={args.seeds}, instance={args.instance_type}")
    for cfg in configs:
        hp = dict(fixed, **cfg)          # per-config values win over the fixed ones
        tag = short_tag(hp)
        log.info(f"  -> {tag}: {hp}")
        if args.dry_run:
            continue
        launch(in_channels=12, batch_size=32, use_frac=1.0, dataset="dandelion", seeds=args.seeds,
               num_ray_workers=1, extra_hp=hp, instance_type=args.instance_type, job_tag=tag)
        time.sleep(SUBMIT_GAP_SEC)
    if args.dry_run:
        log.info("dry run: nothing submitted")
