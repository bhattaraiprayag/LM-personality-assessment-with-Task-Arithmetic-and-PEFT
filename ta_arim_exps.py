# Stand-alone: load every (baseline + LoRA) combo, sweep LoRA scales, evaluate on all 3 inventories,
# and write one growing CSV “ta_results.csv” with the requested columns.

import os, json, gc, argparse
from pathlib import Path
from typing import Dict, Any, List, Union
from tqdm import tqdm

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, set_peft_model_state_dict
# try:
from peft.helpers import rescale_adapter_scale

from src.peft_manager import PEFTManager
from src.eval_manager import EvalManager
from experiment_config import peft_scales, INVENTORIES


# --------------------------------------------------------------------------- #
# metadata helpers
# --------------------------------------------------------------------------- #

def collect_experiment_summary(meta_path: str, outputs_base: str) -> pd.DataFrame:
    with open(meta_path) as f:
        meta: Dict[str, Any] = json.load(f)

    rows: List[Dict[str, Any]] = []
    for exp_id, exp in meta.items():
        use_peft = exp.get("use_peft") or "baseline"
        out_dir  = os.path.join(outputs_base, os.path.basename(exp["exp_out_dir"]))
        files    = os.listdir(out_dir) if os.path.exists(out_dir) else []

        if use_peft == "lora":
            model_files = [f for f in files if f.endswith("lora_final.pt")]
        else:
            model_files = [f for f in files if f.endswith(("model.safetensors", "config.json", ".bin"))]

        rows.append(
            dict(exp_id=exp_id,
                 dataset=exp.get("dataset"),
                 seed=exp.get("seed"),
                 use_peft=use_peft,
                 exp_out_dir=out_dir,
                 model_files=model_files)
        )
    return pd.DataFrame(rows), meta


def make_combinations(df: pd.DataFrame) -> pd.DataFrame:
    pandora_b = df[(df.dataset == "pandora")  & (df.use_peft == "baseline")]
    pandora_l = df[(df.dataset == "pandora")  & (df.use_peft == "lora")]
    emotion_b = df[(df.dataset == "emotion")  & (df.use_peft == "baseline")]
    emotion_l = df[(df.dataset == "emotion")  & (df.use_peft == "lora")]

    combos = []
    for seed in df.seed.dropna().unique():
        for _, pb in pandora_b[pandora_b.seed == seed].iterrows():
            for _, el in emotion_l[emotion_l.seed == seed].iterrows():
                combos.append({"seed": seed, "base-model": pb.exp_id, "lora-model": el.exp_id})
        for _, eb in emotion_b[emotion_b.seed == seed].iterrows():
            for _, pl in pandora_l[pandora_l.seed == seed].iterrows():
                combos.append({"seed": seed, "base-model": eb.exp_id, "lora-model": pl.exp_id})
    return pd.DataFrame(combos)


# --------------------------------------------------------------------------- #
# model loader
# --------------------------------------------------------------------------- #

def load_combined_model(base_row: pd.Series,
                        lora_row: pd.Series,
                        meta: Dict[str, Any],
                        device: str,
                        # dtype: torch.dtype | None,
                        dtype: Union[torch.dtype, None],
                        low_mem: bool = False):
    base_name  = meta[str(base_row.exp_id)]["model_name"]
    base_ckpt  = base_row.exp_out_dir
    model = AutoModelForCausalLM.from_pretrained(
        base_ckpt,
        torch_dtype=dtype,
        device_map=None if device == "cpu" else "auto",
    ).to(device).eval()

    tok = AutoTokenizer.from_pretrained(base_name)
    tok.add_special_tokens({"bos_token": "<|startoftext|>"})
    tok.pad_token = tok.eos_token
    model.resize_token_embeddings(len(tok))

    cfg   = PEFTManager.get_peft_config("lora")
    model = get_peft_model(model, cfg)

    lora_path = next(p for p in lora_row.model_files if p.endswith("lora_final.pt"))
    # state     = torch.load(os.path.join(lora_row.exp_out_dir, lora_path), map_location="cpu")
    state = torch.load(os.path.join(lora_row.exp_out_dir, lora_path), map_location="cpu", weights_only=True)
    set_peft_model_state_dict(model, state, low_cpu_mem_usage=low_mem)
    model.to(device).eval()
    return model, tok


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #

def main(args):
    df_summary, meta = collect_experiment_summary(args.metadata, args.outputs)
    if args.combos and Path(args.combos).exists():
        df_combos = pd.read_csv(args.combos)
    else:
        df_combos = make_combinations(df_summary)
    df_combos = df_combos[df_combos.seed == 183]
    print(f"Found {len(df_combos)} model combinations to evaluate.")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype  = torch.float16 if device == "cuda" else None
    results = []

    if args.max_combos:
        df_combos = df_combos.iloc[:args.max_combos]
        print(f"Found {len(df_combos)} model combinations to evaluate for prototyping.")

    # for _, cmb in df_combos.iterrows():
    for combo_idx, (_, cmb) in enumerate(df_combos.iterrows(), 1):
        print(f"Using combination {combo_idx} / {len(df_combos)}")
        base_row = df_summary[df_summary.exp_id == cmb["base-model"]].iloc[0]
        lora_row = df_summary[df_summary.exp_id == cmb["lora-model"]].iloc[0]
        model, tok = load_combined_model(base_row, lora_row, meta, device, dtype, args.low_mem)
        evaluator  = EvalManager(device=device, model=model, tokenizer=tok)

        # for scale in peft_scales:
        for scale in tqdm(peft_scales, desc=f"Evaluating across LoRA scales..."):
            with rescale_adapter_scale(model, scale):
                for inv in ("BFI10", "PANASX", "IPIP120"):
                    df_inv = evaluator.score_likert(inventory_name=inv,
                                                    include_options="both",
                                                    batch_size=args.bs,
                                                    return_dataframe=True)
                    df_inv.insert(0, "inventory",   inv)
                    df_inv.insert(0, "lora_scale",  scale)
                    df_inv.insert(0, "lora-model",  lora_row.exp_id)
                    df_inv.insert(0, "base-model",  base_row.exp_id)
                    results.append(df_inv)

            torch.cuda.empty_cache(); gc.collect()

    out_df  = pd.concat(results, ignore_index=True)
    cols    = ["base-model", "lora-model", "lora_scale",
               "inventory", "item", "likert_in_prompt", 1, 2, 3, 4, 5]
    out_df  = out_df[cols]
    csv_out = Path(args.output_csv)
    if csv_out.exists():
        prev = pd.read_csv(csv_out)
        out_df = pd.concat([prev, out_df]).drop_duplicates(cols, keep="last")
    out_df.to_csv(csv_out, index=False)
    print(f"Saved results → {csv_out.resolve()}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="outputs_best/experiment_metadata copy.json")
    p.add_argument("--outputs",  default="outputs_best")
    p.add_argument("--combos",   default="", help="optional pre-computed combinations CSV")
    p.add_argument("--output_csv", default="ta_results.csv")
    p.add_argument("--bs", type=int, default=5)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--low_mem", action="store_true")
    p.add_argument("--max_combos", type=int, default=None,
               help="Only run on the first N combinations (for prototyping)")
    a = p.parse_args()
    main(a)
