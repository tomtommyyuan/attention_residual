"""Collect all run results into results/summary.json.

Run from the repo root on the cluster after runs complete:
    python3 analysis/collect_summary.py
Then commit results/summary.json. Pure stdlib -- no venv needed.
"""

import glob
import json
import os

GROUPS = {
    "seed1337_4xA100": {"baseline": "baseline_124m",
        "arms": ["attnres_124m", "sparse_sink_124m_k8", "sparse_sink_124m_k4"]},
    "seed42_4xA100": {"baseline": "baseline_124m_s42",
        "arms": ["attnres_124m_s42", "sparse_sink_124m_k8_s42"]},
    "seed7_1xB200": {"baseline": "baseline_124m_s7_b200",
        "arms": ["attnres_124m_s7_b200", "sparse_k8_s7_b200", "sparse_k4_s7_b200"]},
    "seed8_1xB200": {"baseline": "baseline_124m_s8_b200",
        "arms": ["attnres_124m_s8_b200", "sparse_k8_s8_b200"]},
    "seed9_1xB200": {"baseline": "baseline_124m_s9_b200",
        "arms": ["attnres_124m_s9_b200", "sparse_k8_s9_b200"]},
}


def total_steps(name):
    return 13700 if "350m" in name else 4800


def read_run(name):
    path = f"out/{name}/log.jsonl"
    if not os.path.exists(path):
        return None
    tr, va = [], []
    for line in open(path):
        rec = json.loads(line)
        (tr if "tok_per_s" in rec else va if "val_loss" in rec else []).append(rec)
    if not tr:
        return None
    t = total_steps(name)
    step = max(tr[-1]["step"], va[-1]["step"] if va else 0)
    return {"final_step": step, "total_steps": t,
            "final_val_loss": va[-1]["val_loss"] if va else None,
            "tok_per_s": tr[-1]["tok_per_s"], "mfu": tr[-1]["mfu"],
            "status": "done" if step >= t - 1 else "incomplete"}


def main():
    doc = {"runs": {}, "pairs_124m": {}, "scale_350m": {}, "summary": {}}
    for d in sorted(glob.glob("out/*/log.jsonl")):
        name = d.split("/")[1]
        r = read_run(name)
        if not r:
            continue
        doc["runs"][name] = r
        if "350m" in name:
            doc["scale_350m"][name] = r

    k8, full, k4 = [], [], []
    for gname, g in GROUPS.items():
        base = doc["runs"].get(g["baseline"])
        if not (base and base["status"] == "done" and base["final_val_loss"]):
            continue
        for arm in g["arms"]:
            a = doc["runs"].get(arm)
            if not (a and a["status"] == "done" and a["final_val_loss"]):
                continue
            gap = round(base["final_val_loss"] - a["final_val_loss"], 5)
            doc["pairs_124m"][f"{gname}:{arm}"] = {
                "baseline_val": base["final_val_loss"],
                "arm_val": a["final_val_loss"], "gap": gap}
            (k8 if "k8" in arm else full if "attnres" in arm else k4).append((gname, gap))

    b1 = doc["runs"].get("baseline_124m")
    b2 = doc["runs"].get("baseline_124m_s42")
    doc["summary"] = {
        "sparse_k8_gaps": dict(k8), "full_attnres_gaps": dict(full),
        "sparse_k4_gaps": dict(k4),
        "sparse_k8_mean": round(sum(v for _, v in k8) / len(k8), 5) if k8 else None,
        "full_mean": round(sum(v for _, v in full) / len(full), 5) if full else None,
        "baseline_seed_noise_same_hw": round(
            abs(b1["final_val_loss"] - b2["final_val_loss"]), 5) if b1 and b2 else None,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps(doc["summary"], indent=2))
    incomplete = [n for n, r in doc["runs"].items() if r["status"] != "done"]
    print(f"\nincomplete runs (excluded from pairs): {incomplete or 'none'}")
    print("wrote results/summary.json")


if __name__ == "__main__":
    main()
