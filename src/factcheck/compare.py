"""Per-label, per-answer and per-claim comparison of two development reports."""
import json

from .config import LABELS


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _claim_states(report: dict) -> dict[str, str]:
    """gold claim_id -> correct | wrong:<label> | missed"""
    states = {}
    for a in report["answers"]:
        s = a["scores"]
        for gid in s["missing_gold_ids"]:
            states[gid] = "missed"
        for lr in a["label_results"] or []:
            states[lr["claim_id"]] = "correct" if lr["correct"] else f"wrong:{lr['predicted']}"
    return states


def compare_reports(baseline_path: str, candidate_path: str) -> str:
    b, c = _load(baseline_path), _load(candidate_path)
    for r in (b, c):
        if r["split"] != "development" or r["metrics"] is None:
            raise ValueError("compare needs two scored development reports")
    lines = [f"# {b['agent_version']} vs {c['agent_version']} on development", ""]
    same = {k: b[k] == c[k] for k in ("file_hashes", "settings", "model", "provider")}
    same["evaluator_version"] = b["versions"]["evaluator_version"] == c["versions"]["evaluator_version"]
    lines.append("Identical inputs and scoring: " + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in same.items()))
    lines.append(f"Prompt versions: {b['versions']['prompt_version']} -> {c['versions']['prompt_version']}")
    lines += ["", "| metric | baseline | candidate | delta |", "|---|---|---|---|"]
    bm, cm = b["metrics"], c["metrics"]

    def row(name, x, y):
        lines.append(f"| {name} | {x:.4f} | {y:.4f} | {y - x:+.4f} |")

    for part in ("extraction", "joint"):
        for k in ("precision", "recall", "f1"):
            row(f"{part} {k}", bm[part][k], cm[part][k])
    row("gold claim coverage", bm["gold_claim_coverage"], cm["gold_claim_coverage"])
    row("macro F1", bm["macro_f1"], cm["macro_f1"])
    for label in LABELS:
        for k in ("precision", "recall", "f1"):
            row(f"{label} {k}", bm["per_label"][label][k], cm["per_label"][label][k])
    lines += ["", "| count | baseline | candidate |", "|---|---|---|"]
    for k in ("P", "M", "C", "unmatched_predictions", "answer_errors"):
        lines.append(f"| {k} | {bm['counts'][k]} | {cm['counts'][k]} |")
    lines.append(f"| error types | {b['counts']['error_types']} | {c['counts']['error_types']} |")

    lines += ["", "## Confusion matrices (rows expected, columns predicted)", ""]
    for name, m in (("baseline", bm), ("candidate", cm)):
        cols = list(next(iter(m["confusion_matrix"].values())).keys())
        lines += [f"{name}:", "", "| expected | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for exp, cells in m["confusion_matrix"].items():
            lines.append(f"| {exp} | " + " | ".join(str(cells[k]) for k in cols) + " |")
        lines.append("")

    bs, cs = _claim_states(b), _claim_states(c)
    fixed = sorted(g for g in bs if bs[g] != "correct" and cs.get(g) == "correct")
    broken = sorted(g for g in bs if bs[g] == "correct" and cs.get(g) != "correct")
    changed = sorted(g for g in bs if bs[g] != "correct" and cs.get(g) not in ("correct", bs[g]))
    lines += [f"## Claim-level changes: {len(fixed)} fixed, {len(broken)} regressed, {len(changed)} still wrong but changed", ""]
    for title, ids in (("Fixed", fixed), ("Regressed", broken), ("Changed but still wrong", changed)):
        lines.append(f"{title}:")
        lines += [f"- {g}: {bs[g]} -> {cs.get(g)}" for g in ids] or ["- none"]
        lines.append("")

    b_extra = {x for a in b["answers"] for x in a["scores"]["extra_claim_ids"]}
    c_extra = {x for a in c["answers"] for x in a["scores"]["extra_claim_ids"]}
    lines.append(f"Extra predictions: baseline {len(b_extra)}, candidate {len(c_extra)}; "
                 f"new in candidate {sorted(c_extra - b_extra)}; removed {sorted(b_extra - c_extra)}")
    lines += ["", "## Per-answer joint F1 changes", "", "| answer | baseline | candidate | delta | status |",
              "|---|---|---|---|---|"]
    ca = {a["answer_id"]: a for a in c["answers"]}
    for a in b["answers"]:
        x, y = a["scores"]["joint_f1"], ca[a["answer_id"]]["scores"]["joint_f1"]
        if x != y:
            lines.append(f"| {a['answer_id']} | {x:.3f} | {y:.3f} | {y - x:+.3f} | "
                         f"{a['status']} -> {ca[a['answer_id']]['status']} |")
    return "\n".join(lines) + "\n"
