"""Reveal a prelocked small FIQA comparison only after independent review lock.

The required --review-sha256 must be supplied from the assessor's locked-file
receipt. No automatic locking here: that would not establish blind ordering.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "workspace/.vision-evaluation/efficient-fiqa-20261007"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def direction(value):
    return int(value > 0) - int(value < 0)


def pair_agreement(human, model):
    values = []
    for first, second in itertools.combinations(range(len(human)), 2):
        expected = direction(human[first] - human[second])
        if expected == 0:
            continue
        observed = direction(model[first] - model[second])
        values.append(1.0 if observed == expected else .5 if observed == 0 else 0.0)
    return {"agreement": float(np.mean(values)) if values else None, "nonTiedVisualPairs": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--review-sha256", required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    review_path = args.review or out / "astra-locked-review.json"
    expected = args.review_sha256.lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("Require the assessor's already locked SHA-256 receipt")
    if sha(review_path) != expected:
        raise ValueError("Independent review differs from supplied locked receipt")
    if (out / "reveal.json").exists():
        raise ValueError("Refusing to replace an existing revealed comparison")
    # Validate public review and its complete coverage before reading the key
    # or model scores. A frozen malformed review must be repaired and relocked
    # by the independent assessor, not silently edited by the preparer.
    review = load(review_path)
    manifest, protocol = load(out / "blind/manifest.json"), load(out / "blind/protocol.json")
    lock = load(out / "score-lock.json")
    for field, file in (("protocolSha256", "blind/protocol.json"), ("manifestSha256", "blind/manifest.json")):
        if review.get(field) != sha(out / file) or lock[field] != sha(out / file):
            raise ValueError(f"Review/preparation mismatch: {field}")
    visual = {}
    for group in review["groups"]:
        for item in group["items"]:
            if item["id"] in visual:
                raise ValueError("Duplicated independent score")
            score = item.get("quality")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score) or not 0 <= score <= 4 or score * 2 != round(score * 2):
                raise ValueError("Visual score must follow the prelocked 0-4, .5-step scale")
            visual[item["id"]] = item
    ids = {item["id"] for group in manifest["groups"] for item in group["items"]}
    if set(visual) != ids:
        raise ValueError("Require complete independent scores for all anonymous cases")
    if sha(out / "sealed-key.json") != lock["sealedKeySha256"] or sha(out / "sealed-scores.json") != lock["sealedScoresSha256"]:
        raise ValueError("Sealed key or model score changed after preparation")
    key, scores = load(out / "sealed-key.json"), load(out / "sealed-scores.json")
    key_items = {item["id"]: item for item in key["items"]}
    score_items = {item["id"]: item for item in scores["items"]}
    if set(score_items) != ids:
        raise ValueError("Model score coverage mismatch")
    for group in manifest["groups"]:
        for item in group["items"]:
            if sha(out / "blind/images" / item["file"]) != item["sha256"]:
                raise ValueError("Anonymous pixels changed")
    groups, rho_gains, pair_gains, vetoes = [], [], [], []
    disclosed = []
    for group in manifest["groups"]:
        group_ids = [item["id"] for item in group["items"]]
        human = np.array([visual[id]["quality"] for id in group_ids], dtype=float)
        current = np.array([score_items[id]["current"]["score"] for id in group_ids], dtype=float)
        # Raw linear prediction avoids display clipping creating artificial
        # ranking ties; affine score*100 and raw have identical order in range.
        efficient = np.array([score_items[id]["efficientFiqa"]["provenance"]["rawScore"] for id in group_ids], dtype=float)
        if not np.isfinite(current).all() or not np.isfinite(efficient).all() or np.ptp(human) == 0:
            raise ValueError("Need finite model scores and a scorable visual ranking")
        current_rho = float(spearmanr(human, current).statistic) if np.ptp(current) else 0.0
        efficient_rho = float(spearmanr(human, efficient).statistic) if np.ptp(efficient) else 0.0
        current_pair, efficient_pair = pair_agreement(human, current), pair_agreement(human, efficient)
        rho_gain, pair_gain = efficient_rho - current_rho, efficient_pair["agreement"] - current_pair["agreement"]
        rho_gains.append(rho_gain)
        pair_gains.append(pair_gain)
        variant_index = {key_items[id]["variant"]: ix for ix, id in enumerate(group_ids)}
        original = variant_index["original"]
        sanity = {}
        for variant, ix in variant_index.items():
            if variant == "original":
                continue
            sanity[variant] = {"visualOriginalPreferred": bool(human[original] > human[ix]),
                               "currentOriginalPreferred": bool(current[original] > current[ix]),
                               "efficientOriginalPreferred": bool(efficient[original] > efficient[ix])}
            if variant in ("severe-blur", "clipped-bright") and human[original] > human[ix] and current[original] > current[ix] and efficient[original] < efficient[ix]:
                vetoes.append({"group": group["id"], "variant": variant, "anonymousOriginal": group_ids[original], "anonymousDefect": group_ids[ix]})
        groups.append({"id": group["id"], "sourceId": key_items[group_ids[0]]["sourceId"],
                       "currentSpearman": current_rho, "efficientSpearman": efficient_rho,
                       "rhoGain": rho_gain, "currentPairAgreement": current_pair,
                       "efficientPairAgreement": efficient_pair, "pairAgreementGain": pair_gain,
                       "controlledSanity": sanity})
        for ix, id in enumerate(group_ids):
            disclosed.append({"id": id, "sourceId": key_items[id]["sourceId"], "variant": key_items[id]["variant"],
                              "visualQuality": float(human[ix]), "trainingUsable": visual[id].get("trainingUsable"),
                              "visualReason": visual[id].get("shortReason"), "currentScore": float(current[ix]),
                              "efficientRawScore": float(efficient[ix]), "efficientDisplayScore": score_items[id]["efficientFiqa"]["score"]})
    rng = np.random.default_rng(20261007)
    draws = np.asarray(rho_gains)[rng.integers(0, len(groups), size=(20000, len(groups)))].mean(axis=1)
    interval = np.quantile(draws, [.025, .975]).tolist()
    rule = protocol["predeclaredPromotionRule"]
    summary = {"meanCurrentSpearman": float(np.mean([group["currentSpearman"] for group in groups])),
               "meanEfficientSpearman": float(np.mean([group["efficientSpearman"] for group in groups])),
               "meanRhoGain": float(np.mean(rho_gains)), "meanWithinSourcePairAgreementGain": float(np.mean(pair_gains)),
               "improvedSourceGroups": sum(gain >= rule["substantialRhoDifference"] for gain in rho_gains),
               "substantiallyWorsenedSourceGroups": sum(gain <= -rule["substantialRhoDifference"] for gain in rho_gains),
               "exploratorySourceBootstrap95GainInterval": interval,
               "severeInversionVetoes": vetoes}
    checks = {"rhoGain": summary["meanRhoGain"] >= rule["minimumMeanRhoGain"],
              "pairAgreementGain": summary["meanWithinSourcePairAgreementGain"] >= rule["minimumWithinSourcePairAgreementGain"],
              "improvedSourceGroups": summary["improvedSourceGroups"] >= rule["minimumImprovedSourceGroups"],
              "worsenedSourceGroups": summary["substantiallyWorsenedSourceGroups"] <= rule["maximumSubstantiallyWorsenedSourceGroups"],
              "positiveExploratoryBootstrapLower": interval[0] > 0,
              "noSevereInversion": not vetoes}
    promote = all(checks.values())
    result = {"schemaVersion": 1, "scope": protocol["scope"], "independentReviewSha256": expected,
              "blindProtocolSha256": lock["protocolSha256"], "blindManifestSha256": lock["manifestSha256"],
              "sealedKeySha256": lock["sealedKeySha256"], "sealedScoresSha256": lock["sealedScoresSha256"],
              "caseCount": len(ids), "sourceCount": len(groups), "summary": summary,
              "predeclaredRule": rule, "ruleChecks": checks, "clearBenefitInThisSample": promote,
              "recommendedDefault": "efficient-fiqa" if promote else "current-quality",
              "groups": groups, "revealedItems": disclosed, "sources": key["sources"],
              "originalsUnchanged": all(sha(item["path"]) == item["sha256"] for item in key["sources"]),
              "trainingPerformed": False, "identityAccuracy": None, "annotationAccuracy": None,
              "limitations": protocol["limitations"] + ["Predeclared effect thresholds are pragmatic, not a population significance test", "No training-output quality, identity fidelity or unseen-video generalization proven"]}
    if not result["originalsUnchanged"]:
        raise ValueError("Original source changed")
    (out / "reveal.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reveal": str(out / "reveal.json"), "revealSha256": sha(out / "reveal.json"), "clearBenefitInThisSample": promote,
                      "recommendedDefault": result["recommendedDefault"], "summary": summary, "ruleChecks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
