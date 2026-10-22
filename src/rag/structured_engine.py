"""Structured Data Engine for Dog Breed Classification.

Executes deterministic mathematical and metadata queries over structured artifacts
(CSV, JSON) to eliminate numerical and factual hallucinations.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class StructuredEngine:
    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or PROJECT_ROOT
        self.qual_dir = self.root / "outputs" / "quality_evaluation_2026-10-16"
        self.deploy_v2_dir = self.root / "outputs" / "deployment_oe_v2"
        self.metadata_dir = self.root / "metadata"

        self._class_names: Optional[Dict[int, str]] = None
        self._per_class_metrics: Optional[List[Dict[str, Any]]] = None
        self._top_confusions: Optional[List[Dict[str, Any]]] = None
        self._summary: Optional[Dict[str, Any]] = None
        self._ood_calibration: Optional[Dict[str, Any]] = None

    def get_class_names(self) -> Dict[int, str]:
        """Returns the dictionary of 120 supported class indices and names."""
        if self._class_names is None:
            path = self.deploy_v2_dir / "class_names.json"
            if not path.exists():
                path = self.root / "outputs" / "deployment_ready" / "class_names.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self._class_names = {int(k): str(v) for k, v in raw.items()}
            else:
                self._class_names = {}
        return self._class_names

    def is_breed_supported(self, breed_query: str) -> Tuple[bool, Optional[str], Optional[int]]:
        """Check if breed is among the 120 supported classes.

        Returns (is_supported, canonical_breed_name, class_index).
        """
        normalized = breed_query.strip().lower().replace(" ", "_").replace("-", "_")
        names = self.get_class_names()
        for idx, name in names.items():
            norm_name = name.lower().replace("-", "_")
            if normalized == norm_name or normalized in norm_name:
                return True, name, idx
        return False, None, None

    def get_per_class_metrics(self) -> List[Dict[str, Any]]:
        """Load and cache per-class metrics from per_class_metrics.csv."""
        if self._per_class_metrics is None:
            csv_path = self.qual_dir / "per_class_metrics.csv"
            rows = []
            if csv_path.exists():
                with open(csv_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        # Clean any remaining BOM or whitespace from keys
                        clean_r = {k.lstrip("\ufeff").strip(): v for k, v in r.items() if k}
                        rows.append({
                            "class_index": int(clean_r["class_index"]),
                            "breed": clean_r["breed"],
                            "support": int(clean_r["support"]),
                            "predicted_count": int(clean_r["predicted_count"]),
                            "true_positive": int(clean_r["true_positive"]),
                            "precision": float(clean_r["precision"]),
                            "recall": float(clean_r["recall"]),
                            "f1": float(clean_r["f1"]),
                        })
            self._per_class_metrics = rows
        return self._per_class_metrics

    def get_lowest_f1_breeds(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Returns the top_n breeds with the lowest F1 score."""
        metrics = self.get_per_class_metrics()
        sorted_m = sorted(metrics, key=lambda x: (x["f1"], x["recall"], x["precision"]))
        return sorted_m[:top_n]

    def get_highest_f1_breeds(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Returns the top_n breeds with the highest F1 score."""
        metrics = self.get_per_class_metrics()
        sorted_m = sorted(metrics, key=lambda x: (-x["f1"], -x["recall"], -x["precision"]))
        return sorted_m[:top_n]

    def get_breed_metric(self, breed_name: str) -> Optional[Dict[str, Any]]:
        """Fetch metrics for a specific breed by name."""
        norm_query = breed_name.strip().lower().replace(" ", "_").replace("-", "_")
        metrics = self.get_per_class_metrics()
        for m in metrics:
            norm_b = m["breed"].lower().replace("-", "_")
            if norm_query == norm_b or norm_query in norm_b:
                return m
        return None

    def get_top_confusions(self, breed_name: Optional[str] = None, top_n: int = 15) -> List[Dict[str, Any]]:
        """Load top confusions from top_confusions.csv, optionally filtered by breed."""
        if self._top_confusions is None:
            csv_path = self.qual_dir / "top_confusions.csv"
            rows = []
            if csv_path.exists():
                with open(csv_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        clean_r = {k.lstrip("\ufeff").strip(): v for k, v in r.items() if k}
                        rows.append({
                            "true_breed": clean_r["true_breed"],
                            "predicted_breed": clean_r["predicted_breed"],
                            "count": int(clean_r["count"]),
                            "true_support": int(clean_r["true_support"]),
                            "share_of_true_class": float(clean_r["share_of_true_class"]),
                        })
            self._top_confusions = rows

        if breed_name:
            norm_q = breed_name.strip().lower().replace(" ", "_").replace("-", "_")
            filtered = [
                r for r in self._top_confusions
                if norm_q in r["true_breed"].lower().replace("-", "_")
                or norm_q in r["predicted_breed"].lower().replace("-", "_")
            ]
            return filtered[:top_n]
        return self._top_confusions[:top_n]

    def get_global_metrics(self) -> Dict[str, Any]:
        """Read measured values from the evaluation artifact, never canned constants."""
        with (self.qual_dir / "summary.json").open(encoding="utf-8") as stream:
            summary = json.load(stream)
        m = summary["metrics"]
        c = m["classification_before_ood_rejection"]
        final = m["final_user_result_quality"]
        states = m["state_distribution"]
        non_dog = states["verified_non_dog_combined"]
        latency = summary["latency"]["end_to_end_in_memory_bytes_ms"]
        return {
            "evaluation_date_utc": summary["completed_at_utc"],
            "model_version": summary.get("artifacts", {}).get("model_version", "unknown"),
            "supported_classes": len(self.get_class_names()),
            "test_images_count": c["sample_count"],
            "top1_accuracy": c["top1"]["rate"],
            "top1_correct_count": c["top1"]["count"],
            "top1_misclassified_count": c["sample_count"] - c["top1"]["count"],
            "top5_accuracy": c["top5"]["rate"],
            "top5_correct_count": c["top5"]["count"],
            "top5_misclassified_count": c["sample_count"] - c["top5"]["count"],
            "macro_precision": c["macro_precision"], "macro_recall": c["macro_recall"], "macro_f1": c["macro_f1"],
            "id_dog_rate": final["dog_acceptance"]["rate"],
            "id_dog_count": final["dog_acceptance"]["count"],
            "id_dog_correct_count": final["dog_and_correct_overall"]["count"],
            "id_dog_and_correct_breed_rate": final["dog_and_correct_overall"]["rate"],
            "accuracy_on_dog_subset": final["breed_accuracy_among_DOG"]["rate"],
            "non_dog_test_count": non_dog["sample_count"],
            "non_dog_false_dog_count": non_dog["DOG"]["count"],
            "non_dog_false_dog_rate": non_dog["DOG"]["rate"],
            "non_dog_uncertain_count": non_dog["UNCERTAIN"]["count"],
            "non_dog_not_dog_count": non_dog["NOT_DOG"]["count"],
            "supported_dog_false_not_dog_count": states["supported_dog"]["NOT_DOG"]["count"],
            "auroc": m["ood_continuous_score"]["auroc"],
            "auprc": m["ood_continuous_score"]["auprc_ood_positive"],
            "latency_median_ms": round(latency["median"], 2),
            "latency_p95_ms": round(latency["p95"], 2),
            "source_ref": (self.qual_dir / "summary.json").relative_to(self.root).as_posix(),
        }

    def get_robustness_summary(self) -> Dict[str, Any]:
        with (self.qual_dir / "summary.json").open(encoding="utf-8") as stream:
            summary = json.load(stream)["robustness"]
        variants = {}
        for name, values in summary["transforms"].items():
            n = values["sample_count"]
            fmt = lambda rate: f"{rate*100:.2f}% ({round(rate*n)}/{n})"
            variants[name] = {
                "top1_preserved": fmt(values["top1_agreement_with_source"]),
                "status_preserved": fmt(values["status_agreement_with_source"]),
                "accuracy": fmt(values["top1_accuracy"]),
                "delta_ood_mean": values["mean_absolute_ood_score_change"],
            }
        return {"source_samples":summary["independent_source_images"], "variants":variants,
                "source_ref":(self.qual_dir / "summary.json").relative_to(self.root).as_posix()}

    def get_ood_thresholds(self) -> Dict[str, Any]:
        with (self.deploy_v2_dir / "ood_calibration.json").open(encoding="utf-8") as stream:
            source = json.load(stream)
        return {"detector": source["selected_detector"].lower(),
                "score_definition": source["score_formula"],
                "score_orientation": source["score_orientation"],
                "operating_point": source["operating_point_selected"],
                "T_uncertain": round(source["T_uncertain"],6), "T_not_dog": round(source["T_not_dog"],6),
                "decision_rules": {k.upper():v for k,v in source["decision_rules"].items()},
                "source_file": "outputs/deployment_oe_v2/ood_calibration.json"}

    def get_shiba_inu_evidence(self) -> Dict[str, Any]:
        with (self.qual_dir / "predictions.csv").open(encoding="utf-8-sig",newline="") as stream:
            rows = [r for r in csv.DictReader(stream) if r["evaluation_group"]=="unsupported_dog_breed"]
        shiba = [r for r in rows if r["true_label"]=="shiba_inu"]
        def rates(subset):
            n = len(subset)
            return {s+"_rate":f"{sum(r['status']==s for r in subset)/n*100:.2f}% ({sum(r['status']==s for r in subset)}/{n})"
                    for s in ("DOG","UNCERTAIN","NOT_DOG")} if n else {}
        return {"breed":"shiba_inu","in_supported_120_classes":self.is_breed_supported("shiba_inu")[0],
                "dataset_group":"unsupported_dog_breed","dataset_source":"Oxford-IIIT Pet",
                "group_total_images":len(rows),"group_breeds":sorted({r["true_label"] for r in rows}),
                "group_results":rates(rows),"breed_total_images":len(shiba),"breed_results":rates(shiba),
                "source_ref":(self.qual_dir / "predictions.csv").relative_to(self.root).as_posix()}

    def get_phu_quoc_evidence(self) -> Dict[str, Any]:
        with (self.metadata_dir / "phu_quoc_ridgeback_statistics.json").open(encoding="utf-8") as stream:
            data = json.load(stream)
        supported = self.is_breed_supported("phu_quoc_ridgeback")[0]
        return {"breed":"phu_quoc_ridgeback","in_supported_120_classes":supported,
                "is_recognized_by_model":supported,"retrained_into_model":supported,
                "audit_target":data.get("audit_target"),"wikimedia_images_count":data["total_images"],
                "source_ref":"metadata/phu_quoc_ridgeback_statistics.json"}
