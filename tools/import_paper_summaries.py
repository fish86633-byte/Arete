"""Copy selected saved summaries without recomputing experiments."""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ASSETS = {
    "load_two_stream_equal_means.csv": "experiments/dams_u8_load_030/results_load_030_resume_031_tail13/report/two_stream_equal_means.csv",
    "constellation_200_aggregate_quality.csv": "experiments/dams_constellation_recovery_035/results_recovery_035/report/aggregate_quality.csv",
    "GA1_two_stream_equal_means.csv": "experiments/ga_bhtp_continuous_036/results_ga_036_ga1/report/two_stream_equal_means.csv",
    "NX_six_paired_inputs.csv": "experiments/pcqr_bh_r3_local_repair/results_supplement_013/report/nx_paired.csv",
    "load_plot_data.json": "Demand_Aware_Marginal_Schedulingfor_Multi_Satellite_Beam_Hopping/figures/load_plot_data.json",
    "constellation_plot_data.json": "Demand_Aware_Marginal_Schedulingfor_Multi_Satellite_Beam_Hopping/figures/constellation_plot_data.json",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    rows = []
    for name, original_relative in ASSETS.items():
        original = args.workspace / original_relative
        data = original.read_bytes()
        target = repo / "paper_results" / name
        target.parent.mkdir(exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise SystemExit(f"Existing snapshot differs; review before replacing: {target}")
        target.write_bytes(data)
        rows.append({"workspace_relative_path": original_relative,
                     "original_path": str(original.resolve()),
                     "repository_path": target.relative_to(repo).as_posix(),
                     "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                     "copy_mode": "byte-identical; no new planning, replay, or aggregation"})
    repositories = {}
    for name in ("dams_u8_load_030", "dams_constellation_recovery_035",
                 "ga_bhtp_continuous_036", "pcqr_bh_r3_local_repair"):
        path = args.workspace / "experiments" / name
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                             text=True, capture_output=True)
        repositories[name] = {
            "workspace_relative_path": path.relative_to(args.workspace).as_posix(),
            "archive_head_at_import": git.stdout.strip() if git.returncode == 0 else None,
            "git_metadata_available": git.returncode == 0}
    index = {"workspace_root_at_import": str(args.workspace.resolve()),
             "note": "Archive HEADs locate saved repositories; they are not algorithm version identities.",
             "formal_experiments_rerun": False, "new_formal_planning_calls": 0,
             "raw_trajectories_and_input_banks_copied": False,
             "historical_method_names_preserved": True,
             "method_aliases": {"DAMS": "Arete", "M1-E": "Arete", "M1E": "Arete"},
             "assets": rows, "repositories": repositories}
    (repo / "provenance/experiment_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    print(f"Copied {len(rows)} saved summaries ({sum(r['bytes'] for r in rows)} bytes)")


if __name__ == "__main__":
    main()
