from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run latent reports sequentially for feature-variance cluster-specific configs."
    )
    parser.add_argument(
        "--config-prefix",
        default="latent_report_3_20_26_two_sample_feature_variance10_cluster_",
        help="Prefix for cluster config filenames inside the configs directory.",
    )
    parser.add_argument(
        "--clusters",
        nargs="*",
        type=int,
        help="Optional specific cluster ids to run. Defaults to all configs matching the prefix.",
    )
    parser.add_argument(
        "--configs-dir",
        default="configs",
        help="Directory containing the cluster-specific latent report YAML configs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later clusters if one cluster run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which configs would run without launching the latent report jobs.",
    )
    return parser


def _resolve_config_paths(configs_dir: Path, config_prefix: str, clusters: list[int] | None) -> list[Path]:
    if clusters:
        return [
            (configs_dir / f"{config_prefix}{cluster}.yaml").resolve()
            for cluster in clusters
        ]
    return sorted(configs_dir.glob(f"{config_prefix}*.yaml"))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    configs_dir = (repo_root / args.configs_dir).resolve()
    config_paths = _resolve_config_paths(configs_dir, args.config_prefix, args.clusters)
    if not config_paths:
        raise SystemExit(f"No matching configs found in {configs_dir} for prefix '{args.config_prefix}'")

    missing = [path for path in config_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise SystemExit(f"Missing config files:\n{missing_text}")

    batch_log = repo_root / "configs" / "outputs" / "feature_variance_batch_runs.log"
    batch_log.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Configs to run:")
        for path in config_paths:
            print(f" - {path}")
        print(f"Batch log: {batch_log}")
        return

    with batch_log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now().astimezone().isoformat(timespec='seconds')}] Starting batch run\n")
        handle.write(f"Working directory: {repo_root}\n")
        for index, config_path in enumerate(config_paths, start=1):
            command = [sys.executable, "-m", "map2_patch_discovery.report_cli", "--config", str(config_path)]
            print(f"[batch {index}/{len(config_paths)}] {config_path.name}")
            handle.write(f"[start] {config_path}\n")
            completed = subprocess.run(command, cwd=repo_root)
            handle.write(f"[end] {config_path} | exit_code={completed.returncode}\n")
            handle.flush()
            if completed.returncode != 0 and not args.continue_on_error:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
