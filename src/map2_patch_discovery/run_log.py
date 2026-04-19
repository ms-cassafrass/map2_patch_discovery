from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence


def write_run_metadata(
    output_dir: str | Path,
    *,
    pipeline_name: str,
    command_argv: Sequence[str],
    config_path: str | Path,
    extra_lines: Sequence[str] | None = None,
) -> Path:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path).resolve()
    metadata_path = output_dir / "run_metadata.md"

    command_text = " ".join(str(arg) for arg in command_argv)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    cwd = Path.cwd().resolve()
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        config_text = f"<unable to read config: {exc}>"

    lines = [
        f"# {pipeline_name} Run Metadata",
        "",
        f"- Timestamp: `{timestamp}`",
        f"- Working directory: `{cwd}`",
        f"- Config path: `{config_path}`",
        "",
        "## Command",
        "",
        "```text",
        command_text,
        "```",
        "",
    ]
    if extra_lines:
        lines.extend(["## Run Info", ""])
        for line in extra_lines:
            lines.append(f"- {line}")
        lines.append("")
    lines.extend(
        [
            "## Config Contents",
            "",
            "```yaml",
            config_text.rstrip(),
            "```",
            "",
        ]
    )
    metadata_path.write_text("\n".join(lines), encoding="utf-8")
    return metadata_path
