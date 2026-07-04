"""Deploy the MarketMood Gradio app and runtime artifacts to Hugging Face Spaces."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE_ID = "hw391/AIPI540-MarketMood"
DEFAULT_BUCKET_URI = "hf://buckets/hw391/AIPI540-MarketMood-storage"
DEFAULT_MOUNT_PATH = "/data/marketmood"


@dataclass(frozen=True)
class PathSync:
    """Local path and destination prefix for deployment sync."""

    local_path: str
    remote_path: str


ARTIFACT_SYNCS = [
    PathSync("models", "models"),
    PathSync("data/processed", "data/processed"),
    PathSync("data/prices", "data/prices"),
]

SPACE_UPLOADS = [
    PathSync("app", "app"),
    PathSync("src", "src"),
    PathSync("config.yaml", "config.yaml"),
    PathSync("requirements.txt", "requirements.txt"),
    PathSync("README.md", "README.md"),
    PathSync(
        "outputs/predictions/classical_price_only_test_predictions.csv",
        "outputs/predictions/classical_price_only_test_predictions.csv",
    ),
    PathSync(
        "outputs/predictions/deep_text_price_test_predictions.csv",
        "outputs/predictions/deep_text_price_test_predictions.csv",
    ),
]

UPLOAD_EXCLUDES = [
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.DS_Store",
]


def run_command(command: list[str], dry_run: bool = False) -> None:
    """Run or print a deployment command."""
    printable = " ".join(command)
    if dry_run:
        print(f"[dry-run] {printable}")
        return
    print(f"[run] {printable}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def sync_artifacts(bucket_uri: str, dry_run: bool) -> None:
    """Sync runtime model and data artifacts to the storage bucket."""
    for sync in ARTIFACT_SYNCS:
        local_path = PROJECT_ROOT / sync.local_path
        if not local_path.exists():
            raise FileNotFoundError(f"Missing deployment artifact path: {local_path}")
        command = [
            "hf",
            "buckets",
            "sync",
            str(local_path),
            f"{bucket_uri.rstrip('/')}/{sync.remote_path}",
            "--delete",
        ]
        run_command(command, dry_run=dry_run)


def configure_volume(space_id: str, bucket_uri: str, mount_path: str, dry_run: bool) -> None:
    """Attach the storage bucket as the Space runtime artifact volume."""
    run_command(
        [
            "hf",
            "spaces",
            "volumes",
            "set",
            space_id,
            "-v",
            f"{bucket_uri.rstrip('/')}:{mount_path}",
        ],
        dry_run=dry_run,
    )


def configure_variables(space_id: str, mount_path: str, dry_run: bool) -> None:
    """Set environment variables used by the Space runtime."""
    run_command(
        [
            "hf",
            "spaces",
            "variables",
            "add",
            space_id,
            "-e",
            f"MARKETMOOD_ARTIFACT_ROOT={mount_path}",
            "-e",
            f"HF_HOME={mount_path}/hf-cache",
            "-e",
            "GRADIO_ANALYTICS_ENABLED=False",
        ],
        dry_run=dry_run,
    )


def upload_space_source(space_id: str, dry_run: bool) -> None:
    """Upload the source files needed by the Space."""
    for upload in SPACE_UPLOADS:
        local_path = PROJECT_ROOT / upload.local_path
        if not local_path.exists():
            raise FileNotFoundError(f"Missing Space source path: {local_path}")
        command = [
            "hf",
            "upload",
            space_id,
            upload.local_path,
            upload.remote_path,
            "--repo-type=space",
            "--commit-message",
            "Deploy MarketMood Space",
        ]
        for pattern in UPLOAD_EXCLUDES:
            command.extend(["--exclude", pattern])
        run_command(command, dry_run=dry_run)


def restart_space(space_id: str, factory_reboot: bool, dry_run: bool) -> None:
    """Restart the Space after a deployment."""
    command = ["hf", "spaces", "restart", space_id]
    if factory_reboot:
        command.append("--factory-reboot")
    run_command(command, dry_run=dry_run)


def wait_for_space(space_id: str, dry_run: bool) -> None:
    """Wait until the Space finishes rebuilding or starting."""
    run_command(["hf", "spaces", "wait", space_id], dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    """Parse deployment options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-id", default=DEFAULT_SPACE_ID)
    parser.add_argument("--bucket-uri", default=DEFAULT_BUCKET_URI)
    parser.add_argument("--mount-path", default=DEFAULT_MOUNT_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Print actions without uploading or changing settings.")
    parser.add_argument("--skip-artifacts", action="store_true", help="Skip syncing models and cached data.")
    parser.add_argument("--skip-volume", action="store_true", help="Skip configuring the Space bucket volume.")
    parser.add_argument("--skip-variables", action="store_true", help="Skip configuring Space environment variables.")
    parser.add_argument("--skip-source", action="store_true", help="Skip uploading app/source files.")
    parser.add_argument("--restart", action="store_true", help="Restart the Space after deployment.")
    parser.add_argument("--factory-reboot", action="store_true", help="Restart from scratch without the Space build cache.")
    parser.add_argument("--wait", action="store_true", help="Wait for the Space to finish rebuilding or starting.")
    return parser.parse_args()


def main() -> None:
    """Deploy or redeploy MarketMood to Hugging Face Spaces."""
    args = parse_args()
    if not args.skip_artifacts:
        sync_artifacts(args.bucket_uri, dry_run=args.dry_run)
    if not args.skip_volume:
        configure_volume(args.space_id, args.bucket_uri, args.mount_path, dry_run=args.dry_run)
    if not args.skip_variables:
        configure_variables(args.space_id, args.mount_path, dry_run=args.dry_run)
    if not args.skip_source:
        upload_space_source(args.space_id, dry_run=args.dry_run)
    if args.restart:
        restart_space(args.space_id, factory_reboot=args.factory_reboot, dry_run=args.dry_run)
    if args.wait:
        wait_for_space(args.space_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
