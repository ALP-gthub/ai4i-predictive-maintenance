"""Upload this repository to a Hugging Face Space.

Run by .github/workflows/sync-to-hub.yml. Every check writes a GitHub
annotation, so the cause of a failure is visible on the run's summary page
without opening the full log. The token itself is never printed.
"""

import os
import sys

from huggingface_hub import HfApi


def note(level, title, message):
    print(f"::{level} title={title}::{message}", flush=True)


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    space_id = os.environ.get("HF_SPACE_ID", "").strip()

    missing = []
    if not token:
        missing.append("secret HF_TOKEN")
    if not space_id:
        missing.append("variable HF_SPACE_ID")
    if missing:
        note(
            "error",
            "Missing setting",
            "Not found: " + ", ".join(missing)
            + ". Add under Settings > Secrets and variables > Actions.",
        )
        return 1
    if space_id.count("/") != 1:
        note("error", "Invalid HF_SPACE_ID", f"Expected <username>/<space-name>, got '{space_id}'.")
        return 1

    api = HfApi(token=token)
    try:
        who = api.whoami()
    except Exception as exc:  # invalid, expired or revoked token
        note("error", "Token rejected", f"Hugging Face did not accept HF_TOKEN: {type(exc).__name__}: {exc}")
        return 1

    role = who.get("auth", {}).get("accessToken", {}).get("role", "unknown")
    note("notice", "Token OK", f"Token belongs to '{who.get('name')}', permission: {role}.")

    owner = space_id.split("/")[0]
    if owner != who.get("name"):
        note(
            "warning",
            "Owner mismatch",
            f"Space owner '{owner}' differs from token owner '{who.get('name')}'.",
        )

    try:
        url = api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        note("notice", "Space ready", str(url))
        commit = api.upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=".",
            ignore_patterns=[".git/*", ".github/*", "**/__pycache__/*"],
            commit_message=f"Sync from GitHub {os.environ.get('GITHUB_SHA', '')[:7]}",
        )
    except Exception as exc:
        note("error", "Upload failed", f"{type(exc).__name__}: {exc}")
        response = getattr(exc, "response", None)
        if response is not None:
            body = " ".join(response.text.split())[:900]
            note("error", "Hugging Face response", body or "(empty body)")
        return 1

    note("notice", "Upload complete", f"Space updated: {commit.commit_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
