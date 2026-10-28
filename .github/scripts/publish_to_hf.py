"""Publish this repository to Hugging Face.

Run by .github/workflows/publish-to-hf.yml.

1. Static Space (HF_SPACE_ID): the browser demo in docs/. Static Spaces are
   free; Gradio Spaces on free hardware need a PRO subscription.
2. Model repository (HF_MODEL_ID, default <owner>/ai4i-failure-prediction-model):
   model/model.joblib, its card model/README.md and src/features.py, which
   the saved pipeline imports when it is loaded.

After each upload the Hugging Face history is squashed to a single commit.
The full history is kept on GitHub; the Hub only holds the current version,
so files removed or changed on GitHub do not stay readable there.

Every check writes a GitHub annotation, so results are visible on the run's
summary page without opening the full log. The token is never printed.
"""

import os
import sys

from huggingface_hub import CommitOperationAdd, HfApi


def note(level, title, message):
    print(f"::{level} title={title}::{message}", flush=True)


def explain(exc):
    message = f"{type(exc).__name__}: {exc}"
    response = getattr(exc, "response", None)
    if response is not None:
        body = " ".join(response.text.split())[:900]
        if body:
            message += f" | Hugging Face response: {body}"
    return message


def publish_space(api, space_id, sha):
    api.create_repo(space_id, repo_type="space", space_sdk="static", exist_ok=True)
    commit = api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path="docs",
        commit_message=f"Sync from GitHub {sha}",
    )
    api.super_squash_history(space_id, repo_type="space", commit_message=f"Current version, GitHub {sha}")
    note("notice", "Space updated", f"{commit.commit_url} (history squashed)")


def publish_model(api, model_id, sha):
    api.create_repo(model_id, repo_type="model", exist_ok=True)
    files = {
        "README.md": "model/README.md",
        "model.joblib": "model/model.joblib",
        "src/__init__.py": "src/__init__.py",
        "src/features.py": "src/features.py",
    }
    commit = api.create_commit(
        repo_id=model_id,
        repo_type="model",
        operations=[CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in files.items()],
        commit_message=f"Sync from GitHub {sha}",
    )
    api.super_squash_history(model_id, repo_type="model", commit_message=f"Current version, GitHub {sha}")
    note("notice", "Model updated", f"{commit.commit_url} (history squashed)")


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
        note("error", "Token rejected", f"Hugging Face did not accept HF_TOKEN: {explain(exc)}")
        return 1
    user = who.get("name")
    role = who.get("auth", {}).get("accessToken", {}).get("role", "unknown")
    note("notice", "Token OK", f"Token belongs to '{user}', permission: {role}.")

    owner = space_id.split("/")[0]
    model_id = os.environ.get("HF_MODEL_ID", "").strip() or f"{owner}/ai4i-failure-prediction-model"
    sha = os.environ.get("GITHUB_SHA", "")[:7]

    failed = False
    for label, publish, repo_id in [
        ("Space", publish_space, space_id),
        ("Model", publish_model, model_id),
    ]:
        try:
            publish(api, repo_id, sha)
        except Exception as exc:
            note("error", f"{label} upload failed", f"{repo_id}: {explain(exc)}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
