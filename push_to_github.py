import os
import sys
import json
import base64
import urllib.request
import urllib.error
from pathlib import Path

TOKEN = "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"
REPO = "vip-sk07/Guard-AI"
BRANCH = "main"
PROJECT_DIR = Path(__file__).parent

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "GuardAI-Pusher"
}

def gh_api(endpoint, data=None, method=None):
    url = f"https://api.github.com/repos/{REPO}{endpoint}"
    req_data = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=req_data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        print(f"HTTP Error {e.code} for {url}: {err_msg}")
        raise

def get_latest_commit_and_tree():
    ref = gh_api(f"/git/ref/heads/{BRANCH}")
    commit_sha = ref["object"]["sha"]
    commit_obj = gh_api(f"/git/commits/{commit_sha}")
    tree_sha = commit_obj["tree"]["sha"]
    return commit_sha, tree_sha

def create_blob(filepath: Path):
    with open(filepath, "rb") as f:
        content = f.read()
    b64_content = base64.b64encode(content).decode("ascii")
    payload = {
        "content": b64_content,
        "encoding": "base64"
    }
    res = gh_api("/git/blobs", data=payload)
    return res["sha"]

def collect_files():

    exclude_dirs = {".git", "__pycache__", ".venv", "venv", ".idea", ".pytest_cache"}
    exclude_files = {".DS_Store"}

    files = []
    for root, dirs, filenames in os.walk(PROJECT_DIR):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for fname in filenames:
            if fname in exclude_files or fname.endswith(".pyc"):
                continue
            fpath = Path(root) / fname

            if fname in ["push_to_github.py", "prepare_datasets.py"]:
                pass
            rel_path = fpath.relative_to(PROJECT_DIR).as_posix()
            files.append((rel_path, fpath))
    return files

def main():
    print(f"=== Guard AI GitHub Sync -> {REPO} ({BRANCH}) ===")

    commit_sha, base_tree_sha = get_latest_commit_and_tree()
    print(f"Current HEAD commit: {commit_sha}")
    print(f"Base tree: {base_tree_sha}")

    files = collect_files()
    print(f"Found {len(files)} files to sync.")

    tree_items = []
    for idx, (rel_path, fpath) in enumerate(files, 1):
        size_kb = fpath.stat().st_size / 1024
        print(f"[{idx}/{len(files)}] Uploading blob: {rel_path} ({size_kb:.1f} KB)...")
        blob_sha = create_blob(fpath)
        tree_items.append({
            "path": rel_path,
            "mode": "100644",
            "type": "blob",
            "sha": blob_sha
        })

    print("Creating new Git tree...")
    tree_payload = {
        "base_tree": base_tree_sha,
        "tree": tree_items
    }
    new_tree = gh_api("/git/trees", data=tree_payload)
    new_tree_sha = new_tree["sha"]
    print(f"New tree SHA: {new_tree_sha}")

    print("Creating new commit...")
    commit_message = (
        "feat: complete Guard AI architecture with datasets, trained SLM model & DPO\n\n"
        "- Add datasets/ directory with Anthropic HH-RLHF, PKU-SafeRLHF, AdvBench, Chatbot Safety & MMLU\n"
        "- Add trained artifacts in data/trained/ (SLM classifier, DPO preferences, VDB, GA guardrail)\n"
        "- Integrate trained artifacts into core/guardrail_layer.py and live evaluation\n"
        "- Updated Streamlit dashboard & end-to-end test suite"
    )
    commit_payload = {
        "message": commit_message,
        "tree": new_tree_sha,
        "parents": [commit_sha]
    }
    new_commit = gh_api("/git/commits", data=commit_payload)
    new_commit_sha = new_commit["sha"]
    print(f"New commit SHA: {new_commit_sha}")

    print(f"Updating refs/heads/{BRANCH} to {new_commit_sha}...")
    ref_payload = {
        "sha": new_commit_sha,
        "force": False
    }
    updated_ref = gh_api(f"/git/refs/heads/{BRANCH}", data=ref_payload, method="PATCH")
    print(f"Successfully pushed to GitHub! New HEAD: {updated_ref['object']['sha']}")
    print(f"View repository: https://github.com/{REPO}")

if __name__ == "__main__":
    main()
