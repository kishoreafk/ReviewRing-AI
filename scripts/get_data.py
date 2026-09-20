#!/usr/bin/env python3
"""Dataset acquisition: Kaggle-first with validated fallbacks (spec section 5).

Strategy
--------
1. Kaggle (uses ~/.kaggle/kaggle.json if present) - any candidate must pass
   schema validation, otherwise it is rejected *loudly* (never silently).
2. Fallback A (YelpChi): the canonical CARE-GNN GitHub repository, which hosts
   the original .mat files with per-relation adjacency (net_rur/net_rtr/net_rsr).
3. Fallback B (Track B background): official Amazon Reviews 2023 category file
   from the McAuley-Lab HuggingFace dataset.

Every downloaded file is hashed; hashes are appended to data/raw/source_hashes.txt.

Usage:
    python scripts/get_data.py [--only yelp|amazon] [--data-dir data/raw]
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/YingtongDou/CARE-GNN.git"
AMAZON_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/review_categories/Subscription_Boxes.jsonl"
)
KAGGLE_YELP_CANDIDATES = ["hunhthanhdn/yelpchi-dataset-gat"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[get_data] {msg}", flush=True)


def have_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    env_ok = bool(__import__("os").environ.get("KAGGLE_USERNAME")) and bool(
        __import__("os").environ.get("KAGGLE_KEY")
    )
    return kaggle_json.exists() or env_ok


def validate_yelp_mat(mat_path: Path) -> bool:
    """The benchmark must expose per-relation sparse adjacency (spec 6.1)."""
    try:
        from scipy import sparse
        from scipy.io import loadmat

        mat = loadmat(mat_path)
        ok = all(k in mat for k in ("features", "label", "net_rur", "net_rtr", "net_rsr"))
        if not ok:
            log(f"schema check failed for {mat_path.name}: missing keys")
            return False
        n = mat["features"].shape[0]
        for k in ("net_rur", "net_rtr", "net_rsr"):
            if not sparse.issparse(mat[k]) or mat[k].shape != (n, n):
                log(f"schema check failed: relation {k} invalid")
                return False
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"schema check error: {exc}")
        return False


def try_kaggle_yelpchi(data_dir: Path) -> bool:
    """Attempt Kaggle mirrors; reject any that lose relation identity."""
    if not have_kaggle_credentials():
        log("no Kaggle credentials found (~/.kaggle/kaggle.json); skipping Kaggle")
        return False
    for slug in KAGGLE_YELP_CANDIDATES:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                log(f"trying Kaggle dataset {slug}")
                subprocess.run(
                    [sys.executable, "-m", "kaggle", "datasets", "download", slug, "-p", str(tmp), "--unzip"],
                    check=True,
                    capture_output=True,
                    timeout=600,
                )
                # inspect candidates for a usable .mat
                mats = list(tmp.rglob("*.mat"))
                for mat in mats:
                    if validate_yelp_mat(mat):
                        dest = data_dir / "care" / "YelpChi.mat"
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(mat, dest)
                        log(f"accepted Kaggle mirror -> {dest} (sha256 {sha256_file(dest)[:16]}...)")
                        return True
                # also reject preprocessed .pt bundles explicitly
                pts = list(tmp.rglob("*.pt"))
                if pts or mats:
                    log(
                        f"REJECTED Kaggle mirror {slug}: artifacts are preprocessed "
                        f"({len(pts)} .pt / {len(mats)} .mat) and fail the per-relation "
                        "schema contract (spec 6.1). Documented, not silent. Falling back."
                    )
        except Exception as exc:  # noqa: BLE001
            log(f"Kaggle download of {slug} failed: {exc}")
    return False


def get_yelpchi(data_dir: Path) -> Path:
    dest = data_dir / "care" / "YelpChi.mat"
    if dest.exists() and validate_yelp_mat(dest):
        log(f"YelpChi already present: {dest}")
        return dest
    if try_kaggle_yelpchi(data_dir):
        return dest
    # canonical fallback: CARE-GNN GitHub repository
    log("falling back to the canonical source: github.com/YingtongDou/CARE-GNN")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(tmp / "CARE-GNN")], check=True)
        sha = (
            subprocess.run(["git", "-C", str(tmp / "CARE-GNN"), "rev-parse", "HEAD"], capture_output=True, text=True)
            .stdout.strip()
        )
        log(f"CARE-GNN commit {sha}")
        zips = list((tmp / "CARE-GNN" / "data").glob("*.zip"))
        for z in zips:
            out = tmp / z.stem
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
            for mat in out.glob("*.mat"):
                target = data_dir / "care" / mat.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(mat, target)
                log(f"extracted {mat.name} (sha256 {sha256_file(target)[:16]}...)")
    if not dest.exists():
        raise RuntimeError("YelpChi.mat not available from any source")
    return dest


def get_amazon(data_dir: Path) -> Path:
    dest = data_dir / "amazon" / "Subscription_Boxes.jsonl"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log(f"Amazon category file already present: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log("downloading official Amazon Reviews 2023 category file")
    import urllib.request

    urllib.request.urlretrieve(AMAZON_URL, dest)
    log(f"saved {dest} (sha256 {sha256_file(dest)[:16]}...)")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["yelp", "amazon"], default=None)
    ap.add_argument("--data-dir", default="data/raw")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    if args.only in (None, "yelp"):
        get_yelpchi(data_dir)
    if args.only in (None, "amazon"):
        get_amazon(data_dir)

    hashes_file = data_dir / "source_hashes.txt"
    lines = []
    for mat in sorted(data_dir.rglob("*")):
        if mat.is_file() and mat.suffix in {".mat", ".jsonl", ".jsonl.gz"}:
            lines.append(f"{sha256_file(mat)}  {mat.relative_to(data_dir)}")
    hashes_file.write_text("\n".join(lines) + "\n")
    log(f"hashes written to {hashes_file}")


if __name__ == "__main__":
    main()
