#!/usr/bin/env python
# validate_ecg_npy.py

import argparse
import sys
from pathlib import Path
import re
import numpy as np
from tqdm import tqdm

REQUIRED_KEYS = {"ecg", "report", "path", "orig_sf", "target_sf", "seg_len"}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_PREPROC_RE = re.compile(r"preprocessed_(\d+)(?:_\d+)?$")


def infer_seg_len(fp: Path) -> int | None:
    """Return segment length encoded in an ancestor dir like 'preprocessed_1250_250'."""
    for part in fp.parts:
        m = _PREPROC_RE.match(part)
        if m:
            return int(m.group(1))
    return None


def validate_file(fp: Path, strict: bool = False) -> bool:
    """Return True if *fp* passes all checks."""
    seg_len = infer_seg_len(fp)
    if seg_len is None:
        print(f"[ERROR] {fp}: cannot infer seg_len from directory structure", file=sys.stderr)
        return False
    expected_shape = (12, seg_len)

    # ---------------------------- load dict ---------------------------------
    try:
        obj = np.load(fp, allow_pickle=True)
        data = obj.item() if isinstance(obj, np.ndarray) else obj
    except Exception as exc:
        print(f"[ERROR] {fp}: cannot open ({exc})", file=sys.stderr)
        return False

    # ------------------------- key sanity -----------------------------------
    missing = REQUIRED_KEYS.difference(data.keys())
    if missing:
        print(f"[ERROR] {fp}: missing keys: {sorted(missing)}", file=sys.stderr)
        return False

    # ------------------------ ecg sanity ------------------------------------
    ecg = data["ecg"]
    if not isinstance(ecg, np.ndarray) or ecg.ndim != 2:
        print(f"[ERROR] {fp}: 'ecg' is not a 2-D numpy array", file=sys.stderr)
        return False

    if ecg.shape != expected_shape:
        print(
            f"[ERROR] {fp}: expected shape {expected_shape}, got {ecg.shape}",
            file=sys.stderr,
        )
        return False

    if np.isnan(ecg).any() or np.isinf(ecg).any():
        msg = f"[WARNING] {fp}: 'ecg' contains NaN or Inf values"
        if strict:
            print(msg.replace("WARNING", "ERROR"), file=sys.stderr)
            return False
        print(msg, file=sys.stderr)

    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(root: Path, strict: bool) -> None:
    npy_files = list(root.rglob("*.npy"))
    total = len(npy_files)

    if total == 0:
        print(f"No .npy files found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {total} .npy files\n")

    n_ok = n_fail = 0
    for fp in tqdm(npy_files, desc="Validating", unit="file"):
        if validate_file(fp, strict):
            n_ok += 1
        else:
            n_fail += 1

    print(f"\nSummary: {n_ok} OK  |  {n_fail} failed")
    if n_fail and strict:
        sys.exit(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Validate ECG .npy files")
    p.add_argument("folder", type=Path, help="Root folder to scan")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat any warning as fatal (non-zero exit code)",
    )
    args = p.parse_args()

    if not args.folder.is_dir():
        print(f"{args.folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    main(args.folder, args.strict)
