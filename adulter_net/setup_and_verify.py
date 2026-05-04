#!/usr/bin/env python3
"""Environment verification script."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def check_imports():
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
        ("ultralytics", "ultralytics"),
        ("timm", "timm"),
        ("sklearn", "scikit-learn"),
        ("imblearn", "imbalanced-learn"),
        ("shap", "shap"),
        ("lime", "lime"),
        ("boruta", "boruta"),
        ("mrmr", "mrmr-selection"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
        ("joblib", "joblib"),
        ("pytest", "pytest"),
        ("wandb", "wandb"),
        ("torchmetrics", "torchmetrics"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("optuna", "optuna"),
    ]
    results = []
    for mod_name, pkg_name in packages:
        try:
            mod = __import__(mod_name)
            version = getattr(mod, "__version__", "unknown")
            results.append((pkg_name, version, True))
            print(f"  ✓ {pkg_name:<30} {version}")
        except ImportError as e:
            results.append((pkg_name, str(e), False))
            print(f"  ✗ {pkg_name:<30} MISSING: {e}")
    return all(r[2] for r in results)


def check_device():
    import torch
    print(f"\n[Device Check]")
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device     : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version    : {torch.version.cuda}")
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"  MPS available   : {mps_ok}")
    device = "cuda" if torch.cuda.is_available() else ("mps" if mps_ok else "cpu")
    print(f"  Selected device : {device}")
    return device


def check_data_files(data_dir: str):
    from pathlib import Path
    import os
    base = Path(data_dir)
    ok = True

    required_csvs = [
        "EfficientNetB0_Image_Features.csv",
        "FTIR_Data.csv",
        "Hunter_Lab_colorimeter.csv",
    ]
    print(f"\n[Data File Check] base={base}")
    for fname in required_csvs:
        fpath = base / fname
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"  ✓ {fname} ({size:,} bytes)")
        else:
            print(f"  ✗ {fname} NOT FOUND at {fpath}")
            ok = False

    img_dir = base / "Image_Data"
    if img_dir.exists():
        total = 0
        for root, dirs, files in os.walk(img_dir):
            imgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
            total += len(imgs)
        print(f"  ✓ Image_Data/ exists — {total} images found")
    else:
        print(f"  ✗ Image_Data/ NOT FOUND at {img_dir}")
        ok = False

    return ok


def main():
    import os
    from pathlib import Path

    print("=" * 60)
    print("AdulterNet Environment Verification")
    print("=" * 60)

    print("\n[Package Check]")
    imports_ok = check_imports()

    device = check_device()

    # Determine data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data" / "raw"
    data_ok = check_data_files(str(data_dir))

    print("\n" + "=" * 60)
    if imports_ok and data_ok:
        print("✓ Environment OK")
        status = "OK"
    else:
        issues = []
        if not imports_ok:
            issues.append("missing packages")
        if not data_ok:
            issues.append("missing data files")
        print(f"✗ FAILED: {', '.join(issues)}")
        status = "FAILED"

    # Write report
    log_dir = script_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    report_path = log_dir / "setup_report.txt"

    import io
    import subprocess
    with open(report_path, "w") as f:
        f.write(f"AdulterNet Setup Report\n")
        f.write(f"Status: {status}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Data OK: {data_ok}\n")
        f.write(f"Imports OK: {imports_ok}\n\n")
        f.write("Project Tree:\n")
        for root, dirs, files in os.walk(script_dir):
            # skip .git and __pycache__
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
            level = root.replace(str(script_dir), "").count(os.sep)
            indent = " " * 2 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = " " * 2 * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")

    print(f"\nReport saved to: {report_path}")

    if status == "FAILED" and not data_ok:
        # Don't exit(1) for missing data — it's expected in dev
        print("Note: Data files missing is expected if data not yet copied.")
        sys.exit(0)
    elif not imports_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
