from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).parent
    data_raw = project_root / "data" / "raw"

    errors = []
    print("=" * 60)
    print("AdulterNet Environment Verification")
    print("=" * 60)

    # Check packages
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
        ("pytorch_grad_cam", "pytorch-grad-cam"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("optuna", "optuna"),
    ]

    print("\n[Packages]")
    for import_name, pkg_name in packages:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  ✓ {pkg_name:<30} {version}")
        except ImportError as e:
            print(f"  ✗ {pkg_name:<30} MISSING ({e})")
            errors.append(f"Package missing: {pkg_name}")

    # Check CUDA/MPS
    print("\n[Compute]")
    try:
        import torch
        print(f"  PyTorch version : {torch.__version__}")
        print(f"  CUDA available  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device     : {torch.cuda.get_device_name(0)}")
            print(f"  CUDA memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  MPS available   : {torch.backends.mps.is_available()}")
    except Exception as e:
        errors.append(f"Torch check failed: {e}")

    # Check CSV files
    print("\n[Data Files]")
    csv_files = [
        "EfficientNetB0_Image_Features.csv",
        "FTIR_Data.csv",
        "Hunter_Lab_colorimeter.csv",
    ]
    for f in csv_files:
        path = data_raw / f
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {f:<45} ({size:,} bytes)")
        else:
            print(f"  ✗ {f:<45} NOT FOUND at {path}")
            errors.append(f"Missing CSV: {f}")

    # Check Image_Data
    print("\n[Image Data]")
    image_dir = data_raw / "Image_Data"
    if image_dir.exists():
        total_images = 0
        for subfolder in image_dir.iterdir():
            if subfolder.is_dir():
                imgs = [f for f in subfolder.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}]
                total_images += len(imgs)
                print(f"  ✓ {subfolder.name:<10} {len(imgs)} images")
        print(f"  Total images: {total_images}")
    else:
        print(f"  ✗ Image_Data/ not found at {image_dir}")
        errors.append("Missing Image_Data directory")

    # Print project tree
    print("\n[Project Tree]")
    for root, dirs, files in os.walk(project_root):
        dirs[:] = sorted([d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache"}])
        level = root.replace(str(project_root), "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{Path(root).name}/")
        subindent = "  " * (level + 1)
        for f in sorted(files):
            print(f"{subindent}{f}")

    # Save report
    report_path = project_root / "logs" / "setup_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as fp:
        fp.write("AdulterNet Setup Report\n")
        fp.write("=" * 60 + "\n")
        fp.write(f"Errors: {errors}\n")

    print("\n" + "=" * 60)
    if errors:
        print(f"✗ FAILED: {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("✓ Environment OK")


if __name__ == "__main__":
    main()
