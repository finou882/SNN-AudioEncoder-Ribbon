"""Main Entry Point for SNN-AudioEncoder-Ribbon."""

import sys
from pathlib import Path

# src ディレクトリをパスに追加
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from launch import main as run_launcher


def main():
    run_launcher()


if __name__ == "__main__":
    main()
