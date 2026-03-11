#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RT PatchTST baseline 根目录兼容入口：转发到 scripts.run_rt_baseline。
推荐直接使用: python scripts/run_rt_baseline.py --file <excel>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rt_baseline import main

if __name__ == "__main__":
    main()
