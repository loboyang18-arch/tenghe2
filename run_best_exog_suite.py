#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最佳外生方案根目录兼容入口：转发到 scripts.run_exog_suite。
推荐直接使用: python scripts/run_exog_suite.py --excel <path>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_exog_suite import main

if __name__ == "__main__":
    main()
