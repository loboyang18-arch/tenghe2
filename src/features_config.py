# -*- coding: utf-8 -*-
"""
P5 特征选择配置：从 YAML/JSON 或 dict 读取列列表与 best_methods，避免硬编码。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

# 内置默认（与原 dataio / run_exog_suite 一致）
DEFAULT_PREFERRED_EXOG_ORDER = [
    "系统负荷实际值",
    "风光总加实际值",
    "竞价空间实际值",
    "联络线实际值",
    "地方电厂发电实际值",
    "风电实际值",
    "光伏实际值",
    "自备机组实际值",
    "试验机组实际值",
    "非市场化核电实际值",
    "上旋备用实际值",
    "下旋备用实际值",
]

DEFAULT_EXOG_KEY5 = [
    "系统负荷实际值",
    "光伏实际值",
    "联络线实际值",
    "上旋备用实际值",
    "下旋备用实际值",
]

DEFAULT_EXOG_BEST_METHODS: Dict[str, str] = {
    "系统负荷实际值": "hgbt_residual",
    "光伏实际值": "hgbt_residual",
    "联络线实际值": "use_pred",
    "上旋备用实际值": "hgbt_pure",
    "下旋备用实际值": "hgbt_pure",
}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """从 YAML 或 JSON 文件加载配置；路径不存在或解析失败则返回空 dict。"""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        if p.suffix in (".yaml", ".yml"):
            import yaml

            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        if p.suffix == ".json":
            import json

            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}


def get_preferred_exog_order(config: Optional[Dict[str, Any]] = None) -> List[str]:
    """RT baseline 外生列优先顺序。"""
    if config and "preferred_exog_order" in config:
        order = config["preferred_exog_order"]
        if isinstance(order, list) and order:
            return [str(c) for c in order]
    return list(DEFAULT_PREFERRED_EXOG_ORDER)


def get_exog_key5(config: Optional[Dict[str, Any]] = None) -> List[str]:
    """RT exog5 / 外生 key5 目标列。"""
    if config:
        for key in ("exog_columns", "exog_key5", "target_cols"):
            if key in config:
                val = config[key]
                if isinstance(val, list) and val:
                    return [str(c) for c in val]
    return list(DEFAULT_EXOG_KEY5)


def get_exog_best_methods(config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """外生最佳方案：target_col -> method。"""
    if config and "best_methods" in config:
        b = config["best_methods"]
        if isinstance(b, dict) and b:
            return {str(k): str(v) for k, v in b.items()}
    return dict(DEFAULT_EXOG_BEST_METHODS)
