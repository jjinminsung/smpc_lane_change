"""C++ 노드와 같은 yaml 을 읽어 파라미터를 제공한다.

시뮬레이터가 C++ 과 다른 숫자를 쓰면 비교 실험이 무의미해지므로, 값을 복사하지
않고 config/*.yaml 을 직접 읽는다. yaml 에 없는 키는 C++ 코드 기본값을 아래
CODE_DEFAULTS 에 적어 둔다(출처 줄 번호 포함).
"""
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"

# yaml 에 없고 C++ 에서 pnh_.param 기본값으로만 정해지는 키.
CODE_DEFAULTS = {
    "smpc_decision": {
        "max_lateral_prediction_mps": 0.45,  # smpc_decision_node.cpp:383
    },
}


class Params(SimpleNamespace):
    """속성 접근. 없는 키를 읽으면 AttributeError 로 바로 드러난다."""

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_params(node="smpc_decision", overrides=None):
    path = CONFIG_DIR / f"{node}.yaml"
    with open(path) as f:
        values = yaml.safe_load(f) or {}
    merged = dict(CODE_DEFAULTS.get(node, {}))
    merged.update(values)
    if overrides:
        merged.update(overrides)
    return Params(**merged)
