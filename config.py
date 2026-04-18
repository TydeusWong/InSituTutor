import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"


# Teaching analysis constants
TA_MODEL = "qwen3.5-omni-plus"
TA_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TA_SEGMENT_MINUTES = 5
TA_MAX_RETRIES = 3
TA_REQUEST_TIMEOUT_SEC = 180

TA_MANIFEST_PATH = ROOT / "data" / "processed" / "ingest_manifest.json"
TA_OUTPUT_DIR = ROOT / "data" / "processed" / "analysis"
TA_PREPROCESSED_VIDEO_DIR = ROOT / "data" / "processed" / "preprocessed_videos"
TA_PROMPT_DIR = ROOT / "models" / "prompts"
TA_SCHEMA_PATH = ROOT / "models" / "schemas" / "teaching_knowledge.schema.json"
TA_VIDEO_MAX_MB = 50
TA_API_KEY_ENV_NAMES = [
    "DASHSCOPE_API_KEY",
    "ALIYUN_API_KEY",
    "ALIBABA_CLOUD_API_KEY",
]

# Video ingest constants
VI_PREPROCESSED_VIDEO_DIR = ROOT / "data" / "processed" / "preprocessed_videos"
VI_VIDEO_MAX_MB = 50
VI_INPUT_MANIFEST_PATH = ROOT / "data" / "raw" / "demo_manifest.json"
VI_OUTPUT_MANIFEST_PATH = ROOT / "data" / "processed" / "ingest_manifest.json"

# Pipeline A v2 constants
V2_SEGMENTATION_OUTPUT_PATH = ROOT / "data" / "processed" / "v2" / "segmentation" / "sections_units.json"
V2_STRATEGY_OUTPUT_DIR = ROOT / "data" / "processed" / "v2" / "strategy"
V2_STRATEGY_OUTPUT_PATH = V2_STRATEGY_OUTPUT_DIR / "teaching_strategy_v2.json"
V2_SEGMENTATION_PROMPT_DIR = ROOT / "models" / "prompts"


def load_env_file() -> None:
    if not ENV_PATH.exists():
        return
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cleaned = value.strip()
        # 支持 .env 行尾注释：KEY=value # comment
        # 仅在未使用引号包裹时剔除注释，避免误伤真实值中的 '#'
        if not ((cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'"))):
            hash_index = cleaned.find("#")
            if hash_index >= 0:
                cleaned = cleaned[:hash_index].strip()
        cleaned = cleaned.strip("'").strip('"')
        os.environ.setdefault(key.strip(), cleaned)


def get_api_key() -> str:
    load_env_file()
    for name in TA_API_KEY_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    return ""
