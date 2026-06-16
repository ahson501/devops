import os

MISTRAL_URL = os.getenv(
    "MISTRAL_URL",
    "http://mistral-7b-svc.ai-models.svc.cluster.local:8000"
)

QWEN_URL = os.getenv(
    "QWEN_URL",
    "http://qwen-7b-svc.ai-models.svc.cluster.local:8000"
)

DEEPSEEK_URL = os.getenv(
    "DEEPSEEK_URL",
    "http://deepseek-coder-svc.ai-models.svc.cluster.local:8000"
)
