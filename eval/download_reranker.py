"""下载 bge-reranker-base 到本地 model/ 目录。

修复记录（详见 docs/troubleshooting-hybrid-rerank.md）：
    1. 删除 `resume_download` 与 `local_dir_use_symlinks` —— 这两个参数在
       huggingface_hub >= 0.23 已废弃，在 1.x 中不再生效（会持续告警）。
       `snapshot_download` 写本地目录本就不再使用软链接，无需该参数。
    2. 增加"本地已完整则跳过"的短路：模型若已存在且关键权重文件非空，
       直接返回，不再联网 —— 避免受限网络下的 WinError 10060 等待。
    3. 仅在确实缺失时才联网，并默认走 hf-mirror 镜像。

用法：
    python eval/download_reranker.py
"""
import os

# 国内镜像：必须在导入 huggingface_hub 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "model", "bge-reranker-base"))
REPO_ID = "BAAI/bge-reranker-base"
# 判定"已下载完整"的关键文件（权重 + 配置 + 分词器）
REQUIRED_FILES = ["pytorch_model.bin", "config.json", "tokenizer.json"]


def is_complete(model_dir: str) -> bool:
    for name in REQUIRED_FILES:
        fp = os.path.join(model_dir, name)
        if not os.path.isfile(fp) or os.path.getsize(fp) == 0:
            return False
    return True


def main() -> None:
    if is_complete(MODEL_DIR):
        size_mb = os.path.getsize(os.path.join(MODEL_DIR, "pytorch_model.bin")) / 1024 / 1024
        print(f"✅ 本地模型已完整，跳过下载：{MODEL_DIR}")
        print(f"   pytorch_model.bin = {size_mb:.2f} MB")
        return

    print(f"⬇️  开始下载 {REPO_ID} → {MODEL_DIR}（镜像：{os.environ['HF_ENDPOINT']}）")
    snapshot_download(repo_id=REPO_ID, local_dir=MODEL_DIR)
    print("✅ 下载完成")


if __name__ == "__main__":
    main()
