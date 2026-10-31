FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# プロキシ引数の受け取り
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    UV_CACHE_DIR=/root/.cache/uv

WORKDIR /workspace

# 依存関係定義ファイルを先にコピーして依存関係をキャッシュ
COPY pyproject.toml uv.lock ./

# 依存関係のインストール（仮想環境を /workspace/.venv に作成）
RUN uv sync --frozen

# パスに仮想環境の bin を追加
ENV PATH="/workspace/.venv/bin:$PATH"

# デフォルトコマンド
CMD ["python", "demo.py"]
