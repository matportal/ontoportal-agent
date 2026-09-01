FROM node:22-bookworm-slim AS opencode-builder

RUN npm install --global --no-audit --no-fund opencode-ai@1.15.12 \
    && test "$(opencode --version)" = "1.15.12" \
    && mkdir -p /out \
    && cp --dereference "$(command -v opencode)" /out/opencode

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=opencode-builder /out/opencode /usr/local/bin/opencode
COPY pyproject.toml README.md /app/
COPY src /app/src

RUN chmod 0755 /usr/local/bin/opencode \
    && git --version \
    && test "$(opencode --version)" = "1.15.12" \
    && pip install . \
    && python -c "import ontoportal_agent.server"

EXPOSE 8090

CMD ["python", "-m", "ontoportal_agent.server"]
