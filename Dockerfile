FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NODE_VERSION=22.22.3

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git curl xz-utils default-jre-headless \
    && node_arch="$(dpkg --print-architecture)" \
    && case "${node_arch}" in \
      amd64) node_arch="x64" ;; \
      arm64) node_arch="arm64" ;; \
      *) echo "Unsupported Node.js architecture: ${node_arch}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" -o /tmp/node.tar.xz \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 \
    && rm -f /tmp/node.tar.xz \
    && node --version \
    && npm --version \
    && npm install -g @openai/codex opencode-ai @earendil-works/pi-coding-agent antigravity-claude-proxy \
    && mkdir -p /opt/robot \
    && curl -fsSL -o /opt/robot/robot.jar https://github.com/ontodev/robot/releases/latest/download/robot.jar \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin agent \
    && mkdir -p /workspace /mcp /tmp \
    && chown -R agent:agent /workspace /mcp /tmp /home/agent

EXPOSE 8090

USER 10001:10001

CMD ["python", "-m", "ontoportal_agent.server"]
