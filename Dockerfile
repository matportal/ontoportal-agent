FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git nodejs npm curl default-jre-headless \
    && npm install -g @openai/codex opencode-ai \
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
