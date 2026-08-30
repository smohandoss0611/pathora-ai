# Hugging Face Spaces Dockerfile.
#
# Copy this to the repo root as `Dockerfile` when deploying to Spaces, or point
# the Space at this path. Differs from the main Dockerfile in three ways that
# Spaces requires:
#   * listens on 7860 (the port Spaces proxies)
#   * runs Streamlit rather than the API
#   * writes caches under /home/user, the only writable path in a Space
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/home/user/app/src \
    HOME=/home/user \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user data ./data
COPY --chown=user scripts ./scripts
COPY --chown=user streamlit_app.py ./

# Spaces provides secrets as environment variables, which streamlit_app.py
# already reads before Settings is constructed.
EXPOSE 7860
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py"]
