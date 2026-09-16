# =============================================================================
# VulEvalBench -- container image for the web interface.
#
# Single-stage build: the project is pure Python with no compiled extensions
# and no frontend build step, so a multi-stage build would add complexity
# without shrinking the result.
#
# Build and run:
#   docker compose up --build
# =============================================================================

# `slim` rather than `alpine`: alpine uses musl instead of glibc, which forces
# Python wheels to be compiled from source and turns a fast build into a slow
# one. The version is pinned to the minor release so a future Python 3.13 does
# not silently change the runtime underneath the project.
FROM python:3.12-slim

# PYTHONUNBUFFERED    -- send stdout/stderr straight to the Docker log instead
#                        of buffering, so `docker compose logs -f` shows
#                        pipeline progress live rather than in one burst at the
#                        end.
# PYTHONDONTWRITEBYTECODE -- no .pyc files; they would be written into the
#                        image and into mounted volumes for no benefit.
# PYTHONPATH          -- makes `web.app:app` importable by waitress below.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies are installed BEFORE the source is copied. Docker caches each
# layer, so editing a .py file reuses the cached install step; without this
# split every code change would reinstall every package.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code. Only these two directories are needed at run time; runs,
# logs and ingested code are mounted as volumes.
#
# Credential files are deliberately NOT copied: they are bind-mounted read-only
# at run time (see docker-compose.yml), so a secret never enters an image layer
# where it would persist in the image history even if a later layer deleted it.
COPY src/ ./src/
COPY web/ ./web/

# Run as an unprivileged user. The app handles model-authored and third-party
# source code -- it never executes it, but it does parse it with several
# analysers, so the process should not be root inside the container.
#
# The mount points are created and chowned here so the app can write to them.
# On Docker Desktop (Windows/macOS) bind-mount permissions are handled by the
# VM and this is enough. On native Linux the host directory's ownership wins,
# so if writes fail there, run: sudo chown -R 1000:1000 runs logs ingested
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/runs /app/logs /app/models /app/ingested \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Reports the container as unhealthy if the app stops responding. Uses Python
# rather than curl, which is not present in the slim image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=4)" \
        || exit 1

# waitress rather than Flask's built-in server, which prints a warning and is
# explicitly not intended for anything but local development.
#
# IMPORTANT -- one process, several threads. Job state in web/app.py is an
# in-memory dict, so a multi-process server (gunicorn with workers > 1) would
# give each worker its own copy and the status endpoint would return whichever
# process happened to serve the request. waitress is single-process by design,
# which matches that constraint.
CMD ["waitress-serve", "--listen=0.0.0.0:8000", "--threads=8", "web.app:app"]
