# Build from the repository root. The image contains the core and the Python extras
# selected with ALAMBIQ_EXTRAS; models and datasets are external resources.
ARG PYTHON_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE} AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.10 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/alambiq UV_PYTHON_DOWNLOADS=never
WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ src/
ARG ALAMBIQ_EXTRAS=""
RUN set -eu; set --; \
    for extra in $ALAMBIQ_EXTRAS; do set -- "$@" --extra "$extra"; done; \
    uv sync --locked --no-dev --no-editable --python /usr/local/bin/python "$@"

FROM ${PYTHON_IMAGE} AS runtime
ARG ALAMBIQ_EXTRAS=""
# RDKit's SVG depiction module links these libraries even in headless runs.
RUN set -eu; case " $ALAMBIQ_EXTRAS " in \
    *" chemistry "*|*" routes "*|*" thermodynamics "*) \
        apt-get update && \
        apt-get install -y --no-install-recommends libxrender1 libxext6 libexpat1 && \
        rm -rf /var/lib/apt/lists/* ;; \
    esac
COPY --from=build /opt/alambiq /opt/alambiq
ENV PATH=/opt/alambiq/bin:$PATH \
    HOME=/tmp \
    PREFECT_HOME=/tmp/prefect \
    ALAMBIQ_PREFECT_HOME=/tmp/prefect \
    PREFECT_SERVER_ANALYTICS_ENABLED=false \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
USER 1000:1000
ENTRYPOINT ["alambiq"]
CMD ["--help"]
