# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Pin past the eMudhra/emSign incident. A floating :latest tag can stay cached
# on a pre-0.13.4 digest, which still trusts emSign ECC Root CA - G3.
# https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

WORKDIR /app

# hatch-vcs derives the version from git history, so the build context must
# include .git (keep it out of .dockerignore for this image) and the image
# needs the git binary for setuptools-scm to read it (slim ships without it).
# ComfyUI and the other urllib callers trust this bundle, not certifi.
# Distrust emSign ECC Root CA - G3 and the Comodo AAA cross-sign root.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && sed -i \
        -e 's|^mozilla/emSign_ECC_Root_CA_-_G3|!mozilla/emSign_ECC_Root_CA_-_G3|' \
        -e 's|^mozilla/Comodo_AAA|!mozilla/Comodo_AAA|' \
        /etc/ca-certificates.conf \
    && update-ca-certificates --fresh >/dev/null \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Extras baked into the image. Default keeps it light; build with
# EXTRAS="server,cli,score" for the full scorer stack (torch, insightface, …).
ARG EXTRAS="server,cli"

# pillow + imagehash ride along regardless: they power phash near-dup collapse
# and diversity (honest group pass-rates) at negligible image cost.
RUN python -m pip install --no-cache-dir \
        "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" \
    && uv pip install --system --no-cache "certifi>=2026.10.19" ".[${EXTRAS}]" pillow imagehash

EXPOSE 8104
CMD ["argus-proof", "serve", "--port", "8104", "--cors"]

