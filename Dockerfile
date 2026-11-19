# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Pin past the eMudhra/emSign incident. A floating :latest tag can stay cached
# on a pre-0.13.4 digest, which still trusts emSign ECC Root CA - G3.
# https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

WORKDIR /app

# hatch-vcs derives the version from git history, so the build context
# includes .git (keep it out of .dockerignore for this image) and the image
# needs the git binary for setuptools-scm to read it (slim ships without it).
# A context with no .git (a `git archive` export) passes VERSION instead.
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

# Extras baked into the image. The default (`latest`, `<version>`) stays light
# and scores phash only. The release also publishes `-score` (`latest-score`,
# `<version>-score`), built with EXTRAS="server,cli,score" ORT_GPU=1 (#99).
ARG EXTRAS="server,cli"
# 1 swaps onnxruntime for onnxruntime-gpu, so InsightFace (identity) runs on CUDA.
# No CUDA base image is needed: PyPI's Linux torch wheels bring the CUDA runtime
# as nvidia-* packages, which onnxruntime-gpu loads too (the identity scorer calls
# onnxruntime.preload_dlls()). Without a GPU, both fall back to CPU.
ARG ORT_GPU=0
# pyiqa (in [score]) depends on opencv-python, the GUI build, whose cv2 needs
# libxcb/libGL that slim lacks. The install below swaps it for the same version of
# opencv-python-headless (same cv2 module, no X11), so `import pyiqa` works here.
# Another [score] dependency already installs the headless build, and both packages
# own cv2/, so uninstalling opencv-python deletes it: the reinstall puts it back.
# The release tag without its "v". Empty: read the version from .git.
ARG VERSION=""

# pillow + imagehash ride along regardless: they power phash near-dup collapse
# and diversity (honest group pass-rates) at negligible image cost.
# The version is exported for this RUN only, so it neither joins later layers'
# cache keys nor leaks into the runtime environment.
RUN python -m pip install --no-cache-dir \
        "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" \
    && if [ -n "$VERSION" ]; then export SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION"; fi \
    && uv pip install --system --no-cache "certifi>=2026.10.19" ".[${EXTRAS}]" pillow imagehash \
    && if [ "$ORT_GPU" = 1 ]; then \
        uv pip uninstall --system onnxruntime \
        && uv pip install --system --no-cache "onnxruntime-gpu>=1.19"; \
    fi \
    && cv=$(uv pip show --system opencv-python 2>/dev/null | sed -n 's/^Version: //p') \
    && if [ -n "$cv" ]; then \
        uv pip uninstall --system opencv-python \
        && uv pip install --system --no-cache --reinstall-package opencv-python-headless \
            "opencv-python-headless==$cv"; \
    fi

EXPOSE 8104
CMD ["argus-proof", "serve", "--port", "8104", "--cors"]

