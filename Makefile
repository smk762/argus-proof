UV ?= uv

.PHONY: help install lint format test build clean web-install pages shared-check

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv and install with test extras
	$(UV) venv
	$(UV) pip install -e ".[dev,server,cli]"

lint:  ## Ruff check + format check
	$(UV)x ruff@0.15.16 check src/ tests/
	$(UV)x ruff@0.15.16 format --check src/ tests/

format:  ## Ruff autoformat + autofix
	$(UV)x ruff@0.15.16 format src/ tests/
	$(UV)x ruff@0.15.16 check --fix src/ tests/

test:  ## Run the test suite
	$(UV) run --no-sync pytest --tb=short -q

build:  ## Build sdist + wheel
	$(UV) build

clean:  ## Remove build + cache artifacts
	rm -rf dist build .venv *.egg-info .pytest_cache .ruff_cache src/argus_proof/_version.py

# ── Hosted demo (proof.dragonhound.dev): a static replay of a recorded evaluation, see README "Demo" ──
web-install:  ## Install the demo site's dependencies (web/)
	cd web && npm ci

pages: web-install  ## Static export of the recorded tape to web/out/, with a hash CSP on every page
	cd web && npm run build

shared-check:  ## The vendored Dragonhound shared layer is unedited and matches the published v1 manifest
	node web/vendor/dh/sync-shared.mjs --check . --against https://dragonhound.dev/brand/v1/manifest.json --allow-outdated
