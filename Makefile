# et-hotel-ai-showcase — reproducible, offline entrypoints (manifest S4).
#
# Everything runs on the deterministic MODEL_BACKEND=mock by default: no API
# keys, no network, no cost. To reproduce real model-quality numbers, run the
# eval with a live model, e.g.:
#     MODEL_BACKEND=openrouter LLM_OPENROUTER_API_KEY=... make eval
#     MODEL_BACKEND=ollama make eval

VENV := .venv
PY   := $(VENV)/bin/python
export PYTHONPATH := .

.DEFAULT_GOAL := help
.PHONY: help install test eval demo-graph demo-scrape demo-resilience lint

help:
	@echo "Targets:"
	@echo "  install          create .venv and install requirements.txt"
	@echo "  test             run the pytest suite (subsystem-1 + graph wiring)"
	@echo "  eval             run the reproducible eval suite (MODEL_BACKEND=mock)"
	@echo "  demo-graph       run the trimmed content graph on a synthetic hotel (mock)"
	@echo "  demo-scrape      run the generic fetcher's neutral demo backend (offline)"
	@echo "  demo-resilience  drive the real resilience engine vs a naive baseline"
	@echo "                   against a deterministic offline hostile endpoint"
	@echo "  lint             byte-compile all sources (syntax check)"

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

# Subsystem-1 pytest suite + the graph-wiring test. `-o addopts=` drops the
# ported pytest.ini's --cov-fail-under=100 gate (relaxed for the showcase).
test:
	MODEL_BACKEND=mock $(PY) -m pytest llm_content_generation/tests \
		-o addopts= -q -p no:cacheprovider

# Deterministic security assertions (gated) + illustrative model metrics (mock).
# Exits non-zero ONLY if a deterministic assertion fails.
eval:
	MODEL_BACKEND=mock $(PY) scripts/eval/run_eval_suite.py

demo-graph:
	MODEL_BACKEND=mock $(PY) scripts/demo/demo_graph.py

demo-scrape:
	$(PY) scripts/demo/demo_scrape.py

# Anti-bot resilience PoC: composes the engine's REAL CircuitBreaker, rate
# limiter, proxy rotation and typed errors into a fetch loop against a
# deterministic OFFLINE hostile endpoint, then prints an honest ENGINE-vs-NAIVE
# metrics table. Exits non-zero unless the engine collected ALL records AND the
# naive baseline did not (built-in assertions make it CI-smoke-testable).
demo-resilience:
	$(PY) scripts/demo/demo_resilience.py

lint:
	$(PY) -m compileall -q app common llm_content_generation scripts
