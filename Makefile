.PHONY: lint fmt test prepare-swebench check-toy run-c run-pm clean

# Check for banned toy/mock datasets in source code
check-toy:
	@echo "Checking for banned toy/smoke patterns in source code..."
	@! grep -r -E "(toy_tasks|--toy|smoke_test|--smoke)" . --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=docs --exclude-dir=tests --exclude-dir=.claude --exclude-dir=.pytest_cache --exclude-dir=.github --exclude="Makefile" --exclude="*.parquet" --exclude="*.pyc" --exclude="*.md" --exclude="*.log" 2>/dev/null | grep -v "^[[:space:]]*#" || (echo "ERROR: Found banned patterns in source" && false)
	@echo "✅ No banned patterns found in source code"

lint: check-toy
	ruff check eval tests
	black --check eval tests

fmt:
	ruff check --fix eval tests
	black eval tests

test:
	@if [ -n "$$ARTIFACTS_DIR" ]; then \
		mkdir -p "$$ARTIFACTS_DIR"; \
		python3 -c "import json, platform, subprocess, sys; ver=lambda c: subprocess.check_output(c,shell=True,text=True).strip() if subprocess.run(c,shell=True,capture_output=True).returncode==0 else ''; info={'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'pytest':ver('pytest --version'),'ruff':ver('ruff --version'),'black':ver('black --version')}; import os; open(os.path.join('$$ARTIFACTS_DIR','env.json'),'w').write(json.dumps(info,indent=2))"; \
		pytest -q --junitxml "$$ARTIFACTS_DIR/junit.xml" | tee "$$ARTIFACTS_DIR/pytest_stdout.txt"; \
	else \
		pytest -q; \
	fi

# Prepare SWE-bench Lite data (run once)
prepare-swebench:
	bash scripts/prepare_swebench.sh

# Run Arms for evaluation
run-c:
	HERMES_HERMETIC=1 python -m eval.run_arms --arm C --dataset swebench_lite --instances_file configs/swebench_lite_slice20.txt --seed 123

run-pm:
	HERMES_HERMETIC=1 python -m eval.run_arms --arm PM --dataset swebench_lite --instances_file configs/swebench_lite_slice20.txt --seed 123

# Clean temporary files
clean:
	rm -rf scratch/
	rm -rf runs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete