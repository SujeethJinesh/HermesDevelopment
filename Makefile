.PHONY: lint fmt test

lint:
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