UV ?= uv
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_CACHE_DIR
PYTHON_PROJECT := _project/python
PYTHON_RUN := $(UV) run --project $(PYTHON_PROJECT) --extra test

.PHONY: test test-python test-cpp test-sql

test: test-python test-cpp test-sql

test-python:
	$(PYTHON_RUN) pytest _project/tests/python _project/tests/sql -q --import-mode=importlib

test-cpp:
	@for validator in $$(find _project/tests/cpp -type f -name 'validate_cpp.py' | sort); do \
		$(PYTHON_RUN) python "$$validator" || exit 1; \
	done

test-sql:
	@for validator in $$(find _project/tests/sql -type f \( -name 'validate_sql.py' -o -name 'sql_validation.py' -o -name 'sql_extra_validation.py' \) | sort); do \
		$(PYTHON_RUN) python "$$validator" || exit 1; \
	done
