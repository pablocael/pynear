.PHONY: init-repo
init-repo:
ifeq (, $(shell which clang-format))
	@echo "Installing stylize ..."
	@pip install stylize
ifeq ($(UNAME_S),Linux)
	sudo apt install clang-format
endif

ifeq ($(UNAME_S),Darwin)
	brew install clang-format
endif

else
	@echo "clang-format installed, skipping installation"
endif

.PHONY: fmt
fmt: init-repo
	stylize --exclude_dirs=pyvptree/include/Eigen --yapf_style="{based_on_style: google, column_limit: 150, indent_width: 4}"

.PHONY: check-fmt
check-fmt:
	stylize --check --exclude_dirs=pyvptree/include/Eigen --yapf_style="{based_on_style: google, column_limit: 150, indent_width: 4}"

.PHONY: test
test:
	export PYTHONPATH=$PWD
	pytest pyvptree/tests

.PHONY: cpp-test
cpp-test:
	mkdir -p build && cd build && cmake -G "Unix Makefiles" ../pyvptree && make && make test

.PHONY: benchmarks
benchmarks:
	export PYTHONPATH=$PWD
	pip install -r pyvptree/benchmark/requirements.txt
	python pyvptree/benchmark/run_benchmarks.py
