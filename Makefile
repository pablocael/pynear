.PHONY: init-repo
init-repo:
ifeq (, $(shell which clang-format))
	@echo "Installing formatting tools ..."
	@pip install isort black flake8
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
	clang-format -i --verbose pynear/src/*.cpp pynear/include/*.hpp
	isort -j 0 .
	flake8
	black .

.PHONY: test
test:
	export PYTHONPATH=$PWD
	pytest pynear/tests

.PHONY: cpp-test
cpp-test:
	mkdir -p build && cd build && cmake -G "Unix Makefiles" ../pynear && make && make test
