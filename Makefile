# Variables
PYTHON=python3
PIP=pip3
REQ_FILE=requirements.txt

# Default target
all: install run

# Install dependencies
install:
	$(PIP) install -r $(REQ_FILE)

# Run the tutorial script
run:
	$(PYTHON) tutorial.py

# Run tests
test:
	$(PYTHON) -m unittest discover -s tests

# Clean up __pycache__ and other build files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

.PHONY: all install run test clean