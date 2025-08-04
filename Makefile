.PHONY: install lint test format clean

# Install dependencies
install:
	pip install -r requirements.txt

# Static analysis
lint:
	flake8 src tests

# Run tests
test:
	pytest --maxfail=1 --disable-warnings -q

# Auto-format code
format:
	black .

# Clean build artifacts
clean:
	rm -rf build dist *.egg-info