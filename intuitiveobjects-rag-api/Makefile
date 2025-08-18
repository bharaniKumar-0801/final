# Set the virtual environment activation command
VENV_ACTIVATE = source qbenv/bin/activate

# create virtual environment
create-venv:
	python -m venv qbenv

# Install dependencies
install:
	$(VENV_ACTIVATE) && pip install -r requirements.txt

# Run the FastAPI server
run:
	$(VENV_ACTIVATE) && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Check installed dependencies
deps:
	$(VENV_ACTIVATE) && pip list

# Format code using Black
format:
	$(VENV_ACTIVATE) && black .

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +

# Create requirements.txt
freeze:
	$(VENV_ACTIVATE) && pip freeze > requirements.txt