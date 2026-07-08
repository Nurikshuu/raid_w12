.PHONY: install index serve test

install:
	python -m venv venv
	./venv/Scripts/pip install -r requirements.txt || venv/bin/pip install -r requirements.txt

index:
	./venv/Scripts/python scripts/build_index.py || venv/bin/python scripts/build_index.py

serve:
	./venv/Scripts/python -m uvicorn app.main:app --port 8000 || venv/bin/python -m uvicorn app.main:app --port 8000

test:
	./venv/Scripts/python -m pytest tests/ -v || venv/bin/python -m pytest tests/ -v
