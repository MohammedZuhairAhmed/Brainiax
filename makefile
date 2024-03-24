run:
	poetry run python -m brainiax

dev-windows:
	(set PROFILES=ollama & poetry run python -m uvicorn brainiax.main:app --reload --port 8001)
