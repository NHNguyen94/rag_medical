run-frontend:
	streamlit run src/ui/app.py

run-backend:
	PYTHONPATH=. uvicorn src.main:app --reload --port 8000 --log-level debug

playground:
	PYTHONPATH=. python tests/playground.py

dl:
	PYTHONPATH=. python src/ml_models/lstm.py