run-frontend:
	streamlit run src/ui/app.py

playground:
	PYTHONPATH=. python tests/playground.py