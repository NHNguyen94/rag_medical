run-frontend:
	streamlit run src/ui/app.py

run-backend:
	PYTHONPATH=. uvicorn src.main:app --reload --port 8000 --log-level debug

train-lstm:
	PYTHONPATH=. python src/ml_pipelines/emotion_recognition/train.py

eval-lstm:
	PYTHONPATH=. python src/ml_pipelines/emotion_recognition/eval.py

format:
	PYTHONPATH=. ruff format

test:
	PYTHONPATH=. pytest -s -p no:warnings tests/

run-db:
	sudo mkdir -p postgres_data && sudo docker compose up

clear-db:
	docker compose down -v && rm -rf postgres_data