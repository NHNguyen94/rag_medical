run-frontend:
	streamlit run src/ui/app.py

run-backend:
	PYTHONPATH=. uvicorn src.main:app --reload --port 8000 --log-level debug

train-emotion:
	PYTHONPATH=. python src/pipelines/emotion_recognition/train.py

run-emotion:
	PYTHONPATH=. python src/pipelines/emotion_recognition/inference.py

train-topic:
	PYTHONPATH=. python src/pipelines/topic_clustering/train.py

run-topic:
	PYTHONPATH=. python src/pipelines/topic_clustering/inference.py

format:
	PYTHONPATH=. ruff format

unittest:
	PYTHONPATH=. pytest -s -p no:warnings tests/

run-db:
	sudo mkdir -p postgres_data && sudo docker compose up

clear-db:
	docker compose down -v && rm -rf postgres_data

ingest-data:
	PYTHONPATH=. python src/pipelines/ingest_data/ingest_to_vt_store.py

ingest-medical-data:
	PYTHONPATH=. python src/pipelines/ingest_data/ingest_medical_data.py

compare-rag:
	PYTHONPATH=. python src/pipelines/rag/compare_rag.py