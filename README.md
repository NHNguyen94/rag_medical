# To run the project, please follow these steps:

Note: This application is supported for MacOS and Linux only

## 1. Install dependencies
- Create conda env with Python 3.11
- Run `pip install poetry`
- Run `poetry install` to install all dependencies

## 2. Create vector store

- Download all datasets from here: https://www.kaggle.com/datasets/gvaldenebro/cancer-q-and-a-dataset/data
- Create src/data folder
- Place it with this structure: 

![img.png](img.png)

- Run `make ingest-data` to ingest data into the vector store
- You will need to wait for the process, it will take long time (~ 1-2 hours)

## 3. Train model for Emotion detection

- Download the dataset here: https://www.kaggle.com/datasets/parulpandey/emotion-dataset/data
- Run `make train-emotion` to train the model
- Run `make eval-emotion` to evaluate the model
- Run `make run-emotion` to run the model for examples


## 4. Train model for Topic clustering

- Run `ingest-medical-data` to ingest medical Q&A data for topic clustering
- Run `make train-topic` to train the model
- Run `make eval-topic` to evaluate the model
- Run `make run-topic` to run the model for examples


## 5. Train models for Question recommendation
1. Create or ensure all directories exist.
- mkdir src/data/fine_tune_dataset
- mkdir src/data/processed
- mkdir src/ml_models/model_files

2. Place medical Q&A datasets in the `src/data/fine_tune_dataset` directory. The following dataset files are supported:
    - (DATASET=0) `CancerQA.csv`
    - (DATASET=1) `Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv`
    - (DATASET=2) `Disease_Control_and_PreventionQA.csv`
    - (DATASET=3) `Genetic_and_Rare_DiseasesQA.csv`
    - (DATASET=4) `growth_hormone_receptorQA.csv`
    - (DATASET=5) `Heart_Lung_and_BloodQA.csv`
    - (DATASET=6) `Neurological_Disorders_and_StrokeQA.csv`
    - (DATASET=7) `SeniorHealthQA.csv`
    - (DATASET=8) `OtherQA.csv`

3. make train-question DATASET=0

Where:
- `DATASET=0` corresponds to CancerQA
- `DATASET=1` corresponds to Diabetes and Digestive and Kidney Diseases
- And so on...

4. The training process will:
1. Load the specified dataset
2. Process the questions and create embeddings
3. Build a FAISS index for similarity search
4. Generate training pairs with follow-up questions
5. Fine-tune the Flan-T5 model
6. Save the trained model to the directory `ml_models/model_files`

5. After training is complete, the following files will be generated:
- Trained model: `ml_models/model_files/flant5_<domain>.pth` (e.g., ) `flant5_cancer.pth`
- Training data: `data/processed/training_data.csv`
- Question mappings: `data/processed/questions_mapping.json`


## 6. Run the application
- Run `docker compose up` to start the docker containers with all services
- Run `make run-backend` in another terminal to start the backend server
- Run `make run-frontend` in another terminal to start the frontend server