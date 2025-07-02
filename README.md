# Train vector store

- Download all datasets from here: https://www.kaggle.com/datasets/gvaldenebro/cancer-q-and-a-dataset/data
- Create src/data folder
- Place it with this structure: 

![img.png](img.png)

- Run `make ingest-data` to ingest data into the vector store
- You will need to wait for the process, it will take long time (~ 1-2 hours)

# Emotion detection

- Download the dataset here: https://www.kaggle.com/datasets/parulpandey/emotion-dataset/data
- Run `make train-emotion` to train the model
- Run `make run-emotion` to run the model for examples

# Topic clustering

- Run `ingest-medical-data` to ingest medical Q&A data for topic clustering
- Run `make train-topic` to train the model
- Run `make run-topic` to run the model for examples