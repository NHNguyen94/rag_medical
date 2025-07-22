from pydantic import BaseModel

class IngestCustomFileRequest(BaseModel):
    file_dir_path: str
    index_dir_path: str