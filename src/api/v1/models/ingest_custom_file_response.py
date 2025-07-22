from pydantic import BaseModel


class IngestCustomFileResponse(BaseModel):
    index_dir_path: str
