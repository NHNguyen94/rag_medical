from pydantic import BaseModel, model_validator


class SystemPromptRequest(BaseModel):
    system_prompt: str
    reasoning_effort: str
    temperature: float
    similarity_top_k: int
    yml_file: str

    @model_validator(mode="after")
    def validate_reasoning_effort(self):
        valid_efforts = ["low", "medium", "high"]
        if self.reasoning_effort not in valid_efforts:
            raise ValueError(f"Reasoning effort must be one of {valid_efforts}.")
        return self

    @model_validator(mode="after")
    def validate_temperature(self):
        if not (0 <= self.temperature <= 1):
            raise ValueError("Temperature must be between 0 and 1.")
        return self

    @model_validator(mode="after")
    def validate_similarity_top_k(self):
        if self.similarity_top_k < 1:
            raise ValueError("Similarity top_k must be at least 1.")
        return self
