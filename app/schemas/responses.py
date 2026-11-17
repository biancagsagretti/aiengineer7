from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app: str


class ReadyResponse(BaseModel):
    status: str
    provider: str
    dependencies: dict[str, str]
