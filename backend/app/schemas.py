from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Raw ticket text")


class TicketResponse(BaseModel):
    label: str
    confidence: float
    low_confidence: bool  # flags tickets for human review below the threshold
