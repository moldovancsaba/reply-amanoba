from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=3)
    model: str | None = None
    source: str = Field(default="ask", min_length=1, max_length=32)


class AskResponse(BaseModel):
    status: str
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    confidence: float | None = None
    ticket_id: int | None = None
    reason: str | None = None
    active_model: str | None = None
    switched_from_model: str | None = None


class EditorResponseRequest(BaseModel):
    ticket_id: int
    editor_answer: str = Field(..., min_length=3)
    source_label: str = "editor-validated"

class TicketDismissRequest(BaseModel):
    ticket_id: int
    reason: str = "Removed by editor"


class ModelSelectRequest(BaseModel):
    model: str = Field(..., min_length=1)


class CompareRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=3)
    models: list[str] = Field(..., min_length=2, max_length=6)


class EnrichmentGenerateRequest(BaseModel):
    source_docs: list[str] = Field(default_factory=list, max_length=20)
    questions_per_doc: int = Field(default=5, ge=1, le=10)
    include_ticket_based: bool = True
    include_general_knowledge: bool = True


class EnrichmentAnswerRequest(BaseModel):
    question_id: int
    answer: str = Field(..., min_length=3)
    source_label: str = "editor-clarification"


class EnrichmentEscalateRequest(BaseModel):
    question_id: int
    user_id: str = "editor-unknown"
    reason: str = "Editor could not answer clarification question"


class LanguagePolicyRequest(BaseModel):
    primary_language: str = Field(..., min_length=2, max_length=5)
    allowed_languages: list[str] = Field(..., min_length=1, max_length=8)


class ChatSessionCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    source: str = Field(default="webchat", min_length=1, max_length=32)
    metadata: dict = Field(default_factory=dict)


class ChatMessageRequest(BaseModel):
    session_id: str = Field(..., min_length=10)
    message: str = Field(..., min_length=1)
    model: str | None = None


class ChatHistoryRequest(BaseModel):
    session_id: str = Field(..., min_length=10)
    limit: int = Field(default=100, ge=1, le=1000)


class QAExportRequest(BaseModel):
    format: str = Field(default="jsonl", pattern=r"^(jsonl|csv|md|pdf)$")
    limit: int = Field(default=5000, ge=1, le=200000)
