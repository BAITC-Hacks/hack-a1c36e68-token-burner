"""Thin wrapper over the OpenAI Responses API with structured outputs parsed into pydantic models."""
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from backend.config import settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    def __init__(self, message: str, input_tokens: int = 0, output_tokens: int = 0):
        super().__init__(message)
        self.input_tokens, self.output_tokens = input_tokens, output_tokens


@dataclass
class LLMCall:
    parsed: BaseModel
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float
    attempts: int


@lru_cache
def get_client():
    from openai import OpenAI

    if not settings.openai_api_key:
        raise LLMError("OPENAI_API_KEY не задан")
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def call_structured(*, model: str, reasoning: str | None, instructions: str, input: str,
                    schema: type[T], client=None, max_output_tokens: int = 64000) -> LLMCall:
    """One call + one retry on schema/validation problems. Raises LLMError after the second failure."""
    client = client or get_client()
    tokens_in = tokens_out = 0
    started = time.monotonic()
    last_error = ""
    for attempt in (1, 2):
        kwargs = dict(model=model, instructions=instructions, input=input, text_format=schema,
                      max_output_tokens=max_output_tokens)
        if reasoning:
            kwargs["reasoning"] = {"effort": reasoning}
        try:
            response = client.responses.parse(**kwargs)
        except ValidationError as exc:  # SDK validates the JSON against the schema while parsing
            last_error = f"ответ не прошёл валидацию схемы: {exc.error_count()} ошибок"
            log.warning("%s attempt %d: %s", schema.__name__, attempt, last_error)
            continue
        usage = getattr(response, "usage", None)
        tokens_in += getattr(usage, "input_tokens", 0) or 0
        tokens_out += getattr(usage, "output_tokens", 0) or 0
        parsed = getattr(response, "output_parsed", None)
        if parsed is not None:
            return LLMCall(parsed, model, tokens_in, tokens_out, time.monotonic() - started, attempt)
        details = getattr(response, "incomplete_details", None)
        last_error = f"пустой структурированный ответ (status={getattr(response, 'status', '?')}, {details})"
        log.warning("%s attempt %d: %s", schema.__name__, attempt, last_error)
    raise LLMError(last_error, tokens_in, tokens_out)
