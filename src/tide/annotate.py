"""Cached, budgeted DeepSeek annotation of the controlled source records."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import ValidationError

from tide.config import TideConfig
from tide.env import required_env
from tide.io import write_json
from tide.schemas import AnnotationResponse, ExpandedRecord, SourceRecord
from tide.validate import REQUIRED, protected_for, validate_annotation_response


@dataclass
class UsageBudget:
    max_requests: int
    max_usd: float
    input_per_million: float
    output_per_million: float
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def estimated_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * self.input_per_million
            + self.output_tokens / 1_000_000 * self.output_per_million
        )

    def reserve_request(self) -> None:
        if self.requests >= self.max_requests:
            raise RuntimeError("annotation request cap reached")
        if self.estimated_usd >= self.max_usd:
            raise RuntimeError("annotation cost cap reached")
        self.requests += 1

    def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.input_tokens += prompt_tokens
        self.output_tokens += completion_tokens
        if self.estimated_usd > self.max_usd:
            raise RuntimeError("annotation response exceeded configured cost cap")


def _payload(source: SourceRecord) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "context": source.context,
        "fact_ledger": [fact.model_dump(mode="json") for fact in source.facts],
        "tasks": [{"task_id": task.task_id, "task": task.task} for task in source.tasks],
    }


def cache_key(source: SourceRecord, model: str, system_prompt: str, max_tokens: int) -> str:
    material = json.dumps(
        {
            "payload": _payload(source),
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "prompt": system_prompt,
            "prompt_version": "annotate_v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode()).hexdigest()


def expand_source(
    source: SourceRecord, response: AnnotationResponse, model: str
) -> list[ExpandedRecord]:
    by_task = {item.task_id: item for item in response.annotations}
    expanded: list[ExpandedRecord] = []
    for task in source.tasks:
        annotation = by_task[task.task_id]
        expanded.append(
            ExpandedRecord(
                source_id=source.source_id,
                example_id=f"{source.source_id}_{task.task_id}",
                split=source.split,
                template_id=source.template_id,
                context=source.context,
                facts=source.facts,
                task_id=task.task_id,
                task=task.task,
                rule=task.rule,
                required_fact_ids=REQUIRED[task.task_id],
                protected_fact_ids=protected_for(task.task_id),
                reference_rewrite=annotation.rewritten_context,
                gold_answer=task.gold_answer,
                annotation={
                    "provider": "deepseek",
                    "requested_model": model,
                    "prompt_version": "annotate_v1",
                },
            )
        )
    return expanded


class DeepSeekAnnotator:
    def __init__(self, config: TideConfig, root: Path) -> None:
        self.config = config
        self.root = root
        self.model = os.getenv(config.annotation.model_env, config.annotation.default_model)
        self.system_prompt = (root / "prompts/annotate_system.txt").read_text(encoding="utf-8")
        input_price = required_env("DEEPSEEK_INPUT_USD_PER_MILLION")
        output_price = required_env("DEEPSEEK_OUTPUT_USD_PER_MILLION")
        self.budget = UsageBudget(
            config.annotation.max_api_requests,
            config.annotation.max_estimated_usd,
            float(input_price),
            float(output_price),
        )
        self.client = AsyncOpenAI(
            api_key=required_env("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=float(config.annotation.timeout_seconds),
            max_retries=0,
        )
        self.cache_dir = root / "cache/annotations"
        self._budget_lock = asyncio.Lock()

    async def verify_provider(self) -> None:
        models = await self.client.models.list()
        available = {item.id for item in models.data}
        if self.model not in available:
            raise RuntimeError(
                f"configured DeepSeek model {self.model!r} is not in the provider model list: "
                f"{sorted(available)}"
            )

    async def _request(
        self, source: SourceRecord, repair_errors: list[str] | None
    ) -> tuple[Any, str]:
        user_content = json.dumps(_payload(source), ensure_ascii=False)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        if repair_errors:
            repair_message = (
                f"Your previous JSON failed validation: {repair_errors}. "
                "Correct the annotations using the same original context and tasks. "
                "Return the complete JSON object only."
            )
            messages.append(
                {
                    "role": "user",
                    "content": repair_message,
                }
            )
        async with self._budget_lock:
            self.budget.reserve_request()
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            temperature=0,
            max_tokens=self.config.annotation.max_output_tokens,
        )
        content = response.choices[0].message.content or ""
        if response.usage:
            async with self._budget_lock:
                self.budget.add_usage(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )
        return response, content

    async def annotate_one(self, source: SourceRecord) -> AnnotationResponse:
        key = cache_key(
            source, self.model, self.system_prompt, self.config.annotation.max_output_tokens
        )
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("status") == "valid":
                return AnnotationResponse.model_validate(cached["parsed"])
        repair_errors: list[str] | None = None
        transport_attempt = 0
        semantic_repairs = 0
        while transport_attempt < self.config.annotation.max_transport_attempts:
            transport_attempt += 1
            try:
                response, content = await self._request(source, repair_errors)
                parsed: AnnotationResponse | None
                try:
                    parsed = AnnotationResponse.model_validate_json(content)
                    errors = validate_annotation_response(source, parsed)
                except ValidationError as exc:
                    parsed = None
                    errors = [f"schema:{item['type']}" for item in exc.errors()]
                record = {
                    "status": "valid" if not errors else "invalid",
                    "source_id": source.source_id,
                    "request_id": getattr(response, "id", None),
                    "response_model": getattr(response, "model", None),
                    "finish_reason": response.choices[0].finish_reason,
                    "raw_content": content,
                    "parsed": parsed.model_dump(mode="json") if parsed else None,
                    "validation_errors": errors,
                    "transport_attempt": transport_attempt,
                    "semantic_repairs": semantic_repairs,
                }
                write_json(cache_path, record)
                if not errors and parsed is not None:
                    return parsed
                semantic_repairs += 1
                if semantic_repairs > self.config.annotation.max_semantic_repairs:
                    raise RuntimeError(
                        f"semantic repair cap reached for {source.source_id}: {errors}"
                    )
                repair_errors = errors
            except APIStatusError as exc:
                if exc.status_code in {401, 403}:
                    raise RuntimeError(
                        f"DeepSeek rejected credentials or permission ({exc.status_code})"
                    ) from exc
                if exc.status_code != 429 and exc.status_code < 500:
                    raise
                if transport_attempt >= self.config.annotation.max_transport_attempts:
                    raise RuntimeError(
                        f"annotation HTTP retries exhausted for {source.source_id}"
                    ) from exc
                await asyncio.sleep((2**transport_attempt) + random.random())
            except (APIConnectionError, APITimeoutError, OSError, TimeoutError) as exc:
                if transport_attempt >= self.config.annotation.max_transport_attempts:
                    raise RuntimeError(
                        f"annotation transport failed for {source.source_id}"
                    ) from exc
                await asyncio.sleep((2**transport_attempt) + random.random())
        raise AssertionError("unreachable")

    async def annotate_many(
        self, sources: list[SourceRecord]
    ) -> list[tuple[SourceRecord, AnnotationResponse]]:
        await self.verify_provider()
        semaphore = asyncio.Semaphore(self.config.annotation.concurrency)

        async def bounded(source: SourceRecord) -> tuple[SourceRecord, AnnotationResponse]:
            async with semaphore:
                return source, await self.annotate_one(source)

        return await asyncio.gather(*(bounded(source) for source in sources))

    def write_usage(self) -> None:
        write_json(
            self.root / "runs/overnight/annotation_usage.json",
            {
                "requests": self.budget.requests,
                "input_tokens": self.budget.input_tokens,
                "output_tokens": self.budget.output_tokens,
                "estimated_usd": self.budget.estimated_usd,
                "model": self.model,
            },
        )
