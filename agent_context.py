from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from enum import Enum


MAX_TOTAL_SUBAGENTS = 5
MAX_DEPTH           = 3
MAX_CONCURRENT      = 2


class ModelTier(str, Enum):
    LIGHT    = "light"
    STANDARD = "standard"
    HEAVY    = "heavy"


TIER_MODEL_MAP: dict[ModelTier, str] = {
    ModelTier.LIGHT:    "phi3-mini",
    ModelTier.STANDARD: "qwen2.5:3b",
    ModelTier.HEAVY:    "gemma4",
}

REQUIRES_CONFIRMATION: dict[ModelTier, bool] = {
    ModelTier.LIGHT:    False,
    ModelTier.STANDARD: False,
    ModelTier.HEAVY:    True,
}

TIER_DESCRIPTIONS: dict[ModelTier, str] = {
    ModelTier.LIGHT: (
        "phi3-mini (Light) — simple single-step tasks: short text generation, "
        "formatting, basic lookups. Fastest, lowest resource. "
        "Risk: may fail on complex reasoning."
    ),
    ModelTier.STANDARD: (
        "qwen2.5:3b (Standard) — moderate tasks: multi-step reasoning, "
        "code generation, summarisation. Balanced speed and capability. "
        "Recommended default for most sub-tasks."
    ),
    ModelTier.HEAVY: (
        "gemma4 (Heavy) — use ONLY for tasks requiring deep reasoning where "
        "Standard has failed or is clearly insufficient. "
        "WARNING: high resource usage, significant speed reduction. "
        "Requires explicit user confirmation before the run starts."
    ),
}


def requires_confirmation(model_name: str) -> bool:
    for tier, name in TIER_MODEL_MAP.items():
        if name == model_name:
            return REQUIRES_CONFIRMATION.get(tier, True)
    return True


@dataclass
class AgentContext:
    orchestrator_brief: str         = ""
    restrictions: str               = ""
    confirmed_models: set[str]      = field(default_factory=set)

    _spawned_count: int             = field(default=0,    repr=True)
    _current_depth: int             = field(default=0,    repr=False)
    _lock: asyncio.Lock | None      = field(default=None, repr=False)
    _semaphore: asyncio.BoundedSemaphore | None = field(default=None, repr=False)

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def semaphore(self) -> asyncio.BoundedSemaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.BoundedSemaphore(MAX_CONCURRENT)
        return self._semaphore

    def is_approved(self, model_name: str) -> bool:
        if not requires_confirmation(model_name):
            return True
        return model_name in self.confirmed_models

    def resolve_model(self, tier: ModelTier | str) -> str | None:
        if isinstance(tier, str):
            try:
                tier = ModelTier(tier.lower())
            except ValueError:
                tier = ModelTier.STANDARD

        model_name = TIER_MODEL_MAP[tier]

        if not self.is_approved(model_name):
            return None

        return model_name

    def can_spawn(self) -> bool:
        return (
            self._spawned_count < MAX_TOTAL_SUBAGENTS
            and self._current_depth < MAX_DEPTH
        )

    async def claim_spawn(self) -> bool:
        async with self.lock:
            if not self.can_spawn():
                return False
            self._spawned_count += 1
            return True

    def child_context(self, new_brief: str = "") -> AgentContext:
        return AgentContext(
            orchestrator_brief=new_brief or self.orchestrator_brief,
            restrictions=self.restrictions,
            confirmed_models=self.confirmed_models,
            _spawned_count=self._spawned_count,
            _current_depth=self._current_depth + 1,
            _lock=self._lock,
            _semaphore=self._semaphore,
        )
