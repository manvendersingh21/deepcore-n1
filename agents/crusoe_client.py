"""Crusoe Cloud LLM client (OpenAI-compatible) plus the deterministic fallback patch synthesizer.

No network call is ever attempted when CRUSOE_API_KEY is missing; every failure degrades to `None`.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://api.crusoe.ai/v1"
DEFAULT_MODEL = "nvidia/Llama-3.1-Nemotron-70B-Instruct"

try:  # .env is optional; never override variables already exported by the shell
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover - python-dotenv is in requirements
    pass

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)


def estimate_tokens(text: str) -> int:
    """Shared-brain token estimate: ceil(len/4)."""
    return math.ceil(len(text or "") / 4)


def extract_code(text: Optional[str]) -> Optional[str]:
    """Return the fenced python block that defines `class ExcavationPipeline` (unfenced source accepted too)."""
    if not text:
        return None
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(text)]
    for lang, body in sorted(blocks, key=lambda b: b[0] not in ("python", "py")):
        if lang in ("python", "py", "") and "class ExcavationPipeline" in body:
            return body.strip() + "\n"
    if not blocks and re.search(r"^class ExcavationPipeline\b", text, re.MULTILINE):
        return text.strip() + "\n"
    return None


class CrusoeClient:
    """Thin wrapper over `openai.OpenAI` pointed at Crusoe's OpenAI-compatible endpoint."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.environ.get("CRUSOE_API_KEY", "").strip()
        self.base_url = base_url or os.environ.get("CRUSOE_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("CRUSOE_MODEL", DEFAULT_MODEL)
        self.last_error: Optional[str] = None
        self.last_usage: dict = {}
        self._client = None

    @property
    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def _get(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=1)
        return self._client

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 4096, timeout: float = 60,
             thinking: Optional[bool] = None) -> Optional[str]:
        """Return the assistant text, or None on missing key / any error. Never raises."""
        if not self.available:
            self.last_error = "CRUSOE_API_KEY not set"
            return None
        try:
            options = {}
            # Nemotron 3 supports this template switch. Bounded edit replies need
            # their small output budget for the JSON rather than hidden reasoning.
            if thinking is not None and 'nemotron-3' in self.model.lower():
                options['extra_body'] = {'chat_template_kwargs': {'enable_thinking': thinking}}
            resp = self._get().chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout,
                **options
            )
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            prompt = "".join(str(m.get("content", "")) for m in messages)
            self.last_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None) or estimate_tokens(prompt),
                "completion_tokens": getattr(usage, "completion_tokens", None) or estimate_tokens(text),
                "finish_reason": resp.choices[0].finish_reason,
            }
            self.last_error = None
            return text or None
        except Exception as exc:  # network, auth, rate limit, malformed response
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None


def default_validator(source: str) -> list[str]:
    """Static AST guard of the locked verifier + compile check. Empty list means valid."""
    try:
        from core.verifier_harness import static_check
    except Exception as exc:
        return [f"verifier unavailable: {exc}"]
    problems = list(static_check(source))
    try:
        compile(source, "<candidate>", "exec")
    except SyntaxError as exc:
        problems.append(f"SyntaxError: {exc}")
    return problems


class FallbackSynthesizer:
    """Produces candidate sources: LLM first (validated), otherwise the curated fallback patch for the generation."""

    def __init__(self, client: Optional[CrusoeClient] = None, brain=None, patches: Optional[list[dict]] = None,
                 validator: Callable[[str], list[str]] = default_validator):
        self.client = client
        self.brain = brain
        self._patches = patches
        self.validator = validator
        self.last_problems: list[str] = []

    @property
    def patches(self) -> list[dict]:
        if self._patches is None:
            from agents.fallback_library import FALLBACK_PATCHES

            self._patches = list(FALLBACK_PATCHES)
        return self._patches

    def next_patch(self, generation: int, autopsy: Optional[dict] = None) -> dict:
        """Generation 1 -> v1, 2 -> v2, ...; clamps to the last patch."""
        if not self.patches:
            raise RuntimeError("fallback library is empty")
        return self.patches[min(max(int(generation), 1), len(self.patches)) - 1]

    def patch_source(self, patch: dict) -> str:
        path = Path(patch["path"])
        return (path if path.is_absolute() else REPO_ROOT / path).read_text()

    def _log(self, agent: str, op: str, target: str, tokens: int, detail: dict) -> None:
        if self.brain is not None:
            self.brain.log_transaction(agent, op, target, tokens=tokens, detail=detail)

    def try_llm(self, system: str, user: str, agent: str = "llm", temperature: float = 0.2) -> Optional[str]:
        """Ask the LLM for a full pipeline file; return it only if it passes static_check and compiles."""
        self.last_problems = []
        if self.client is None or not self.client.available:
            return None
        text = self.client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                temperature=temperature, max_tokens=6000, timeout=120)
        usage = dict(self.client.last_usage) if text else {}
        self._log(agent, "write", f"llm:{self.client.model}", estimate_tokens(system + user) + estimate_tokens(text or ""),
                  {"ok": text is not None, "error": self.client.last_error, **usage})
        code = extract_code(text)
        if code is None:
            self.last_problems = ["no ExcavationPipeline code block in LLM reply"] if text else [str(self.client.last_error)]
            return None
        self.last_problems = self.validator(code)
        return None if self.last_problems else code

    def try_edits(self, system: str, user: str, source: str, agent: str = "llm",
                  temperature: float = 0.2) -> Optional[str]:
        """Apply a bounded JSON edit proposal, then validate the complete candidate."""
        self.last_problems = []
        if self.client is None or not self.client.available:
            self.last_problems = ["LLM client unavailable"]
            return None
        text = None
        usage: dict = {}
        parsed_edit_count = 0
        candidate = None
        try:
            text = self.client.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature, max_tokens=1400, timeout=60, thinking=False)
            usage = dict(self.client.last_usage) if text else {}
            if not text:
                raise ValueError(self.client.last_error or "empty LLM edit reply")
            payload = text.strip()
            fenced = _FENCE.fullmatch(payload)
            if fenced:
                if fenced.group(1).lower() not in ("json", ""):
                    raise ValueError("edit proposal fence must be JSON")
                payload = fenced.group(2)
            proposal = json.loads(payload)
            if not isinstance(proposal, dict) or set(proposal) - {"edits", "rationale"}:
                raise ValueError("edit proposal must be an object with edits and optional rationale")
            if "rationale" in proposal and not isinstance(proposal["rationale"], str):
                raise ValueError("edit rationale must be a string")
            edits = proposal.get("edits")
            if not isinstance(edits, list):
                raise ValueError("edits must be a list")
            parsed_edit_count = len(edits)
            if not 1 <= parsed_edit_count <= 3:
                raise ValueError("edit proposal must contain 1 to 3 edits")
            updated = source
            seen = set()
            for index, edit in enumerate(edits, 1):
                if not isinstance(edit, dict) or set(edit) != {"old", "new"}:
                    raise ValueError(f"edit {index} must contain exactly old and new")
                old, new = edit["old"], edit["new"]
                if not isinstance(old, str) or not isinstance(new, str):
                    raise ValueError(f"edit {index} old and new must be strings")
                if not old or max(len(old), len(new)) > 5000:
                    raise ValueError(f"edit {index} requires nonempty old and strings of at most 5000 characters")
                if old in seen:
                    raise ValueError(f"edit {index} duplicates an earlier replacement target")
                seen.add(old)
                matches = updated.count(old)
                if matches != 1:
                    raise ValueError(f"edit {index} old matches {matches} times in current candidate; expected exactly once")
                updated = updated.replace(old, new, 1)
            if updated == source:
                raise ValueError("edit proposal leaves source unchanged")
            tree = ast.parse(updated, filename="<candidate>")
            if not any(isinstance(node, ast.ClassDef) and node.name == "ExcavationPipeline"
                       for node in tree.body):
                raise ValueError("edited source must retain class ExcavationPipeline")
            self.last_problems = list(self.validator(updated))
            if self.validator is not default_validator:
                try:
                    compile(updated, "<candidate>", "exec")
                except (SyntaxError, ValueError, TypeError) as exc:
                    self.last_problems.append(f"{type(exc).__name__}: {exc}")
            if not self.last_problems:
                candidate = updated
        except Exception as exc:
            self.last_problems.append(f"{type(exc).__name__}: {exc}")
        finally:
            prompt_tokens = usage.get("prompt_tokens", estimate_tokens(system + user))
            completion_tokens = usage.get("completion_tokens", estimate_tokens(text or ""))
            self._log(agent, "write", f"llm:{self.client.model}", prompt_tokens + completion_tokens,
                      {"proposal_mode": "edits", "parsed_edit_count": parsed_edit_count,
                       "ok": candidate is not None, "error": "; ".join(self.last_problems) or None,
                       "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens})
        return candidate

    def synthesize(self, generation: int, system: str, user: str, autopsy: Optional[dict] = None,
                   agent: str = "llm") -> tuple[str, str]:
        """Return (source_code, origin) with origin 'llm:<model>' or 'fallback_patch:<name>'."""
        code = self.try_llm(system, user, agent=agent)
        if code is not None:
            return code, f"llm:{self.client.model}"
        patch = self.next_patch(generation, autopsy)
        src = self.patch_source(patch)
        self._log(agent, "read", f"fallback_patch:{patch['name']}", estimate_tokens(src), {"generation": generation})
        return src, f"fallback_patch:{patch['name']}"
