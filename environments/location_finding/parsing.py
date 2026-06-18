from __future__ import annotations

import json
import re

from helpers import _strip_code_fences
from .types import Location, LocationStrategyCandidate, SourceConfig, _dedupe_source_configs, normalize_location, normalize_source_config


def _clean_json_completion(text: str) -> str:
    return re.sub(r"<eos>\s*$", "", _strip_code_fences(text).strip()).strip()


def _extract_first_json_value(text: str) -> str | None:
    stripped = _clean_json_completion(text)
    start_idx: int | None = None
    opening = ""
    closing = ""
    depth = 0
    in_string = False
    escaped = False

    for idx, char in enumerate(stripped):
        if start_idx is None:
            if char == "{":
                start_idx = idx
                opening = "{"
                closing = "}"
                depth = 1
            elif char == "[":
                start_idx = idx
                opening = "["
                closing = "]"
                depth = 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return stripped[start_idx:idx + 1]

    return None


def _extract_json_values(text: str) -> list[object]:
    cleaned = _clean_json_completion(text)
    decoder = json.JSONDecoder()
    values: list[object] = []
    for idx, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            payload, _end_idx = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        values.append(payload)
    return values


def _loads_json_value(text: str) -> object:
    stripped = _clean_json_completion(text)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError) as exc:
        extracted = _extract_first_json_value(text)
        if extracted is None:
            raise ValueError(f"Could not find JSON value in completion: {text!r}") from exc
        try:
            return json.loads(extracted)
        except (json.JSONDecodeError, TypeError) as nested_exc:
            raise ValueError(f"Invalid JSON in completion: {text!r}") from nested_exc


def _collect_source_configs_from_payload(payload: object, num_sources: int, dim: int) -> list[SourceConfig]:
    configs: list[SourceConfig] = []
    if isinstance(payload, dict):
        for key in ("hypotheses", "source_configurations", "sources", "configs"):
            if key in payload:
                configs.extend(_collect_source_configs_from_payload(payload[key], num_sources, dim))
                return configs
        for value in payload.values():
            configs.extend(_collect_source_configs_from_payload(value, num_sources, dim))
        return configs

    if isinstance(payload, list):
        try:
            configs.append(normalize_source_config(payload, num_sources, dim))
            return configs
        except ValueError:
            pass
        for item in payload:
            configs.extend(_collect_source_configs_from_payload(item, num_sources, dim))
    return configs


def _extract_partial_source_configs(text: str, num_sources: int, dim: int) -> list[SourceConfig]:
    cleaned = _clean_json_completion(text)
    decoder = json.JSONDecoder()
    configs: list[SourceConfig] = []
    for idx, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            payload, _end_idx = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        configs.extend(_collect_source_configs_from_payload(payload, num_sources, dim))
    return _dedupe_source_configs(configs)


def parse_source_hypotheses(completion: str, num_sources: int, dim: int) -> list[SourceConfig]:
    try:
        payload = _loads_json_value(completion)
    except ValueError:
        configs = _extract_partial_source_configs(completion, num_sources, dim)
        if configs:
            return configs
        raise

    if isinstance(payload, dict):
        for key in ("hypotheses", "source_configurations", "sources", "configs"):
            if key in payload:
                payload = payload[key]
                break

    if not isinstance(payload, list):
        raise ValueError("Source hypothesis completion must decode to a JSON list or object")

    configs: list[SourceConfig] = []
    for item in payload:
        raw_config = item.get("sources") if isinstance(item, dict) else item
        try:
            configs.append(normalize_source_config(raw_config, num_sources, dim))
        except ValueError:
            continue
    if configs:
        return _dedupe_source_configs(configs)
    return _extract_partial_source_configs(completion, num_sources, dim)


def parse_candidate_locations(
    completion: str,
    dim: int,
    bounds: tuple[float, float] | None = None,
) -> list[Location]:
    payload = _loads_json_value(completion)
    if isinstance(payload, dict):
        for key in ("locations", "candidates", "queries", "points", "location"):
            if key in payload:
                payload = payload[key]
                break

    if not isinstance(payload, list):
        raise ValueError("Candidate location completion must decode to a JSON list or object")

    locations: list[Location] = []
    seen: set[tuple[float, ...]] = set()
    for item in payload:
        raw_location = item.get("location") if isinstance(item, dict) else item
        try:
            location = normalize_location(raw_location, dim)
        except ValueError:
            continue
        key = tuple(round(value, 6) for value in location)
        if key in seen:
            continue
        seen.add(key)
        locations.append(location)
    return locations


def parse_single_location(
    completion: str,
    dim: int,
    bounds: tuple[float, float] | None = None,
) -> Location:
    del bounds
    payload = _loads_json_value(completion)
    if isinstance(payload, dict):
        for key in ("location", "query", "point"):
            if key in payload:
                payload = payload[key]
                break
    location = normalize_location(payload, dim)
    return location


def parse_single_location_from_completion(
    completion: str,
    dim: int,
    bounds: tuple[float, float] | None = None,
) -> Location:
    for payload in reversed(_extract_json_values(completion)):
        raw_location: object = payload
        if isinstance(payload, dict):
            matched = False
            for key in ("location", "query", "point"):
                if key in payload:
                    raw_location = payload[key]
                    matched = True
                    break
            if not matched:
                continue
        try:
            location = normalize_location(raw_location, dim)
        except ValueError:
            continue
        return location
    try:
        return parse_single_location(completion, dim, bounds)
    except ValueError:
        pass
    raise ValueError("No valid location JSON found in completion")


def parse_best_source_estimate_from_completion(completion: str, num_sources: int, dim: int) -> SourceConfig:
    for payload in reversed(_extract_json_values(completion)):
        configs = _collect_source_configs_from_payload(payload, num_sources, dim)
        if configs:
            return configs[-1]
    try:
        estimates = parse_source_hypotheses(completion, num_sources, dim)
    except ValueError:
        estimates = []
    if estimates:
        return estimates[-1]
    raise ValueError("No valid source estimate JSON found in completion")


_STRATEGY_MIN_LENGTH = 5


def _clean_strategy_text(raw_strategy: object) -> str | None:
    if not isinstance(raw_strategy, str):
        return None
    cleaned = re.sub(r"\s+", " ", raw_strategy.strip())
    cleaned = re.sub(r"^(?:[-*]|\d+[.)])\s*", "", cleaned).strip()
    if len(cleaned) < _STRATEGY_MIN_LENGTH:
        return None
    return cleaned


def _collect_strategy_texts(payload: object) -> list[str]:
    if isinstance(payload, str):
        cleaned = _clean_strategy_text(payload)
        return [] if cleaned is None else [cleaned]
    if isinstance(payload, list):
        strategies: list[str] = []
        for item in payload:
            strategies.extend(_collect_strategy_texts(item))
        return strategies
    if isinstance(payload, dict):
        for key in ("strategies", "plans", "candidates"):
            if key in payload:
                return _collect_strategy_texts(payload[key])
        if "strategy" in payload:
            return _collect_strategy_texts(payload["strategy"])
        strategies = []
        for value in payload.values():
            strategies.extend(_collect_strategy_texts(value))
        return strategies
    return []


def _strategy_key(strategy: str) -> str:
    return re.sub(r"\s+", " ", strategy.strip()).lower()


def _collect_strategy_root_candidates(
    payload: object,
    dim: int,
    bounds: tuple[float, float] | None = None,
) -> list[LocationStrategyCandidate]:
    candidates: list[LocationStrategyCandidate] = []
    if isinstance(payload, list):
        for item in payload:
            candidates.extend(_collect_strategy_root_candidates(item, dim, bounds))
        return candidates
    if not isinstance(payload, dict):
        return candidates

    for key in ("strategies", "plans", "candidates"):
        if key in payload:
            return _collect_strategy_root_candidates(payload[key], dim, bounds)

    raw_strategy = payload.get("strategy")
    strategy = _clean_strategy_text(raw_strategy)
    if strategy is None:
        for value in payload.values():
            candidates.extend(_collect_strategy_root_candidates(value, dim, bounds))
        return candidates

    raw_root = None
    for key in ("root_query", "root_location", "first_query", "first_location", "query", "location"):
        if key in payload:
            raw_root = payload[key]
            break
    if raw_root is None:
        return candidates

    try:
        root_query = normalize_location(raw_root, dim)
    except ValueError:
        return candidates
    candidates.append(LocationStrategyCandidate(strategy=strategy, root_query=root_query))
    return candidates


def parse_location_strategies(completion: str) -> list[str]:
    payload = _loads_json_value(completion)
    strategies: list[str] = []
    seen: set[str] = set()
    for strategy in _collect_strategy_texts(payload):
        key = strategy.lower()
        if key in seen:
            continue
        seen.add(key)
        strategies.append(strategy)
    return strategies


def parse_location_strategy_roots(
    completion: str,
    dim: int,
    bounds: tuple[float, float] | None = None,
) -> list[LocationStrategyCandidate]:
    payload = _loads_json_value(completion)
    candidates: list[LocationStrategyCandidate] = []
    seen: set[str] = set()
    for candidate in _collect_strategy_root_candidates(payload, dim, bounds):
        if candidate.root_query is None:
            continue
        key = _strategy_key(candidate.strategy)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def parse_strategy_location(
    completion: str,
    dim: int,
    bounds: tuple[float, float] | None = None,
) -> Location:
    payload = _loads_json_value(completion)
    raw_location = payload
    if isinstance(payload, dict):
        for key in ("location", "query", "point"):
            if key in payload:
                raw_location = payload[key]
                break
        else:
            candidate_locations = parse_candidate_locations(completion, dim, bounds)
            if candidate_locations:
                return candidate_locations[0]
            raise ValueError("Strategy location completion must include location, query, or point")

    location = normalize_location(raw_location, dim)
    return location
