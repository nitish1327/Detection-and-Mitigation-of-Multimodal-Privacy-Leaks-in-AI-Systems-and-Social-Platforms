"""Shared data models for LeakWatch."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class Span(BaseModel):
    start: int
    end: int


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectedEntity(BaseModel):
    entity_id: str | None = None
    modality: Modality
    label: str
    confidence: float
    text: str | None = None
    span: Span | None = None
    bbox: BoundingBox | None = None
    mitigation: Literal["mask", "blur", "replace", "none"] = "none"
    explanation: str | None = None
    detection_method: Literal["regex", "nlp", "heuristic", "unknown"] = "unknown"


class DetectionResult(BaseModel):
    source_path: Path
    modality: Modality
    entities: list[DetectedEntity] = Field(default_factory=list)
    mitigated_output: Path | None = None
    audit_log: Path | None = None
    artifacts: list[Path] = Field(default_factory=list)
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    """Lightweight representation of a graph node for future GNN work."""

    identifier: str
    modality: Modality
    payload: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str = "related"


class GraphContext(BaseModel):
    """Container for token/region graphs; GraphSAGE stays a future extension."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

    def adjacency(self) -> dict[str, list[str]]:
        adjacency: dict[str, list[str]] = {node.identifier: [] for node in self.nodes}
        for edge in self.edges:
            adjacency.setdefault(edge.source, []).append(edge.target)
        return adjacency

    # NOTE: GraphSAGE / GNN aggregation intentionally omitted in Phase 2.
