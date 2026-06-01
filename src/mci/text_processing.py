"""Interfaces for headline text normalisation and deduplication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class TextProcessingSpec:
    """Input and output paths for deterministic text-processing steps."""

    input_path: Path
    output_path: Path
    text_column: str = "headline"


def normalise_text(text: str) -> str:
    """Return a normalised version of text for matching and deduplication."""

    raise NotImplementedError("Text normalisation is not implemented in the scaffold.")


def deduplicate_headlines(records: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """Deduplicate headline records while preserving source metadata."""

    raise NotImplementedError("Headline deduplication is not implemented in the scaffold.")


def build_annotation_sample(spec: TextProcessingSpec) -> Path:
    """Export a mixed candidate and non-candidate sample for K's annotation."""

    raise NotImplementedError("Annotation sample export is not implemented in the scaffold.")

