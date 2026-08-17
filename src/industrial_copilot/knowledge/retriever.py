"""Local TF-IDF retrieval for a small reviewed industrial knowledge corpus."""

from __future__ import annotations

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from industrial_copilot.knowledge.schemas import KnowledgeHit


class DomainKnowledgeRetriever:
    """Retrieve explanatory knowledge; retrieved text never becomes executable instruction."""

    def __init__(self, skills_directory: Path) -> None:
        self.skills_directory = Path(skills_directory)
        self._chunks = self._load_chunks()
        corpus = [self._search_text(chunk) for chunk in self._chunks]
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._matrix = self._vectorizer.fit_transform(corpus) if corpus else None

    def search(self, query: str, *, failure_mode: str | None = None, top_k: int = 3) -> list[KnowledgeHit]:
        if not query.strip() or not self._chunks or self._matrix is None:
            return []
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        results: list[KnowledgeHit] = []
        for index in sorted(range(len(scores)), key=lambda item: scores[item], reverse=True):
            chunk = self._chunks[index]
            if failure_mode and chunk["failure_modes"] and failure_mode not in chunk["failure_modes"]:
                continue
            if scores[index] <= 0:
                continue
            results.append(KnowledgeHit(**chunk, score=float(scores[index])))
            if len(results) >= max(1, min(top_k, 5)):
                break
        return results

    def _load_chunks(self) -> list[dict[str, object]]:
        chunks: list[dict[str, object]] = []
        for path in sorted(self.skills_directory.glob("*.md")):
            metadata, body = _split_frontmatter(path.read_text(encoding="utf-8"))
            sections = re.split(r"(?m)^##\s+", body)
            for index, section in enumerate(sections):
                content = section.strip()
                if not content:
                    continue
                lines = content.splitlines()
                heading = "Overview" if index == 0 else lines[0].strip()
                text = content if index == 0 else "\n".join(lines[1:]).strip()
                if text:
                    chunks.append(
                        {
                            "id": f"{path.stem}:{index}",
                            "title": metadata.get("title", path.stem),
                            "section": heading,
                            "text": text[:1_800],
                            "authority": metadata.get("authority", "engineering_reference"),
                            "source": metadata.get("source", path.name),
                            "failure_modes": _csv(metadata.get("failure_modes", "")),
                            "signals": _csv(metadata.get("signals", "")),
                        }
                    )
        return chunks

    @staticmethod
    def _search_text(chunk: dict[str, object]) -> str:
        return " ".join(str(value) for value in chunk.values())


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata, parts[2]


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
