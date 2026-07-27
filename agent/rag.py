"""
RAG 代码检索
────────────
CodeIndex：对代码仓库建立向量索引，支持自然语言语义检索。

  - 分块：按行滑动窗口（chunk_lines + overlap），保留文件路径与行号
  - 向量：OpenAI-compatible /embeddings 接口（EMBEDDING_BASE_URL/API_KEY/MODEL）
  - 存储：FAISS（faiss-cpu，若已安装）否则 numpy 暴力余弦相似度兜底
  - 索引：内存态 + 可选磁盘缓存（.rag_index.npz），文件 mtime 变化自动重建

面向大仓库场景：Agent 不必通读全部源码，用 search_code 工具按需检索相关片段。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

_CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
              ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".md"}
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


@dataclass
class Chunk:
    path: str          # 相对路径
    start_line: int
    end_line: int
    text: str


class CodeIndex:
    def __init__(
        self,
        root: str,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        chunk_lines: int = 60,
        chunk_overlap: int = 10,
    ) -> None:
        self.root = Path(root).resolve()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.chunk_lines = chunk_lines
        self.chunk_overlap = chunk_overlap

        self.chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None   # (N, dim) L2 归一化
        self._faiss_index = None                  # faiss.IndexFlatIP

    # ── 分块 ──────────────────────────────────────────────────────────
    def _iter_files(self) -> list[Path]:
        files = []
        for p in self.root.rglob("*"):
            if not p.is_file() or p.suffix not in _CODE_EXTS:
                continue
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            files.append(p)
        return sorted(files)

    def _chunk_file(self, path: Path) -> list[Chunk]:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return []
        rel = str(path.relative_to(self.root))
        chunks, step = [], max(1, self.chunk_lines - self.chunk_overlap)
        for start in range(0, len(lines), step):
            end = min(start + self.chunk_lines, len(lines))
            text = "\n".join(lines[start:end]).strip()
            if text:
                chunks.append(Chunk(rel, start + 1, end, text))
            if end == len(lines):
                break
        return chunks

    # ── 向量接口 ──────────────────────────────────────────────────────
    def _embed(self, texts: list[str]) -> np.ndarray:
        """调用 OpenAI-compatible embeddings 接口，返回 L2 归一化向量。"""
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        vecs = np.array([d["embedding"] for d in resp.json()["data"]], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    # ── 索引构建（含磁盘缓存）─────────────────────────────────────────
    def _cache_key(self) -> str:
        """按文件路径+mtime 生成指纹，变化则重建。"""
        sig = [(str(p), p.stat().st_mtime) for p in self._iter_files()]
        return hashlib.md5(json.dumps(sig).encode()).hexdigest()

    def build(self) -> None:
        cache_file = self.root / ".rag_index.npz"
        key = self._cache_key()
        if cache_file.exists():
            try:
                data = np.load(cache_file, allow_pickle=True)
                if str(data["key"]) == key:
                    self.chunks = [Chunk(*c) for c in data["chunks"].tolist()]
                    self._load_vectors(data["vectors"])
                    return
            except Exception:
                pass  # 缓存损坏则重建

        self.chunks = [c for f in self._iter_files() for c in self._chunk_file(f)]
        if not self.chunks:
            return
        # 分批 embedding，避免单次请求过大
        vecs = []
        batch = 64
        for i in range(0, len(self.chunks), batch):
            texts = [f"{c.path}:{c.start_line}\n{c.text}" for c in self.chunks[i:i + batch]]
            vecs.append(self._embed(texts))
        self._load_vectors(np.vstack(vecs))
        np.savez(cache_file, key=key, vectors=self._vectors,
                 chunks=np.array([(c.path, c.start_line, c.end_line, c.text)
                                  for c in self.chunks], dtype=object))

    def _load_vectors(self, vectors: np.ndarray) -> None:
        self._vectors = vectors
        if _HAS_FAISS:
            self._faiss_index = faiss.IndexFlatIP(vectors.shape[1])
            self._faiss_index.add(vectors)

    # ── 检索 ──────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.chunks or self._vectors is None:
            return []
        q = self._embed([query])
        k = min(top_k, len(self.chunks))
        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(q, k)
            idxs, scores = idxs[0], scores[0]
        else:
            sims = (self._vectors @ q[0])
            idxs = np.argpartition(-sims, k - 1)[:k]
            idxs = idxs[np.argsort(-sims[idxs])]
            scores = sims[idxs]
        return [
            {
                "path": self.chunks[i].path,
                "lines": f"{self.chunks[i].start_line}-{self.chunks[i].end_line}",
                "score": round(float(s), 4),
                "text": self.chunks[i].text,
            }
            for i, s in zip(idxs, scores)
        ]
