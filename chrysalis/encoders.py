import re
import threading
import time
import zipfile
from pathlib import Path

import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class Ledger:
    """Busy encoder seconds and item counts per category."""

    def __init__(self, path=None):
        self.lock = threading.Lock()
        self.totals = {}
        self.f = open(path, "a") if path else None

    def record(self, category, seconds, count, device):
        with self.lock:
            s, c = self.totals.get(category, (0.0, 0))
            self.totals[category] = (s + seconds, c + count)
            if self.f:
                self.f.write(f"{time.time():.6f}\t{category}\t{seconds:.6f}\t{count}\t{device}\n")
                self.f.flush()

    def per_item(self, category):
        with self.lock:
            s, c = self.totals.get(category, (0.0, 0))
            return s / c if c else 0.0

    def seconds(self, exclude=("served_query",)):
        with self.lock:
            return sum(s for k, (s, _) in self.totals.items() if k not in exclude)


class Encoder:
    """Pinned Hugging Face checkpoint; FP16 on CUDA, unit-normalized outputs."""

    def __init__(self, spec, device="cuda", batch_size=128, cache=None, ledger=None):
        import torch

        self.torch, self.spec, self.device, self.batch_size, self.ledger = torch, spec, device, batch_size, ledger
        self.kind, self.dim = spec["kind"], spec["dim"]
        self.lock = threading.Lock()
        if self.kind == "clip":
            from transformers import CLIPModel, CLIPProcessor

            self.processor = CLIPProcessor.from_pretrained(spec["id"], revision=spec["revision"], cache_dir=cache)
            self.model = CLIPModel.from_pretrained(spec["id"], revision=spec["revision"], cache_dir=cache).to(device).eval()
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(spec["id"], revision=spec["revision"], cache_folder=cache, device=device)
            self.model.max_seq_length = spec["max_tokens"]
        if device.startswith("cuda"):
            self.model.half()

    def _batch(self, items, role):
        torch = self.torch
        with torch.inference_mode():
            if self.kind == "clip":
                if role == "query":
                    inputs = self.processor(text=items, padding=True, truncation=True,
                                            max_length=self.spec["max_tokens"], return_tensors="pt").to(self.device)
                    out = self.model.get_text_features(**inputs)
                else:
                    from PIL import Image

                    images = []
                    for path in items:
                        with Image.open(path) as im:
                            images.append(im.convert("RGB"))
                    inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                    if self.device.startswith("cuda"):
                        inputs["pixel_values"] = inputs["pixel_values"].half()
                    out = self.model.get_image_features(**inputs)
                return out.float().cpu().numpy()
            if self.kind == "e5":
                items = [("query: " if role == "query" else "passage: ") + t for t in items]
            return self.model.encode(items, batch_size=len(items), convert_to_numpy=True,
                                     normalize_embeddings=False, show_progress_bar=False)

    def encode(self, items, role="item", category="item"):
        out = []
        with self.lock:
            for start in range(0, len(items), self.batch_size):
                batch = list(items[start:start + self.batch_size])
                if self.device.startswith("cuda"):
                    self.torch.cuda.synchronize()
                t = time.perf_counter()
                v = self._batch(batch, role)
                if self.device.startswith("cuda"):
                    self.torch.cuda.synchronize()
                if self.ledger:
                    self.ledger.record(category, time.perf_counter() - t, len(batch), self.device)
                out.append(unit(v))
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)


class GloveEncoder:
    """Average of GloVe 6B.300d vectors over lowercase word tokens. Out-of-vocabulary tokens are
    skipped; an item with no known token gets the vocabulary mean."""

    def __init__(self, spec, path, ledger=None):
        self.spec, self.dim, self.ledger, self.table = spec, spec["dim"], ledger, {}
        self.mean = None
        path = Path(path)
        f = zipfile.ZipFile(path).open(spec["member"]) if path.suffix == ".zip" else open(path, "rb")
        with f:
            for line in f:
                word, *vals = line.decode("utf-8").rstrip().split(" ")
                self.table[word] = np.asarray(vals, dtype=np.float32)

    def encode(self, items, role="item", category="item"):
        t = time.perf_counter()
        out = np.zeros((len(items), self.dim), np.float32)
        for j, text in enumerate(items):
            vecs = [self.table[w] for w in re.findall(r"\w+", text.lower()) if w in self.table]
            if not vecs:
                if self.mean is None:
                    self.mean = np.mean(np.stack(list(self.table.values())), axis=0)
                vecs = [self.mean]
            out[j] = np.mean(vecs, axis=0)
        if self.ledger:
            self.ledger.record(category, time.perf_counter() - t, len(items), "cpu")
        return unit(out)


def load(name, models, device="cuda", batch_size=128, cache=None, ledger=None, glove_path=None):
    spec = models[name]
    if spec["kind"] == "glove":
        return GloveEncoder(spec, glove_path, ledger)
    return Encoder(spec, device, batch_size, cache, ledger)
