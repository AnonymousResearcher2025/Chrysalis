"""Actual PyTorch inference. Evaluation oracles never enter this worker API."""
import re
import os
import time
import numpy as np
from .math import normalize

MODEL_SPECS = {
    'minilm': ('sentence-transformers/all-MiniLM-L6-v2', 'text', 256),
    'mpnet': ('sentence-transformers/all-mpnet-base-v2', 'text', 384),
    'e5-base': ('intfloat/e5-base-v2', 'e5', 512),
    'e5-large': ('intfloat/e5-large-v2', 'e5', 512),
    'clip-b32': ('openai/clip-vit-base-patch32', 'clip', 77),
    'clip-l14': ('openai/clip-vit-large-patch14', 'clip', 77),
}
PAIRS = {'P1': ('minilm', 'mpnet'), 'P2': ('e5-base', 'e5-large'),
         'P3': ('clip-b32', 'clip-l14'), 'P4': ('glove', 'mpnet')}


class Encoder:
    def __init__(self, key, lock, device='cpu', batch_size=32, cache='models', account=None):
        # Do not load arbitrary site-installed accelerator plugins in artifacts.
        os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
        import torch
        self.torch = torch
        self.device, self.batch_size, self.account = device, batch_size, account
        self.key, self.spec = key, lock[key]
        if not re.fullmatch(r'[0-9a-f]{40}', self.spec['revision']):
            raise ValueError('model revision must be immutable commit SHA')
        self.kind = self.spec['kind']
        if self.kind == 'clip':
            from transformers import CLIPModel, CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(self.spec['id'], revision=self.spec['revision'], cache_dir=cache)
            self.model = CLIPModel.from_pretrained(self.spec['id'], revision=self.spec['revision'], cache_dir=cache).to(device).eval()
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.spec['id'], revision=self.spec['revision'], cache_folder=cache, device=device)
            self.model.max_seq_length = self.spec['max_tokens']
        if device.startswith('cuda'):
            self.model.half()

    def encode(self, raw, role='corpus', category='background'):
        if role not in ('corpus', 'query'):
            raise ValueError('embedding role')
        if not raw:
            raise ValueError('empty inference batch')
        output = []
        for start in range(0, len(raw), self.batch_size):
            batch = raw[start:start + self.batch_size]
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()
            t = time.perf_counter()
            with self.torch.inference_mode():
                if self.kind == 'clip':
                    if role == 'corpus':
                        from PIL import Image
                        # Explicit local immutable images; processor implements the
                        # pinned CLIP resize/center-crop/RGB normalization pipeline.
                        images = []
                        for path in batch:
                            if isinstance(path, dict):
                                import base64
                                import hashlib
                                import io
                                data = base64.b64decode(path['image_base64'], validate=True)
                                if hashlib.sha256(data).hexdigest() != path['sha256']:
                                    raise ValueError('RPC image checksum mismatch')
                                path = io.BytesIO(data)
                            with Image.open(path) as im:
                                images.append(im.convert('RGB').copy())
                        inputs = self.processor(images=images, return_tensors='pt').to(self.device)
                        value = self.model.get_image_features(**inputs)
                    else:
                        inputs = self.processor(text=batch, padding=True, truncation=True,
                                                max_length=self.spec['max_tokens'], return_tensors='pt').to(self.device)
                        value = self.model.get_text_features(**inputs)
                    value = value.float().cpu().numpy()
                else:
                    if self.kind == 'e5':
                        prefix = 'query: ' if role == 'query' else 'passage: '
                        batch = [prefix + x for x in batch]
                    value = self.model.encode(batch, batch_size=len(batch), convert_to_numpy=True,
                                              normalize_embeddings=False, show_progress_bar=False)
            if self.device.startswith('cuda'):
                self.torch.cuda.synchronize()
            seconds = time.perf_counter() - t
            if self.account:
                self.account(category, seconds, len(batch), self.device)
            output.append(normalize(value))
        return np.concatenate(output)


class GloveEncoder:
    """Averaged GloVe 6B.300d, explicit local asset hash checked by caller.

    Reconstruction preprocessing: lowercase Unicode word tokens; skip OOV;
    all-OOV inputs are rejected, not mapped to fabricated unit vectors.
    """
    def __init__(self, path, wanted_text):
        from .datasets import sha256
        self.spec = dict(id='GloVe-6B.300d-averaged', sha256=sha256(path),
                         preprocessing='lowercase_unicode_word_tokens_skip_oov_reject_all_oov')
        wanted = {w for t in wanted_text for w in re.findall(r'\w+', t.lower())}
        self.table = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                word, *values = line.split()
                if word in wanted:
                    v = np.asarray(values, dtype='float32')
                    if v.shape != (300,):
                        raise ValueError('GloVe must have 300 dimensions')
                    self.table[word] = v

    def encode(self, raw, role='corpus', category='initial'):
        output = []
        for text in raw:
            found = [self.table[w] for w in re.findall(r'\w+', text.lower()) if w in self.table]
            if not found:
                raise ValueError('all-OOV GloVe item')
            output.append(np.mean(found, axis=0))
        return normalize(output)
