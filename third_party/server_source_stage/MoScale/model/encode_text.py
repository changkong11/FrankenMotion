import hashlib
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5EncoderModel


class T5TextEncoder:
    """Frozen T5 encoder with an optional deterministic on-disk embedding cache."""
    def __init__(self, device, local_files_only, from_pretrained=None,
                 model_max_length=120, cache_dir=None):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained, local_files_only=local_files_only, legacy=False)
        self.model = T5EncoderModel.from_pretrained(
            from_pretrained, local_files_only=local_files_only).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model_max_length = model_max_length
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)

    def _cache_path(self, text):
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return self.cache_dir / f'{digest}.pt'

    def _load_cached(self, text):
        if not self.cache_dir:
            return None
        path = self._cache_path(text)
        if not path.is_file():
            return None
        try:
            item = torch.load(path, map_location='cpu', weights_only=True)
            embedding, mask = item['embedding'], item['mask']
            if embedding.shape != (self.model_max_length, self.model.config.d_model):
                return None
            if mask.shape != (self.model_max_length,):
                return None
            return embedding, mask
        except Exception:
            return None

    def _store_cached(self, text, embedding, mask):
        if not self.cache_dir:
            return
        path = self._cache_path(text)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        torch.save({'embedding': embedding.cpu(), 'mask': mask.cpu()}, tmp)
        os.replace(tmp, path)

    @torch.no_grad()
    def get_text_embeddings(self, texts):
        # Parent training mode must never reactivate dropout in this frozen encoder.
        self.model.eval()
        cached = [self._load_cached(text) for text in texts]
        missing = [i for i, item in enumerate(cached) if item is None]
        if missing:
            missing_texts = [texts[i] for i in missing]
            tokenized = self.tokenizer(
                missing_texts, max_length=self.model_max_length, padding='max_length',
                truncation=True, return_attention_mask=True, add_special_tokens=True,
                return_tensors='pt')
            input_ids = tokenized['input_ids'].to(self.device)
            masks = tokenized['attention_mask'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=masks)['last_hidden_state'].detach()
            for offset, i in enumerate(missing):
                # BF16 is the training precision; caching it avoids changing subsequent forward inputs.
                item = (outputs[offset].to(dtype=torch.bfloat16).cpu(), masks[offset].to(dtype=torch.uint8).cpu())
                self._store_cached(texts[i], *item)
                cached[i] = item
        embeddings = torch.stack([item[0] for item in cached]).to(self.device, non_blocking=True)
        attention_mask = torch.stack([item[1] for item in cached]).to(self.device, non_blocking=True)
        return embeddings, attention_mask