"""Detached, valid-token attention statistics; does not consume random numbers."""
import math
import torch
import torch.nn.functional as F


class CrossAttentionDiagnostics:
    def __init__(self, denoiser, scope):
        self.scope = scope
        self.enabled = False
        self.records = {}
        self.mask = None
        self.handles = [denoiser.register_forward_pre_hook(self.capture_mask, with_kwargs=True)]
        for name in ('part_to_global_attn', 'global_to_part_attn'):
            self.handles.append(getattr(denoiser, name).register_forward_pre_hook(
                self.capture(name), with_kwargs=True, prepend=True))

    def capture_mask(self, module, args, kwargs):
        if self.enabled:
            y = kwargs.get('y', args[1] if len(args) > 1 else None)
            self.mask = y['mask'].detach().bool()

    def capture(self, name):
        def hook(module, args, kwargs):
            if not self.enabled:
                return
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                q, k, v = [kwargs[key].detach().float() for key in ('query', 'key', 'value')]
                kmask = ~kwargs['key_padding_mask'].bool()
                qmask = self.mask
                if name == 'global_to_part_attn':
                    n_parts = q.shape[1] // qmask.shape[1]
                    qmask = qmask[:, :, None].expand(-1, -1, n_parts).reshape(q.shape[:2])
                def stats(x, mask):
                    valid = x[mask]
                    return dict(rms=float(valid.square().mean().sqrt()), absmax=float(valid.abs().max()))
                record = {'raw_query': stats(q, qmask), 'raw_key': stats(k, kmask),
                          'raw_value': stats(v, kmask)}
                scope = self.scope.get(name, 'none') if isinstance(self.scope, dict) else self.scope
                if scope in ('qk', 'qkv'):
                    q = F.layer_norm(q, (q.shape[-1],), eps=1e-5)
                    k = F.layer_norm(k, (k.shape[-1],), eps=1e-5)
                # Sample at most 4 sequences and 8 evenly spaced valid queries;
                # include every valid key, all heads, and the learned projections.
                dim, heads = module.embed_dim, module.num_heads
                w = module.in_proj_weight.detach().float()
                bias = module.in_proj_bias
                bias = bias.detach().float() if bias is not None else None
                samples = []
                for b in range(min(4, q.shape[0])):
                    qi = qmask[b].nonzero().flatten()
                    ki = kmask[b].nonzero().flatten()
                    if not len(qi) or not len(ki):
                        continue
                    qi = qi[torch.linspace(0, len(qi)-1, min(8, len(qi)), device=qi.device).long()]
                    pq = F.linear(q[b, qi], w[:dim], None if bias is None else bias[:dim])
                    pk = F.linear(k[b, ki], w[dim:2*dim], None if bias is None else bias[dim:2*dim])
                    pq = pq.reshape(-1, heads, dim//heads).transpose(0, 1)
                    pk = pk.reshape(-1, heads, dim//heads).transpose(0, 1)
                    logits = (pq @ pk.transpose(-1, -2)) / math.sqrt(dim//heads)
                    probs = logits.softmax(-1)
                    entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
                    samples.append(dict(logit_absmax=float(logits.abs().max()),
                                        logit_rms=float(logits.square().mean().sqrt()),
                                        mean_max_probability=float(probs.max(-1).values.mean()),
                                        normalized_entropy=float(entropy.mean()/math.log(max(2, len(ki))))))
                record['sampled_projected_attention'] = samples
                self.records[name] = record
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
