import torch
import torch.nn as nn

from .modules.positional_encoding import PositionalEncoding, TimestepEmbedder
from einops import repeat
from src.bptools import get_indexes_body_parts


def _masked_mean(x, mask, dim):
    mask = mask.to(dtype=x.dtype)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return (x * mask).sum(dim=dim) / denom


class TransformerDenoiser(nn.Module):
    name = "transformer"

    def __init__(
        self,
        nfeats: int,
        tx_dim: int,
        latent_dim: int = 512,
        noutfeats: int = 205,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        nb_registers: int = 2, 
        activation: str = "gelu",
    ):
        super().__init__()

        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.nb_registers = nb_registers
        self.tx_dim = tx_dim

        if tx_dim != 0:
            # Linear layer for the condition
            self.tx_embedding = nn.Sequential(
                nn.Linear(tx_dim, 2 * latent_dim),
                nn.GELU(),
                nn.Linear(2 * latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )
        else: 
            self.tx_embedding = None

        # Linear layer for the skeletons
        self.skel_embedding = nn.Linear(nfeats, latent_dim)

        # register for aggregating info
        if nb_registers > 0:
            self.registers = nn.Parameter(torch.randn(nb_registers, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout, batch_first=True
        )

        # MLP for the timesteps
        self.timestep_encoder = TimestepEmbedder(latent_dim, self.sequence_pos_encoding)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            norm_first=True,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

        # Final layer to go back to skeletons
        self.to_skel_layer = nn.Linear(latent_dim, nfeats)

    def forward(self, x, y, t, tf = None):
        device = x.device
        x_mask = y["mask"].to(device)
        bs, nframes, nfeats = x.shape

        # Time embedding
        time_emb = self.timestep_encoder(t)
        time_mask = torch.ones(bs, dtype=bool, device=device)
        # put all the additionnal here
        info_emb = time_emb[:, None]
        info_mask = time_mask[:, None]

        if tf is not None:
            time_f_emb = self.timestep_encoder(tf)
            time_f_mask = torch.ones(bs, dtype=bool, device=device)
            info_emb = torch.cat((info_emb, time_f_emb[:, None]), 1)
            info_mask = torch.cat((info_mask, time_f_mask[:, None]), 1)


        if "tx" in y:
            assert "tx" in y

            # Condition part (can be text/action etc)
            tx_x = y["tx"]["x"]
            tx_mask = y["tx"]["mask"]

            tx_emb = self.tx_embedding(tx_x)

            info_emb = torch.cat((info_emb, tx_emb), 1)
            info_mask = torch.cat((info_mask, tx_mask), 1)

        # add registers
        if self.nb_registers > 0:
            registers = repeat(self.registers, "nbtoken dim -> bs nbtoken dim", bs=bs)
            registers_mask = torch.ones(
                (bs, self.nb_registers), dtype=bool, device=device
            )
            # add the register
            info_emb = torch.cat((info_emb, registers), 1)
            info_mask = torch.cat((info_mask, registers_mask), 1)

        x = self.skel_embedding(x)
        number_of_info = info_emb.shape[1]

        # adding the embedding token for all sequences
        xseq = torch.cat((info_emb, x), 1)

        # add positional encoding to all the tokens
        xseq = self.sequence_pos_encoding(xseq)

        # create a bigger mask, to allow attend to time and condition as well
        aug_mask = torch.cat((info_mask, x_mask), 1)

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        # extract the important part
        output = self.to_skel_layer(final[:, number_of_info:])
        return output


class ResidualTCN(nn.Module):
    def __init__(self, latent_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        residual = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + residual)


class LGTransformerDenoiser(nn.Module):
    name = "lg_transformer"

    def __init__(
        self,
        nfeats: int,
        tx_dim: int,
        latent_dim: int = 512,
        motion_nfeats: int = 205,
        text_part_dim: int = 51,
        text_part_names=(
            "action",
            "trajectory",
            "spine",
            "head",
            "left_arm",
            "right_arm",
            "left_leg",
            "right_leg",
        ),
        motion_part_names=(
            "trajectory",
            "spine",
            "head",
            "left_arm",
            "right_arm",
            "left_leg",
            "right_leg",
        ),
        motion_featsname: str = "smplrifke",
        ff_size: int = 2048,
        num_layers: int = 4,
        local_layers: int = 2,
        global_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        coord_gate_init: float = -4.0,
        refine_gate_init: float = -4.0,
        use_temporal_refine: bool = True,
    ):
        super().__init__()

        self.nfeats = nfeats
        self.motion_nfeats = motion_nfeats
        self.text_nfeats = nfeats - motion_nfeats
        self.text_part_dim = text_part_dim
        self.text_part_names = list(text_part_names)
        self.motion_part_names = list(motion_part_names)
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.tx_dim = tx_dim
        self.use_temporal_refine = use_temporal_refine

        expected_text_dim = len(self.text_part_names) * text_part_dim
        if self.text_nfeats != expected_text_dim:
            raise ValueError(
                f"Expected {expected_text_dim} text features from "
                f"{len(self.text_part_names)} parts x {text_part_dim}, "
                f"but nfeats-motion_nfeats is {self.text_nfeats}."
            )

        if tx_dim != 0:
            self.tx_embedding = nn.Sequential(
                nn.Linear(tx_dim, 2 * latent_dim),
                nn.GELU(),
                nn.Linear(2 * latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )
        else:
            self.tx_embedding = None

        self.motion_global_embedding = nn.Linear(motion_nfeats, latent_dim)
        self.action_embedding = nn.Linear(text_part_dim, latent_dim)
        self.part_text_embedding = nn.Sequential(
            nn.Linear(text_part_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        indexes_bp = get_indexes_body_parts(motion_featsname)
        self.motion_part_embeddings = nn.ModuleList()
        for idx, part_name in enumerate(self.motion_part_names):
            key = part_name.replace("_", " ")
            if key not in indexes_bp:
                raise ValueError(f"Unknown motion body part {part_name!r}")
            indexes = torch.as_tensor(indexes_bp[key], dtype=torch.long)
            self.register_buffer(f"motion_part_index_{idx}", indexes, persistent=False)
            self.motion_part_embeddings.append(nn.Linear(len(indexes), latent_dim))

        self.part_id_embedding = nn.Embedding(len(self.motion_part_names), latent_dim)
        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout, batch_first=True
        )
        self.timestep_encoder = TimestepEmbedder(latent_dim, self.sequence_pos_encoding)

        local_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            norm_first=True,
            activation=activation,
            batch_first=True,
        )
        self.local_temporal_encoder = nn.TransformerEncoder(
            local_layer, num_layers=local_layers
        )

        part_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            norm_first=True,
            activation=activation,
            batch_first=True,
        )
        self.part_interaction_encoder = nn.TransformerEncoder(part_layer, num_layers=1)

        global_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            norm_first=True,
            activation=activation,
            batch_first=True,
        )
        self.global_temporal_encoder = nn.TransformerEncoder(
            global_layer, num_layers=global_layers
        )

        self.part_to_global_attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.global_to_part_attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.coord_projection = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.coord_gate = nn.Parameter(torch.tensor(float(coord_gate_init)))

        self.base_motion_head = nn.Linear(latent_dim, motion_nfeats)
        self.text_head = nn.Linear(latent_dim, self.text_nfeats)
        self.refine_tcn = ResidualTCN(latent_dim, dropout=dropout)
        self.refine_motion_head = nn.Linear(latent_dim, motion_nfeats)
        self.refine_gate = nn.Parameter(torch.tensor(float(refine_gate_init)))

    def _split_input(self, x):
        motion = x[..., : self.motion_nfeats]
        text = x[..., self.motion_nfeats :]
        bs, nframes, _ = x.shape
        text = text.reshape(
            bs, nframes, len(self.text_part_names), self.text_part_dim
        )
        action_text = text[:, :, 0]
        part_text = text[:, :, 1:]
        return motion, action_text, part_text

    def _embed_motion_parts(self, motion):
        parts = []
        for idx, layer in enumerate(self.motion_part_embeddings):
            indexes = getattr(self, f"motion_part_index_{idx}")
            parts.append(layer(motion.index_select(-1, indexes)))
        return torch.stack(parts, dim=2)

    def _embed_sequence_condition(self, y, device, bs):
        if self.tx_embedding is None or "tx" not in y:
            return torch.zeros(bs, self.latent_dim, device=device)

        tx_x = y["tx"]["x"]
        tx_mask = y["tx"]["mask"].to(device)
        tx_emb = self.tx_embedding(tx_x)
        return _masked_mean(tx_emb, tx_mask, dim=1)

    def forward(self, x, y, t, tf=None):
        device = x.device
        x_mask = y["mask"].to(device)
        bs, nframes, _ = x.shape

        motion, action_text, part_text = self._split_input(x)
        time_emb = self.timestep_encoder(t)
        if tf is not None:
            time_emb = time_emb + self.timestep_encoder(tf)

        c_seq = self._embed_sequence_condition(y, device, bs)
        c_action = self.action_embedding(action_text)
        c_part = self.part_text_embedding(part_text)

        h_part = self._embed_motion_parts(motion)
        part_ids = torch.arange(
            len(self.motion_part_names), device=device, dtype=torch.long
        )
        h_part = h_part + c_part
        h_part = h_part + self.part_id_embedding(part_ids)[None, None]
        h_part = h_part + time_emb[:, None, None]

        n_parts = h_part.shape[2]
        local_tokens = h_part.permute(0, 2, 1, 3).reshape(bs * n_parts, nframes, -1)
        local_tokens = self.sequence_pos_encoding(local_tokens)
        local_mask = x_mask[:, None].expand(bs, n_parts, nframes).reshape(
            bs * n_parts, nframes
        )
        local_tokens = self.local_temporal_encoder(
            local_tokens, src_key_padding_mask=~local_mask
        )
        local_tokens = local_tokens.reshape(bs, n_parts, nframes, -1).permute(
            0, 2, 1, 3
        )

        part_mix_tokens = local_tokens.reshape(bs * nframes, n_parts, -1)
        part_mix_tokens = self.part_interaction_encoder(part_mix_tokens)
        local_tokens = part_mix_tokens.reshape(bs, nframes, n_parts, -1)

        global_tokens = self.motion_global_embedding(motion)
        global_tokens = global_tokens + c_action + c_seq[:, None] + time_emb[:, None]
        global_tokens = self.sequence_pos_encoding(global_tokens)
        global_tokens = self.global_temporal_encoder(
            global_tokens, src_key_padding_mask=~x_mask
        )

        local_flat = local_tokens.reshape(bs, nframes * n_parts, -1)
        local_key_mask = x_mask[:, :, None].expand(bs, nframes, n_parts).reshape(
            bs, nframes * n_parts
        )
        global_delta, _ = self.part_to_global_attn(
            query=global_tokens,
            key=local_flat,
            value=local_flat,
            key_padding_mask=~local_key_mask,
            need_weights=False,
        )
        global_enhanced = global_tokens + global_delta

        part_delta, _ = self.global_to_part_attn(
            query=local_flat,
            key=global_tokens,
            value=global_tokens,
            key_padding_mask=~x_mask,
            need_weights=False,
        )
        part_enhanced = (local_flat + part_delta).reshape(bs, nframes, n_parts, -1)
        part_summary = part_enhanced.mean(dim=2)

        coord_delta = self.coord_projection(
            torch.cat((global_enhanced, part_summary), dim=-1)
        )
        coord_gate = torch.sigmoid(self.coord_gate)
        coord_features = global_tokens + coord_gate * coord_delta

        motion_base = self.base_motion_head(coord_features)
        if self.use_temporal_refine:
            refine_features = self.refine_tcn(coord_features)
            motion_delta = self.refine_motion_head(refine_features)
            motion_output = motion_base + torch.sigmoid(self.refine_gate) * motion_delta
        else:
            motion_output = motion_base

        text_output = self.text_head(coord_features)
        return torch.cat((motion_output, text_output), dim=-1)
