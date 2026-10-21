import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DynamicKinematicGraphBlock(nn.Module):
    """Text-conditioned message passing over body parts and a support-state node."""

    def __init__(
        self,
        latent_dim,
        num_heads,
        dropout,
        static_adjacency,
        topk,
        use_contact_balance_state=False,
    ):
        super().__init__()
        self.topk = topk
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.condition = nn.Linear(latent_dim, latent_dim)
        self.global_context = nn.Linear(latent_dim, latent_dim)
        self.relation_norm = nn.LayerNorm(latent_dim)
        self.message_projection = nn.Linear(latent_dim, latent_dim)
        self.part_norm = nn.LayerNorm(latent_dim)
        self.part_ffn = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * latent_dim, latent_dim),
        )
        self.part_ffn_norm = nn.LayerNorm(latent_dim)
        self.global_read = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.global_norm = nn.LayerNorm(latent_dim)
        self.global_to_part = nn.Linear(latent_dim, latent_dim)
        # Start the full graph residual almost closed. This prevents a randomly
        # initialized message path from immediately perturbing the LG-FM trunk.
        self.residual_gate = nn.Parameter(torch.tensor(-4.0))
        self.global_gate = nn.Parameter(torch.tensor(-2.0))
        self.part_gate = nn.Parameter(torch.tensor(-2.0))
        self.use_contact_balance_state = use_contact_balance_state

        if self.use_contact_balance_state:
            self.state_context = nn.Linear(latent_dim, latent_dim)
            self.root_state_context = nn.Linear(latent_dim, latent_dim)
            self.contact_state_context = nn.Linear(2, latent_dim)
            self.balance_state_context = nn.Linear(2, latent_dim)
            self.state_read = nn.MultiheadAttention(
                latent_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.state_norm = nn.LayerNorm(latent_dim)
            self.state_to_global = nn.Linear(latent_dim, latent_dim)
            self.state_to_part = nn.Linear(latent_dim, latent_dim)
            self.state_global_norm = nn.LayerNorm(latent_dim)
            self.state_part_norm = nn.LayerNorm(latent_dim)
            self.state_gate = nn.Parameter(torch.tensor(-2.0))
            self.state_global_gate = nn.Parameter(torch.tensor(-2.0))
            self.state_part_gate = nn.Parameter(torch.tensor(-2.0))

        adjacency = torch.as_tensor(static_adjacency, dtype=torch.bool)
        # Non-kinematic edges can still be selected when text/motion demands it,
        # but start at a lower prior than direct kinematic or bilateral links.
        self.register_buffer("static_adjacency", adjacency, persistent=False)
        self.static_edge_bias = nn.Parameter(torch.tensor(1.0))
        self.non_kinematic_bias = nn.Parameter(torch.tensor(-1.0))

    def forward(
        self,
        part_tokens,
        global_tokens,
        part_conditions,
        frame_mask,
        support_state=None,
        root_state=None,
        contact_state=None,
        balance_state=None,
        state_feedback_scale=None,
        state_feedback_gain_cap=1.0,
        dynamic_relation_scale=None,
        graph_residual_scale=None,
        graph_residual_gain_cap=1.0,
    ):
        """Return updated part/global/state tokens and frame-wise relation weights."""
        batch_size, nframes, n_parts, latent_dim = part_tokens.shape
        if state_feedback_scale is None:
            state_feedback_scale = part_tokens.new_ones(batch_size, 1, 1)
        else:
            state_feedback_scale = state_feedback_scale.to(
                device=part_tokens.device, dtype=part_tokens.dtype
            ).reshape(batch_size, 1, 1)
        state_feedback_scale_4d = state_feedback_scale.unsqueeze(-1)
        state_feedback_gain_cap = min(max(float(state_feedback_gain_cap), 0.0), 1.0)
        if dynamic_relation_scale is None:
            dynamic_relation_scale = part_tokens.new_ones(batch_size, 1, 1, 1)
        else:
            dynamic_relation_scale = dynamic_relation_scale.to(
                device=part_tokens.device, dtype=part_tokens.dtype
            ).reshape(batch_size, 1, 1, 1)
        if graph_residual_scale is None:
            graph_residual_scale = part_tokens.new_ones(batch_size, 1, 1)
        else:
            graph_residual_scale = graph_residual_scale.to(
                device=part_tokens.device, dtype=part_tokens.dtype
            ).reshape(batch_size, 1, 1)
        graph_residual_gain_cap = min(max(float(graph_residual_gain_cap), 0.0), 1.0)
        graph_residual_gain = (
            graph_residual_scale
            * graph_residual_gain_cap
            * torch.sigmoid(self.residual_gate)
        )
        graph_residual_gain_4d = graph_residual_gain.unsqueeze(-1)
        state_input_scale = None
        state_input_scale_4d = None
        if support_state is not None:
            # The state route is a residual control signal, never an unrestricted
            # replacement for the motion/text token path.
            state_input_scale = (
                state_feedback_scale
                * state_feedback_gain_cap
                * torch.sigmoid(self.state_gate)
            )
            state_input_scale_4d = state_input_scale.unsqueeze(-1)
        source = (
            part_tokens
            + self.condition(part_conditions)
            + self.global_context(global_tokens)[:, :, None]
        )
        if support_state is not None:
            source = source + state_input_scale_4d * self.state_context(support_state)[:, :, None]
        if root_state is not None:
            source = source + state_input_scale_4d * self.root_state_context(root_state)[:, :, None]
        if contact_state is not None:
            source = source + state_input_scale_4d * self.contact_state_context(contact_state)[:, :, None]
        if balance_state is not None:
            source = source + state_input_scale_4d * self.balance_state_context(balance_state)[:, :, None]
        # Keep graph routing in fp32 even under mixed precision. The dot-product
        # logits are the numerically sensitive part of this block; clamping only
        # affects already-saturated softmax probabilities.
        with torch.autocast(device_type=part_tokens.device.type, enabled=False):
            source_fp32 = self.relation_norm(source.float())
            query = self.query(source_fp32)
            key = self.key(source_fp32)
            value = self.value(part_tokens.float())
            logits = torch.einsum("btpd,btqd->btpq", query, key) / (latent_dim ** 0.5)
            edge_bias = torch.where(
                self.static_adjacency,
                self.static_edge_bias,
                self.non_kinematic_bias,
            ).float()
            logits = (logits + edge_bias[None, None]).clamp_(-30.0, 30.0)

            # The static kinematic graph is always available. At noisy diffusion
            # steps the learned router is blended out entirely; this prevents
            # unstable edges inferred from x_t from being propagated globally.
            static_allowed_mask = self.static_adjacency[None, None].expand_as(logits)
            static_logits = (edge_bias[None, None]).expand_as(logits).masked_fill(
                ~static_allowed_mask, torch.finfo(logits.dtype).min
            )
            static_weights = static_logits.softmax(dim=-1)

            # The learned router may add non-kinematic edges, but never removes
            # a fixed chain or bilateral relation.
            relation_allowed_mask = static_allowed_mask
            if self.topk:
                dynamic_logits = logits.masked_fill(
                    static_allowed_mask, torch.finfo(logits.dtype).min
                )
                dynamic_topk = min(self.topk, n_parts)
                dynamic_idx = dynamic_logits.topk(dynamic_topk, dim=-1).indices
                dynamic_mask = torch.zeros_like(logits, dtype=torch.bool)
                dynamic_mask.scatter_(-1, dynamic_idx, True)
                relation_allowed_mask = relation_allowed_mask | dynamic_mask
            dynamic_logits = logits.masked_fill(
                ~relation_allowed_mask, torch.finfo(logits.dtype).min
            )
            dynamic_weights = dynamic_logits.softmax(dim=-1)
            relation_weights = torch.lerp(
                static_weights, dynamic_weights, dynamic_relation_scale.float()
            )
            relation_weights = relation_weights * frame_mask[:, :, None, None].to(logits.dtype)
            messages = torch.einsum("btpq,btqd->btpd", relation_weights, value)
        messages = messages.to(dtype=part_tokens.dtype)
        proposed_parts = self.part_norm(part_tokens + self.message_projection(messages))
        parts = part_tokens + graph_residual_gain_4d * (proposed_parts - part_tokens)
        proposed_parts = self.part_ffn_norm(parts + self.part_ffn(parts))
        parts = parts + graph_residual_gain_4d * (proposed_parts - parts)

        parts_flat = parts.reshape(batch_size * nframes, n_parts, latent_dim)
        global_flat = global_tokens.reshape(batch_size * nframes, 1, latent_dim)
        part_mask = frame_mask.reshape(batch_size * nframes, 1).expand(-1, n_parts)
        global_delta, _ = self.global_read(
            global_flat,
            parts_flat,
            parts_flat,
            key_padding_mask=~part_mask,
            need_weights=False,
        )
        global_delta = global_delta.reshape(batch_size, nframes, latent_dim)
        proposed_global_tokens = self.global_norm(
            global_tokens + torch.sigmoid(self.global_gate) * global_delta
        )
        global_tokens = global_tokens + graph_residual_gain * (
            proposed_global_tokens - global_tokens
        )
        parts = parts + graph_residual_gain_4d * torch.sigmoid(self.part_gate) * (
            self.global_to_part(global_tokens)[:, :, None]
        )
        if support_state is not None:
            state_flat = support_state.reshape(batch_size * nframes, 1, latent_dim)
            state_delta, _ = self.state_read(
                state_flat,
                parts_flat,
                parts_flat,
                key_padding_mask=~part_mask,
                need_weights=False,
            )
            state_delta = state_delta.reshape(batch_size, nframes, latent_dim)
            support_state = self.state_norm(
                support_state
                + state_input_scale
                * state_delta
            )
            global_tokens = self.state_global_norm(
                global_tokens
                + state_feedback_scale
                * state_feedback_gain_cap
                * torch.sigmoid(self.state_global_gate)
                * self.state_to_global(support_state)
            )
            parts = self.state_part_norm(
                parts
                + state_feedback_scale_4d
                * state_feedback_gain_cap
                * torch.sigmoid(self.state_part_gate)
                * self.state_to_part(support_state)[:, :, None]
            )
        return parts, global_tokens, relation_weights, support_state, relation_allowed_mask


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
        use_local_branch: bool = True,
        use_global_branch: bool = True,
        use_local_global_fusion: bool = True,
        coord_gate_init: float = -4.0,
        refine_gate_init: float = -4.0,
        use_temporal_refine: bool = True,
        use_dynamic_kinematic_graph: bool = False,
        dynamic_graph_layers: int = 2,
        dynamic_graph_topk: int = 3,
        dynamic_graph_timesteps: int = 100,
        use_contact_balance_state: bool = False,
        state_feedback_gain_cap: float = 1.0,
        dynamic_relation_gain_cap: float = 1.0,
        dynamic_relation_static_start_timestep_ratio: float = 0.75,
        graph_residual_gain_cap: float = 1.0,
        use_part_specific_decoder: bool = False,
        part_decoder_gate_init: float = -4.0,
        cross_attention_norm: str = "none",
    ):
        super().__init__()
        if cross_attention_norm not in ("none", "qk_g2p", "qk_p2g", "qk_both", "qkv_both"):
            raise ValueError(f"Unknown cross_attention_norm: {cross_attention_norm}")
        self.cross_attention_norm = cross_attention_norm
        self.dynamic_graph_timesteps = max(2, int(dynamic_graph_timesteps))

        self.nfeats = nfeats
        self.motion_nfeats = motion_nfeats
        self.text_nfeats = nfeats - motion_nfeats
        self.text_part_dim = text_part_dim
        self.text_part_names = list(text_part_names)
        self.motion_part_names = list(motion_part_names)
        self.motion_featsname = motion_featsname
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.tx_dim = tx_dim
        self.use_local_branch = use_local_branch
        self.use_global_branch = use_global_branch
        self.use_local_global_fusion = use_local_global_fusion
        self.use_temporal_refine = use_temporal_refine
        self.use_dynamic_kinematic_graph = use_dynamic_kinematic_graph
        self.use_contact_balance_state = use_contact_balance_state
        self.state_feedback_gain_cap = min(max(float(state_feedback_gain_cap), 0.0), 1.0)
        self.dynamic_relation_gain_cap = min(
            max(float(dynamic_relation_gain_cap), 0.0), 1.0
        )
        self.dynamic_relation_static_start_timestep_ratio = min(
            max(float(dynamic_relation_static_start_timestep_ratio), 0.0), 1.0
        )
        self.graph_residual_gain_cap = min(
            max(float(graph_residual_gain_cap), 0.0), 1.0
        )
        # A resumed checkpoint can retain its original routing initially, then
        # transition to the high-noise static graph without an abrupt shift.
        self.dynamic_relation_transition_progress = 1.0
        self.use_part_specific_decoder = use_part_specific_decoder

        if not self.use_local_branch and not self.use_global_branch:
            raise ValueError(
                "LGTransformerDenoiser needs at least one enabled branch: "
                "use_local_branch or use_global_branch."
            )
        if self.use_dynamic_kinematic_graph and not (
            self.use_local_branch and self.use_global_branch
        ):
            raise ValueError(
                "Dynamic kinematic coordination requires both local and global branches."
            )
        if self.use_dynamic_kinematic_graph and dynamic_graph_layers < 1:
            raise ValueError("dynamic_graph_layers must be at least 1 when enabled.")
        if self.use_contact_balance_state and not self.use_dynamic_kinematic_graph:
            raise ValueError(
                "Contact/balance state requires use_dynamic_kinematic_graph=true."
            )
        if self.use_contact_balance_state and "trajectory" not in self.motion_part_names:
            raise ValueError("Contact/balance state requires a trajectory motion part.")

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

        if self.use_dynamic_kinematic_graph:
            adjacency = self._build_kinematic_adjacency()
            self.dynamic_graph_blocks = nn.ModuleList(
                DynamicKinematicGraphBlock(
                    latent_dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    static_adjacency=adjacency,
                    topk=dynamic_graph_topk,
                    use_contact_balance_state=use_contact_balance_state,
                )
                for _ in range(dynamic_graph_layers)
            )
            if self.use_contact_balance_state:
                self.contact_balance_state_init = nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.GELU(),
                    nn.LayerNorm(latent_dim),
                )
                self.trajectory_part_index = self.motion_part_names.index("trajectory")
                self.root_state_projection = nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.GELU(),
                    nn.LayerNorm(latent_dim),
                )
                self.contact_state_head = nn.Linear(latent_dim, 2)
                self.balance_state_head = nn.Linear(latent_dim, 2)

        if self.use_part_specific_decoder:
            self.part_motion_heads = nn.ModuleList(
                nn.Linear(latent_dim, len(getattr(self, f"motion_part_index_{idx}")))
                for idx in range(len(self.motion_part_names))
            )
            self.part_decoder_gate = nn.Parameter(
                torch.full((len(self.motion_part_names),), float(part_decoder_gate_init))
            )

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

    def set_dynamic_relation_transition_progress(self, progress: float):
        self.dynamic_relation_transition_progress = min(max(float(progress), 0.0), 1.0)

    def _embed_motion_parts(self, motion):
        parts = []
        for idx, layer in enumerate(self.motion_part_embeddings):
            indexes = getattr(self, f"motion_part_index_{idx}")
            parts.append(layer(motion.index_select(-1, indexes)))
        return torch.stack(parts, dim=2)

    def _build_kinematic_adjacency(self):
        """Static graph prior: hierarchy plus bilateral coordination edges."""
        n_parts = len(self.motion_part_names)
        adjacency = torch.eye(n_parts, dtype=torch.bool)
        links = [
            ("trajectory", "spine"),
            ("trajectory", "left_leg"),
            ("trajectory", "right_leg"),
            ("spine", "head"),
            ("spine", "left_arm"),
            ("spine", "right_arm"),
            ("left_leg", "right_leg"),
            ("left_arm", "right_arm"),
        ]
        for first, second in links:
            if first in self.motion_part_names and second in self.motion_part_names:
                i = self.motion_part_names.index(first)
                j = self.motion_part_names.index(second)
                adjacency[i, j] = True
                adjacency[j, i] = True
        return adjacency

    def _embed_sequence_condition(self, y, device, bs):
        if self.tx_embedding is None or "tx" not in y:
            return torch.zeros(bs, self.latent_dim, device=device)

        tx_x = y["tx"]["x"]
        tx_mask = y["tx"]["mask"].to(device)
        tx_emb = self.tx_embedding(tx_x)
        return _masked_mean(tx_emb, tx_mask, dim=1)

    supports_part_alignment_aux = True
    supports_dynamic_coordination_aux = True

    def _cross_attention_inputs(self, query, key, value, direction):
        mode = self.cross_attention_norm
        if mode not in ("qk_both", "qkv_both", f"qk_{direction}"):
            return query, key, value
        def normalize(tensor):
            return F.layer_norm(
                tensor.float(), (tensor.shape[-1],), eps=1e-5
            ).to(tensor.dtype)
        return normalize(query), normalize(key), normalize(value) if mode == "qkv_both" else value

    def forward(self, x, y, t, tf=None, return_aux: bool = False):
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

        local_tokens = None
        if self.use_local_branch:
            h_part = self._embed_motion_parts(motion)
            part_ids = torch.arange(
                len(self.motion_part_names), device=device, dtype=torch.long
            )
            h_part = h_part + c_part
            h_part = h_part + self.part_id_embedding(part_ids)[None, None]
            h_part = h_part + time_emb[:, None, None]

            n_parts = h_part.shape[2]
            local_tokens = h_part.permute(0, 2, 1, 3).reshape(
                bs * n_parts, nframes, -1
            )
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

        global_tokens = None
        if self.use_global_branch:
            global_tokens = self.motion_global_embedding(motion)
            global_tokens = global_tokens + c_action + c_seq[:, None] + time_emb[:, None]
            global_tokens = self.sequence_pos_encoding(global_tokens)
            global_tokens = self.global_temporal_encoder(
                global_tokens, src_key_padding_mask=~x_mask
            )

        part_motion_tokens = local_tokens
        if self.use_local_branch and self.use_global_branch:
            n_parts = local_tokens.shape[2]
            local_flat = local_tokens.reshape(bs, nframes * n_parts, -1)

            if self.use_local_global_fusion:
                local_key_mask = x_mask[:, :, None].expand(
                    bs, nframes, n_parts
                ).reshape(bs, nframes * n_parts)
                p2g_query, p2g_key, p2g_value = self._cross_attention_inputs(
                    global_tokens, local_flat, local_flat, "p2g"
                )
                global_delta, _ = self.part_to_global_attn(
                    query=p2g_query,
                    key=p2g_key,
                    value=p2g_value,
                    key_padding_mask=~local_key_mask,
                    need_weights=False,
                )
                global_enhanced = global_tokens + global_delta

                g2p_query, g2p_key, g2p_value = self._cross_attention_inputs(
                    local_flat, global_tokens, global_tokens, "g2p"
                )
                part_delta, _ = self.global_to_part_attn(
                    query=g2p_query,
                    key=g2p_key,
                    value=g2p_value,
                    key_padding_mask=~x_mask,
                    need_weights=False,
                )
                part_enhanced = (local_flat + part_delta).reshape(
                    bs, nframes, n_parts, -1
                )
                part_summary = part_enhanced.mean(dim=2)
                part_motion_tokens = part_enhanced
            else:
                global_enhanced = global_tokens
                part_summary = local_tokens.mean(dim=2)

            coord_delta = self.coord_projection(
                torch.cat((global_enhanced, part_summary), dim=-1)
            )
            coord_gate = torch.sigmoid(self.coord_gate)
            coord_features = global_tokens + coord_gate * coord_delta
        elif self.use_global_branch:
            coord_features = global_tokens
        else:
            coord_features = local_tokens.mean(dim=2)

        relation_weights = None
        relation_allowed_mask = None
        support_state = None
        root_state = None
        contact_state_logits = None
        balance_state = None
        state_confidence = None
        if self.use_dynamic_kinematic_graph:
            # Dynamic state is least reliable at the noisiest diffusion steps.
            # The fixed graph path remains active; only state-conditioned routing
            # and feedback are faded out there.
            state_confidence = (
                1.0
                - t.float().reshape(bs)
                / float(self.dynamic_graph_timesteps - 1)
            ).clamp_(0.0, 1.0)
            state_feedback_scale = state_confidence[:, None, None]
            noise_ratio = 1.0 - state_confidence
            static_start = self.dynamic_relation_static_start_timestep_ratio
            if static_start >= 1.0:
                dynamic_relation_scale = torch.ones_like(state_confidence)
            else:
                dynamic_relation_scale = (
                    1.0 - (noise_ratio - static_start) / (1.0 - static_start)
                ).clamp_(0.0, 1.0)
            dynamic_relation_scale = (
                dynamic_relation_scale[:, None, None, None]
                * self.dynamic_relation_gain_cap
            )
            dynamic_relation_scale = 1.0 + self.dynamic_relation_transition_progress * (
                dynamic_relation_scale - 1.0
            )
            if self.use_contact_balance_state:
                support_state = self.contact_balance_state_init(coord_features)
                root_state = self.root_state_projection(
                    part_motion_tokens[:, :, self.trajectory_part_index]
                )
                contact_state_logits = self.contact_state_head(support_state)
                balance_state = torch.tanh(self.balance_state_head(support_state))
            for graph_block in self.dynamic_graph_blocks:
                (
                    part_motion_tokens,
                    coord_features,
                    relation_weights,
                    support_state,
                    relation_allowed_mask,
                ) = graph_block(
                    part_motion_tokens,
                    coord_features,
                    c_part,
                    x_mask,
                    support_state=support_state,
                    root_state=root_state,
                    contact_state=(
                        torch.sigmoid(contact_state_logits).detach()
                        if contact_state_logits is not None
                        else None
                    ),
                    balance_state=(
                        balance_state.detach()
                        if balance_state is not None
                        else None
                    ),
                    state_feedback_scale=state_feedback_scale,
                    state_feedback_gain_cap=self.state_feedback_gain_cap,
                    dynamic_relation_scale=dynamic_relation_scale,
                    graph_residual_scale=state_feedback_scale,
                    graph_residual_gain_cap=self.graph_residual_gain_cap,
                )
                if support_state is not None:
                    root_state = self.root_state_projection(
                        part_motion_tokens[:, :, self.trajectory_part_index]
                    )
                    contact_state_logits = self.contact_state_head(support_state)
                    balance_state = torch.tanh(self.balance_state_head(support_state))

        motion_base = self.base_motion_head(coord_features)
        if self.use_temporal_refine:
            refine_features = self.refine_tcn(coord_features)
            motion_delta = self.refine_motion_head(refine_features)
            motion_output = motion_base + torch.sigmoid(self.refine_gate) * motion_delta
        else:
            motion_output = motion_base

        if self.use_part_specific_decoder:
            part_residual = torch.zeros_like(motion_output)
            for idx, part_head in enumerate(self.part_motion_heads):
                indexes = getattr(self, f"motion_part_index_{idx}")
                values = part_head(part_motion_tokens[:, :, idx])
                part_residual[..., indexes] = values
            for idx in range(len(self.motion_part_names)):
                indexes = getattr(self, f"motion_part_index_{idx}")
                motion_output[..., indexes] = (
                    motion_output[..., indexes]
                    + torch.sigmoid(self.part_decoder_gate[idx]) * part_residual[..., indexes]
                )

        text_output = self.text_head(coord_features)
        output = torch.cat((motion_output, text_output), dim=-1)
        if not return_aux:
            return output

        aux = {
            "part_motion_tokens": part_motion_tokens,
            "part_text_tokens": c_part,
            "part_names": self.motion_part_names,
            "relation_weights": relation_weights,
            "relation_mask": x_mask,
            "relation_static_adjacency": (
                self.dynamic_graph_blocks[-1].static_adjacency
                if self.use_dynamic_kinematic_graph
                else None
            ),
            "relation_allowed_mask": relation_allowed_mask,
            "dynamic_relation_scale": (
                dynamic_relation_scale if self.use_dynamic_kinematic_graph else None
            ),
            "state_confidence": state_confidence,
            "root_state": root_state,
            "contact_state_logits": contact_state_logits,
            "balance_state": balance_state,
        }
        return output, aux
