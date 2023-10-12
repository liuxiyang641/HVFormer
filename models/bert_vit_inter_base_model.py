from typing import Any, Optional, Tuple
import math
import torch.cuda
from transformers.modeling_outputs import TokenClassifierOutput, BaseModelOutput, BaseModelOutputWithPooling
from transformers.activations import ACT2FN
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
)
from torch import nn, Tensor, device


# some function
def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
        device: (:obj:`torch.device`):
            The device of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.long)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def get_head_mask(
        head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
) -> Tensor:
    """
    Prepare the head mask if needed.

    Args:
        head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
        num_hidden_layers (:obj:`int`):
            The number of hidden layers in the model.
        is_attention_chunked: (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the attentions scores are computed by chunks or not.

    Returns:
        :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
        list with :obj:`[None]` for each layer.
    """
    head_mask = [None] * num_hidden_layers

    return head_mask


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))  # [CLS] embedding

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

        self.aux_position_embedding = nn.Embedding(48, self.embed_dim)
        self.register_buffer("aux_position_ids", torch.arange(48).expand((1, -1)))

        self.rcnn_position_embedding = nn.Embedding(12, self.embed_dim)
        self.register_buffer("rcnn_position_ids", torch.arange(12).expand((1, -1)))

    def forward(self, pixel_values, aux_embeddings=None, rcnn_embeddings=None):
        batch_size = pixel_values.shape[0]

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = class_embeds
        if aux_embeddings is not None:
            aux_embeds = []
            for aux_embedding in aux_embeddings:
                aux_embed = self.patch_embedding(aux_embedding)
                aux_embed = aux_embed.flatten(2).transpose(1, 2).flatten(0, 1)  # 3*16, 768 3
                aux_embeds.append(aux_embed)
            aux_embeds = torch.stack(aux_embeds)  # bsz, 48, 768
            aux_embeds = aux_embeds + self.aux_position_embedding(self.aux_position_ids)
            embeddings = torch.cat((embeddings, aux_embeds), dim=1)
        if rcnn_embeddings is not None:
            rcnn_embeds = []
            for rcnn_embedding in rcnn_embeddings:
                rcnn_embed = self.patch_embedding(rcnn_embedding)
                rcnn_embed = rcnn_embed.flatten(2).transpose(1, 2).flatten(0, 1)  # 3*4, 768 3
                rcnn_embeds.append(rcnn_embed)
            rcnn_embeds = torch.stack(rcnn_embeds)  # bsz, 12, 768
            rcnn_embeds = rcnn_embeds + self.rcnn_position_embedding(self.rcnn_position_ids)
            embeddings = torch.cat((embeddings, rcnn_embeds), dim=1)
        return embeddings


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# VIT multi-head layer
class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: bool = False,
            current_layer: int = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale  # [bsz, tgt, emb_dim]
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)  # [bsz, num_heads, tgt, head_dim]
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)  # [bsz, num_heads, tgt, head_dim]

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)  # (bsz * num_heads, tgt, head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)  # [bsz, tgt, emb_dim]

        query_states = query_states.view(*proj_shape)  # (bsz * num_heads, tgt, head_dim)
        key_states = key_states.view(*proj_shape)  # (bsz * num_heads, tgt, head_dim)
        value_states = value_states.view(*proj_shape)  # (bsz * num_heads, tgt, head_dim)

        src_len = key_states.size(1)  # src_len
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # (bsz * num_heads, tgt, src_len)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # (bsz * num_heads, tgt, src_len)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)  # (bsz * num_heads, tgt, head_dim)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)  # (bsz, num_heads, tgt, head_dim)
        attn_output = attn_output.transpose(1, 2)  # (bsz, tgt, num_heads, head_dim)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)  # (bsz, tgt, emb_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# VIT FFN layer
class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: bool = False,
            current_layer: int = None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # multi-head attention layer
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
            current_layer=current_layer,
        )
        hidden_states = residual + hidden_states  # attention output + residual
        # FFN layer
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads  # 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 768

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            current_layer=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # bsz, 128, 768

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            current_layer=None
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            current_layer
        )
        # Add and Layer Normalization in self-attention block
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Bert FFN
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # Add and Norm in FFN block
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            current_layer=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            current_layer=current_layer,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Vision2TextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, args=None):
        super().__init__()
        self.config = config
        self.args = args
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5
        # self.dropout = config.attention_dropout  # only for vision config
        self.dropout = config.attention_probs_dropout_prob  # only for text config

        # self.text_k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.text_v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.text_q_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.vision_k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.vision_v_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.vision_and_text_k_proj = nn.Linear(2 * self.embed_dim, self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            text_hidden_states: torch.Tensor,
            vision_hidden_states: torch.Tensor,
            output_attentions: bool = False,
            current_layer: int = None,
            text2vision_hidden_states: torch.Tensor = None,
    ):
        bsz, tgt_len, embed_dim = text_hidden_states.size()

        # get query proj
        query_states = self.text_q_proj(text_hidden_states) * self.scale  # [bsz, tgt, emb_dim]
        key_states = self._shape(self.vision_k_proj(vision_hidden_states), -1, bsz)  # [bsz, num_heads, src, head_dim]
        value_states = self._shape(self.vision_v_proj(vision_hidden_states), -1, bsz)  # [bsz, num_heads, src, head_dim]
        if text2vision_hidden_states is not None:
            key_states = self._shape(
                self.vision_and_text_k_proj(torch.cat((text2vision_hidden_states, text2vision_hidden_states), dim=-1)),
                -1,
                bsz)  # [bsz, num_heads, src, head_dim]

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)  # (bsz * num_heads, -1, head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)  # [bsz, tgt, emb_dim]

        query_states = query_states.view(*proj_shape)  # (bsz * num_heads, src + tgt, head_dim)
        key_states = key_states.view(*proj_shape)  # (bsz * num_heads, src + tgt, head_dim)
        value_states = value_states.view(*proj_shape)  # (bsz * num_heads, src + tgt, head_dim)

        src_tgt_len = key_states.size(1)  # src + tgt
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # (bsz * num_heads, tgt, src + tgt)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_tgt_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_tgt_len)}, but is {attn_weights.size()}"
            )
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # (bsz * num_heads, tgt, src + tgt)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_tgt_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_tgt_len)
        else:
            attn_weights_reshaped = None

        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_weights, value_states)  # (bsz * num_heads, tgt, head_dim)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)  # (bsz, num_heads, tgt, head_dim)
        attn_output = attn_output.transpose(1, 2)  # (bsz, tgt, num_heads, head_dim)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)  # (bsz, tgt, emb_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class Vision2TextAttOnlyLayer(nn.Module):
    def __init__(self, config, args, layer: int = None, ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.args = args
        self.layer = layer
        self.self_attn = Vision2TextAttention(config, args)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            text_hidden_states: torch.Tensor,
            all_vision_hidden_states: Tuple,
            output_attentions: bool = False,
    ):
        # MHA layer
        vision_hidden_states = all_vision_hidden_states[self.layer]
        attentional_hidden_states, attn_weights = self.self_attn(
            text_hidden_states=text_hidden_states,
            vision_hidden_states=vision_hidden_states,
            output_attentions=output_attentions,
            current_layer=self.layer,
        )
        attentional_hidden_states = self.layer_norm1(attentional_hidden_states)

        outputs = (attentional_hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class VisualLayerWiseMoE(nn.Module):
    def __init__(self, config, aggred_layer_num=1):
        super().__init__()
        self.config = config
        # self.n_experts = aggred_layer_num
        self.n_experts = 3
        self.experts1 = nn.Parameter(torch.Tensor(self.n_experts, 2 * config.hidden_size, config.hidden_size),
                                     requires_grad=True)
        torch.nn.init.xavier_uniform(self.experts1.data)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.experts2 = nn.Parameter(torch.Tensor(self.n_experts, config.hidden_size, config.hidden_size),
                                     requires_grad=True)
        torch.nn.init.xavier_uniform(self.experts2.data)

        self.expert_router = nn.Linear((1 + self.n_experts) * config.hidden_size, self.n_experts + 1)

    def forward(self,
                text_hidden_states: torch.Tensor = None,
                layer_wise_visual_hidden_states: Tuple = None,
                attention_mask: torch.Tensor = None, ):
        expert_router_input = torch.cat((text_hidden_states,
                                         layer_wise_visual_hidden_states[0],
                                         layer_wise_visual_hidden_states[6],
                                         layer_wise_visual_hidden_states[12],), dim=-1)  # text+img
        expert_weights = self.expert_router(expert_router_input)
        expert_weights = torch.softmax(expert_weights, dim=-1)
        bsz, tgt_len, embed_dim = text_hidden_states.size()
        # layer_wise_visual_hidden_states = torch.stack(list(layer_wise_visual_hidden_states), dim=2)
        layer_wise_visual_hidden_states = torch.stack((layer_wise_visual_hidden_states[0],
                                                       layer_wise_visual_hidden_states[6],
                                                       layer_wise_visual_hidden_states[12],), dim=2)
        input_hidden_states = torch.cat((text_hidden_states.unsqueeze(2).expand(layer_wise_visual_hidden_states.shape),
                                         layer_wise_visual_hidden_states), dim=-1)
        experts_out = torch.einsum('bsni,nio->bsno', input_hidden_states, self.experts1)
        # expert network layer 2
        experts_out = self.activation_fn(experts_out)
        experts_out = torch.einsum('bsni,nio->bsno', experts_out, self.experts2)
        # text_only expert
        text_hidden_states = text_hidden_states.unsqueeze(2)
        experts_out = experts_out + text_hidden_states.expand(layer_wise_visual_hidden_states.shape)
        # experts fusion
        experts_out = torch.cat([text_hidden_states, experts_out], dim=2)
        experts_out = torch.einsum('bsno,bsn->bsno', experts_out, expert_weights)
        moe_fusion = torch.sum(experts_out, dim=2)

        return moe_fusion


class Vision2TextLayerWiseAgg(nn.Module):
    def __init__(self, config, args, aggred_layer_num=1):
        super().__init__()
        self.args = args
        self.cross_modal_att_layer = nn.ModuleList(
            [Vision2TextAttOnlyLayer(config, args, layer=i) for i in range(aggred_layer_num)])
        self.moe_layer_wise_agg = VisualLayerWiseMoE(config, aggred_layer_num)
        self.aggred_layer_num = aggred_layer_num

    def forward(
            self,
            all_text_hidden_states: Tuple = None,
            all_vision_hidden_states: Tuple = None,
            attention_mask: torch.Tensor = None,
            output_attentions: bool = False,
    ):
        layer_wise_visual_hidden_states = ()
        all_cross_modal_attentions = () if output_attentions else None
        text_hidden_states = all_text_hidden_states[-1]
        detached_vision_hidden_states = ()
        for i in range(self.aggred_layer_num):
            detached_vision_hidden_states += (all_vision_hidden_states[i].detach(),)
        for i in range(self.aggred_layer_num):
            attention_outputs = self.cross_modal_att_layer[i](text_hidden_states,
                                                              detached_vision_hidden_states,
                                                              output_attentions)
            if output_attentions:
                all_cross_modal_attentions += (attention_outputs[1],)
            layer_wise_visual_hidden_states += (attention_outputs[0],)
        cross_modal_hidden_state = self.moe_layer_wise_agg(text_hidden_states=text_hidden_states,
                                                           layer_wise_visual_hidden_states=layer_wise_visual_hidden_states,
                                                           attention_mask=attention_mask)
        outputs = (cross_modal_hidden_state,)
        if output_attentions:
            all_cross_modal_attentions = torch.stack(list(all_cross_modal_attentions), dim=1)
            outputs += (all_cross_modal_attentions,)
        return outputs


class BertVitInterEncoder(nn.Module):
    def __init__(self, vision_config, text_config, args):
        super().__init__()
        self.vision_config = vision_config
        self.text_config = text_config
        self.args = args

        self.vision_layers = nn.ModuleList(
            [CLIPEncoderLayer(vision_config) for _ in range(vision_config.num_hidden_layers)])
        self.text_layer = nn.ModuleList([BertLayer(text_config) for _ in range(text_config.num_hidden_layers)])

        self.vision2text_layer_agg = Vision2TextLayerWiseAgg(text_config, args, aggred_layer_num=13)

    def forward(
            self,
            vision_embeds=None,
            text_embeds=None,
            attention_mask=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            image_mask=None,
    ):
        assert self.vision_config.num_hidden_layers == self.text_config.num_hidden_layers

        all_vision_hidden_states = () if output_hidden_states else None
        all_text_hidden_states = () if output_hidden_states else None
        all_vision_attentions = () if output_attentions else None
        all_text_attentions = () if output_attentions else None

        vision_hidden_states = vision_embeds
        text_hidden_states = text_embeds
        for idx in range(self.vision_config.num_hidden_layers):
            if output_hidden_states:
                all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states,)
                all_text_hidden_states = all_text_hidden_states + (text_hidden_states,)

            # vision VIT
            vision_layer_module = self.vision_layers[idx]
            vision_layer_output = vision_layer_module(
                vision_hidden_states,
                output_attentions=output_attentions,
                current_layer=idx,
            )
            vision_hidden_states = vision_layer_output[0]

            # text BERT
            layer_head_mask = head_mask[idx] if head_mask is not None else None
            text_layer_module = self.text_layer[idx]
            text_layer_output = text_layer_module(
                text_hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
                current_layer=idx,
            )
            text_hidden_states = text_layer_output[0]
            if output_attentions:
                all_vision_attentions = all_vision_attentions + (vision_layer_output[1],)
                all_text_attentions = all_text_attentions + (text_layer_output[1],)

        if output_hidden_states:
            all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states,)
            all_text_hidden_states = all_text_hidden_states + (text_hidden_states,)
            if torch.cuda.is_available():
                for i in range(len(all_vision_hidden_states)):
                    all_text_hidden_states[i].to(self.args.device)
                    all_vision_hidden_states[i].to(self.args.device)

        vision2text_outputs = self.vision2text_layer_agg(all_text_hidden_states=all_text_hidden_states,
                                                         all_vision_hidden_states=all_vision_hidden_states,
                                                         attention_mask=attention_mask,
                                                         output_attentions=output_attentions, )
        last_hidden_state = vision2text_outputs[0]

        all_cross_modal_attentions = vision2text_outputs[1] if output_attentions else None

        if not return_dict:
            return tuple(
                v for v in [
                    text_hidden_states,
                    all_text_hidden_states,
                    all_text_attentions,
                ] if v is not None)
        vision_hidden_states = vision_hidden_states
        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=all_text_hidden_states,
            attentions=all_cross_modal_attentions,
        ), vision_hidden_states, text_hidden_states


class BertVitInterBaseModel(nn.Module):
    def __init__(self, vision_config, text_config, args, add_pooling_layer=False):
        super(BertVitInterBaseModel, self).__init__()
        self.args = args

        # vision model
        self.vision_config = vision_config
        self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
        self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)
        # text model
        self.text_config = text_config
        self.text_embeddings = BertEmbeddings(text_config)
        self.text_pooler = BertPooler(text_config) if add_pooling_layer else None

        # all
        self.encoder = BertVitInterEncoder(vision_config, text_config, args)

        self.device = 'cuda' if torch.cuda.is_available() and self.args.device == 'cuda' else 'cpu'

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,

            pixel_values=None,
            aux_values=None,
            rcnn_values=None,

            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

            image_mask=None,
    ):
        # pre vision, extract visual features, CLIPVisionEmbeddings Class
        vision_embedding_output = self.vision_embeddings(pixel_values, aux_values, rcnn_values)
        vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)

        # pre text
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            raise ValueError("token_type_ids is None!")

        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = get_head_mask(head_mask, self.text_config.num_hidden_layers)  # [None]*12
        # extract textual features, BertEmbeddings Class
        text_embedding_output = self.text_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        encoder_outputs, vision_hidden_states, text_hidden_states = self.encoder(
            vision_embeds=vision_embedding_output,
            text_embeds=text_embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_mask=image_mask,
        )
        sequence_output = encoder_outputs[0]
        # BertPooler Class
        pooled_output = self.text_pooler(sequence_output) if self.text_pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return (BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ), vision_hidden_states, text_hidden_states)

    def _init_text_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_embeddings.word_embeddings = value

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()  # original BERT token embedding size: 30522
        # resize token embedding to 30526, with new tokens ['<s>', '</s>', '<o>', '</o>']
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

    def _get_resized_embeddings(
            self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`torch.nn.Embedding`` module of the model without doing anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
        if new_num_tokens is None:
            return old_embeddings
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
                f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim).to(
            self.device, dtype=old_embeddings.weight.dtype
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_text_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings
