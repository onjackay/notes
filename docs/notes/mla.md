# Multi-head Latent Attention (MLA)

## Transformers

### DeepSeekV3 Configuration
```
hidden_size = 7168
q_lora_rank = 1536
num_heads   = 128
qk_nope_head_dim = 128
qk_rope_head_dim = 64
qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192
```

### DeepSeekV3 Attention

```python
query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

if self.q_lora_rank is None:
    q_states = self.q_proj(hidden_states)  # [b, s, h * qk_head_dim]
else:
    q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        # q_a_proj: [b, s, d] -> [b, s, q_lora_rank]   (16x down_proj)
        # q_b_proj: [b, s, q_lora_rank] -> [b, s, h * qk_head_dim]  (16x up_proj)

q_states = q_states.view(query_shape).transpose(1, 2)  # [b, h, s, qk_head_dim]
q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    # q_pass: [b, h, s, qk_nope_head_dim]
    # q_rot:  [b, h, s, qk_rope_head_dim]

compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [b, s, kv_lora_rank + qk_rope_head_dim]
k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    # k_pass: [b, s, kv_lora_rank]
    # k_rot:  [b, s, qk_rope_head_dim]

k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
    # k_pass: [b, h, s, qk_nope_head_dim + v_head_dim]
k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    # k_pass:       [b, h, s, qk_nope_head_dim]
    # value_states: [b, h, s, v_head_dim]

k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)  # [b, 1, s, qk_rope_head_dim]

cos, sin = position_embeddings
if self.config.rope_interleave:  # support using interleaved weights for efficiency
    q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
else:
    q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
k_rot = k_rot.expand(*k_pass.shape[:-1], -1)  # [b, h, s, qk_rope_head_dim]

query_states = torch.cat((q_pass, q_rot), dim=-1)  # [b, h, s, qk_head_dim]
key_states = torch.cat((k_pass, k_rot), dim=-1)  # [b, h, s, qk_head_dim]

# Normal attention
attn_output, attn_weights = attention_interface(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    dropout=0.0 if not self.training else self.attention_dropout,
    scaling=self.scaling,
    **kwargs,
)
```

MLA 的 KV Cache 里存的是 `compressed_kv` 和 `k_rot`，大大减少了 KV Cache 的所需空间。
以 DeepSeekV3 为例，每个 layer 每个 token 仅需一个长度为 192 的 Cache。

> 为什么 Q 和 K 要分成 pass 和 rot 两部分，前者不经过 RoPE，后者经过 RoPE？
>
> 我们想让 `compressed_kv` 经过的 `up_proj` 和 KV 权重矩阵（`w_k`，`w_v`）融合，但是 RoPE 夹在这两个矩阵乘中间阻止了融合。
> MLA 的方案是让 KV 的 `pass` 部分不经过 RoPE，融合两个矩阵乘；让 `rot` 部分不参与矩阵乘，经过 RoPE 后直接与 `pass` 部分 concat。
