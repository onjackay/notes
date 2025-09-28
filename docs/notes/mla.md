# Multi-head Latent Attention (MLA)

## Transformers

```python
query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

q_states = self.q_proj(hidden_states)
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
