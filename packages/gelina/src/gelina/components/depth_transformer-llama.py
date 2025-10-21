from torchtune.models import llama3_2
import torchtune
import torch
from torch import nn
from torchtune.modules.common_utils import disable_kv_cache

def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )

def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    decoder = llama3_2_100M().to(device)
    decoder.tok_embeddings = nn.Identity().to(device)
    decoder.output = nn.Identity().to(device)
    with next(decoder.parameters()).device:
        decoder.setup_caches(128, torch.float32, decoder_max_seq_len=6)

    decoder_causal_mask =  _create_causal_mask(6, device)

    decoder.reset_caches()

    b,n,d = 32,100,1024

    embeddings = torch.randn((b,n,d), dtype=torch.float32).to(device) # sequence of embeddings (b,length,d)
    embeddings = embeddings.reshape(-1,d) # flatten sequence length

    target_codes
    curr_decoder_mask = decoder_causal_mask.unsqueeze(0).expand(b, -1, -1)
    curr_pos = torch.arange(0, n, device=device).unsqueeze(0).repeat(b, 1)
    print(curr_pos.shape)
    print(curr_decoder_mask.shape)
    with disable_kv_cache(decoder):
        out_emb = decoder(embeddings, input_pos=curr_pos, mask=curr_decoder_mask)
    print(out_emb.shape)

    # do for loop