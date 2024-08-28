import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Attention module with support for Flash attention.
Reference:
    - https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    - https://youtu.be/1RaIS98jj1Q
"""


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        enable_math: bool = False,
        enable_mem_efficient: bool = False,
        enable_flash: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.enable_math = enable_math
        self.enable_mem_efficient = enable_mem_efficient
        self.enable_flash = enable_flash

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            with torch.backends.cuda.sdp_kernel(
                enable_math=self.enable_math,
                enable_mem_efficient=self.enable_mem_efficient,
                enable_flash=self.enable_flash,
            ):
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == "__main__":

    batch_size = 2
    seq_len = 4
    embed_dim = 768

    # Create a dummy input tensor
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Create an Attention layer
    attn = Attention(dim=embed_dim, fused_attn=True, enable_flash=True)

    # Pass the input tensor through the Attention layer
    output = attn(x)

    # Print the shape of the output tensor
    print(output.shape)
    # Output: torch.Size([2, 4, 768])
