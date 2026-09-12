import torch
import math
from torch import nn
from einops import rearrange

def fn_silu(in_features):
    return in_features*torch.sigmoid(in_features)


def fn_softmax(x: torch.Tensor, dim:int = -1)->torch.Tensor:
    x_max = torch.max(x,dim = dim,keepdim=True).values
    # for numerical stability
    x_stable = x-x_max

    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x,dim = dim, keepdim=True)
    return exp_x/sum_exp


def fn_scaled_dot_product_attetion(
        keys: torch.Tensor, 
        queries: torch.Tensor, 
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
)-> torch.Tensor:
    # keys and queries: (batch_size, ..., seq_len, d_k)
    # values: (batch_size, ..., seq_len, d_v)
    # output: (batch_size, ..., seq_len, d_v)
    d_k = queries.shape[-1]
    scores = queries @ keys.transpose(-2,-1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attention_weights = fn_softmax(scores,dim=-1)
    return attention_weights @ values

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None,dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.trunc_normal_(
            self.weight,
            mean = 0,
            std = 1,
            a=-3,
            b=3,
        )

    def forward(self,token_ids:torch.Tensor):
        return self.weight[token_ids]


class Linear(nn.Module):
    def __init__(self,in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        std = math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std=std,
            a=-3*std,
            b = 3*std,
        )

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        return x @ self.weight.T


class RMSNorm(nn.Module):
    def __init__(self, d_model:int,eps:float=1e-5,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # initialize the "gain" to 1
        self.gain =nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self,x:torch.Tensor)->torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((x**2).sum(dim = -1,keepdim=True)/self.d_model + self.eps)
        result = x*self.gain/rms

        return result.to(in_dtype)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        powers = torch.arange(0,d_k,2,device=device).float() / d_k
        freqs = 1.0 / (theta ** powers)

        t = torch.arange(max_seq_len, device=device).float()
        freqs_matrix = torch.outer(t,freqs)
        self.register_buffer("cos_cached", freqs_matrix.cos(),persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(),persistent=False)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 维度对齐
        # 只有当 x 是 4D (含 Head 维) 且 cos 是 3D (含 Batch 维) 时，才需要手动插入 Head 维。
        # 对于 test_rope 这种 3D x vs 2D cos 的情况，PyTorch 会自动左侧补 1，无需操作。
        if x.ndim > cos.ndim and cos.ndim>=3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # 确保类型一致
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        x_even = x[...,0::2]
        x_odd = x[...,1::2]
        output = torch.empty_like(x)
        output[...,0::2] = x_even*cos - x_odd*sin
        output[...,1::2] = x_even*sin + x_odd*cos

        return output


class SwiGLU(nn.Module):
    def __init__(self,d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        # w1 和 w3 是升维层，w2 是降维层
        # 按照 (d_out, d_in) 来定义
        self.w1 = nn.Parameter(
            torch.empty(d_ff,d_model,device=device,dtype=dtype)
        )

        self.w3 = nn.Parameter(
            torch.empty(d_ff,d_model,device=device,dtype=dtype)
        )

        self.w2 = nn.Parameter(
            torch.empty(d_model,d_ff,device=device,dtype=dtype)
        )

    def forward(self,x:torch.Tensor):
        gate = fn_silu(x@self.w1.T)
        linear_transform = x@self.w3.T
        dot_product = gate*linear_transform
        return dot_product@self.w2.T


class multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 bias: bool = False, 
                 context_length=None, theta=None, 
                 device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is not None and context_length is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, context_length, device=device)
        else:
            self.rope = None

    def forward(
        self,
        x:torch.Tensor,
        token_positions: torch.Tensor = None,
    )->torch.Tensor:
        s =  x.shape[-2]
        q = rearrange(self.q_proj(x),'... s (h d) -> ... h s d', h=self.num_heads)
        k = rearrange(self.k_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)
        v = rearrange(self.v_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                batch_dims = x.shape[:-2]
                token_positions = torch.arange(s,device=x.device).expand(*batch_dims, s)
            q = self.rope(q,token_positions)
            k = self.rope(k,token_positions)

        mask = torch.tril(torch.ones(s,s,device=x.device,dtype=torch.bool))
        attn_out = fn_scaled_dot_product_attetion(k,q,v,mask=mask)
        attn_out = rearrange(attn_out,'... h s d -> ... s (h d)')
        return self.output_proj(attn_out)

    def load_weights(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        o_weight: torch.Tensor,
    ):
        with torch.no_grad():
            self.q_proj.weight.copy_(q_weight)
            self.k_proj.weight.copy_(k_weight)
            self.v_proj.weight.copy_(v_weight)
            self.output_proj.weight.copy_(o_weight)

class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, context_length: int,
                 theta: float, device=None, dtype=None, 
                 use_rms_norm: bool = True,
                 norm_mode: str = "pre",   # 选项: "pre", "post"
                 ffn_type: str = "swiglu",  # 选项: "swiglu", "silu"
                 weights: dict[str, torch.Tensor] | None = None,
                ):
        super().__init__()
        self.use_rms_norm = use_rms_norm
        self.norm_mode = norm_mode
        self.ffn_type = ffn_type

        # 1. 初始化 Attention
        # RoPE 的开关由外部传入的 theta 是否为 None 控制
        self.attn = multihead_self_attention(
            d_model=d_model, 
            num_heads=num_heads, 
            context_length=context_length, 
            theta=theta,
            device=device, 
            dtype=dtype
        )

        # 2. 初始化 Norm 层 (Ablation 1)
        if use_rms_norm:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            # 如果禁用 Norm，使用 Identity 占位，它直接返回输入，不改变任何东西
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

        # 3. 初始化 FFN (Ablation 4)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            # 标准 FFN: x -> Linear -> SiLU -> Linear -> out
            # 注意: 为了公平对比，通常 SiLU FFN 的 d_ff 应该是 4 * d_model
            d_ff = 4 * d_model
            self.ffn = nn.Sequential(
                Linear(d_model, d_ff, device=device, dtype=dtype),
                nn.SiLU(),
                Linear(d_ff, d_model, device=device, dtype=dtype)
            )
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm (Llama 默认, 也是作业基准)
        # 公式: x = x + Sublayer(Norm(x))
        if self.norm_mode == "pre":
            x = x + self.attn(self.ln1(x), token_positions=token_positions)
            x = x + self.ffn(self.ln2(x))
        
        # Post-norm (原始 Transformer, Ablation 2)
        # 公式: x = Norm(x + Sublayer(x))
        elif self.norm_mode == "post":
            # 注意: Post-norm 通常很难训练，需要 Warmup
            x = self.ln1(x + self.attn(x, token_positions=token_positions))
            x = self.ln2(x + self.ffn(x))
            
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, 
                 num_layers: int, num_heads: int, d_ff: int, rope_theta: float, 
                 device=None, dtype=None,
                 # 新增实验参数
                 use_rms_norm: bool = True,
                 norm_mode: str = "pre",
                 ffn_type: str = "swiglu"):
        super().__init__()
        self.context_length = context_length
        
        # 1. Token Embedding 层
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # 2. 堆叠 Transformer Blocks
        # 将实验参数透传给每一个 Block
        self.layers = nn.ModuleList([
            transformer_block(
                d_model, num_heads, d_ff, context_length, rope_theta, 
                device=device, dtype=dtype,
                use_rms_norm=use_rms_norm,
                norm_mode=norm_mode,
                ffn_type=ffn_type
            )
            for _ in range(num_layers)
        ])

        # 3. 最终的输出层
        # 如果全局禁用了 Norm，这里的 Final Norm 也要变成 Identity
        if use_rms_norm:
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            """
            forward(input):
                return input
            """
            self.ln_final = nn.Identity()
   
        # 最后是一个 Linear 层映射回词表大小 (LM Head)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        b, s = token_ids.shape
        
        # 准备位置信息用于 RoPE, shape: [S] -> [1, S] -> [B,S]
        token_positions = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b, s)
        
        # 1. Embedding
        x = self.token_embeddings(token_ids)
        
        # 2. 逐层通过 Transformer Blocks
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
            
        # 3. 最终归一化 (如果 use_rms_norm=False，这里就是直通)
        x = self.ln_final(x)
        
        # 4. 投影到词表空间得到 logits
        return self.lm_head(x)
