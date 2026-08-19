import torch
import math
from torch import nn

class MultiHeadAttention(nn.Module):
    """
    A MultiHeadAttention module with optional bias and optional gating.
    """

    def __init__(self, c_in, c, N_head, attn_dim, gated=False, is_global=False, use_bias_for_embeddings=False):
        """
        Initializes the module. MultiHeadAttention theoretically consists of 
        N_head separate linear layers for the query, key and value embeddings.
        However, the embeddings can be computed jointly and split afterwards,
        so we only need one query, key and value layer with larger c_out.

        Args:
            c_in (int): Input dimension for the embeddings.
            c (int): Embedding dimension for each individual head.
            N_head (int): Number of heads.
            attn_dim (int): The dimension in the input tensor along which
                the attention mechanism is performed.
            gated (bool, optional): If True, an additional sigmoid-activated 
                linear layer will be multiplicated against the weighted 
                value vectors before feeding them through the output layer. 
                Defaults to False.
            is_global (bool, optional): If True, global calculation will be performed.
                For global calculation, key and value embeddings will only use one head,
                and the q query vectors will be averaged to one query vector.
                Defaults to False.
            use_bias_for_embeddings (bool, optional): If True, query, 
                key, and value embeddings will use bias, otherwise not. 
                Defaults to False.
        """
        super().__init__()

        self.c_in = c_in
        self.c = c
        self.N_head = N_head
        self.gated = gated
        self.attn_dim = attn_dim
        self.is_global = is_global

        c_out = c*N_head
        self.linear_q = nn.Linear(c_in, c_out, bias = use_bias_for_embeddings)

        c_kv = c if is_global else c_out
        self.linear_k = nn.Linear(c_in, c_kv, bias = use_bias_for_embeddings)
        self.linear_v = nn.Linear(c_in, c_kv, bias = use_bias_for_embeddings)
        self.linear_o = nn.Linear(c_out, c_in)

        if gated:
            self.linear_g = nn.Linear(c_in, c_out)


    def prepare_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Splits the embeddings into individual heads and transforms the input
        shapes of form (*, q/k/v, *, N_head*c) into the shape 
        (*, N_head, q/k/v, c). The position of the q/k/v dimension 
        in the original tensors is given by attn_dim.

        Rearranges the tensors with the following changes:                    
           - (*, q/k/v, *, N_head*c) -> (*, q/k/v, N_head*c) with movedim       
           - (*, q/k/v, N_head*c) -> (*, q/k/v, N_head, c)                      
           - (*, q/k/v, N_head, c) -> (*, N_head, q/k/v, c)                     

        Args:
            q (torch.Tensor): Query embedding of shape (*, q, *, N_head*c).
            k (torch.Tensor): Key embedding of shape (*, k, *, N_head*c).
            v (torch.Tensor): Value embedding of shape (*, v, *, N_head*c).

        Returns:
            tuple: The rearranged embeddings q, k, and v of 
                shape (*, N_head, q/k/v, c) respectively.
        """


        
        q = q.movedim(self.attn_dim, -2)
        qkv_shape = q.shape[:-1] + (self.N_head, self.c)
        q = q.view(qkv_shape)
        q = q.transpose(-2, -3)

        k = k.movedim(self.attn_dim, -2)
        k = k.view(qkv_shape)
        k = k.transpose(-2, -3)

        v = v.movedim(self.attn_dim, -2)
        v = v.view(qkv_shape)
        v = v.transpose(-2, -3)

        return q, k, v

    def prepare_qkv_global(self, q, k, v):
        """
        Prepares the query, key and value embeddings with the following 
        differences to the non-global version:
            - key and value embeddings use only one head.
            - the query vectors are contracted into one, average query vector.
        Rearranges the tensors to match the output dimensions. 
        

        Args:
            q (torch.tensor): Query embeddings of shape (*, q, *, N_head*c).
            k (torch.tensor): Key embeddings of shape (*, k, *, c).
            v (torch.tensor): Value embeddings of shape (*, v, *, c).

        Returns:
            tuple: The rearranged embeddings q, k, and v of
                shape (*, N_head, 1, c) for q and shape (*, 1, k, c) for k and v. 
        """

        q = q.movedim(self.attn_dim, -2)
        q_shape = q.shape[:-1] + (self.N_head, self.c)
        q = q.view(q_shape)
        q = q.transpose(-2, -3)
        q = torch.mean(q, dim = -2, keepdim=True)

        k = k.movedim(self.attn_dim, -2)
        kv_shape = q.shape[:-1] + (1, self.c)
        k = k.view(kv_shape)
        k = k.transpose(-2, -3)

        v = v.movedim(self.attn_dim, -2)
        v = v.view(kv_shape)
        v = v.transpose(-2, -3)


        return q, k, v

    def forward(self, x, bias=None, attention_mask=None):
        """
        Forward pass through the MultiHeadAttention module.

        Args:
            x (torch.tensor): Input tensor of shape (*, q/k/v, *, c_in).
            bias (torch.tensor, optional): Optional bias tensor of shape
                (*, N_head, q, k) that will be added to the attention weights. 
                Defaults to None.
            attention_mask (torch.tensor, optional): Optional attention mask
                of shape (*, k). If set, the keys with value 0 in the mask will
                not be attended to.

        Returns:
            torch.tensor: Output tensor of shape (*, q/k/v, *, c_in)
        """

        out = None

        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        if self.is_global:
            q,k,v = self.prepare_qkv_global(q, k, v)
        else:
            q,k,v = self.prepare_qkv(q, k, v)

        q = q*(1/math.sqrt(self.c))

        a = torch.einsum('...nqc,...nkc->...nqk', q, k)

        if bias is not None:
            n = a.ndim - bias.ndim
            bias_batch_shape = bias.shape[:-3]
            bias_full_shape = bias_batch_shape + (1,) * n + bias.shape[-3:]
            bias = bias.view(bias_full_shape)
            a = a + bias

        if attention_mask is not None:
            attention_mask = attention_mask[..., None, None, :]
            offset = (attention_mask == 0) *1e8
            a = a + offset

        a = torch.Softmax(a, dim = -1)

        o = torch.einsum('...qk,...kc->...qc', a, v)
        o = o.transpose(-2, -3)
        o = torch.flatten(o, start_dim = -2)
        o = o.movedim(-2, self.attn_dim)

        if self.gated:
            g = torch.sigmoid(self.linear_g(o))
            o = o * g

        out = self.linear_o(o)

        return out
