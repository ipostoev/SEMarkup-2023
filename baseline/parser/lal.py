from typing import Optional, Dict, Any
from overrides import overrides

from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder, TokenEmbedder
from allennlp.nn import util
import numpy as np

import torch
import torch.nn.init as init
import torch.nn as nn

from .lal_misc import ScaledDotProductAttention
from .lal_misc import LayerNormalization
from .lal_misc import FeatureDropout
from .lal_misc import BatchIndices

import sys

DTYPE = torch.uint8 if float(sys.version[:3]) < 3.7 else torch.bool

@TokenEmbedder.register("lal")
class LabelAttention(TokenEmbedder):
    """
    Single-head Attention layer for label-specific representations
    """

    def __init__(self, model_name: str, train_parameters: bool = True) -> None:

        super().__init__()

        self.bert_embedder = PretrainedTransformerMismatchedEmbedder(model_name)
        residual_dropout = 0.1
        attention_dropout = 0.1
        self.d_k = 128
        self.d_v = 128
        self.d_l = 40 # Number of Labels
        self.d_model = self.bert_embedder.get_output_dim() # Model Dimensionality
        self.d_proj = 64 # Projection dimension of each label output
        self.use_resdrop = False # Using Residual Dropout?
        self.q_as_matrix = False # Using a Matrix of Q to be multiplied with input instead of learned q vectors
        d_model = self.d_model
        d_k = self.d_k
        d_v = self.d_v
        
        self.w_qs = nn.Parameter(torch.FloatTensor(self.d_l, d_k), requires_grad=True)
        self.w_ks = nn.Parameter(torch.FloatTensor(self.d_l, d_model, d_k), requires_grad=True)
        self.w_vs = nn.Parameter(torch.FloatTensor(self.d_l, d_model, d_v), requires_grad=True)

        init.xavier_normal_(self.w_qs)
        init.xavier_normal_(self.w_ks)
        init.xavier_normal_(self.w_vs)

        self.attention = ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = nn.Linear(d_v, d_model, bias=False) # input dimension does not match, should be d_l * d_v
        self.reduce_proj = nn.Linear(d_model, self.d_proj, bias=False)
        self.residual_dropout = FeatureDropout(residual_dropout)

    @overrides
    def get_output_dim(self) -> int:
        return self.reduce_proj * self.d_l

    def split_qkv_packed(self, inp, k_inp=None):
        len_inp = inp.size(0)
        v_inp_repeated = inp.repeat(self.d_l, 1).view(self.d_l, -1, inp.size(-1)) # d_l x len_inp x d_model

        if k_inp is None:
            k_inp_repeated = v_inp_repeated
        else:
            k_inp_repeated = k_inp.repeat(self.d_l, 1).view(self.d_l, -1, k_inp.size(-1)) # d_l x len_inp x d_model

        
        q_s = self.w_qs.unsqueeze(1) # d_l x 1 x d_k
        k_s = torch.bmm(k_inp_repeated, self.w_ks) # d_l x len_inp x d_k
        v_s = torch.bmm(v_inp_repeated, self.w_vs) # d_l x len_inp x d_v
       
        return q_s, k_s, v_s
    
    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs):
        # Input is padded representation: n_head x len_inp x d
        # Output is packed representation: (n_head * mb_size) x len_padded x d
        # (along with masks for the attention and output)
        n_head = self.d_l
        d_k, d_v = self.d_k, self.d_v

        len_padded = batch_idxs.max_len
        mb_size = batch_idxs.batch_size
        q_padded = q_s.repeat(mb_size, 1, 1) # (d_l * mb_size) x 1 x d_k
        k_padded = k_s.new_zeros((n_head, mb_size, len_padded, d_k))
        v_padded = v_s.new_zeros((n_head, mb_size, len_padded, d_v))
        invalid_mask = q_s.new_ones((mb_size, len_padded), dtype=DTYPE)

        for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
            if self.q_as_matrix:
                q_padded[:,i,:end-start,:] = q_s[:,start:end,:]
            k_padded[:,i,:end-start,:] = k_s[:,start:end,:]
            v_padded[:,i,:end-start,:] = v_s[:,start:end,:]
            invalid_mask[i, :end-start].fill_(False)

        attn_mask = invalid_mask.unsqueeze(1).repeat(n_head, 1, 1)
        
        output_mask = (~invalid_mask).repeat(n_head, 1)

        return(
            q_padded,
            k_padded.view(-1, len_padded, d_k),
            v_padded.view(-1, len_padded, d_v),
            attn_mask,
            output_mask,
            )

    def combine_v(self, outputs):
        # Combine attention information from the different labels
        d_l = self.d_l
        outputs = outputs.view(d_l, -1, self.d_v) # d_l x len_inp x d_v
    
        outputs = torch.transpose(outputs, 0, 1)#.contiguous() #.view(-1, d_l * self.d_v)
        # Project back to residual size

        outputs = self.proj(outputs) # Becomes len_inp x d_l x d_model
        return outputs

    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:

        emb_output = self.bert_embedder(token_ids, mask, offsets, wordpiece_mask, type_ids, segment_concat_mask)

        packed_len = sum([(len(emb_sentence)) for emb_sentence in emb_output])
        batch_idxs = np.zeros(packed_len, dtype=int)
        words_embs = []

        i = 0
        for snum, emb_sentence in enumerate(emb_output):
            for word_emb in emb_sentence:
                batch_idxs[i] = snum
                words_embs.append(word_emb)
                i += 1
        assert i == packed_len

        batch_idxs = BatchIndices(batch_idxs)
        unpacked_emb = torch.stack(words_embs)
        
        #lal_outputs, _ = self.lal_attention(unpacked_emb, batch_idxs)

        inp = unpacked_emb
        k_inp=None
        residual = inp # len_inp x d_model
        len_inp = inp.size(0)

        # While still using a packed representation, project to obtain the
        # query/key/value for each head
        q_s, k_s, v_s = self.split_qkv_packed(inp, k_inp=k_inp)
        # d_l x len_inp x d_k
        # q_s is d_l x 1 x d_k
        
        # Switch to padded representation, perform attention, then switch back
        q_padded, k_padded, v_padded, attn_mask, output_mask = self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs)
        # q_padded, k_padded, v_padded: (d_l * batch_size) x max_len x d_kv
        # q_s is (d_l * batch_size) x 1 x d_kv
        
        outputs_padded, attns_padded = self.attention(
            q_padded, k_padded, v_padded,
            attn_mask=attn_mask,
            )
  
        # outputs_padded: (d_l * batch_size) x max_len x d_kv
        # in LAL: (d_l * batch_size) x 1 x d_kv
        # on the best model, this is one value vector per label that is repeated max_len times
        
        if not self.q_as_matrix:
            outputs_padded = outputs_padded.repeat(1,output_mask.size(-1),1)

        #outputs = outputs_padded[output_mask]
        torch.cuda.empty_cache()
        outputs = self.combine_v(outputs_padded)
  
        for l in range(self.d_l):
            outputs[:, l, :] = outputs[:, l, :] + inp
        
        outputs = self.layer_norm(outputs) # len_inp x d_l x d_proj
        outputs = self.reduce_proj(outputs)
        outputs = outputs.view(len_inp, -1).contiguous() # len_inp x (d_l * d_proj)
        
        outputs_padded = outputs.new_zeros((1, batch_idxs.batch_size, batch_idxs.max_len, outputs.shape[-1]))
        for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
            outputs_padded[:, i, :end-start,:] = outputs[start:end,:]
        outputs_padded = outputs_padded.squeeze(0)
        outputs_padded = torch.nn.functional.normalize(outputs_padded, dim=2)
        return outputs_padded