import numpy as np
from numpy.random import default_rng
import matplotlib as mlp
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import scipy
from scipy.stats import unitary_group
import os
import random
import math
from copy import copy
import pandas as pd
from tqdm import tqdm
from time import time
from IPython import display
from collections import Counter

from datasets import load_dataset
import string
import re
from sklearn.model_selection import train_test_split
from functools import partial

import torch
import torchvision
import torchinfo
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
import torch.nn as nn
from torch import linalg
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


from sklearn.metrics import (confusion_matrix, 
                             ConfusionMatrixDisplay,
                             roc_curve, 
                             roc_auc_score, 
                             RocCurveDisplay,
                             precision_recall_curve,
                             average_precision_score, 
                             PrecisionRecallDisplay,
                             classification_report, 
                             accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score)


torch.set_default_dtype(torch.float64)
plt.rcParams["font.family"] = 'sans-serif'
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Get cpu or gpu device for training.
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ',device)

np.random.seed(0)
torch.manual_seed(0)




##############################################
##############################################
##############################################
'''
General Functions
'''


def direct_sum(A,B):
  ds_tensor = np.zeros((A.shape[0]+B.shape[0],A.shape[1]+B.shape[1]))
  ds_tensor[:A.shape[0],:A.shape[1]] = A
  ds_tensor[A.shape[0]:,A.shape[1]:] = B
  return ds_tensor

def lin_comb(w,A):
  return torch.einsum('i,ijk',w,A)

# project a matrix M onto a preferred spin rep basis BS using projections coefficients from B
def project(M,B,BS):
    # find coeffient vector and dot that into the preferred basis
    return np.einsum('i,ijk->jk',projection_coefficients(M,B),BS)

def projection_coefficients(M,B):
    return np.trace(np.einsum('lij,jk->lik',np.transpose(np.conjugate(B), (0,2,1)),M),axis1=1,axis2=2)

def projection_coefficients_real(M,B):
    return np.trace(np.einsum('lij,jk->lik',np.transpose(np.conjugate(B), (0,2,1)),M),axis1=1,axis2=2).real

##############################################
##############################################
##############################################
'''
Data Preprocessing
'''

# Standardizers
def english_standardize(s: str):
    s = s.lower().strip()
    s = re.sub(f"[{re.escape(string.punctuation)}]", "", s)
    return s

def spanish_standardize(s: str):
    s = s.strip().lower()
    
    # # Preserve special tokens
    # if s in ["[start]", "[end]"]:
    #     return s
    
    # Remove punctuation (keep letters, numbers, whitespace)
    s = re.sub(f"[{re.escape(string.punctuation)}]", "", s)
    return s

# Tokenizers
def tokenize_text(s: str):
    return s.split()

def find_max_seq_length(sequences):
    max_seq_length = 0
    for seq in sequences:
        if len(seq)>max_seq_length:
            max_seq_length = len(seq)
    return max_seq_length

# Indexing (Vocabulary)
def build_vocab(token_lists, max_size=15000, min_freq=1):
    counter = Counter(tok for toks in token_lists for tok in toks)
    vocab = {"<pad>": 0, "<unk>": 1, "[start]": 2, "[end]": 3}
    idx = len(vocab)
    # counter.most_common(max_size) automatically limits the size of the vocab being enumerated
    for token, freq in counter.most_common(max_size):
        # min_freq, if 1 or less, sometimes not meaningful 
        # and you can stop adding words once below that threshold too
        if freq < min_freq:
            break
        if token in vocab:
            continue
        vocab[token] = idx
        idx += 1
    inv_vocab = {i: w for w, i in vocab.items()}
    return vocab, inv_vocab

# Vectorize
def tokens_to_ids(tokens, vocab):
    return [vocab.get(t, vocab["<unk>"]) for t in tokens]

# Dataset class to comply with PyTorch Dataloader
class TranslationDataset(Dataset):
    def __init__(self, pairs, source_vocab, target_vocab):
        self.pairs = pairs
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]
        source_ids = tokens_to_ids(tokenize_text(source), self.source_vocab)
        target_ids = tokens_to_ids(tokenize_text(target), self.target_vocab)
        # vector lists; collate will pad  and create shifted target
        return {"source": source_ids, "target": target_ids}
    
def collate_seq2seq(batch, source_pad_idx, target_pad_idx):
    # Lists of source and target idx vectors
    source_seqs = [torch.tensor(item["source"], dtype=torch.long) for item in batch]
    target_seqs = [torch.tensor(item["target"], dtype=torch.long) for item in batch]
    # Tensors of source and target seq lengths
    max
    source_lengths = torch.tensor([len(s) for s in source_seqs], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in target_seqs], dtype=torch.long)
    # pad, return torch Tensor
    source_padded = pad_sequence(source_seqs, batch_first=True, padding_value=source_pad_idx)
    target_padded = pad_sequence(target_seqs, batch_first=True, padding_value=target_pad_idx)
    # build target_input and target_output
    # target_input: all tokens except final (since they include [start] .. [end])
    # target_output: all tokens except the first (model must predict next token)
    target_input = target_padded[:, :-1].contiguous()
    target_output = target_padded[:, 1:].contiguous()
    target_input_lengths = (target_lengths - 1).clamp(min=0)  # lengths for input to decoder

    return {
        "source": source_padded,
        "source_lengths": source_lengths,
        "target_input": target_input,
        "target_output": target_output,
        "target_lengths": target_input_lengths
    }

##############################################
##############################################
##############################################
'''
Data pipeline
'''



##############################################
##############################################
##############################################
'''
Models
'''
class Seq2SeqGRU(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, embed_dim, enc_hidden, dec_hidden, pad_idx_source, pad_idx_target):
        super().__init__()
        self.source_embed = nn.Embedding(source_vocab_size, embed_dim, padding_idx=pad_idx_source)
        self.target_embed = nn.Embedding(target_vocab_size, embed_dim, padding_idx=pad_idx_target)
        self.encoder = nn.GRU(input_size=embed_dim, hidden_size=enc_hidden, batch_first=True, bidirectional=True)
        # decoder will be unidirectional
        self.decoder = nn.GRU(input_size=embed_dim, hidden_size=dec_hidden, batch_first=True, bidirectional=False)
        # We must map encoder final (2*enc_hidden) to decoder initial (dec_hidden)
        self.enc_to_dec = nn.Linear(2*enc_hidden, dec_hidden)
        self.out_proj = nn.Linear(dec_hidden, target_vocab_size)  # logits per token

    def encode(self, source, source_lengths):
        # source: [B, source_len]
        embedded = self.source_embed(source)  # [B, source_len, E]
        packed = pack_padded_sequence(embedded, source_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.encoder(packed)
        # h_n: [num_layers * num_directions, B, enc_hidden] -> num_layers=1, shape [2, B, enc_hidden]
        # combine the forward/backward final states
        # forward = h_n[0], backward = h_n[1]
        h_n = h_n.view(1, 2, source.size(0), -1) if False else h_n  #[2,B,enc_hidden]
        # combine along hidden dim
        h_forward = h_n[0] if False else h_n[0]  # shape [B, enc_hidden]  (but indexing above unused)
        # simpler: h_n is [2, B, enc_hidden] -> take 0 and 1
        h_f = h_n[0]  # [B, enc_hidden]
        h_b = h_n[1]  # [B, enc_hidden]
        h_combined = torch.cat([h_f, h_b], dim=1)  # [B, 2*enc_hidden]
        # map to decoder initial
        dec_init = torch.tanh(self.enc_to_dec(h_combined)).unsqueeze(0)  # [1, B, dec_hidden]
        return dec_init, packed_out

    def decode(self, target_input, dec_init, target_lengths):
        # target_input: [B, target_seq_len] 
        embedded = self.target_embed(target_input)  # [B, target_seq_len, E]
        packed = pack_padded_sequence(embedded, target_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.decoder(packed, dec_init)  # packed_out is PackedSequence
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, target_seq_len, dec_hidden]
        logits = self.out_proj(out)  # [B, target_seq_len, target_vocab_size]
        return logits

    def forward(self, source, source_lengths, target_input, target_input_lengths):
        dec_init, _ = self.encode(source, source_lengths)
        logits = self.decode(target_input, dec_init, target_input_lengths)
        return logits

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                               num_heads=num_heads,
                                               dropout=0.0,
                                               bias=True,
                                               batch_first=True)

        self.dense_proj = nn.Sequential(nn.Linear(embed_dim, hidden_dim, dtype=torch.float64),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, embed_dim, dtype=torch.float64)
                                        )

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, source_key_padding_mask=None):
        """
        x: [B, seq_len, E] (already embedded input)
        key_padding_mask: [B, seq_len] boolean mask for padding
        """
        att_out, att_weights = self.attention(query=x,
                                              key=x,
                                              value=x,
                                              key_padding_mask=source_key_padding_mask,
                                              need_weights=True)

        proj_in = self.layernorm1(x + att_out)   # residual
        proj_out = self.dense_proj(proj_in)
        out = self.layernorm2(proj_in + proj_out)  # residual
        return out, att_weights

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(embed_dim, hidden_dim, dtype=torch.float64),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, embed_dim, dtype=torch.float64)
                                    )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, target_mask=None, target_key_padding_mask=None, memory_key_padding_mask=None):
        # Masked self-attention (decoder cannot peek forward)
        self_attn_out, _ = self.self_attn(x, x, x, attn_mask=target_mask, key_padding_mask=target_key_padding_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # Cross-attention over encoder memory
        cross_attn_out, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # Feedforward
        ff_out = self.linear(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

class TransformerSeq2Seq(nn.Module):
    def __init__(self, source_vocab_size, 
                 target_vocab_size, 
                 embed_dim, 
                 hidden_dim, 
                 num_heads, 
                 num_layers, 
                 pad_idx_source, 
                 pad_idx_target, 
                 max_source_len, 
                 max_target_len):
        super().__init__()

        self.source_embed = nn.Embedding(source_vocab_size, embed_dim, padding_idx=0)#vocab["<pad>"])
        self.target_embed = nn.Embedding(target_vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = SinusoidalPositionalEmbedding(embed_dim, max_len=max_source_len) #5000
        # self.source_embed = PositionalEmbedding(max_source_len, embed_dim, pad_idx_source)
        # self.target_embed = PositionalEmbedding(max_target_len, embed_dim, pad_idx_target)

        self.encoder_layers = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoder(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)])

        self.out_proj = nn.Linear(embed_dim, target_vocab_size, dtype=torch.float64)
        self.pad_idx_source = pad_idx_source
        self.pad_idx_target = pad_idx_target

    def make_pad_mask(self, seq, pad_idx):
        return seq == pad_idx  # [B, T]

    # mask the upper half of the pairwise attention matrix to prevent 
    # the model from paying any attention to information from the future
    def make_subsequent_mask(self, size):
        return torch.triu(torch.ones(size, size), diagonal=1).bool()

    def forward(self,  source, source_lengths, target_input, target_input_lengths):
        # masks
        source_key_padding_mask = self.make_pad_mask(source, self.pad_idx_source)  # [B, source_len]
        target_key_padding_mask = self.make_pad_mask(target_input, self.pad_idx_target)  # [B, target_len]
        target_mask = self.make_subsequent_mask(target_input.size(1)).to(source.device)  # [target_len, target_len]

        # [B, source_len] → embeddings
        # source = self.source_embed(source)  # [B, source_len, E]
        # target = self.target_embed(target_input)  # [B, target_len, E]
        source = self.pos_embed(self.source_embed(source))
        target = self.pos_embed(self.target_embed(target_input))

        # Encoder
        memory = source
        for layer in self.encoder_layers:
            memory, _ = layer(memory, source_key_padding_mask=source_key_padding_mask)

        # Decoder
        out = target
        for layer in self.decoder_layers:
            out = layer(out, memory, target_mask=target_mask, target_key_padding_mask=target_key_padding_mask, memory_key_padding_mask=source_key_padding_mask)

        # Output projection
        logits = self.out_proj(out)  # [B, target_len, vocab_size]
        return logits
    
class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_dim, vocab):
        super(PositionalEmbedding, self).__init__()
        self.embed_words = nn.Embedding(num_embeddings=len(vocab),embedding_dim=embed_dim,padding_idx=vocab["<pad>"])
        self.embed_positions = nn.Embedding(num_embeddings=max_seq_len,embedding_dim=embed_dim)

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device)          # [seq]
        positions = positions.unsqueeze(0).expand(batch_size, -1)   # [batch, seq]
        embedded_words = self.embed_words(x)                        # [batch, seq, embed_dim]
        embedded_positions = self.embed_positions(positions)        # [batch, seq, embed_dim]
        return embedded_words + embedded_positions
    
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim

        # Precompute the sinusoidal embeddings up to max_len
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(1.0*10e4) / embed_dim) ) # [embed_dim/2]
        # Even indices → sine, odd indices → cosine
        pe[:, 0::2] = torch.sin(position * div_term) # [max_len,embed_dim/2]
        pe[:, 1::2] = torch.cos(position * div_term) # [max_len,embed_dim/2]

        # Register as buffer so it's not a parameter but moves with the model (cpu/gpu)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, embed_dim]

    def forward(self, x):
        """
        x: [batch_size, seq_len, embed_dim] (already word-embedded input)
        returns: x + positional_encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :] #self.pe.squeeze(0)[None, :seq_len, :]

##############################################
##############################################
##############################################
'''
Training
'''

def training_seq2seq(train_data,
                     val_data,
                     epochs,
                     loss_fn,
                     model,
                     optimizer,
                     regularize=False):

    history = {
        'train_loss': [],
        'val_loss': [],
        'weights': []
    }
    best_val_loss = float("inf")

    start = time()
    for epoch in range(epochs):
        last_step = True if (epoch == epochs - 1) else False
        
        # ---- Training ----
        running_train_loss = 0.0
        model.train()
        for batch in train_data:
            source = batch['source'].to(device)                 # [B, source_len]
            source_lengths = batch['source_lengths'].to(device) # [B, 1]
            target_inp = batch['target_input'].to(device)       # [B, target_len]
            target_out = batch['target_output'].to(device)      # [B, target_len]
            target_lengths = batch['target_lengths'].to(device)

            train_loss, _, trained_weights = train_seq2seq(source, 
                                                           source_lengths, 
                                                           target_inp, 
                                                           target_out,
                                                           target_lengths,
                                                           model, 
                                                           loss_fn, 
                                                           optimizer,
                                                           backprop=True,
                                                           last_epoch=last_step,
                                                           regularize=regularize)
            
            running_train_loss += train_loss.item() * source.size(0)

        train_loss = running_train_loss / len(train_data.dataset)

        # ---- Validation ----
        running_val_loss = 0.0
        model.eval()
        for batch in val_data:
            source = batch['source'].to(device)
            source_lengths = batch['source_lengths'].to(device)
            target_inp = batch['target_input'].to(device)
            target_out = batch['target_output'].to(device)
            target_lengths = batch['target_lengths'].to(device)

            val_loss, _, _ = train_seq2seq(source, 
                                           source_lengths, 
                                           target_inp, 
                                           target_out,
                                           target_lengths,
                                           model, 
                                           loss_fn, 
                                           optimizer,
                                           backprop=False,
                                           last_epoch=last_step,
                                           regularize=regularize)
            
            running_val_loss += val_loss.item() * source.size(0)

        val_loss = running_val_loss / len(val_data.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if last_step:
            history['weights'].append(trained_weights)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"Epoch {epoch+1} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Best Val Loss: {best_val_loss:.4f}")

    end = time()
    print(f"Training Complete. Total Time (s): {end-start:.2f}")
    return {'history': history}

def train_seq2seq(source, 
                  source_lengths, 
                  target_inp, 
                  target_out,
                  target_lengths,
                  model, 
                  loss_fn, 
                  optimizer,
                  backprop=True, 
                  last_epoch=False, 
                  regularize=False):
    """
    source:        [B, source_len]   - English
    source_lengths:[B]               - true lengths
    target_inp:    [B, target_len]   - Spanish shifted right ([start] + ...)
    target_out:    [B, target_len]   - Spanish shifted left (... + [end])
    """

    if backprop:
        model.train()
        logits = model(source, source_lengths, target_inp, target_lengths)      # [B, target_len, vocab_size]
        loss = loss_fn(logits.transpose(1, 2), target_out)  # CE expects [B, vocab, target_len]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        model.eval()
        with torch.no_grad():
            logits = model(source, source_lengths, target_inp, target_lengths)
            loss = loss_fn(logits.transpose(1, 2), target_out)

    trained_weights = None
    if last_epoch:
        trained_weights = {name: param.clone().detach().cpu()
                           for name, param in model.named_parameters()}

    return loss, logits, trained_weights


##############################################
##############################################
##############################################
'''
Visualizations
'''
