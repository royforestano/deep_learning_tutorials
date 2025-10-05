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
# General Functions

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
# Data Preprocessing


# ----------------------------- Manual -------------------------------
class Vectorizer:
    def standardize(self,text):
        text = text.lower()
         # remove punctuation and rejoin characters
        return "".join(char for char in text if char not in string.punctuation)
    
    def tokenize(self, text):
       text = self.standardize(text)
       return text.split()
    
    def make_vocabulary(self, dataset):
        self.vocabulary = {"":0, "[UNK]": 1}
        for text in dataset:
          text = self.standardize(text)
          tokens = self.tokenize(text)
          for token in tokens:
             if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)
        self.inverse_vocabulary = dict((value, key) for key, value in self.vocabulary.items())
    
    def encode(self,text):
       text = self.standardize(text)
       tokens = self.tokenize(text)
       return [self.vocabulary.get(token,1) for token in tokens]
    
    def decode(self, int_sequence):
       return " ".join(self.inverse_vocabulary.get(i,"[UNK]") for i in int_sequence)

# ----------------------------- Integrated for Hugging Face/PyTorch -------------------------------
# Function to map tokens → binary vectors
# Preprocessing function
def preprocess(batch):
    # Lowercase
    text = batch["text"].lower()
    # Remove punctuation (keep only letters, numbers, spaces)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return {"text": text}

def tokenize(batch,ngram_size):
    if ngram_size==1:
        return {"tokens": batch["text"].split()}
    else:
        tokens = batch["text"].split()
        ngrams = []
        for i,token in enumerate(tokens):
            for j in range(0,ngram_size,1):
                if i+j+1<len(tokens)-1:
                    ngrams.append( " ".join(tokens[i:i+j+1]) )
        return {"tokens": ngrams}

def create_vocab(dataset,tokenizer,maximum_words_in_vocab):
    # Build a list of all posible words (tokens)
    train_tokens_list = []
    max_seq_length = 0
    for sample in dataset["train"].map(tokenizer, batched=False):
        train_tokens_list.append(sample["tokens"])
        if len(sample["tokens"])>max_seq_length:
            max_seq_length = len(sample["tokens"])

    # Build a dictionary ordered by word frequency
    counter = Counter(token for tokens in train_tokens_list for token in tokens)

    # Build vocab dictionary with at most 20k of the most frequent
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, word in enumerate(counter.keys(), start=2):
        if i>=maximum_words_in_vocab:
            break
        vocab[word] = [i,counter[word]]
    return vocab, max_seq_length

class VecFuncs():
    def __init__(self, vocabulary):
        self.vocab = vocabulary

    def vectorize_fn_multihot(self,tokens):
        temp_list = [0]*len(self.vocab)
        for token in tokens:
            temp_list[self.vocab.get(token,(1,1))[0]] = 1
        return temp_list

    # Term Frequency- Inverse Document Frequency Normalization
    def vectorize_fn_tf_idf(self,tokens):
        temp_list = [0]*len(self.vocab)
        for token in tokens:
            temp_list[self.vocab.get(token,(1,1))[0]] = self.td_idf(token,tokens)
        return temp_list

    # Function to map tokens → IDs (vectors)
    def vectorize_fn(self,tokens):
        return [self.vocab.get(token,(1,1))[0] for token in tokens]

    def td_idf(self, term, tokens):
        term_freq = tokens.count(term)
        doc_freq = math.log(float(self.vocab.get(term,(1,1))[1])+1.)
        return term_freq / doc_freq

def collate_fn(batch,vocab,vect_function):
    # for each sample in the batch, take the tokens and vectorize them based on the dictionary
    token_ids = [torch.tensor(vect_function(item["tokens"])) for item in batch]
    # Compute true lengths before padding
    lengths = torch.tensor([len(seq) for seq in token_ids])
    # create a label tensor
    labels = torch.tensor([item["label"] for item in batch])

    # Pad sequences in the batch
    token_ids_padded = pad_sequence(token_ids, 
                                    batch_first=True, # puts the batch dimension as first one in shape, rather than longest sequence length
                                    padding_value=vocab["<pad>"]) # pad with 0s

    return {"tokens": token_ids_padded, "lengths": lengths, "label": labels}


##############################################
##############################################
##############################################
# Data pipeline

'''
Do it with Hugging Face Datasets / PyTorch and avoid the manuals step

Hugging Face Datasets: https://huggingface.co/docs/datasets/en/index
'''
def get_data(ngram_size,max_vocab_length,batch_size): # vectorizing_function 
    # Load IMDB dataset
    dataset = load_dataset("imdb")

    # Apply preprocessing to *entire dataset*
    dataset = dataset.map(preprocess)

    # Now split into train/validation/test
    split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_data = split_dataset["train"]
    val_data   = split_dataset["test"]
    test_data  = dataset["test"]

    # Example: inspect a sample
    print('Text:',train_data[0]['text'])  # {'text': 'A very, very, very slow-moving, aimless movie...', 'label': 0}
    print('Label:',train_data[0]['label'])
    # Tokenization (basic example), N-grams with binary encoding
    # ngram_size = 2
    tokenize_with_ngrams = partial(tokenize, ngram_size = ngram_size)

    # Apply tokenization
    train_tokens = train_data.map(tokenize_with_ngrams, batched=False)
    val_tokens = val_data.map(tokenize_with_ngrams, batched=False)
    test_tokens = test_data.map(tokenize_with_ngrams, batched=False)

    print('Tokens:',train_tokens[0]["tokens"][:10])

    torch.manual_seed(0)
    # max_vocab_length = 20000
    vocab, max_seq_length = create_vocab(dataset,tokenizer=tokenize_with_ngrams,maximum_words_in_vocab=max_vocab_length)

    vectorizing_function = VecFuncs(vocab).vectorize_fn
    collate_with_vocab = partial(collate_fn, vocab = vocab, vect_function=vectorizing_function)

    # batch_size = 32
    train_loader = DataLoader(train_tokens, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            collate_fn=collate_with_vocab)

    val_loader = DataLoader(val_tokens, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            collate_fn=collate_with_vocab)

    test_loader = DataLoader(test_tokens, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            collate_fn=collate_with_vocab)
    
    
    return train_loader, val_loader, test_loader, vocab, max_seq_length



##############################################
##############################################
##############################################
# Models

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(BinaryClassifier, self).__init__()
        self.bc = nn.Sequential(nn.Linear(in_features = input_size, out_features = hidden_dim, bias = True, dtype = torch.float64),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(in_features = hidden_dim, out_features = 1, bias = True, dtype = torch.float64),
                                nn.Sigmoid())


    def forward(self, x, lengths): # lengths not needed in original model
        #logits w/o sigmoid
        probs = self.bc(x)
        return probs
    
class BinaryClassifierLSTM(nn.Module):
    def __init__(self, hidden_dim, vocab, embed_dim):
        super(BinaryClassifierLSTM, self).__init__()
        self.vocab = vocab
        # self.embed = OneHot(num_embeddings=len(vocab))  # or nn.Embedding # make embed size len(vocab)
        self.embed = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_dim, padding_idx=vocab["<pad>"]) # change embed_size
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features = 2*hidden_dim, out_features = 1, bias = True, dtype = torch.float64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, lengths):
        # x1: [batch, seq] LongTensor
        x = self.embed(x1.long())  # -> [batch, seq, embed_dim]
        # If using OneHot
        # mask = (x1 != self.vocab["<pad>"]).unsqueeze(-1) # shape [batch, seq, 1]. # make True all indices from original padded batch tensor that are not zero
        # x = x * mask  # Multiplies False (0) by all padded elements

        # lstm_out, _ = self.lstm(x.to(torch.float64))  # lstm_out: [batch, seq, 2*hidden_dim]
        # last_hidden = lstm_out[:, -1, :]  # take last timestep (or use pooling)

        # Pack
        packed = pack_padded_sequence(x, 
                                     lengths.cpu(), 
                                     batch_first=True, 
                                     enforce_sorted=False)

        # LSTM
        packed_out, (h_n, c_n) = self.lstm(packed)

        # If you want the last hidden state (handles variable lengths correctly):
        # h_n shape: [num_directions, batch, hidden_dim]
        # last_hidden = h_n.transpose(0,1).reshape(x.size(0), -1)  # [batch, 2*hidden_dim]

        # unpack
        unpacked, _ = pad_packed_sequence(packed_out, batch_first=True)  # [batch, max_len, 2*hidden_dim]

        # mask for mean pooling
        #[None, :] ensures [1,2,3,4,5,... ] is broadcast down (row) (each column has same value)
        #[:,None] emnsures the length for each sample in the batch is broadcast across (columns) (each row has same value)
        # unsqueeze adds an extra dimension back tyo broadcast correctly over the last index (vector one) of unpacked
        mask = (torch.arange(unpacked.size(1), device=lengths.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        masked_out = unpacked * mask
        pooled = masked_out.sum(dim=1) / lengths.unsqueeze(1)

        out = self.dropout(pooled)
        # out = self.dropout(last_hidden)
        out = self.fc(out)
        # logits w/o sigmoid
        return self.sigmoid(out)  # [batch, 1]

class OneHot(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, x):
        # x is expected to be LongTensor of word indices
        return F.one_hot(x, num_classes=self.num_embeddings).long()
    
class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, vocab, embed_dim, num_heads):
        super(TransformerEncoder, self).__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(num_embeddings=len(vocab), 
                                  embedding_dim=embed_dim, 
                                  padding_idx=vocab["<pad>"])
        
        self.attention = torch.nn.MultiheadAttention(embed_dim=embed_dim,
                                                    num_heads=num_heads, 
                                                    dropout=0.0, 
                                                    bias=True,
                                                    batch_first=True)

        self.dense_proj = nn.Sequential(nn.Linear(in_features = embed_dim, 
                                                  out_features = hidden_dim, 
                                                  bias = True, 
                                                  dtype = torch.float64),
                                                  nn.ReLU(),
                                                  nn.Linear(in_features = hidden_dim, 
                                                  out_features = embed_dim, 
                                                  bias = True, 
                                                  dtype = torch.float64)
                                            )
        
        self.layernorm1 = nn.LayerNorm(normalized_shape=(embed_dim))
        self.layernorm2 = nn.LayerNorm(normalized_shape=(embed_dim))

    def forward(self, x, lengths):
        batch_size, seq_len, embed_dim = x.size()
        key_padding_mask = torch.arange(seq_len, device=lengths.device)[None, :] >= lengths[:, None]
        # print('x',x.shape)
        att_out, att_weights = self.attention(query=x,
                                key=x,
                                value=x,
                                key_padding_mask=key_padding_mask, 
                                need_weights=True, 
                                attn_mask=None)
        # print('att',att_out.shape)
        proj_in = self.layernorm1(x+att_out[0]) # in this case, att_out and x have same shape
                                                # can change with different query, key, value
        # print('pin',proj_in.shape)
        proj_out = self.dense_proj(proj_in)
        # print('pout',proj_out.shape)
        out = self.layernorm2(proj_in+proj_out)
        return out, att_weights
    

class TransformerClassifier(nn.Module):
    def __init__(self, hidden_dim, vocab, embed_dim, num_heads, max_seq_length):
        super(TransformerClassifier, self).__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.embed = PositionalEmbedding(max_seq_length, embed_dim, vocab)
        # self.embed = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_dim, padding_idx=vocab["<pad>"])
        self.transformer_encoder = TransformerEncoder(hidden_dim, vocab, embed_dim, num_heads)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.dense = nn.Linear(in_features = embed_dim, 
                                                  out_features = 1, 
                                                  bias = True, 
                                                  dtype = torch.float64)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, lengths):
        # x: [batch, seq]
        x = self.embed(x.long())  # [batch, seq, embed_dim]

        # transformer outputs: [batch, seq, embed_dim]
        x, att_weights = self.transformer_encoder(x, lengths)

        # MAX POOLING: pool across sequence dimension
        x = x.permute(0, 2, 1)  # -> [batch, embed_dim, seq]
        x = self.global_max_pool(x).squeeze(-1)  # -> [batch, embed_dim]

        # MEAN POOLING: mask out padding before averaging
        # mask = (x.new_ones(x.size(0), x.size(1))).bool()
        # mask[torch.arange(x.size(0)).unsqueeze(1), lengths.unsqueeze(1)-1:] = False
        # # set pad positions to 0
        # x = x * mask.unsqueeze(-1)
        # pooled = x.sum(dim=1) / lengths.unsqueeze(1)

        # classifier head
        x = self.dropout(x)
        x = self.dense(x)
        return self.sigmoid(x)
    
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

    
##############################################
##############################################
##############################################
# Training
    
def training(train_data,
             val_data,
             epochs,
             loss_fn,
             model,
             optimizer,
             regularize = False):

    history = { 'train_loss':[],
               'val_loss':[],
               'accuracy':[],
               'weights':[], 
               'components_loss':[]
               }
    best_val_loss = torch.inf

    start = time()
    for epoch in tqdm(range(epochs)):
        last_step = False
        if epoch==epochs-1:
            last_step = True
        
        running_train_loss = 0.0
        for batch_idx, batch in enumerate(train_data):#tqdm(enumerate(train_loader)):
            # batched data
            input_data = batch['tokens'].to(device).to(torch.float64)
            lengths_data = batch['lengths'].to(device)
            label_data = batch['label'].to(device).to(torch.float64).unsqueeze(1)
            # Use a train function for separate backprop vs. no-backprop    
            train_loss, predictions, trained_weights = train(input_data, 
                                                            label_data, 
                                                            lengths_data,
                                                            model, 
                                                            loss_fn,
                                                            optimizer,
                                                            backprop=True, 
                                                            last_epoch = last_step, 
                                                            regularize = regularize)
            
            running_train_loss += train_loss.item() * input_data.shape[0] # weight loss by size of each batch

        train_loss = running_train_loss / len(train_data.dataset) # compute global average, center of "loss" (mass)

        running_val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        model.eval()
        for batch_idx, batch in enumerate(val_data):#tqdm(enumerate(train_loader)):
            # batched data
            input_data = batch['tokens'].to(device).to(torch.float64)
            lengths_data = batch['lengths'].to(device)
            label_data = batch['label'].to(device).to(torch.float64).unsqueeze(1)

            val_loss, predictions, _ = train(input_data, 
                                            label_data, 
                                            lengths_data,
                                            model, 
                                            loss_fn,
                                            optimizer, 
                                            backprop=False, 
                                            last_epoch = last_step, 
                                            regularize = regularize)
            
            running_val_loss += val_loss.item()*input_data.shape[0]
            predictions = (predictions>0.5).long()
            correct_predictions += (label_data.long()==predictions).sum().item()
            total_predictions += label_data.shape[0]
        
        val_loss = running_val_loss / len(val_data.dataset)
        accuracy = correct_predictions / total_predictions

        history['train_loss'].append(train_loss)#.cpu().detach().numpy())
        history['val_loss'].append(val_loss)#.cpu().detach().numpy())
        history['accuracy'].append(accuracy)
        if last_step:
            history['weights'].append(trained_weights) #.cpu().detach().numpy())
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if epoch % 1 == 0 or epoch==epochs-1: #args.test_interval == 0:
          print("Epoch: %d | Train Loss: %.8f | Val Loss: %.8f | Best Val Loss: %.8f | Accuracy: %.8f" % (epoch, train_loss, val_loss, best_val_loss, accuracy))

    end = time()
    print(f'Training Complete. Total Time (s): {end-start}')
    return {'history': history}

def train(input_data, label_data, lengths_data, model, loss_fn, optimizer, backprop=True, last_epoch = False, regularize = False):
    if backprop:
        model.train()
        predictions = model(input_data, lengths_data)
        loss = loss_fn(predictions, label_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        model.eval()
        with torch.inference_mode():#.no_grad():
            predictions = model(input_data, lengths_data)
            loss = loss_fn(predictions, label_data)

    trained_weights = None
    if last_epoch:
        # for i,param in enumerate(model.parameters()):
        #     trained_weights = param
        trained_weights = model.parameters()

    return loss, predictions, trained_weights
        
##############################################
##############################################
##############################################
# Visualizations

# -------------- True, predictions, and probabilities --------------
def outputs(model,dataloader,device = device):
    model.to(device)
    model.eval()
    y_true = []
    y_prob = []
    with torch.inference_mode():
        for batch in dataloader:
            input_data = batch['tokens'].to(device).to(torch.float64)
            lengths_data = batch['lengths'].to(device)
            label_data = batch['label'].to(device).to(torch.float64).unsqueeze(1)

            probabilities = model(input_data,lengths_data).squeeze()
            y_prob.append(probabilities.cpu().numpy())
            y_true.append(label_data.cpu().numpy())

    # create numpy tensors of true, predicted, and pred probabilies
    y_prob = np.concatenate(y_prob, axis = 0)
    y_true = np.concatenate(y_true, axis = 0)
    y_pred = (y_prob>=0.5).astype(np.int32)

    return y_true, y_pred, y_prob

# -------------- Scalar metrics --------------
def compute_metrics(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)                    # ACC
    prec = precision_score(y_true, y_pred, zero_division=0) # PPV
    rec = recall_score(y_true, y_pred, zero_division=0)     # TPR
    f1 = f1_score(y_true, y_pred, zero_division=0)          # F1
    roc_auc = roc_auc_score(y_true, y_probs)                # area under the ROC curve
    pr_auc = average_precision_score(y_true, y_probs)       # area under PR curve

    report = classification_report(y_true, y_pred, digits=4)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "report": report
    }

# -------------- Confusion matrix, ROC, PR curves --------------
def plot_confusion_matrix(y_true, y_pred, labels=["neg", "pos"], normalize=None):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # fig = plt.subplots(1,1,figsize=(5,4),constrained_layout = True)
    # plt.subplot(1,1,1)
    plt.figure(figsize=(5,4),constrained_layout = True)
    disp.plot(cmap=plt.cm.Blues, colorbar=True) # you can remove cmap if you want default
    plt.title("Confusion matrix" + (f" (normalized={normalize})" if normalize else ""))
    plt.show()

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
    plt.figure(figsize=(5,4),constrained_layout = True)
    # fig = plt.subplots(1,1,figsize=(5,4),constrained_layout = True)
    # plt.subplot(1,1,1)
    disp.plot()
    plt.title(f"ROC curve (AUC = {auc:.4f})")
    plt.show()

def plot_precision_recall(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=ap)
    plt.figure(figsize=(5,4),constrained_layout = True)
    # fig = plt.subplots(1,1,figsize=(5,4),constrained_layout = True)
    # plt.subplot(1,1,1)
    disp.plot()
    plt.title(f"Precision-Recall curve (AP = {ap:.4f})")
    plt.show()

# -------------- Threshold sweep to find best F1 --------------
def threshold_sweep(y_true, y_probs, thresholds=np.linspace(0.0, 1.0, 101)):
    best_t = 0.5
    best_f1 = -1
    results = []
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        results.append((t, f1, prec, rec))
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1, results



# ------------------- Image Plotting Function Example ------------------------
# def plot_single_generator(g,title, y_label):
#   max = torch.max(torch.view_as_real(g))
#   if g.type()=='torch.ComplexFloatTensor':
#     fig = plt.subplots(1,2,figsize=(6,2.8),constrained_layout = True)
#     plt.subplot(1,2,1)
#     plt.imshow(g.real,cmap='bwr_r',vmin=-max,vmax=max)
#     plt.title('Re('+title+')',fontsize=25)
#     # plt.axis('off')
#     plt.yticks([])
#     plt.xticks([])
#     plt.ylabel(y_label,fontsize=25)
#     plt.subplot(1,2,2)
#     im = plt.imshow(g.imag,cmap='bwr_r',vmin=-max,vmax=max)
#     plt.title('Im('+title+')',fontsize=25)
#     # plt.axis('off')
#     plt.yticks([])
#     plt.xticks([])
#     plt.colorbar()
#   else:
#     fig = plt.subplots(1,1,figsize=(4,3),constrained_layout = True)
#     plt.subplot(1,1,1)
#     plt.imshow(g,cmap='bwr_r',vmin=-max,vmax=max)
#     plt.title(title,fontsize=25)
#     plt.ylabel(y_label,fontsize=25)
#     # plt.axis('off')
#     plt.yticks([])
#     plt.xticks([])
#   plt.show()
