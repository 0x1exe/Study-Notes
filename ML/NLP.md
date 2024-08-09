
# Implementations
## Token embeddings
```
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
```
## Positional embeddings
```
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        return self.encoding[:seq_len, :]
        # it will add with tok_emb : [128, 30, 512]
        
```
## Transformer embeddings
```
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
```




## ScaleDotProduct attention
```
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v, mask=None, e=1e-12):
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        k_t = k.transpose(2, 3)  
        score = (q @ k_t) / math.sqrt(d_tensor) 
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score)
        v = score @ v
        return v,score

```
## MultiHead attention
```
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
```
## PositionWise FeedForward
```
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```



## Encoder layer
```
class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        _x = x
        x = self.ffn(x)
      
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
```

## Decoder layer
```
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
```
## Encoder and Decoder
```

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
    def forward(self, x, src_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
		
```


```
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        output = self.linear(trg)
        return output
```
## Transformer block
```

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
```

# 1. Learning representations that convey semantic and syntactic information

A central problem in NLP is how to represent words as input to any of our models. We need some notions of similarity and distance between words. Intuitively, we do feel that queen is closer to king than cat for instance.

Thus, we want to encode word tokens into some vectors that represent points in some 'word space'.

**Objective**: Finding a N-dimensional space (where N is between 100 and 1000) that is sufficient to encode all semantics of our language. In this word space, each dimension would encode some meaning that we transfer using speech (e.g. tense, count, gender...)
- **What are the four main steps of text processing?**
    
    1) **Loading**: Load text as string into memory
    
    2) **Cleaning**: Cleaning the text, correct misspellings, remove emojis
    
    3) **Tokenization**: Split strings into tokens, where a token could be a word or a character
    
    4) **Vectorization**: Map all the tokens in data into vectors for ease of feeding into models
    

- **What does bag of words refer to?**
    
    When we put all the words of a document in a _'bucket'_ we call such a bucket a bag of words. Simply put, a bag of words is a set of all the words in a document.
    

- **What are stop words?**
    
    Stop words are essentially high-frequency generic words that do not convey context-specific sense. E.g.: 'the', 'of', ...

- **What is TF-IDF vectorizer?**
    
    TF-IDF stands for term frequency - inverse data frequency. TF is the number of times a word appears in a document divided by the total number of words in the document. IDF is the log of the numbers of documents divided by the number of documents that contain the word w. TF-IDF is the product of those two quantities.
    

- **What is a word embedding?**
    
    Word embedding is a dense representation of words in the form of numeric vectors.

## 1.1. One-hot vector, a naïve approach (denotational semantics)
- **What is one-hot word representation?**
    
    Every word is represented as a V-dimensional vector (where V is the size of the vocabulary), with all 0s and 1 at the index of that word in the sorted English language.
    

- **What is denotational semantics?**
    
    The concept of representing an idea as a symbol. It is sparse and cannot capture similarity.
    

- **What is the issue of one-hot encoding?**
    
    No notion of similarity (no cosine similarity for instance).
## 1.2 SVD-based methods (distributional semantics)
- **What is distributional semantics?**
    
    The concept of representing the meaning of a word based on the context in which it usually appears. It is dense and can better capture similarity.
    
- **How do SVD-based methods work?**
    
    Loop over a massive dataset and accumulate word co-occurence counts in a matrix X. Then use SVD where the word vectors are the columns of U.
     ```
     def compute_co_occurrence_matrix(data, vocab_size):
    co_occurrence_matrix = [[0]*vocab_size for _ in range(vocab_size)]
    for document in data:
        for i in range(len(document)):
            word_i = document[i]
            for j in range(i + 1, len(document)):
                word_j = document[j]
                co_occurrence_matrix[word_i][word_j] += 1
                co_occurrence_matrix[word_j][word_i] += 1
    
    return co_occurrence_matrix

    ```
    
- **What is latent semantic analysis (LSA)?**
    
    LSA is a technique of analyzing relationships between a set of documents and the terms they contain by producing a set of concepts related to the documents and terms. To do so, we build a matrix whose coefficients are word counts per document and use SVD to reduce the number of rows while preserving the similarity structure among columns. To compare documents, we compute the cosine similarity between the column vectors representing them.


### 1.2.1. Word-document matrix
- **What is the assumption of a word-document matrix?**
    
    We make the bold conjecture that words that are related will often appear in the same documents. For instance, _banks_, _bonds_ and _stocks_ are probably likely to appear together. But _banks_, _octopus_ and _hockey_ would probably not consistently appear together.

- **What is the issue of a word-document matrix?**
    
    A very large matrix that scales with the number of documents.
### 1.2.2. Window-based co-occurrence matrix
- **What is a window-based co-occurrence matrix?**
    
    A matrix stores co-occurrences of words thereby becoming an affinity matrix. In this method, we count the number of times each word appears inside a window of a particular size around the word of interest.
    

- **How to extract word vectors from a word-word co-occurrence matrix?**
    
    Generate a VxV co-occurrence matrix X. Apply SVD on X. Select the first k columns of U to get a k-dimensional word vectors. The ratio of the variances indicates the amount of variance captured by the first k dimensions.
    

- **What are the problems faced with such a method?**
    - The dimensions of the matrix change very often
    - The matrix is sparse
    - Very high dimensional
    - Quadratic cost to train

```
def compute_co_occurrence_matrix(data, vocab_size, window_size):
    co_occurrence_matrix = [[0]*vocab_size for _ in range(vocab_size)]
    for document in data:
        for i, word_i in enumerate(document):
            start = max(0, i - window_size)
            end = min(len(document), i + window_size + 1)
            for j in range(start, end):
                if j == i:
                    continue
                word_j = document[j]
                co_occurrence_matrix[word_i][word_j] += 1
                co_occurrence_matrix[word_j][word_i] += 1
    return co_occurrence_matrix

```

## 1.3. Word2Vec
- **What is an iteration-based model?**
    
    A model that is able to learn one iteration at a time and eventually be able to encode the probability of a word given its context.

- **What is Word2Vec?**
    
    A model whose parameters are the word vectors. Train the model on a certain objective. At every iteration, we run our model, evaluate the errors and backpropagate the gradients in the model.

- **What are the initial embeddings of Word2Vec model?**
    
    The embedding matrix is initialized randomly using a Normal or uniform distribution. Then, the embedding of word _i_ in the vocabulary is the row _i_ of the embedding matrix.

- **What are the two algorithms used by Word2Vec? Explain how they work.**
    
    Continuous bag-of-words (CBOW):
	    The model predicts the current word (target word) based on the context words (surrounding words). The input to the model is a fixed-size window of context words, and the output is the target word. CBOW treats the context words as the input and predicts the target word. CBOW is typically faster to train compared to Skip-gram because it requires predicting one target word from multiple context words. CBOW tends to perform better when the task involves predicting frequent words because it averages the context words to predict the target word. CBOW tends to produce smoother word representations because it averages the context words.
	Skip-gram:
		In the Skip-gram architecture, the model predicts the context words (surrounding words) based on the current word (target word). The input to the model is the target word, and the output is a set of context words. Skip-gram treats the target word as the input and predicts the context words. Skip-gram is slower to train compared to CBOW because it requires predicting multiple context words for each target word. Skip-gram tends to perform better when the task involves predicting infrequent words or capturing fine-grained semantic relationships because it learns to differentiate between different context words. Skip-gram tends to produce richer word representations because it learns to distinguish between different context words.

- **What are the two training methods used?**

    Hierarchical softmax
    Negative sampling
    
- **What is the advantage of Word2Vec over SVD-based methods?**    
    Much faster to compute and capture complex linguistic patterns beyond word similarity

- **What is the limitation of Word2Vec?**
    Fails to make use of global co-occurrence statistics. It only relies on local statistics (words in the neighborhood of word _i_).
    E.g.: The cat sat on the mat. Word2Vec doesn't capture if _the_ is a special word in the context of cat or just a stop word.

### Word2Vec [CBOW]
```
class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )
    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x
```
### Word2Vec [Skip-gram]
```
class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )
    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x
```
## 1.4 GloVe
- **What is GloVe?**
    
    GloVe aims to combine the SVD-based approach and the context-based skip-gram model.
    
- **How to build a co-occurence matrix for GloVe? What can we calculate with such a matrix?**
    
    Let X be a word-word co-occurence matrix (coefficients are the number of times word _i_ appears in the context of word _j_). With this matrix, we can compute the probability of word _i_ appearing in the context of word j: _Pij = Xij / Xi_
    
- **How is GloVe built?**
    
    After building the co-occurence matrix, GloVe computes the ratios of co-occurrence probabilities (non-zero). The intuition is that the word meanings are capture by the ratios of co-occurrence probabilities rather than the probabilities themselves. The global vector models the relationship between two words regarding to the third context word as:
	$$
	F(w_i, w_j, \tilde{w}_k) = \frac{p_{\text{co}}(\tilde{w}_k \vert w_i)}{p_{\text{co}}(\tilde{w}_k \vert w_j)}
	$$
    F is designed to be a function of the linear difference between two words _wi_ and _wj_. It is an exponential function.

- **What are the pros of GloVe?**
    
    The GloVe model efficiently leverages global statistical information by training only on non-zero elements in a word-word co-occurence matrix, and produces a vector space with meaningful substructure.
    
- **What is window classification and why is it important?**
    
    Natural languages tend to use the same word for very different meanings and we typically need to know the context of the word usage to discriminate between meanings.
    
    E.g.: 'to sanction' means depending on the context 'to permit' or 'to punish'
    
    A sequence is a central word vector preceded and succeeded by context word vectors. The number of words in the context is also known as the context window size and varies depending on the problem being solved.
    
-  **How do window size relate to performance?**
     
    Generally, narrower window size lead to better performance in syntactic tests while wider windows lead to better performance in semantic tests.




# 2. How to create language models
## 2.1. N-gram models

Objective: Compute the probability of occurrence of a number of words in particular sequence looking at the n previous words.

- **What are the 2 limitations of n-gram language models?**
    - **Sparsity problems**: if n-gram never appears in corpus, then probability is 0.
    - **Storage problems**: as n increases or the corpus size increases, the model size increases as well.

Since we only consider the n previous words in the sequence, we cannot take advantage of the full information conveyed in a sequence.

## 2.2. RNNs
Objective: Condition the language model on all previous words in the corpus.

- **How is organized RNN?**
    
    A RNN is organized in a series of hidden layers holding a number of neurons, each of which performs a linear matrix operation on its inputs followed by a non-linear operation.
    
    At each time-step, there are two inputs to the hidden layer: the output of the previous layer _h(t-1)_, and the input at that timestep _x(t)_.

$$
h_{t} = \sigma(W^{(hh)}h_{t-1}+W^{(hx)}x_{t})
$$
$$
\hat{y_{t}}=softmax(W^{(S)}h_{t})
$$
- **How does a RNN solve the curse of dimensionality problem incurred by n-gram language models?**
    
    It is solved since the weight matrices are applied at every step of the network. Hence the model parameters don't grow proportionally to the input sequence size. The number of parameters is independent of the sequence length.
    
- **What is the loss function of a RNN?**
    
    Cross-entropy summed over a corpus of size T and a vocabulary of size V.
    
- **What is the perplexity of a RNN and what does it mean to have a low perplexity?**
    
    The perplexity of a RNN is 2 to the negative log probability of the cross entropy loss function.
    
    Perplexity is a measure of confusion where lower values imply more confidence in predicting the next word in the sequence.
    
    A perplexity for a language model of 247 means that the model is as confused/perplex as if it has to choose uniformly and independently among 247 possibilities for each word.
- Problems? Vanishing gradient
- **How to solve vanishing gradient problem?**
    
    **Technique 1**: Instead of initializing W randomly, start off from an identity matrix initialization.
    
    **Technique 2**: Use ReLU as an activation function since the derivative of the gradient is either 0 or 1. This way, gradients would flow through the neurons whose derivatives is 1 without getting attenuated while propagating back through time-steps.
    

- **What are exploding gradients and give a technique on how to solve them?**
    
    The explosion occurs through exponential growth by repeatedly multiplying gradients through the network layers that have values larger than 1.0.
    
    A technique to solve exploding gradients is gradient clipping. Gradient clipping is a simple heuristic invented by Thomas Mikolov to counter the effect of exploding gradient. That is, whenever the gradient reach a certain threshold, they are set back to a small number.

## 2.3. Deep bidirectional RNNs
## 2.4 LSTMs and GRUs

- **Why do we need GRU?**
    Although RNNs can theoretically capture long-term dependencies, they are very hard to actually train to do this. GRU are designed in a manner to have more persistent memory thereby making it easier for RNNs to capture long-term dependencies.
- **What are the 4 gates of a GRU cell?**
    - New memory generation
    - Reset gate
    - Update gate
    - Hidden state
- **What are the new memory stage and reset gate in a GRU?**
	The reset signal is responsible for determining how important h(t-1) is to the summarization of h(t). The reset gate has the ability to completely diminish past hidden state if it finds that h(t-1) is irrelevant to the computation of the new memory.
	$$
	r_{t}=\sigma(W^{(r)}x_{t}+U^{(r)}h_{t-1})
	$$
	This stage is the one who knows the recipe of combining a newly observed word with the past hidden state h(t-1) to summarize this new word in light of the contextual past.
	$$h_{t}=tanh(r_{t}\circ Uh_{t-1}+Wx_{t})$$
- **What is the update gate in a GRU?**
	The update gate is responsible for how much of h(t-1) should be carried forward to the next state. $$
z_{t}=\sigma(W^{(z)}x_{t}+U^{(z)}h_{t-1})$$
- **Give the equation of the new hidden state in a GRU cell.**
	$$ h_{t}=(1-z_{t})\circ \hat{h}_{t}+z_{t}\circ h_{t-1}$$
- **What are the 6 memory cells of LSTM?**
    - Input gate
    - Forget gate
    - Output/exposure gate
    - New memory cell
    - Final memory cell
- **What is the difference between GRU and LSTM cells?**
    
    The GRU controls the flow of information like the LSTM unit, but without having a memory unit. It just exposes the full hidden content without any control.
    
    GRU performance is on par with LSTM, but computationally more efficient (less complex structure as pointed out).
### LSTM code 
```
import math

class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, 
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
```
# 3. Seq2Seq
## 3.1. Seq2Seq models
- **What is the advantage of Seq2Seq models over regular RNNs?**
    
    Seq2Seq models can generate arbitrary output sequences after seeing the entire input. They can even focus in on specific parts of the input automatically to help generate a useful translation.
    
    E.g.: 1 encoder and 1 decoder (both LSTM or bi-LSTM)
- **What does an encoder do in a Seq2Seq model?**
    
    It reads the input sequence and generate a fixed-dimensional context vector C for the sequence.
    
    The encoder stacks multiple RNNs on top of each other to effectively compress an arbitrary-length sequence into a fixed-size vector. The final layer's LSTM hidden state will be used as a context vector.
    
- **What does a decoder do in a Seq2Seq model?**
    
    It uses the context vector as a _'seed'_ from which to generate an output sequence.

- **In which order does an encoder read a sentence?**
    Reverse
- **How does a decoder work in a Seq2Seq model?**
    
    We initialize the hidden state of our first layer with the context vector, we then pass an token appended to the end of the input. We then run the three-stacked RNN, following up with a softmax on the final layer's output to generate the first word.
### Seq2Seq code
```
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
	    # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell
```

```
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
    
```
## 3.2 Attention mechanism

- **How does attention work?**
    
    We compute an attention vector. It is a vector of weights that are used to compute the context vector as a weighted average of the hidden states generated by the encoder at time steps _1, ..., n_.
    
    The decoder network is provided with a look at the entire input sequence at every decoding step since it is fed with the context vector. The decoder can then decide what input words are important.
    
- **What is global attention?**
    
    Instead of giving to the decoder's LSTM cells a single context vector which is a weighted average of all the hidden states of the encoder (attention). LSTM cells in the decoder are fed with the **concatenation** of the encoder hidden states at time i and the context vector.
    
- **Give 4 algorithms that can be used by decoders to search translations.**
    
    Exhaustive search, ancestral sampling, greedy search, beam search
    

- **How does beam search work?**
    
    We maintain K candidates at each time step. To do so, we compute H(t+1) by expanding H(t) and keeping the best K candidates (the ones with the highest probability).


# 4. Dealing with large vocabularies 
Seq2Seq models have a hard time dealing with large vocabulary size. These models predict the next word in the sequence by computing a target probabilistic distribution over the entire vocabulary using softmax. **Softmax is expensive to compute and its complexity scales proportionally to the vocabulary size**.
## 4.1. Scaling softmax

- **Give two techniques to scale softmax**.
    
    Noise contrastive estimation and hierarchical softmax
    
- **What is the issue with both techniques?**
    
    Both methods save computation during training step (when target word is known). At test time, one still has to compute the probability of all words in the vocabulary in order to make predictions.
    

## 4.2. Word and character-based models

Phonology posits a small set or sets of distinctive, categorical units that we call **phonemes**. Let's try to change our perspective: instead of working with words or even splitting words into characters, let's now consider **n-gram characters**.

- **Name three families of techniques to deal with unknown or rare words.**
    
    Word segmentation (adaptation of Byte-pair encoding), character-based model, hybrid model
    

### 4.2.1. Word segmentation

- **How does word segmentation solve the issue related to large vocabulary?**
    
    Word segmentation represents unknown or rare words as a sequence of subword units.
    

- **What is Byte-pair encoding?**
    
    Byte-pair encoding is a lossless compression algorithm. In NLP, it is adapted to transform rare words into character tokens.
    
    For instance: athazagoraphobia becomes ['_ath', 'az', 'agor', 'aphobia']
    
    It follows the following procedure:
    
    1. Represent each word in the corpus as a combination of the characters along with the special end of word token </w>
    2. Iteratively count character pairs in all tokens of the vocabulary
    3. Merge every occurrence of the most frequent pair, add the new character n-gram to the vocabulary
    4. Repeat step 3 until the desired number of merge operations are completed or the desired vocabulary size is achieved
        
        > {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
        > 
        > {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w es t </w>': 6, 'w i d es t </w>': 3}
        > 
        > {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est </w>': 6, 'w i d est </w>': 3}
        

- **How does word segmentation work?**
    
    Start with a vocabulary of characters and keep extending the vocabulary with most frequent n-gram pairs in the data set. This process is repeated until all n-gram pairs are selected or vocabulary size reaches some threshold.
    

- **What is the difference between WordPiece and SentencePiece?**
    
    WordPiece is an algorithm that creates smaller units than words (sub-word level).
    
    SentencePiece - created by Google - is an algorithm that combines sub-word level tokenization (BPE) as well as unigram tokenization.
    

### 4.2.2. Character-based models

- **How are character-level embeddings generated?**
    
    Character-level embeddings use one-dimensional CNN to find numeric representation of words by looking at their character-level compositions.
    

### 4.2.3. FastText embeddings

- **Give a brief explanation of FastText embeddings.**
    
    Simply put, FastText is a word2vec-like algorithm that leverages n-gram word pieces to encode a large and open vocabulary and to capture word morphology.
    

- **How are words represented in FastText?**
    
    They are represented as char n-grams augmented with boundary symbols and as whole word.
    
    E.g.: where = <wh, whe, her, ere, re>, < where >
    

- **How are FastText embeddings generated?**
    
    A bi-LSTM is trained to compute embeddings.
    

### 4.2.4. Hybrid NMT

- **How does a hybrid Neural Machine Translation work?**
    
    The system translates mostly at word-level and consults the character components for rare words.
    

- **Give the structure of a hybrid NMT.**
    
    A word-based translation as a backbone
    
    A source character-based representation
    
    A target character-level generation
    

- **What is the purpose of a word-based translation as a backbone in a hybrid NMT?**
    
    This is the core of a hybrid NMT (LSTM encoder-decoder) that translates at the word level. We use to represent OOV words.
    

- **What is the purpose of a source character-based representation model in a hybrid NMT?**
    
    A deep LSTM model that learns over characters of rare words and use the final hidden state of the LSTM as the representation for the rare word.
    

- **What is the purpose of a target character-level generation model?**
    
    The goal is to create a coherent representation that handles unlimited output vocabulary. To do so, we have a separate deep LSTM that _'translates'_ at the character-level given the current word-level state.

# 5. How to create contextual word embeddings?

So far, we have always the same representation for a word type regardless of the context in which a word token occurs. Word embeddings are context-free with GloVe, Word2Vec or FastText. Those language models are trained to predict the next word but they are producing context-specific word representations at each position.

E.g.: I ate an apple. I have an Apple iPhone.

In both cases, the word _'apple'_ shares the same word embedding. However, they mean two radically different things.

## 5.1. ELMo (Embeddings from language model)

- **What is ELMo?**
    
    ELMo learns contextualized word representation by pre-training a language model in an unsupervised way.
    

- **How is ELMo trained?**
    
    The bidirectional language model is the foundation of ELMo. While the input is a sequence of **n** tokens, the language model learns to predict the probability of next token given the history.
    The model is trained to minimize the negative log-likelihood in both directions:

- **How does ELMo learn task-specific representations?**
    
    On top of a L-layer biLM, ELMo stacks all the hidden states across layers together by learning a task-specific linear combination.
    
    The weights, s_task, in the linear combination are learned for each end task and normalized by softmax. The scaling factor γtask is used to correct the misalignment between the distribution of biLM hidden states and the distribution of task specific representations.
    
	    $$
	    v_i = f(R_i; \Theta^\text{task}) = \gamma^\text{task} \sum_{\ell=0}^L s^\text{task}_i \mathbf{h}_{i,\ell}
	    $$
    

- **To which tasks correspond which layers?**
    
    The comparison study indicates that syntactic information is better represented at lower layers while semantic information is captured by higher layers. Because different layers tend to carry different type of information, **stacking them together helps**.
    

## 5.2. ULMFiT

- **What is innovative with ULMFiT?**
    
    ULMFiT is the first model to introduce the idea of generative pretrained language models that is fine-tuned for a specific task.
    

- **What are the three steps to achieve good transfer learning results?**
    
    1. General LM pre-training (already done by [fast.ai](http://fast.ai/) researchers) on Wikipedia
    2. Target task LM fine-tuning: finetuning LM on a specific vocabulary
    3. Train a target task classifier with 2 fully-connected layers
- **What are the two techniques used when fine-tuning LM?**
    
    1. **Discriminative fine-tuning**: tune each layer with different learning rates
    2. **Slanted triangular learning rates**: triangular learning rate schedule
- **What are the two techniques used when fine-tuning the classifier?**
    
    1. **Concat pooling** extracts max-polling and mean-pooling over the history of hidden states and concatenates them with the final hidden state.
    2. **Gradual unfreezing** extracts max-polling and mean-pooling over the history of hidden states and concatenates them with the final hidden state.

## 5.3. Transformers

### 5.3.1. OpenAI GPT

- **What does GPT stand for?**
    
    Generative Pre-training transformer
    

- **What are the two major difference between ELMo and OpenAI GPT?**
    1. The model architectures are different: ELMo uses a shallow concatenation of independently trained left-to-right and right-to-left multi-layer LSTMs, while GPT is a multi-layer transformer decoder.
    2. ELMo feeds embeddings into models customized for specific tasks as additional features, while GPT fine-tunes the same base model for all end tasks.

- **What is the major upgrade brought by OpenAI GPT?**
    
    To get rid of the task-specific model and use the pre-trained language model directly.
    
    Hence we don't need new a new design for specific tasks. We just need to modify the input sequence by adding custom tags. At the first stage, generative pre-training on a language model can absorb as much free text as possible. Then at the second stage, the model is fine-tuned on specific tasks with a small labeled dataset and a minimal set of new parameters to learn.
    

- **What is the loss of OpenAI GPT?**
    
    The loss is the negative log-likelihood for true labels to which we add the LM loss as an auxiliary loss.
    $$
    \mathcal{L}_\text{cls} = \sum_{(\mathbf{x}, y) \in \mathcal{D}} \log P(y\mid x_1, \dots, x_n) = \sum_{(\mathbf{x}, y) \in \mathcal{D}} \log \text{softmax}(\mathbf{h}_L^{(n)}(\mathbf{x})\mathbf{W}_y)
    $$
    $$
\mathcal{L}_\text{LM} = -\sum_{i} \log p(x_i\mid x_{i-k}, \dots, x_{i-1})
    $$
    

- **What is a limitation of OpenAI GPT?**
    
    Its unidirectional nature.


### 5.3.2. BERT
- **What is the biggest difference between OpenAI GPT and BERT? Which limitation does it solve?**
    
    OpenAI is unidirectional. BERT is bidirectional.
    

- **What is the base component of BERT?**
    
    The model architecture of BERT is a multi-layer bidirectional Transformer encoder. It is composed of Multi-headed self attention, feed-forward layers, layer norm and residuals and positional embeddings.
    

- **What are the tasks on which BERT is trained?**
    
    1. **Mask language model (MLM)**: Randomly mask 15% of tokens in each sequence.
        
        BERT employed several heuristic tricks:
        
        - (a) with 80% probability, replace the chosen words with `[MASK]`;
        - (b) with 10% probability, replace with a random word;
        - (c) with 10% probability, keep it the same.
    2. **Next sentence prediction**: tell whether one sentence is the next sentence of the other. It is used to learn relationships between sentences.
        

- **Give the 3 components that constitutes BERT input embedding.**
    
    1. WordPiece embeddings (cf. previously)
    2. Segment embeddings
    3. Position embeddings (are learned)

- **What is the structure of BERT base?**
    
    12 layers, 768-dimensional output hidden state, 12 heads
    

- **What is the structure of BERT large?**
    
    24 layers, 1024-dimensional output hidden state, 16 heads
    

- **What is the [SEP] token used for?**
    
    [SEP] token is used when building a sequence from multiple sequences
    
    E.g.: two sequences for sequence classification or for a text and a question for question answering
    

- **What is the [PAD] token used for?**
    
    The token used for padding, for example when batching sequences of different lengths
    

- **What is the [CLS] token used for?**
    
    The classifier token which is used when doing sequence classification. It is the first token of the sequence when built with special tokens.
    

### 5.3.3. RoBERTa

- **What does RoBERTa stand for?**
    
    Robustly optimized BERT approach
    

### 5.3.4. ALBERT
- **What are the 3 innovations brought by ALBERT?**
    1. Factorized Embedding parametrization
    2. Cross-layer parameter sharing
    3. Sentence order prediction

- **What is factorized embedding parametrization?**
    
    In BERT, the WordPiece tokenization embedding size **_E_** is configured to be the same as the hidden state size **_H_**. That is saying, if we want to increase the model size (larger **_H_**), we need to learn a larger tokenization embedding too, which is expensive because it depends on the vocabulary size (**_V_**).
    
    Conceptually, because the tokenization embedding is expected to learn _context-independent_ representation and the hidden states are _context-dependent_, it makes sense to separate the size of the hidden layers from the size of vocabulary embedding. Using factorized embedding parameterization, the large vocabulary embedding matrix of size **_V×H_** is decomposed into two small matrices of size **_V×E_** and **_E×H_**.
    

### 5.3.5. ELECTRA

### 5.3.6. DistilBERT
- **How does distillation work?**
    
    - Train "Teacher": Use SOTA pre-training + fine-tuning technique to train model with maximum accuracy
        
    - Label a large amount of unlabeled input examples with Teacher
        
    - Train "Student": much smaller model which is trained to mimic Teacher output
        
    - Student objective is typically MSE or cross-entropy
        

### 5.3.7. XLNet
- **What are two innovations of XLNet?**
    
    1. Relative position embeddings
    2. Permutation language modelling


# Approximating softmax
Recall that the softmax calculates the probability of a word w given its context c and can be computed using the following equation:
$$
p(w | c) = \dfrac{\text{exp}({h^\top v'_w})}{\sum_{w_i \in V} \text{exp}({h^\top v'_{w_i}})}
$$
Computing the softmax is expensive as the inner product between hℎ and the output embedding of every word w_i in the vocabulary V needs to be computed as part of the sum in the denominator in order to obtain the normalized probability of the target word w given its context c.
Softmax-based approaches are methods that keep the softmax layer intact, but modify its architecture to improve its efficiency. Sampling-based approaches on the other hand completely do away with the softmax layer and instead optimise some other loss function that approximates the softmax.

## Hierarchical softmax

Hierarchical softmax (H-Softmax) is an approximation inspired by binary trees that was proposed by Morin and Bengio (2005) (https://www.ruder.io/word-embeddings-softmax/#fn3). H-Softmax essentially replaces the flat softmax layer with a hierarchical layer that has the words as leaves.

This allows us to decompose calculating the probability of one word into a sequence of probability calculations, which saves us from having to calculate the expensive normalization over all words.

We can think of the regular softmax as a tree of depth 11, with each word in V as a leaf node. Computing the softmax probability of one word then requires normalizing over the probabilities of all |V| leaves. If we instead structure the softmax as a binary tree, with the words as leaf nodes, then we only need to follow the path to the leaf node of that word, without having to consider any of the other nodes.

Since a balanced binary tree has a depth of log2(|V|), we only need to evaluate at most log2(|V|) nodes to obtain the final probability of a word. Note that this probability is already normalized, as the probabilities of all leaves in a binary tree sum to 11 and thus form a probability distribution. To informally verify this, we can reason that at a tree's root node, the probabilities of branching decisions must sum to 11. At each subsequent node, the probability mass is then split among its children, until it eventually ends up at the leaf nodes, i.e. the words. Since no probability is lost along the way and since all words are leaves, the probabilities of all words must necessarily sum to 11 and hence the hierarchical softmax defines a normalized probability distribution over all words in V.

We can now calculate the probability of going right (or left) at a given node n given the context c the following way:
$$
p(\text{right} \| n, c) = \sigma (h^\top v'_{n})
$$
 Conversely, the probability of turning left is simply $$1 - p(  \text{right}   \|    n,c)$$
 Notably, we are only able to obtain this speed-up during training, when we know the word we want to predict (and consequently its path) in advance. During testing, when we need to find the most likely prediction, we still need to calculate the probability of all words, although narrowing down the choices in advance helps here.
#### A note on the information content of words

Recall that the information content I(w) of a word w is the negative logarithm of its probability p(w):
$$
I(w) = -\log_2 p(w)
$$
The entropy H of all words in a corpus is then the expectation of the information content of all words in the vocabulary:
$$
H = \sum_{i\in V} p(w_i) I(w_i)
$$

For a balanced binary tree, where we treat every word equally, the word entropy H equals the information content I(w)\ of every word w, as each word has the same probability. The average word entropy H in a balanced binary tree with |V|=10000 thus coincides with its average path length. We saw before that the structure of the tree is important. Notably, we can leverage the tree structure not only to gain better performance, but also to speed up computation: If we manage to encode more information into the tree, we can get away with taking shorter paths for less informative words. Morin and Bengio point out that leveraging word probabilities should work even better; as some words are more likely to occur than others, they can be encoded using less information
#### Hierarchical softmax [Code]  
```
class HierarchicalSoftmax(nn.Module):
    def __init__(self, ntokens, nhid, ntokens_per_class = None):
        super(HierarchicalSoftmax, self).__init__()
        self.ntokens = ntokens
        self.nhid = nhid

        if ntokens_per_class is None:
            ntokens_per_class = int(np.ceil(np.sqrt(ntokens)))

        self.ntokens_per_class = ntokens_per_class

        self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))
        self.ntokens_actual = self.nclasses * self.ntokens_per_class

        self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True)
        self.layer_top_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)

        self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.nclasses, self.nhid, self.ntokens_per_class), requires_grad=True)
        self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.nclasses, self.ntokens_per_class), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):

        initrange = 0.1
        self.layer_top_W.data.uniform_(-initrange, initrange)
        self.layer_top_b.data.fill_(0)
        self.layer_bottom_W.data.uniform_(-initrange, initrange)
        self.layer_bottom_b.data.fill_(0)


    def forward(self, inputs, labels = None):

        batch_size, d = inputs.size()

        if labels is not None:

            label_position_top = labels / self.ntokens_per_class
            label_position_bottom = labels % self.ntokens_per_class

            layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
            layer_top_probs = self.softmax(layer_top_logits)

            layer_bottom_logits = torch.squeeze(torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top]), dim=1) + self.layer_bottom_b[label_position_top]
            layer_bottom_probs = self.softmax(layer_bottom_logits)

            target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * layer_bottom_probs[torch.arange(batch_size).long(), label_position_bottom]

            return target_probs

        else:
            # Remain to be implemented
            layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
            layer_top_probs = self.softmax(layer_top_logits)

            word_probs = layer_top_probs[:,0] * self.softmax(torch.matmul(inputs, self.layer_bottom_W[0]) + self.layer_bottom_b[0])

            for i in range(1, self.nclasses):
                word_probs = torch.cat((word_probs, layer_top_probs[:,i] * self.softmax(torch.matmul(inputs, self.layer_bottom_W[i]) + self.layer_bottom_b[i])), dim=1)

            return word_probs
```

## Differentiated softmax
### CNN-Softmax

# Sampling based approaches
While the approaches discussed so far still maintain the overall structure of the softmax, sampling-based approaches on the other hand completely do away with the softmax layer. They do this by approximating the normalization in the denominator of the softmax with some other loss that is cheap to compute. However, sampling-based approaches are only useful at training time -- during inference, the full softmax still needs to be computed to obtain a normalised probability.

In order to gain some intuitions about the softmax denominator's impact on the loss, we will derive the gradient of our loss function w.r.t. the parameters of our model.

The crux of most sampling-based approach now is to approximate the negative reinforcement in some way to make it easier to compute, since we don't want to sum over the probabilities for all words in V.
## Importance sampling
We can approximate the expected value E of any probability distribution using the Monte Carlo method, i.e. by taking the mean of random samples of the probability distribution. If we knew the network's distribution, i.e. P(w), we could thus directly sample m words w1,⋯,wm1,⋯, from it and approximate the above expectation with:
$$
\mathbb{E}_{w_i \sim P}[\nabla_\theta \mathcal{E}(w_i)] \approx \dfrac{1}{m} \sum\limits^m_{i=1} \nabla_\theta \mathcal{E}(w_i)
$$
However, in order to sample from the probability distribution P, we need to compute P�, which is just what we wanted to avoid in the first place. We therefore have find some other distribution Q (we call this the proposal distribution), from which it is cheap to sample and which can be used as the basis of Monte-Carlo sampling. Preferably, Q  should also be similar to P, since we want our approximated expectation to be as accurate as possible. A straightforward choice in the case of language modelling is to simply use the unigram distribution of the training set for Q
## Noise Contrastive Estimation

 A more stable sampling method than Importance Sampling (IS), as we have seen that IS poses the risk of having the proposal distribution Q� diverge from the distribution P that should be optimized. In contrast to the former, NCE does not try to estimate the probability of a word directly. Instead, it uses an auxiliary loss that also optimises the goal of maximizing the probability of correct words.
 
We train a model to differentiate the target word from noise. We can thus reduce the problem of predicting the correct word to a binary classification task, where the model tries to distinguish positive, genuine data from noise samples

For every word w_i given its context c_i of n previous words  in the training set, we thus generate k noise samples w_ik from a noise distribution Q. As in IS, we can sample from the unigram distribution of the training set. As we need labels to perform our binary classification task, we designate all correct words w_i given their context c_i as true (y=1) and all noise samples w_ik as false (y=0).

We can now use logistic regression to minimize the negative log-likelihood, i.e. cross-entropy of our training examples against the noise (conversely, we could also maximize the _positive_ log-likelihood as some papers do). Instead of computing the expectation E_w_ik∼Q of our noise samples, which would still require summing over all words in V to predict the normalised probability of a negative label, we can again take the mean with the Monte Carlo approximation:
$$
J_\theta = - \sum_{w_i \in V} [ \text{log}    P(y=1  |  w_i,c_i) + k    \sum_{j=1}^k \dfrac{1}{k}   \text{log}    P(y=0   |  \tilde{w}_{ij},c_i)]
$$

By generating k noise samples for every genuine word w_i given its context c, we are effectively sampling words from two different distributions: Correct words are sampled from the empirical distribution of the training set P_train  and depend on their context c, whereas noise samples come from the noise distribution Q. We can thus represent the probability of sampling either a positive or a noise sample as a mixture of those two distributions, which are weighted based on the number of samples that come from each:
$$
P(y, w    \|    c) = \dfrac{1}{k+1} P_{\text{train}}(w    \|    c)+ \dfrac{k}{k+1}Q(w)
$$
Given this mixture, we can now calculate the probability that a sample came from the training Ptrain�train distribution as a conditional probability of y given w and c:
$$
P(y=1  \|  w,c)= \dfrac{P_{\text{train}}(w    \|    c)}{P_{\text{train}}(w    \|    c) + k    Q(w)}
$$
As we don't know P_train (which is what we would like to calculate), we replace P_train with the probability of our model P.
Note that computing P(w|c), i.e. the probability of a word w given its context c is essentially the definition of our softmax.

For notational brevity and unambiguity, let us designate the denominator of the softmax with Z(c), since the denominator only depends on hℎ, which is generated from c (assuming a fixed V). The softmax then looks like this:
$$
P(w    \|    c) = \dfrac{\text{exp}({h^\top v'_{w}})}{Z(c)}
$$
We can treat the normalisation denominator Z(c) as a parameter that the model can learn.  
Mnih and Teh (2012) and Vaswani et al. (2013 actually keep Z(c) fixed at 11, which they report does not affect the model's performance. This assumption has the nice side-effect of reducing the model's parameters, while ensuring that the model self-normalises by not depending on the explicit normalisation in Z(c).

$$
P(w    \|    c) = \text{exp}({h^\top v'_{w}})
$$
We can now insert this term in the above equation to compute P(y=1|w,c):
$$
P(y=1  \|  w,c)= \dfrac{\text{exp}({h^\top v'_{w}})}{\text{exp}({h^\top v'_{w}}) + k    Q(w)}
$$
Inserting this term in turn in our logistic regression objective finally yields the full NCE loss:
$$
J_\theta = - \sum_{w_i \in V} [ \text{log}    \dfrac{\text{exp}({h^\top v'_{w_i}})}{\text{exp}({h^\top v'_{w_i}}) + k    Q(w_i)} + \sum_{j=1}^k   \text{log}    (1 - \dfrac{\text{exp}({h^\top v'_{\tilde{w}_{ij}}})}{\text{exp}({h^\top v'_{\tilde{w}_{ij}}}) + k    Q(\tilde{w}_{ij})})]
$$
One caveat of NCE is that as typically different noise samples are sampled for every training word w�, the noise samples and their gradients cannot be stored in dense matrices, which reduces the benefit of using NCE with GPUs, as it cannot benefit from fast dense matrix multiplications. Jozefowicz et al. (2016) and Zoph et al. (2016) independently propose to share noise samples across all training words in a mini-batch, so that NCE gradients can be computed with dense matrix operations, which are more efficient on GPUs.
## Negative sampling

## Self-normalization and Infrequent normalization

We previously mentioned in passing that by setting the denominator Z(c) of the NCE loss to 11, the model essentially self-normalises. This is a useful property as it allows us to skip computing the expensive normalisation in Z(c).
$$
J_\theta    P(w    \|    c) = - \sum\limits_i [h^\top v'_{w_i} + \text{log}    Z(c)]
$$
If we are able to constrain our model so that it sets Z(c)=1 or similarly logZ(c)=0, then we can avoid computing the normalisation in Z(c) altogether. Devlin et al. (2014) thus propose to add a squared error penalty term to the loss function that encourages the model to keep logZ(c) as close as possible to 0:
$$
J_\theta = - \sum\limits_i [h^\top v'_{w_i} + \text{log} Z(c) - \alpha (\text{log}(Z(c)) - 0)^2]
$$
$$
J_\theta = - \sum\limits_i [h^\top v'_{w_i} + \text{log}    Z(c) - \alpha    \text{log}^2 Z(c)]
$$
where α allows us to trade-off between model accuracy and mean self-normalisation. By doing this, we can essentially guarantee that Z(c) will be as close to 11 as we want. At decoding time in their MT system, Devlin et al. (2014) then set the denominator of the softmax to 11 and only use the numerator for computing P(w|c) together with their penalty term:
$$
J_\theta = - \sum\limits_i [h^\top v'_{w_i} - \alpha \text{log}^2 Z(c)]
$$
**Infrequent normalization**

 It should even be sufficient to only normalize a fraction of the training examples and still obtain approximate self-normalising behaviour. They thus propose Infrequent Normalization (IN), which down-samples the penalty term, making this a sampling-based approach.
 
Let us first decompose the sum of the previous loss J_θ into two separate sums:
$$
J_\theta = - \sum\limits_i h^\top v'_{w_i} +  \alpha \sum\limits_i \text{log}^2 Z(c)
$$
We can now down-sample the second term by only computing the normalisation for a subset C of words w_j and thus of contexts c_j (as Z(c) only depends on the context c in the training data:
$$
J_\theta = - \sum\limits_i h^\top v'_{w_i} +  \dfrac{\alpha}{\gamma} \sum\limits_{c_j \in C} \text{log}^2 Z(c_j)
$$



# Word segmentation/Byte pair encoding models

Byte-Pair Encoding (BPE) was initially developed as an algorithm to compress texts, and then used by OpenAI for tokenization when pretraining the GPT model.

BPE training starts by computing the unique set of words used in the corpus (after the normalization and pre-tokenization steps are completed), then building the vocabulary by taking all the symbols used to write those words.

As a very simple example, let’s say our corpus uses these five words:

`"hug", "pug", "pun", "bun", "hugs"` 

The base vocabulary will then be `["b", "g", "h", "n", "p", "s", "u"]`. For real-world cases, that base vocabulary will contain all the ASCII characters, at the very least, and probably some Unicode characters as well. If an example you are tokenizing uses a character that is not in the training corpus, that character will be converted to the unknown token. That’s one reason why lots of NLP models are very bad at analyzing content with emojis, for instance.

> The GPT-2 and RoBERTa tokenizers (which are pretty similar) have a clever way to deal with this: they don’t look at words as being written with Unicode characters, but with bytes. This way the base vocabulary has a small size (256), but every character you can think of will still be included and not end up being converted to the unknown token. This trick is called _byte-level BPE_.

After getting this base vocabulary, we add new tokens until the desired vocabulary size is reached by learning _merges_, which are rules to merge two elements of the existing vocabulary together into a new one. So, at the beginning these merges will create tokens with two characters, and then, as training progresses, longer subwords.

**At any step during the tokenizer training, the BPE algorithm will search for the most frequent pair of existing tokens (by “pair,” here we mean two consecutive tokens in a word). That most frequent pair is the one that will be merged, and we rinse and repeat for the next step.**

Thus, the first merge rule learned by the tokenizer is `("u", "g") -> "ug"`, which means that `"ug"` will be added to the vocabulary, and the pair should be merged in all the words of the corpus. At the end of this stage, the vocabulary and corpus look like this:

```
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug"]
Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```
## Tokenization algorithm

1. Normalization
2. Pre-tokenization
3. Splitting the words into individual characters
4. Applying the merge rules learned in order on those splits

```
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
```
Next, we need to pre-tokenize that corpus into words. Since we are replicating a BPE tokenizer (like GPT-2), we will use the `gpt2` tokenizer for the pre-tokenization:
```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)
```
The next step is to create a base vocabulaty formed by all characters in the corpus.
```
alphabet = []
for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()
print(alphabet)
```
Split each word:
```
splits = {word: [c for c in word] for word in word_freqs.keys()}
```

Compute frequencies
```
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs
```
Now, finding the most frequent pair only takes a quick loop:
```
best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

print(best_pair, max_freq)
```
So the first merge to learn is `('Ġ', 't') -> 'Ġt'`, and we add `'Ġt'` to the vocabulary: To continue, we need to apply that merge in our `splits` dictionary. Let’s write another function for this::

```
merges = {("Ġ", "t"): "Ġt"}
vocab.append("Ġt")
```
```
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits
```
Now we have everything we need to loop until we have learned all the merges we want. Let’s aim for a vocab size of 50:
```
vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])
```
Then to tokenize the new text we pre-tokenize, normalize it and apply the merger rules:
```
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])
```

> ⚠️ Our implementation will throw an error if there is an unknown character since we didn’t do anything to handle them. GPT-2 doesn’t actually have an unknown token (it’s impossible to get an unknown character when using byte-level BPE), but this could happen here because we did not include all the possible bytes in the initial vocabulary. This aspect of BPE is beyond the scope of this section, so we’ve left the details out.

*Pre-tokenization is a pre-processing step that occurs before passing the text to the tokenization algorithm. It involves breaking down text into more granular chunks, which can be based on linguistic rules like splitting on punctuation marks or spaces to ensure that the individual learned tokens are meaningful and promote compositional re-use*
# Character-based models
Character-based models in natural language processing (NLP) are neural language models that operate at the character level, predicting the next character in a sequence based on the characters that have come before it. These models offer several advantages over word-based models:
#### Advantages of Character-Based Models:

1. **Small Vocabulary**: Character-based models have a smaller discrete space compared to word-based models, making them computationally efficient[
2. **Flexibility**: They can handle any words, punctuation, and document structures, providing flexibility in text processing[
3. **Resilience**: Character-based models are resilient to spelling mistakes and other anomalies in human text, enhancing their robustness[
4. **Handling Large Vocabularies**: They can handle arbitrarily large vocabularies, which is beneficial for languages with complex morphologies or rich vocabularies

#### Disadvantages of Character-Based Models:

1. **Training Complexity**: Character-based models may require larger models and longer training times compared to word-based models due to the increased granularity of characters.
2. **Sequential Processing**: Processing input sequentially in character-based models can be counterintuitive and may restrict the choice of architecture, especially when using convolutions or Transformers

#### Subword tokenization


# HuggingFace: NLP course
#### Tokenizers 
```
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
Usage example: 
```
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```
##### Padding the inputs
We’ll use _padding_ to make our tensors have a rectangular shape. Padding makes sure all our sentences have the same length by adding a special word called the _padding token_ to the sentences with fewer values.

The padding token ID can be found in `tokenizer.pad_token_id`. Let’s use it and send our two sentences through the model individually and batched together
```
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits
```
There’s something wrong with the logits in our batched predictions: the second row should be the same as the logits for the second sentence, but we’ve got completely different values!

This is because the key feature of Transformer models is attention layers that _contextualize_ each token. These will take into account the padding tokens since they attend to all of the tokens of a sequence. To get the same result when passing individual sentences of different lengths through the model or when passing a batch with the same sentences and padding applied, we need to tell those attention layers to ignore the padding tokens. This is done by using an attention mask.

_Attention masks_ are tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s: 1s indicate the corresponding tokens should be attended to, and 0s indicate the corresponding tokens should not be attended to (i.e., they should be ignored by the attention layers of the model).

```
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
```
##### Longer sequences
Most models handle sequences of up to 512 or 1024 tokens, and will crash when asked to process longer sequences. There are two solutions to this problem:

- Use a model with a longer supported sequence length.
- Truncate your sequences.

#### AutoModel
The output of the Transformer model is sent directly to the model head to be processed.
There are many different architectures available in 🤗 Transformers, with each one designed around tackling a specific task. Here is a non-exhaustive list:

- `*Model` (retrieve the hidden states)
- `*ForCausalLM`
- `*ForMaskedLM`
- `*ForMultipleChoice`
- `*ForQuestionAnswering`
- `*ForSequenceClassification`
- `*ForTokenClassification`
- and others 🤗

For our example, we will need a model with a sequence classification head (to be able to classify the sentences as positive or negative). So, we won’t actually use the `AutoModel` class, but `AutoModelForSequenceClassification`:
```
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
```
#### PostProcessing the output
The values we get as output from our model don’t necessarily make sense by themselves.
```
print(outputs.logits)
```
Our model predicted `[-1.5607, 1.6123]` for the first sentence and `[ 4.1692, -3.3464]` for the second one. Those are not probabilities but _logits_, the raw, unnormalized scores outputted by the last layer of the model. To be converted to probabilities, they need to go through a [SoftMax] layer (all 🤗 Transformers models output the logits, as the loss function for training will generally fuse the last activation function, such as SoftMax, with the actual loss function, such as cross entropy)

```
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```
Now we can see that the model predicted `[0.0402, 0.9598]` for the first sentence and `[0.9995, 0.0005]` for the second one. These are recognizable probability scores.

To get the labels corresponding to each position, we can inspect the `id2label` attribute of the model config (more on this in the next section):

```
model.config.id2label
```
Now we can conclude that the model predicted the following:

- First sentence: NEGATIVE: 0.0402, POSITIVE: 0.9598
- Second sentence: NEGATIVE: 0.9995, POSITIVE: 0.0005

#### Models
The `AutoModel` class and all of its relatives are actually simple wrappers over the wide variety of models available in the library. It’s a clever wrapper as it can automatically guess the appropriate model architecture for your checkpoint, and then instantiates a model with this architecture.

However, if you know the type of model you want to use, you can use the class that defines its architecture directly. Let’s take a look at how this works with a BERT model.
```
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)
```
Creating a model from the default configuration initializes it with **random values**.

The model can be used in this state, but it will output gibberish; it needs to be trained first. We could train the model from scratch on the task at hand, but this would require a long time and a lot of data, and it would have a non-negligible environmental impact. To avoid unnecessary and duplicated effort, it’s imperative to be able to share and reuse models that have already been trained.

```
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```
Saving a model is as easy as loading one — we use the `save_pretrained()` method, which is analogous to the `from_pretrained()` method:
```
model.save_pretrained("directory_on_my_computer")
```




#### Finetuning a pretrained model
##### Process the data
Here is how we would train a sequence classifier on one batch in PyTorch:
```
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

**Loading a dataset from hub**
```
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
```
**Preprocessing dataset data**
We can feed the tokenizer one sentence or a list of sentences, so we can directly tokenize all the first sentences and all the second sentences of each pair like this:
```
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
```
However, we can’t just pass two sequences to the model and get a prediction of whether the two sentences are paraphrases or not. We need to handle the two sequences as a pair, and apply the appropriate preprocessing. Fortunately, the tokenizer can also take a pair of sequences and prepare it the way our BERT model expects:
```
inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs
```
Or better way:
```
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```
This works well, but it has the disadvantage of returning a dictionary (with our keys, `input_ids`, `attention_mask`, and `token_type_ids`, and values that are lists of lists). It will also only work if you have enough RAM to store your whole dataset during the tokenization (whereas the datasets from the 🤗 Datasets library are [Apache Arrow](https://arrow.apache.org/) files stored on the disk, so you only keep the samples you ask for loaded in memory).

To keep the data as a dataset, we will use the [`Dataset.map()`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.map) method. This also allows us some extra flexibility, if we need more preprocessing done than just tokenization. The `map()` method works by applying a function on each element of the dataset, so let’s define a function that tokenizes our inputs:

```
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```
Note that we’ve left the `padding` argument out in our tokenization function for now. This is because padding all the samples to the maximum length is not efficient: it’s better to pad the samples when we’re building a batch, as then we only need to pad to the maximum length in that batch, and not the maximum length in the entire dataset. This can save a lot of time and processing power when the inputs have very variable lengths!

ere is how we apply the tokenization function on all our datasets at once. We’re using `batched=True` in our call to `map` so the function is applied to multiple elements of our dataset at once, and not on each element separately. This allows for faster preprocessing:
```
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```
Now back to "padding during batching part":
```
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
```
##### Setup a trainer/Pytorch training
```
from transformers import TrainingArguments
training_args = TrainingArguments("test-trainer")

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```
Note that when you pass the `tokenizer` as we did here, the default `data_collator` used by the `Trainer` will be a `DataCollatorWithPadding` as defined previously, so you can skip the line `data_collator=data_collator` in this call.
To fine-tune the model on our dataset, we just have to call the `train()` method of our `Trainer`:

```
trainer.train()
```

This will start the fine-tuning (which should take a couple of minutes on a GPU) and report the training loss every 500 steps. It won’t, however, tell you how well (or badly) your model is performing. This is because:

1. We didn’t tell the `Trainer` to evaluate during training by setting `evaluation_strategy` to either `"steps"` (evaluate every `eval_steps`) or `"epoch"` (evaluate at the end of each epoch).
2. We didn’t provide the `Trainer` with a `compute_metrics()` function to calculate a metric during said evaluation (otherwise the evaluation would just have printed the loss, which is not a very intuitive number).
```
import evaluate

predictions = trainer.predict(tokenized_datasets["validation"])
preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```
A compute metric function to provide to the Trainer is:
```
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```
Final Trainer is:
```
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

**On the other hand , the use of API is unnecessary and can be replaced with pytorch pipeline**:
```
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
Before actually writing our training loop, we will need to define a few objects. The first ones are the dataloaders we will use to iterate over batches. But before we can define those dataloaders, we need to apply a bit of postprocessing to our `tokenized_datasets`, to take care of some things that the `Trainer` did for us automatically. Specifically, we need to:

- Remove the columns corresponding to values the model does not expect (like the `sentence1` and `sentence2` columns).
- Rename the column `label` to `labels` (because the model expects the argument to be named `labels`).
- Set the format of the datasets so they return PyTorch tensors instead of lists.
```
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names
```
Now the Dataloaders: 
```
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
```
Now the model:
```
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

## To check sure everything goes smoothly
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)


from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
```
**Training loop**:
```
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```
**Evaluation loop**:
```
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```
##### Training with accelerate 
```
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```
Then the main bulk of the work is done in the line that sends the dataloaders, the model, and the optimizer to `accelerator.prepare()`. This will wrap those objects in the proper container to make sure your distributed training works as intended. The remaining changes to make are removing the line that puts the batch on the `device` (again, if you want to keep this you can just change it to use `accelerator.device`) and replacing `loss.backward()` with `accelerator.backward(loss)`.

> ⚠️ In order to benefit from the speed-up offered by Cloud TPUs, we recommend padding your samples to a fixed length with the `padding="max_length"` and `max_length` arguments of the tokenizer.

To try it out in your distributed setup, run the command:

`accelerate config`

which will prompt you to answer a few questions and dump your answers in a configuration file used by this command:

`accelerate launch train.py`

f you want to try this in a Notebook (for instance, to test it with TPUs on Colab), just paste the code in a `training_function()` and run a last cell with:
```
from accelerate import notebook_launcher
notebook_launcher(training_function)
```

# GPT tokenizer [from scratch]
```
import tiktoken
from .regex import RegexTokenizer


def bpe(mergeable_ranks, token, max_rank):
    # helper function used in get_gpt4_merges() to reconstruct the merge forest
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def recover_merges(mergeable_ranks):
    # the `merges` are already the byte sequences in their merged state.
    # so we have to recover the original pairings. We can do this by doing
    # a small BPE training run on all the tokens, in their order.
    # also see https://github.com/openai/tiktoken/issues/60
    # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # skip raw bytes
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class GPT4Tokenizer(RegexTokenizer):
    """Lightweight wrapper on RegexTokenizer that matches GPT-4's tokenizer."""

    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        # get the official tokenizer and its merges
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        # the merges are those of gpt4, but we have to recover them
        self.merges = recover_merges(mergeable_ranks)
        # reconstruct the vocab from the merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        # now here is another tricky part.
        # for some reason, the tokens corresponding to individual bytes
        # are permuted in a different order. This is completely non-sensical
        # and probably historical, but therefore we have to deal with it here.
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        # finally register the special tokens
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, text_bytes):
        # before we start processing bytes, we have to permute them
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids = super()._encode_chunk(text_bytes)
        return ids

    def decode(self, ids):
        # we have to un-permute the bytes before we decode
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    # this is a pretrained tokenizer, it is not intended to be trained
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    # save/load would require some thought.
    # we'd have to change save/load of base to add support for byte_shuffle...
    # alternatively, we could move byte_shuffle to base class, but that would
    # mean that we're making ugly our beautiful Tokenizer just to support
    # the GPT-4 tokenizer and its weird historical quirks around byte_shuffle.
    def save(self, file_prefix):
        raise NotImplementedError("GPT4Tokenizer cannot be saved.")

    def load(self, model_file):
        raise NotImplementedError("GPT4Tokenizer cannot be loaded.")

    def save_vocab(self, vocab_file):
        # just for visualization purposes let's output the GPT-4 tokens
        # in the exact same format as the base class would.
        # simple run as:
        # python -c "from minbpe import GPT4Tokenizer; GPT4Tokenizer().save_vocab('gpt4.vocab')"
        from .base import render_token
        # build vocab being mindful of the byte shuffle
        vocab = {idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # now merge the shuffled bytes and write to file
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")
```









# Optimizing inference 
## Decoder only inference
- GPT-like models are decoder only models
- No encoder-decoder multi-head attention

- Input ptocessing (aka prefill): highly parallel -> Multi-head attention computes the keys and values
- Large matrix multiplication, high usage of hardware accelerator 

- Output: sequential and slow (one token at a time until EOS) -> KV cache
- 

## KV - cache 
- Can we avoid computing KV values again for the inout tokens
- Cache is stored on accelerator RAM with `Cache size (FP16) = 2*2*batch_ize*seq_length*num_layers*embeddings_length` -> gigabytes
## Continuous batching
- Input and output length can greatly vary, leading to very different generation times

## Speculative decoding
- Again, one token generation issue
- Memory-bound: idle resources available
- **with small model we predit several potential completions in parallel** (aka speculativa sampling) and then **with large model we evaluate those and pick the best to correct it**
- Each iteration generates at least one valid token and potentially more 

- How to build a small model?
	- Select small off-the-shelf version of the large model
	- Use a basic n-gram to generate tokens found in prompt
	- Fine-tune small model (on top\together)

## Speculative decoding: Medusa
- Add decoding heads to the LLM to predict multiple outputs in parallel


# Подбор гиперпараметров RAG-системы с помощью Optuna
## Задача 
Но для начала формализуем задачу, т.к. от этого сильно зависит архитектура системы.

Итак, наша RAG-система должна выполнять следующие функции:

- Поиск в различной технической документации: ТЗ, инструкции, регламенты и пр.
    
- Поддержка только трех форматов документов: DOCX, PDF и TXT.
    
- В одном документе могут быть ответы на многие вопросы. Но ответ на любой вопрос содержится только в одном документе, в одной его части (например, абзаце).  
    Т.е. не может быть ситуации, что ответ на один и тот же вопрос можно найти в разных файлах.
    
- Язык документации, преимущественно, - русский. Исключения - имена собственные на английском.
    
- Работа в закрытом контуре (т.е. никаких OpenAI и прочих товарищей).
    
- Работа на одной GPU A100 40Мб.

## Архитектура

**1. Парсинг документов**
![[Pasted image 20240718130953.png]]

**2. Ответ на вопрос**
![[Pasted image 20240718131027.png]]


**Компоненты**

В этой архитектуре можно выделить три основных компонента:

1. Bi-encoder - производит кодирование строки в вектор.  Для чего это нужно? Две строки закодированные в вектор можно сравнивать между собой посредством косинусного расстояния и сказать насколько они похожи. Таким способом, например, можно приблизительно подобрать ответы на вопрос.
2. Векторная БД. Нужна для хранения векторов. А еще она может очень быстро находить (и возвращать) близкие вектора по косинусному (и некоторым другим) расстоянию.  В вектора мы будем кодировать чанки документов и вопросы пользователей. И по косиносному расстоянию будем искать наиболее похожие на вопрос чанки. И уже эти чанки будем скармливать LLM. **В качестве векторной БД будем использовать [Qdrant](https://qdrant.tech/)**
3. LLM

На второй картинке вы можете увидеть (затенен) еще один компонент - Cross-encoder (еще его называют re-ranker). В этом решении он не используется, но он часто встречается в других решениях. Его основная функция - дополнительная фильтрация отобранных чанков. Cross-encoder типа умнее bi-энкодера, но работает заметно дольше bi-энкодера. А если у вас миллионы векторов - это может быть существенно снизить производительность системы. Поэтому поступают так: с помощью косинусного расстояния отбирают, например, топ-100 чанков. Их скармливают сross-encoder’у, который отбирает из них 3-5 чанков, которые уже поступают в LLM.  
  
Или еще более хитрые варианты:  
1. LLM просят переформулировать вопрос пользователя 2 раза. По каждому из получившихся 3 вопросов ищут чанки в БД (например, по 30 чанков на вопрос). Затем cross-encoder отбирает из общего списка чанков 3-5 лучших.  
2. Выполнять два поиска: векторный и по ключевым словам (key-word), например, TF-IDF или BM-25. И обе выборки скормить cross-encoder'у.


## Тесты

Чтобы оценить качество работы RAG-системы необходимо подготовить тестовые вопросы и ответы к ним. И лучше чтобы в их составлении участвовали конечные пользователи RAG-системы (или заказчики), поскольку ваше представление о “прекрасном” может отличаться от необходимого. Вопросов нужно порядка 20-30 на 5-10 файлов.

В результате у вас должна получится примерно такая таблица:

| #   | Вопрос | Правильный ответ | Контекст | Файл | № страницы |
| --- | ------ | ---------------- | -------- | ---- | ---------- |
|     |        |                  |          |      |            |

Вопросы желательно подбирать так, чтобы протестировать различные варианты ответов:
- Простые факты из документов.
- Вопросы на суммаризацию.
- Описания процессов.
- Перечисления фактов.
- Вопросы с условием.
- Числа, даты, имена собственные.
- И пр.

## Оценка
Учитывая задачу, мы хотим в результате тестирования получить ответ на три вопроса:

1. Найден ли правильный файл?
2. Найден ли правильный контекст?
3. Оценить ответ LLM.

Метрик для оценки ответов LLM довольно много: BERTScore, BLEURT, METEOR и пр.

Правильность контекста мы будем оценивать по пересечению. Т.е. будем искать, какой наибольший кусок контекста содержится в отобранных чанках.

Файл мы будем оценивать просто по факту его нахождения: вернула ли нам БД нужный файл (к каждому чанку у нас будет привязано название файла из которого он взят).

## Код

### Chunks
![[Pasted image 20240718132642.png]]

Первые гиперпараметры:

- sep - разделитель по которому мы будем шинковать файл.
- chunk_size - размер чанков (в символах).
- chunk_overlap - с каким перехлестом будут делаться чанки.

### Bi-encoder
![[Pasted image 20240718132800.png]]

Тут стоит поподробнее остановиться на двух важных свойствах bi-encoder’а:

- Сколько текста он может скушать за раз - остальное будет отброшено.  Если у вас длинные чанки и/или длинные вопросы, то вам, возможно, стоит подобрать bi-encoder с бОльшей длиной контекста.
- Какого размера вектора он возвращает. При прочих равных, чем больше длина вектора, тем больше информации в нем можно закодировать. 

Оба этих параметра “зашиты” в bi-энкодер и оба важны для RAG-системы. Плюс bi-энкодеры обладают разным "интеллектом". Поэтому у нас сам bi-энкодер будет гиперпараметром. Т.е. мы будем пробовать разные bi-энкодеры и смотреть какой из них лучше себя покажет.

	Есть даже лидерборд bi-энкодеров, в котором вы можете подобрать нужный вам: [https://github.com/avidale/encodechka]



### Qdrant
![[Pasted image 20240718133431.png]]
![[Pasted image 20240718133538.png]]

Обратите внимание:
- Коллекцию мы создаем такого же размера, какого размера вектора возвращает bi-encoder.
- Вместе с вектором мы будем хранить сам чанк, из которого он сформирован и название файла, из которого он взят.


	Да, мы будет при каждом прогоне теста заново разбивать все файлы на чанки и создавать из них коллекцию. Но не стоит об этом переживать. На фоне скорости генерации ответа LLM это происходит почти мгновенно :)

### Vector search
![[Pasted image 20240718134008.png]]

	Тут на будущее напрашивается функционал остановки поиска. Все найденные вектора возвращаются с оценкой косинусного расстояния. И если это расстояние меньше определенного порога, то можно прерывать пайплайн и возвращать что-то вроде "Ответ не найден".

### LLM
```
import torchfrom transformers import AutoTokenizer, AutoModelForCausalLM
model_id = 'IlyaGusev/saiga_llama3_8b'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype = torch.bfloat16,device_map = "auto")
```

![[Pasted image 20240718135455.png]]

Прибавилось гиперпараметров и все они относятся к LLM:

- max_new_tokens - максимальное количество токенов, которое будет сгенерировано LLM (не считая токены в промте).
- temperature - определяет насколько “творческим” будет ответ LLM. Чем выше значение, тем выше “творчество”.
- top_p - также отвечает за степень детерминированности модели. Чем больше значение, тем более разнообразными будут ответы. Меньшие значения будут давать более точные и фактические ответы.  З.Ы. Рекомендуется изменить либо температуру, либо top_p, но не оба сразу.
- top_k - ограничивает количество вариантов, которые модель рассматривает при генерации следующего токена.

З.Ы. Здесь напрашивается в качестве гиперпараметра системный промт. Точнее различные варианты его формулировки. Но пока оставим его статичным.

> Т.к. ответы у нас состоят из одного-двух слов - просим ламу отвечать очень кратко.

> Также в качестве эксперемна на будущее, можно дополнить промт фразой: "Если ответа нет в контексте, напиши 'Ответ не найден.'"

### Оценка

Оценкой ответа LLM (от 0 до 100) будут заниматься две функции.
![[Pasted image 20240718135734.png]]

Оценка контекста (от 0 до 100) будет также производится метрикой Rouge. Точнее одной из ее версий (Rouge-L), которая измеряет максимальную общую длину между двумя строками:
![[Pasted image 20240718135859.png]]

### Тест
![[Pasted image 20240718141641.png]]

	С оценкой можно поиграться. Например можно усреднить оценку файла, контекста и LLM. Или можно усреднить только оценки LLM и контекста и умножить все это на оценку файла (если файл найден неправильно, то вся оценка занулится). В общем простор для фантазии большой.


**Обратите внимание, что код внутри функции _run_one_test_ мы обернули в try-except. Это нужно обязательно сделать на случай, если оптуна захочет передать в RAG-систему слишком жирные параметры. От которых модель просто упадет. В этом случаем мы не прекращаем обучение, а просто возвращаем скор 0. Оптуна быстро выучит границы дозволенного и суваться за их пределы почти не будет.**


## Запуск Optuna
![[Pasted image 20240718142324.png]]

	В качестве разделителя для чанков мы используем только один символ (точку, запятую или пробел). Но в функцию RecursiveCharacterTextSplitter можно передать сразу последовательность символов. Примерно так: ['/n', '.', ',', ' ']. Тогда, она сначала попробует первый знак, затем перейдет ко второму и т.д. Так можно получить чанки более равномерной длины. Ну а возможности комбинаций этих символов оставляют большой простор для фантазии :)

![[Pasted image 20240718142629.png]]


# Архитектура RAG

## Нюансы 

Поскольку знания в модели не зашиты, качество ответов ну очень сильно зависят от того, что найдет Retriver, и в какой форме. 

Задача не тривиальная, так как в типичном бардаке документов компании обычно и люди разбираются с трудом. Документы и знания как правило хранятся в плохо структурированном виде, в разных местах, иногда в виде изображений, графиков и записок от руку и т.п.

Часто информация в одном месте противоречит информации в другом, и во всем этом зоопарке надо как-то разбираться. Часть информация просто не имеет смысла без контекста, как например сокращения, аббревиатуры принятые в компании, имена и фамилии.
## Что делать? 

1. Первоначальную обработку и очистку вопроса пользователя
2. Поиск данных в хранилищах
3. Ранжирование полученных результатов из хранилища
4. Обработка и комбинирование результатов в ответ
5. Оценка ответа
6. Применение форматирования, стилистики и тона

В качестве простого решения - можно просить другую LLM переформулировать ответ пользователя, но есть техники и мощнее

К пре-обработке запроса пользователя так-же относится его классификация. Например, запросы могут подразделяться на вопросы, жалобы, просьбы и так далее. Можно далее классифицировать запросы на срочные, не срочные, спам, фрод. Можно классифицировать по подразделениям (например бухгалтерия, производство, HR) и так далее. Это все позволяет сузить круг поиска информации и соответственно повышает скорость и качество ответа.

## RAG Fusion 

Берем несколько вариантов вопроса пользователя, делаем по ним поиск, объединяем, предварительно ранжируя через **Cross Encoder**.

Векторные базы используют **Bi-encoder** модели чтобы вычислить похожесть двух понятий в векторном пространстве. Эти модели обучаются представлять данные в виде векторов и, соответственно, при поиске запрос пользователя тоже превращается в вектор, и при поиске возвращаются вектора ближайшие к запросу. Но такая близость не гарантирует что это наилучший ответ.

**Cross Encoder** работает по другому. Он принимает два объекта (текста, изображения и т.п.) и возвращает их релевантность (similarity) относительно друг друга. Его точность как правило [лучше](https://arxiv.org/abs/1908.10084) чем у Bi Encoder. Обычно, из векторной базы возвращают больше результатов, чем нужно (на всякий случай, допустим 30) и потом ранжируют их, используя Cross Encoder или подобные техники, и возвращают первые 3.

## Отступление о векторных базах

Самые популярные векторные базы (на текущий момент):

- QDrant - база с открытым исходным кодом 
- Pinecone - cloud-native (читай - сдерут 3 шкуры) база
- Chroma - еще одна база с открытым исходным кодом (Apache-2.0 license)
- Weaviate - открыта по BSD-3-Clause license
- Milvus - открыта под Apache-2.0 license
- [FAISS](https://github.com/facebookresearch/faiss) 

## Ансамбль ретриверов и/или источников данных

Как пример - использование нескольких типов ретриверов из [langchain](https://python.langchain.com/docs/modules/data_connection/retrievers/). Ансамблинг особенно полезен при обьединении sparse (даже не знаю как это перевести, поэтому пусть будет так)  ретриверов(например [BM25](https://python.langchain.com/docs/integrations/retrievers/bm25)) ретриверов и dense ретриверов(работающих на основе embdedding similarity, например те же [векторные](https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore) базы) потому что они хорошо дополняют друг друга.

**Dense Retriever** - использует обычно трансформеры, например BERT, для кодирования как запросов, так и документов в векторы в многомерном пространстве.

**Sparse Retriever -** использует традиционные методы информационного поиска, такие как TF-IDF (Частотность Термина) или BM25.

[EnsembleRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble) затем ранжирует и объединяет результаты используя, например, [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf):

![[Pasted image 20240718171620.png]](Как выбрать правильную стратегию из всего этого зоопарка? Экспериментировать. Или воспользоваться фреймворком, например [https://github.com/Marker-Inc-Korea/AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG).)

## RELP

Это еще один способ поиска данных, Retrieval Augmented Language Model based Prediction.

Отличается тут шаг поиска - после того как мы находим информацию в векторном хранилище, в том числе используя техники выше, - мы используем ее не для генерации ответа с помощью LLM а для генерации примеров ответов (с помощью [few-shot prompting](https://en.wikipedia.org/wiki/Few-shot_learning)) для LLM, и на основе этих примеров LLM как бы учиться, и отвечает на основе этого мини-обучения на заданный вопрос. Эта техника является формой **динамического обучения**, что намного менее затратно чем до-обучение модели стандартными методами.

![[Pasted image 20240718172752.png]]
## few-shot (learning) prompting

Есть две похожие техники промптинга LLM: zero shot и few-shot. **Zero-shot** это когда вы спрашиваете LLM свой вопрос не приводя никаких примеров. Например:
![[Pasted image 20240718173054.png]]


**Few-shot** — это когда сначала LLM дается несколько примеров, на которых она обучается. Это значительно повышает вероятность получить релевантный ответ, в релевантной форме. Например:

![[Pasted image 20240718173127.png]]

## Ражирование, обьединение и оценка

Для ренкинга (ранжирования) используются разные подходы. Наиболее частые:

1. Использование **Cross Encoder** (описано выше) для ре-ранжирования полученных результатов и отбрасывания наименее релевантных (например достаем топ 30 результатов из векторной базы (top k), ранжируем Cross Encoder’ом, берем первые 10).
    

Есть уже готовые решения для этих целей, например от [Cohere](https://cohere.com/rerankhttps:/cohere.com/rerank).

2. [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). Основная идея RRF заключается в том, чтобы придать большее значение элементам, занимающим более высокие позиции в каждом наборе результатов поиска. В RRF оценка каждого элемента рассчитывается на основе его места в индивидуальных результатах поиска. Обычно это делается с использованием формулы 1/(k + ранг), где "ранг" - это позиция элемента в определенном наборе результатов поиска, а "k" - это константа (часто устанавливаемая на уровне около 60). Эта формула обеспечивает высокую оценку для элементов с более высоким рангом.
    

Оценки для каждого элемента в разных наборах результатов затем суммируются для получения итоговой оценки. Элементы сортируются по этим итоговым оценкам для формирования комбинированного списка результатов.

RRF особенно полезен, потому что он не зависит от абсолютных оценок, присвоенных отдельными поисковыми системами, которые могут значительно различаться по своей шкале и распределению. RRF эффективно объединяет результаты из разных систем таким образом, что подчеркивает наиболее последовательно высоко ранжированные элементы.

3. LLM based ренкинг и оценка: можно не напрягаться и просто попросить LLM отранжировать и оценить результат 🙂. Последние версии OpenAI довольно хорошо с этим справляются, однако их использование для этого накладно.


## Оценка результатов поиска в Vector Store
Оценить, насколько результаты релевантны запросу можно используя следующие метрики: P@K, MAP@K, NDCG@K (и подобные). Обычно они возвращают число от 0 до 1, где 1 - это самая высокая точность.  Они похожи по смыслу, отличия в деталях:

**P@K** означает precision at K, то есть точность на k элементах. Допустим на запрос о зайцах система нашла 4 документа:

_[ “Зайцы в дикой природе”, “Как ездить зайцем в автобусе”, “Трактат о Моркови”, “Зайка моя: мемуары by Киркоров”]_

Так как жизнеописание Киркорова или автобусы к зайцам отношения не имеют, в этих позициях стоит 0, и общая точность получается:

![[Pasted image 20240718191014.png]]
Однако, тут не учитывается порядок. Что если возвращаемый список выглядит так:

_[ “Как ездить зайцем в автобусе”, “Зайка моя: мемуары by Киркоров”, “Зайцы в дикой природе”, “Трактат о Моркови”]_

P@K все еще равен 0.5, но как мы знаем порядок релевантный и нерелевантных результатов имеет значение! (как для людей так и для LLM, которая будет их использовать).

Поэтому мы берем **AP@K** или average precision at K. Идея простая, надо модифицировать формулу так чтобы порядок учитывался, и релевантные результаты в конце не увеличивали общую оценку меньше, чем те что вначале списка:

![[Pasted image 20240718191333.png]]Тут возникает пару вопросов: а как мы оценили релевантность индивидуальных элементов, чтобы посчитать эти метрики

В контексте RAG мы чаще всего просим LLM или другую модель сделать оценку. То есть мы спрашиваем LLM по каждому элементу - этот документ что мы нашли в векторном хранилище - оно вообще релевантно вот этому запросу? И т.д , вопросов может быть много.

Это нужно агрегировать, и на сцену выходит:

**MAP@K** (Mean Average Precision at K) — это среднее от суммы AP@K для всех вопросов.

**NDCG@K** означает normalized discounted cumulative gain at K, даже не буду переводить 🙂. Описание посмотрите в интернете сами.

## Оценка ответов LLM
Не все знают, но LLM (включая Llama и OpenAI) можно попросить вернуть не токены (текст) а логиты (logits). Т.е. по факту можно попросить ее вернуть распределение токенов с их вероятностью, и поглядеть - а насколько вообще модель уверенна в том, чего она набредила (посчитав token level uncertainty). Если вероятности в распределении низкие (что считать низким зависит от задачи), то скорее всего модель начала выдумывать (галлюцинировать) и совсем не уверенна в своем ответе. Это может использоваться для оценки ответа, и возвращения юзеру честного “Я не знаю”.


## Access control

Первая это то что доступ к данным может быть не равномерным, т.е. В том же вики могут быть роли и права и не каждый юзер потенциально может видеть всю информацию. Эта же проблема, на самом деле существует и для поиска в векторной базе. Тоесть встаёт проблема управления доступом. Эта проблема еще и усложняется тем, что есть много разных подходов и их гибридов, и например, кто работал с SharePoint тот в цирке не смеётся. 

Есть как минимум Role-Based Access Control (RBAC), Attribute-Based Access Control (ABAC), и Relationship-Based Access Control (ReBAC) и их сочетания.

Вообще говоря, User Directories (тот же Active Directory), например, тоже представляет собой граф, в котором вопрос доступа примерно звучит как “Есть ли путь от ноды юзер U к ноде ресурса R”. Если такой путь есть - доступ разрешен.

Права и категории - тоже форма метаданных, и для того чтобы это все хозяйство работало - эти метеданные нужно сохранить на шаге Data Ingestion в граф знаний и векторную базу.

И, соответственно, при поиске в векторной базе, нужно проверять на найденных документах соответствует ли роль или другие атрибуты доступа тому что доступно юзеру.

## Ingestion and parsing
Есть разные фреймворки и библиотеки, которые делают это с разным уровнем успеха:

- [LLama parse](https://github.com/run-llama/llama_parse)
- PyPDF2
- PdfMiner
- Tabula
- PDFQuery
- PyMyPDF
- Pytesseract

## Corrective Retrieval Augmented Generation (CRAG)

По сути это еще один граф, реализующий машину состояний (сюрприз 🙂), который выглядит примерно так: ![[Pasted image 20240719061752.png]]
## Self-RAG
Self-reflective RAG базируется на данном [исследовании](https://arxiv.org/abs/2310.11511), которое утверждает что данный подход даёт лучшие результаты чем обычный RAG.

Идея в том чтобы зафайнтюнить LLM на генерацию токенов саморефлексии, в дополнение к обычным. Токены используются чтобы направить процесс поиска ответа, то есть в каком-то смысле процесс управляется самой LLM

Это очень удобно, так как не приходится догадываться насколько LLM уверенна и что с этим делать. Токены генерируются следующие:

- **Retrieve** токен определяет нужно ли достать D чанков для данного промпта x. Варианты Yes, No, Continue
- **ISREL** токен определяет, является ли чанк d из D релевантным ответом для данного промпта x. Варианты relevant и irrelevant
- **ISSUP** токен определяет релевантен ли ответ(генрация) y LLM на чанк d, тоесть подтверждает ли чанк d ответ y. Варианты fully supported, partially supported, no support. 
- **ISUSE** токен определяет, является ли ответ LLM на каждый чанк d полезным ответом на запрос x. Варианты представляют шкалу полезности от 5 до 1.

![[Pasted image 20240719062035.png]]Подробнее [тут](https://blog.langchain.dev/agentic-rag-with-langgraph/).

## HyDe

Hyde расшифровывается как Hypothetical Document Embeddings и базируется на исследовании [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://paperswithcode.com/paper/precise-zero-shot-dense-retrieval-without).

Идея очень простая - вместо того чтобы использовать вопрос юзера для поиска в векторной базе мы используем LLM чтобы сгенерировать ответ (виртуальный гипотетический документ), и затем ответ используем для поиска в векторной базе (чтобы найти похожие ответы).

**Иногда вопросы юзера слишком абстрактны, и требуется больше контекста, который LLM может дать, и без которого поиск в базе не имеет никакого смысла.**


## Как улучшить? 

1. **Предварительная выборка:** эффективная загрузка эмбеддингов в векторное хранилище.
2. **Выборка:** точный и быстрый поиск релевантного контента.
3. **Пост-обработка:** грамотная предварительная обработка результатов перед тем, как отправить их в LLM.
4. **Генерация:** максимально эффективное использование контекста для решения задачи пользователя.
5. **Маршрутизация:** оптимизация маршрутизации запроса, например, использование агентного подхода — разбиения вопроса на части и организация их последовательной обработки моделью.

![[Pasted image 20240719071100.png]]
# Abacus embedding [Transformers can do math with the right embeddings]

Key contributions:

- propose a new positional embedding called Abacus Embeddings to better capture the significance of each digit, which leads to near-perfect in-distribution generalization.
- combining Abacus Embeddings with input injection and looped transformers performance further improves, increasing from 92.9% to 99.1% in out of distribution accuracy, an 87% reduction in error compared to using the embeddings with standard architectures alone.
- pushed length generalization beyond existing work and showed that authors models can solve problems with six times as many digits as the largest samples in the training set, whereas the previous state of the art is only two and a half times.

## Length generalization for addition

range of methods were studied for improving the arithmetic capabilities centered on two main hypotheses:
- the positional information for individual digits within numbers is being lost
- recurrence can improve the reasoning abilities of transformer architectures on multi-step arithmetic reasoning problems.

Experimental setup:
- decoder-only causal language models were trained to solve addition problems
- inputs are formatted least significant digit first, e.g. 98282 + 3859172 = 2787472
- Unlike prior work, authors did not added any padding between digits and did not padded any numbers with zeros, neither in the case of carry digits , nor to make all operands the same length
- To facilitate training of many models from scratch, authors used a language model cramming setup [2] and limit each training run to 8 exaFLOP of compute (a single Nvidia RTXA4000 GPU for 24 hours); for multiplication results authors allowed 64 exaFLOP(eight Nvidia RTXA4000 GPUs for 24 hours)
- During training, authors masked the input question and only compute loss on the answer digits

Two standard transformer architectures considered:
- First, authors used a standard autoregressive transformer model where multiple decoder layers are stacked in a feedforward manner.
- Second, authors enhanced this standard transformer model by incorporating input injection, where the embedded inputs are added to the input of each decoder layer.

## Solving addition

- **first hypothesis is that the significance of each digit**  (i.e. each digit’s position relative to the beginning of the number) is not easy for transformers to represent, and that this sub-problem presents more of a hurdle than the actual addition itself.
- uthors designed a specially built positional embedding that encodes the location of each digit relative to the start of the current number, and called it as **Abacus Embeddings**.
- inspiration drawn from Randomized Embeddings, but instead of using random ascending indices to represent positions in a sample, it use consecutive ascending indices with a random starting position to allow for length generalization
 ![[Pasted image 20240807082523.png]]
- Abacus Embeddings improve generalization performance up to 100 digits and beyond for standard transformer architectures
- Unless otherwise stated, authors used a maximally recurrent architecture, i.e. only one unique layer recurred to achieve the effective depth. Authors also employed input injection, skip-connections that propagate a copy of the input to each layer in the network
- Figure below compare all architecture variants using both FIRE and NoPE embeddings trained on addition over operands with up to 40 digits![[Pasted image 20240807082738.png]]
- Despite having approximately 10× fewer parameters than the other models, we see that the looped transformer (recurrent, with input injection and progressive loss), achieves the best out of distribution performance using either position embedding
## Solving multiplication
- First, authors removed the input injection from inside the recurrent block and second, we divide the gradients in the recurrent block by the number of recurrences, down-weighing the gradient update from batches with many recurrences.
![[Pasted image 20240807082939.png]]
# Fast geometric ensembling 

Traditionally the loss surfaces of deep neural networks are thought of as having multiple isolated local optima (see the left panel of the figure below). We show however, that the optima are in fact connected by simple curves, such as a polygonal chain with only one bend, over which training and test accuracy are nearly constant (see the middle and right panels of the figure below) and propose a method to find such curves. Inspired by this geometric observation we propose Fast Geometric Ensembling (FGE), an ensembling method that aims to explore the loss surfaces along the curves of low loss.

**The method consists of running SGD with a cyclical learning rate schedule starting from a pre-trained solution, and averaging the predictions of the traversed networks. We show that FGE outperforms ensembling independently trained networks and the recently proposed [Snapshot Ensembling](https://arxiv.org/abs/1704.00109) for any given computational budget.**

# Rotary Embeddings: A Relative Revolution
## Problem statement

When applying self-attention to a given domain, the choice of position encoding typically involves tradeoffs between simplicity, flexibility, and efficiency. For example, learned absolute positional encoding is very simple, but may not generalize and are not always particularly meaningful due to the common practices [1, 3, 9, 15] of packing short sentences and phrases together in a single context and breaking up sentences across contexts.

Another major limitation of existing methods is that they do not work with efficient transformers. Methods like T5's relative positional bias [10] require constructing the full N×N attention matrix between positions, which is not possible when using many of the efficient alternatives to softmax attention, including kernelized variants like FAVOR+ [2].

## Intuition 

We would like to find a positional encoding function f(x,ℓ) for an item x and its position ℓ such that, for two items q and k at positions m and n, the inner product between f(q,m) and f(k,n) is sensitive only to the values of q, k, and their relative position m−n. This is related in spirit to the kernel trick: we are searching for a feature map such that its kernel has certain properties. A key piece of information is the geometric definition of the dot product between Euclidean vectors:
$$
\mathbf{q} \cdot \mathbf{k} = \lVert \mathbf{q} \rVert \lVert \mathbf{k} \rVert \cos(\theta_{qk})
$$
- With this in mind, the intuition behind RoPE is that we can represent the token embeddings as complex numbers and their positions as pure rotations that we apply to them.

If we shift both the query and key by the same amount, changing absolute position but not relative position, this will lead both representations to be additionally rotated in the same manner---as we will see in the derivation---thus the angle between them will remain unchanged and thus the dot product will also remain unchanged. By exploiting the nature of rotations, the dot product used in self-attention will have the property we are looking for, preserving relative positional information while discarding absolute position.

The following is an example illustrating the core idea of RoPE—a more rigorous derivation is presented in a subsequent section. Some arbitrary 0<ε≤π/2N is chosen, where N is the maximum sequence length. When viewed elementwise on q and k, with j as the element index, RoPE can be viewed as follows:
$$
\begin{align}
\mathrm{RoPE}(x, m) &= xe^{mi\varepsilon} \\
\langle \mathrm{RoPE}(q_j, m), \mathrm{RoPE}(k_j, n)\rangle &= \langle q_j e^{mi\varepsilon}, k_j e^{ni\varepsilon} \rangle \\
&= q_j k_j e^{mi\varepsilon} \overline{e^{ni\varepsilon}} \\
&= q_j k_j e^{(m - n)i\varepsilon} \\
&= \mathrm{RoPE}(q_j k_j, m - n)
\end{align}
$$
## Derivation

We begin with absolute positional information: for each token, we know where it is in the sequence. However, dot products (and therefore attention) do not preserve absolute positional information, so if we encode that positional information in the absolute position of the embeddings, we will lose a significant amount of information. -> On the other hand, dot products do preserve relative position, so if we can encode the absolute positional information into the token embeddings in a way that only leverages relative positional information, that will be preserved by the attention function.

Instead of working in the usual Rd, we will work in Cd/2 by considering consecutive pairs of elements of the query and key vectors to be a single complex number. Specifically, instead of viewing q=(q1,q2,q3,q4,…,qd) 
as a d-dimensional real vector we view it as $$ q=(q1+iq2,q3+iq4,…qd−1+iqd)∈Cd/2. $$ As we will see, casting it in this fashion will make discussing the rotary embeddings easier. If d is odd, we can pad it with a dummy coordinate to ensure things line up correctly. Alternatively, we can simply increase d by one

Let q and k be query and key vectors respectively and let m and n be the absolute positions of the corresponding tokens. Let f(x,ℓ) be the function that takes the token embedding x in position ℓ and outputs a new embedding that contains (in some fashion) the relative positional information. Our goal is to find a "nice" function f that does this. Once the positional information is encoded, we need to compute the inner product like so:

$$
\langle f(\mathbf{q}, m),f(\mathbf{k},n) \rangle = g(\mathbf{q},\mathbf{k}, m - n)
$$

'
where g(q,k,m−n) now represents the pre-softmax logit of the usual attention equation. Writing these three functions in exponential form gives:
$$
\begin{align*}
f(\mathbf{q}, m) &= R_f(\mathbf{q}, m)e^{i\Theta_f(\mathbf{q}, m)}\\
f(\mathbf{k}, n) &= R_f(\mathbf{k}, n)e^{i\Theta_f(\mathbf{k}, n)}\\
g(\mathbf{q}, \mathbf{k}, m - n) &= R_g(\mathbf{q}, \mathbf{k}, m - n)e^{i\Theta_g(\mathbf{q}, \mathbf{k}, m - n)}
\end{align*}
$$
![[Pasted image 20240807213443.png]]Computing the inner product and equating corresponding components yields:
$$
\begin{align*}
R_f(\mathbf{q}, m) R_f(\mathbf{k}, n) &= R_g(\mathbf{q}, \mathbf{k}, m - n)\\
\Theta_f(\mathbf{q}, m) - \Theta_f(\mathbf{k}, n) &= \Theta_g(\mathbf{q}, \mathbf{k}, m - n)\\
\end{align*}
$$
Substituting m=n and applying the initial condition f(x,0)=x gives:
$$ R_f(\mathbf{q}, m) R_f(\mathbf{k}, m) = R_g(\mathbf{q}, \mathbf{k}, 0) = R_f(\mathbf{q}, 0) R_f(\mathbf{k}, 0) = \mathbf{q}\mathbf{k} $$
Long story short, final formula is:
$$
f(\mathbf{q}, m) = R_f(\mathbf{q}, m)e^{i\Theta_f(\mathbf{q}, m)}=\mathbf{q}e^{i(\Theta(\mathbf{q})+m\mathbf{\theta})} = \sum_{j=1}^{d/2} q_je^{im\theta_j} \vec{e_j}
$$
and likewise for k. Since computers tend to like real numbers and matrices more than complex numbers, its convenient to convert this expression into the matrix equation:
$$
f(\mathbf{q}, m) =
\begin{pmatrix}
M_1 & & & \\
& M_2 & & \\
& & \ddots & \\
& & & M_{d/2}
\end{pmatrix}
\begin{pmatrix}
q_1\\
q_2\\
\vdots\\
q_d
\end{pmatrix} = \mathbf{\Theta_m Q_m} = \mathbf{\Theta_m W_q X_m}
$$
where: 
$$
M_j=\begin{pmatrix}\cos m\theta_j & -\sin m\theta_j \\sin m\theta_j & \cos m\theta_j\end{pmatrix}
$$
Θm is the block diagonal rotation matrix, Wq is the learned query weights, and Xm is the embedding of the m token. Again, we also have the corresponding equation for k.
## How is this different from the sinusoidal embeddings used in "Attention is All you Need"

There are two ways that rotary embeddings are different from sinusoidal embeddings:

1. Sinusoidal embeddings apply to each coordinate individually, while rotary embeddings mix pairs of coordinates
2. Sinusoidal embeddings add a cos⁡(mθ) or sin⁡(mθ) term, while rotary embeddings use a multiplicative factor.

## Implementation

The original implementations of RoPE are available in [roformer](https://github.com/ZhuiyiTechnology/roformer) and [bert4keras](https://github.com/bojone/bert4keras).

More naive implementation is the following: 
![[Pasted image 20240807214313.png]]