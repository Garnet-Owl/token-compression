# LLM-Optimized Text Compression: Mathematical Specification

## 1. Introduction

This document provides the mathematical foundation for a text compression system designed for Large Language Model (LLM) processing efficiency. The core innovation is that the compressed output remains in a text format that an LLM can process directly, with the attention mechanism serving as a selective decompression function.

## 2. Mathematical Framework

### 2.1 Base Definitions

Let:
- $S = s_1 s_2 ... s_n$ be an input string of length $n$
- $\Sigma$ be the alphabet (character set)
- $w$ be the sliding window size
- $l$ be the lookahead buffer size
- $m_{min}$ be the minimum match length

### 2.2 Sliding Window Function

The sliding window at position $i$ is defined as:

$$W(i) = S[\max(0, i-w) : i]$$

### 2.3 Lookahead Buffer Function

The lookahead buffer at position $i$ is defined as:

$$L(i) = S[i : \min(n, i+l)]$$

### 2.4 Match Function

For position $i$, we find the optimal match $(d^*, l^*)$ where:

$$M(i) = \arg\max_{(d,l)} \{ l : S[i-d:i-d+l] = S[i:i+l] \}$$

Subject to constraints:
- $1 \leq d \leq \min(i, w)$ (distance within window)
- $m_{min} \leq l \leq l_{max}$ (valid match length)

### 2.5 Encoding Function

The encoding function $E$ maps input to compressed output:

$$E(s_i, M(i)) = \begin{cases} 
\langle d^*, l^* \rangle & \text{if } |M(i)| \geq m_{min} \\
s_i & \text{otherwise}
\end{cases}$$

### 2.6 Compression Ratio

The compression ratio $CR$ is defined as:

$$CR = \frac{|S|}{|C(S)|}$$

Where $C(S)$ is the compressed representation.

## 3. Dictionary Enhancement

### 3.1 Common Phrase Dictionary

Define dictionary $D = \{(p_1, t_1), (p_2, t_2), ..., (p_k, t_k)\}$ where:
- $p_i$ = common phrase
- $t_i$ = compact token replacement

The dictionary substitution function:

$$D(S) = S' \text{ where } \forall (p, t) \in D: p \rightarrow t$$

### 3.2 Token Efficiency

For phrase $p$ with length $|p|$ and token $t$ with length $|t|$:

$$\text{Savings}(p, t) = |p| - |t|$$

Total dictionary savings:

$$\text{TotalSavings} = \sum_{i=1}^{k} \text{count}(p_i) \times \text{Savings}(p_i, t_i)$$

## 4. LLM Attention as Decompression

### 4.1 Selective Decompression Model

In a transformer architecture, attention at layer $l$ for query $q$ over keys $K$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 4.2 Compression-Aware Attention

For compressed input with reference tokens, we propose modified attention:

$$\text{Attention}_{compress}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot \text{Decompress}(V)$$

Where $\text{Decompress}(V)$ expands reference tokens only when attended to:

$$\text{Decompress}(v_i) = \begin{cases}
\text{Expand}(v_i) & \text{if } \alpha_i > \theta \\
v_i & \text{otherwise}
\end{cases}$$

Here $\alpha_i$ is the attention weight and $\theta$ is the attention threshold.

### 4.3 Lazy Decompression Theorem

**Theorem**: For a compressed sequence $C$ with $k$ reference tokens, if the average attention sparsity is $s$ (fraction of tokens attended per step), then the effective decompression work is:

$$W_{decompress} = s \times k \times \bar{l}$$

Where $\bar{l}$ is the average reference length.

**Proof**: 
- Total reference tokens: $k$
- Per inference step, fraction attended: $s$
- Average expansion length: $\bar{l}$
- Work per step: $s \times k \times \bar{l}$

For sparse attention ($s << 1$), this is significantly less than full decompression ($k \times \bar{l}$).

## 5. Reference Token Format

### 5.1 Compact Format

Reference token: $\langle d, l \rangle$ encoded as: `«d,l»`

- $d$: distance (offset backwards)
- $l$: length to copy

### 5.2 Information Content

Each reference token carries:

$$I_{ref} = \log_2(w) + \log_2(l_{max})$$ bits of information

But represents $l$ characters, giving compression efficiency:

$$\eta = \frac{l \times \log_2(|\Sigma|)}{\log_2(w) + \log_2(l_{max})}$$

## 6. Theoretical Compression Bounds

### 6.1 Best Case

For highly repetitive text with repeat length $r$ and period $p$:

$$CR_{best} \approx \frac{r}{|«p,r»|}$$

### 6.2 Worst Case

For random text with no repetitions:

$$CR_{worst} = 1$$ (no compression)

### 6.3 Expected Case

For natural language with entropy $H$:

$$CR_{expected} \approx \frac{1}{H/\log_2(|\Sigma|)}$$

## 7. Algorithm Complexity

### 7.1 Time Complexity

- Naive matching: $O(n \times w \times l)$
- With hash table: $O(n \times l)$ expected
- With suffix array: $O(n \log n)$ preprocessing, $O(n)$ matching

### 7.2 Space Complexity

- Sliding window: $O(w)$
- Hash table (optional): $O(w)$
- Output buffer: $O(n)$

## 8. Token Reduction Analysis

### 8.1 LLM Tokenization

For tokenizer $T$ that maps text to tokens:

$$\text{TokenCount}(S) = |T(S)|$$

### 8.2 Compression Impact

For compressed text $C(S)$:

$$\text{TokenReduction} = \frac{|T(S)| - |T(C(S))|}{|T(S)|} \times 100\%$$

### 8.3 Target Metrics

For the system to be effective:
- Token reduction: $> 30\%$
- Processing overhead: $< 10\%$
- Semantic preservation: $100\%$

## 9. Implementation Considerations

### 9.1 Training Requirements

For an LLM to natively process compressed text:
1. Include compressed examples in training data
2. Train attention to recognize reference patterns
3. Implement dynamic expansion in forward pass

### 9.2 Inference Pipeline

```
Input → Dictionary Compression → LZ77 Compression → Compressed Tokens
                                                            ↓
                                                    LLM Processing
                                                    (with lazy decompression)
                                                            ↓
Decompressed Output ← Dictionary Expansion ← LZ77 Expansion ← Response
```

## 10. Future Extensions

### 10.1 Semantic Compression

Extend beyond syntactic patterns to semantic similarity:

$$\text{Compress}_{semantic}(s_i...s_j) = \text{embed}(s_i...s_j) \rightarrow t_{semantic}$$

### 10.2 Learned Compression

Train end-to-end compression with LLM objective:

$$\mathcal{L} = \mathcal{L}_{LM} + \lambda \times \mathcal{L}_{compression}$$

Where:
- $\mathcal{L}_{LM}$ = language modeling loss
- $\mathcal{L}_{compression}$ = compression ratio loss
- $\lambda$ = balance hyperparameter
