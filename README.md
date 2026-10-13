# Self-Attention from Scratch, with Real Numbers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seyedvahidmirnezami/self-attention-from-scratch/blob/main/self_attention_from_scratch.ipynb)
[![Course on Udemy](https://img.shields.io/badge/Full%20course-Udemy-A435F0?logo=udemy&logoColor=white)](https://www.udemy.com/course/master-transformers-from-scratch-bert-vit-ai-search/)

Self-attention explained with numbers small enough to check on paper, then implemented in NumPy and PyTorch.
Every number below is reproduced exactly by the code, and the tests check it.

## The running example

Three tokens, "bank of river", with 4-dimensional toy embeddings:

```
X = [ 1 0 1 0 ]   bank
    [ 0 1 0 1 ]   of
    [ 1 1 0 0 ]   river
```

### Step 1: score, normalize, mix (no weights)

```
S = X Xᵀ          A = softmax(S)               Y = A X
[ 2 0 1 ]         [ 0.665 0.090 0.245 ]        [ 0.910 0.335 0.665 0.090 ]  bank
[ 0 2 1 ]   →     [ 0.090 0.665 0.245 ]   →    [ 0.335 0.910 0.090 0.665 ]  of
[ 1 1 2 ]         [ 0.212 0.212 0.576 ]        [ 0.788 0.788 0.212 0.212 ]  river
```

"bank" keeps most of itself but pulls in some "river". Its vector now carries context.

### Step 2: learnable queries, keys and values

```
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V      where Q = X W_Q, K = X W_K, V = X W_V
```

With the example weights from the notebook, the focus of "bank" changes from `[0.665, 0.090, 0.245]` to `[0.707, 0.070, 0.223]`.
Training adjusts `W_Q`, `W_K` and `W_V` until the focus pattern helps the task.

### Step 3: multi-head attention

Two heads with different weights look at the same sentence differently:
head 1 makes "bank" attend mostly to "of" (0.768), head 2 mostly to itself (0.576).
Their outputs are concatenated and projected back to the input size, so blocks can be stacked.

## What's in this repo

| File | What it is |
|---|---|
| [`self_attention_from_scratch.ipynb`](self_attention_from_scratch.ipynb) | Step-by-step walkthrough of all three steps. Open it in Colab, no setup needed |
| [`self_attention.py`](self_attention.py) | NumPy: `simple_self_attention`, `scaled_dot_product_attention`, `multi_head_attention` |
| [`self_attention_torch.py`](self_attention_torch.py) | PyTorch `SelfAttention` and `MultiHeadAttention` modules, batched and trainable |
| [`tests/test_course_numbers.py`](tests/test_course_numbers.py) | Checks the code against every hand-worked number above |

## Run it locally

```bash
pip install -r requirements.txt
python self_attention.py          # prints steps 1 and 2
python self_attention_torch.py    # multi-head attention on a random batch
pytest                            # verifies the course numbers
```

## Learn the full transformer

This repo covers Sessions 2–4 of **[Master Transformers from Scratch: BERT, ViT & AI Search](https://www.udemy.com/course/master-transformers-from-scratch-bert-vit-ai-search/)**.
The course continues from here to positional encoding, the encoder block, BERT pretraining in PyTorch,
fine-tuning for classification and NER, Vision Transformers, and two-tower semantic search.

More from the course:
- [transformer-encoder-pytorch](https://github.com/seyedvahidmirnezami/transformer-encoder-pytorch): the full encoder block, and why positional encoding matters
- [vision-transformer-cassava](https://github.com/seyedvahidmirnezami/vision-transformer-cassava): fine-tune a Vision Transformer on Kaggle
- [semantic-search-two-tower](https://github.com/seyedvahidmirnezami/semantic-search-two-tower): build two-tower retrieval from scratch

If this helped you understand attention, a ⭐ helps other learners find it.

## License

[MIT](LICENSE)
