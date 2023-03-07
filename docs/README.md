```
$python inference.py
Loading
type(checkpoint): <class 'dict'>
LLaMA Core model:
 Transformer(
  (tok_embeddings): Embedding(32000, 4096)
  (layers): ModuleList(
    (0): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (1): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (2): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (3): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (4): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (5): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (6): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (7): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (8): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (9): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (10): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (11): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (12): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (13): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (14): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (15): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (16): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (17): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (18): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (19): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (20): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (21): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (22): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (23): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (24): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (25): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (26): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (27): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (28): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (29): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (30): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (31): TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=4096, bias=False)
        (wv): Linear(in_features=4096, out_features=4096, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=11008, bias=False)
        (w2): Linear(in_features=11008, out_features=4096, bias=False)
        (w3): Linear(in_features=4096, out_features=11008, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
  )
  (norm): RMSNorm()
  (output): Linear(in_features=4096, out_features=32000, bias=False)
)
Loaded in 19.94 seconds
I believe the meaning of life is to appreciate everything you have.
This is a journey for me to follow my heart and do what I love. I believe that life is to be lived in the moment and to give yourself the opportunity to dream and have the courage to pursue those dreams.
Everything I do in my life is based on living in the moment. I think it is important to remember that we only have the moment with us. Life is fragile and there is no time to waste. We should all live in the moment and make the most of our lives.
I am not a believer in good or bad. Everything that happens in our lives is the right thing for us to have at that time. We are always moving and growing and learning and life is a wonderful journey.
My role as an artist is to try and depict my feelings and what I am going through. I hope that when people view my art they will also feel my emotions and connect to the pieces.
I am an Australian born artist living in Perth, Australia. I work with acrylic paints, inks, gouache, oil paints and mixed media. I have been a full-time artist since 2008. I have exhibited
==================================
```
