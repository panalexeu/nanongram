### nanongram

ngrams are the most minimal language models possible.
They work by calculating the probability of some word `w_n`
appearing after some `w_n-1` words, where `n` is the
ngram size, e.g. 2, 3, 4, etc.

ngrams and LLMs share a common training objective: both predict
the most probable next word, though ngrams are purely statistical
models.

The current implementation is minimal and dependency-free. It defines
text preprocessing (preproc.py), tokenization (tokenizer.py), and the statistical model
(model.py), along with scripts to train the model and export it in pkl format (train.py),
and to sample from a trained model, either from a randomly chosen start ngram or by specifying the first n-1 tokens. Sampling
also supports text streaming (sample.py).
preproc.py implements only a preprocessor to collect messages from a Telegram chat, allowing
you to train a model on your friend's messages from a conversation and then have fun sampling from it.
That said, it's easy to implement a preprocessor for your own use case.

train.py usage example:
```bash
python ./train.py <USER_ID> <TG_CNV_EXPORT_PATH> --preproc_export_path ./out.txt --ngram 2 --export_prob_path ./outprobs.pkl
```

sample.py usage example:
```bash
python ./sample.py ./outprobs.pkl
```
