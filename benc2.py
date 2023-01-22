from tokenizers import (ByteLevelBPETokenizer,
                            CharBPETokenizer,
                            SentencePieceBPETokenizer,
                            BertWordPieceTokenizer)
import time
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

fn="wikitext-103/wiki.train.tokens"

t1 = time.time()
with open(fn, 'rt') as inf:
    txt = inf.read()
t2 = time.time()
print(f"Reading txt took {t2-t1} sec.")

tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)

t1 = time.time()
output = tokenizer.encode_batch(txt.split('\n'))
t2 = time.time()
print(f"Encoding took {t2-t1} sec.")