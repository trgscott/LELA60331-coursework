# Product Review Helpfulness Classifier

`MLP_from_scratch.py` implements the forward and backward passes with NumPy rather than using a machine-learning framework.

The script includes:

- One-hot bag-of-words softmax regression with L2 regularization
- A two-layer neural network with a ReLU hidden layer and softmax output
- Optional Word2Vec embeddings followed by softmax regression
- Macro-averaged precision and recall, plus per-class TP, FP, and FN counts
- Optional training-loss plots

## Requirements

Python 3.10 or newer is recommended. Install the basic dependency with:

```bash
python3 -m pip install numpy
```

Matplotlib is only required when using `--plot`:

```bash
python3 -m pip install matplotlib
```

The Word2Vec model additionally requires `gensim` and downloads `word2vec-google-news-300`, which is several gigabytes:

```bash
python3 -m pip install gensim
```

## Dataset

The default input filename is `Compiled_Reviews.txt`. The original dataset can be downloaded with:

```bash
curl -L -o Compiled_Reviews.txt \
  https://raw.githubusercontent.com/cbannard/lela60331_24-25/refs/heads/main/coursework/Compiled_Reviews.txt
```

The file must be tab-separated with a header and four columns in this order:

1. Review text
2. Sentiment rating
3. Product type
4. Helpfulness rating

Helpfulness labels must be `neutral`, `helpful`, or `unhelpful`.

## Usage

Run the one-hot softmax model:

```bash
python3 MLP_from_scratch.py --data Compiled_Reviews.txt --model softmax
```

Run the neural network:

```bash
python3 MLP_from_scratch.py --data Compiled_Reviews.txt --model mlp
```

Run both one-hot models:

```bash
python3 MLP_from_scratch.py --data Compiled_Reviews.txt --model all
```

Run the Word2Vec experiment separately:

```bash
python3 MLP_from_scratch.py --data Compiled_Reviews.txt --model word2vec
```

Display a loss plot after training:

```bash
python3 MLP_from_scratch.py --data Compiled_Reviews.txt --model mlp --plot
```

Use `--help` to see every available option:

```bash
python3 MLP_from_scratch.py --help
```

## Options

| Option | Default | Description |
| --- | ---: | --- |
| `--data` | `Compiled_Reviews.txt` | Path to the input TSV file |
| `--model` | `all` | `softmax`, `mlp`, `word2vec`, or `all` |
| `--vocabulary-size` | `4804` | Number of most frequent vocabulary items |
| `--hidden-size` | `2` | MLP hidden-layer size |
| `--softmax-iterations` | `1200` | One-hot softmax training iterations |
| `--mlp-iterations` | `300` | MLP training iterations |
| `--embedding-iterations` | `4000` | Word2Vec softmax training iterations |
| `--seed` | `10` | Random seed for reproducible splits and initialization |
| `--plot` | disabled | Display training-loss plots |

## Notes

The one-hot feature matrix can require substantial memory for the full dataset. The Word2Vec option is intentionally separate because downloading and loading the pretrained model is expensive. Results can vary if the seed or training settings are changed.
