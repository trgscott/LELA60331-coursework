"""
The models use NumPy only for their forward and backward passes.  The input
file is the tab-separated ``Compiled_Reviews.txt``:
review, sentiment rating, product type, helpfulness rating.

Examples:
    python MLP_from_scratch.py --data Compiled_Reviews.txt --model all
    python MLP_from_scratch.py --data Compiled_Reviews.txt --model mlp --plot
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-z]{2,}(?:-[a-z]+)*(?:'[a-z]+)*")
STOP_WORDS = {
    "no", "up", "on", "i'll", "quot", "there's", "us", "she's", "he's",
    "it's", "i've", "i'm", "im", "would", "upon", "also", "i", "me", "my",
    "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "as", "until", "while", "of", "at", "by",
    "for", "with", "about", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "in", "out", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "other",
    "some", "such", "only", "own", "same", "so", "than", "too", "can", "will",
    "just", "should", "now",
}
LABELS = ("neutral", "helpful", "unhelpful")


def load_reviews(path: Path) -> tuple[list[str], list[str]]:
    reviews, ratings = [], []
    with path.open(encoding="utf-8") as data_file:
        next(data_file, None)
        for line_number, line in enumerate(data_file, start=2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise ValueError(f"Expected four tab-separated fields on line {line_number}")
            reviews.append(fields[0])
            ratings.append(fields[3].strip().lower())
    unknown = sorted(set(ratings) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown helpfulness labels: {unknown}")
    if not reviews:
        raise ValueError(f"No reviews found in {path}")
    return reviews, ratings


def tokenize(review: str) -> list[str]:
    review = review.lower()
    review = re.sub(r"[,.;:?!()\"\[\]]+\s*", " ", review)
    tokens = TOKEN_PATTERN.findall(review)
    return [token for token in tokens if token not in STOP_WORDS]


def build_features(reviews: list[str], vocabulary_size: int) -> tuple[np.ndarray, list[str]]:
    tokenized = [tokenize(review) for review in reviews]
    counts = Counter(token for review in tokenized for token in review)
    vocabulary = [token for token, _ in counts.most_common(vocabulary_size)]
    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    features = np.zeros((len(reviews), len(vocabulary)), dtype=np.float64)
    for row, tokens in enumerate(tokenized):
        for token in set(tokens):
            if token in word_to_index:
                features[row, word_to_index[token]] = 1.0
    return features, vocabulary


def split_data(features: np.ndarray, ratings: list[str], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ratings))
    train_end = int(len(indices) * 0.8)
    dev_end = train_end + int(len(indices) * 0.1)
    train_indices, dev_indices, test_indices = (
        indices[:train_end], indices[train_end:dev_end], indices[dev_end:]
    )
    labels = np.array([LABELS.index(rating) for rating in ratings])
    return {
        "x_train": features[train_indices], "x_dev": features[dev_indices],
        "x_test": features[test_indices], "y_train": labels[train_indices],
        "y_dev": labels[dev_indices], "y_test": labels[test_indices],
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def metrics(predictions: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    true_positive = np.array([np.sum((predictions == label) & (truth == label)) for label in range(3)])
    false_positive = np.array([np.sum((predictions == label) & (truth != label)) for label in range(3)])
    false_negative = np.array([np.sum((predictions != label) & (truth == label)) for label in range(3)])
    precision = np.divide(true_positive, true_positive + false_positive, out=np.zeros(3, dtype=float), where=true_positive + false_positive != 0)
    recall = np.divide(true_positive, true_positive + false_negative, out=np.zeros(3, dtype=float), where=true_positive + false_negative != 0)
    return {"precision": float(precision.mean()), "recall": float(recall.mean()), "tp": true_positive.tolist(), "fp": false_positive.tolist(), "fn": false_negative.tolist()}


def train_softmax(x_train: np.ndarray, y_train: np.ndarray, iterations: int, learning_rate: float, seed: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    weights = rng.random((x_train.shape[1], 3))
    bias = np.zeros(3)
    targets = np.eye(3)[y_train]
    losses = []
    for _ in range(iterations):
        probabilities = softmax(x_train @ weights + bias)
        losses.append(float(-np.mean(np.log2(np.maximum((targets * probabilities).sum(axis=1), 1e-15)))))
        gradient = x_train.T @ (probabilities - targets) / len(x_train) + 0.01 * weights
        weights -= learning_rate * gradient
        bias -= learning_rate * (probabilities - targets).mean(axis=0)
    return weights, bias, losses


def train_mlp(x_train: np.ndarray, y_train: np.ndarray, iterations: int, learning_rate: float, hidden_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    input_weights = rng.random((x_train.shape[1], hidden_size))
    output_weights = rng.random((hidden_size, 3))
    targets = np.eye(3)[y_train]
    losses = []
    for _ in range(iterations):
        hidden = np.maximum(x_train @ input_weights, 0)
        probabilities = softmax(hidden @ output_weights)
        losses.append(float(-np.log2(np.maximum((targets * probabilities).sum(axis=1), 1e-15)).sum()))
        output_error = probabilities - targets
        output_gradient = hidden.T @ output_error
        input_gradient = x_train.T @ ((output_error @ output_weights.T) * (hidden > 0))
        input_weights -= learning_rate * input_gradient
        output_weights -= learning_rate * output_gradient
    return input_weights, output_weights, losses


def report(name: str, predictions: np.ndarray, truth: np.ndarray) -> None:
    result = metrics(predictions, truth)
    print(f"{name}: macro precision={result['precision']:.6f}, macro recall={result['recall']:.6f}")
    print(f"  TP={result['tp']} FP={result['fp']} FN={result['fn']}")


def run(args: argparse.Namespace) -> None:
    reviews, ratings = load_reviews(args.data)
    features, vocabulary = build_features(reviews, args.vocabulary_size)
    data = split_data(features, ratings, args.seed)
    print(f"Loaded {len(reviews)} reviews and {len(vocabulary)} vocabulary items")
    print("Training labels:", {label: int(np.sum(data["y_train"] == index)) for index, label in enumerate(LABELS)})

    if args.model in ("softmax", "all"):
        weights, bias, losses = train_softmax(data["x_train"], data["y_train"], args.softmax_iterations, 0.2, args.seed)
        report("One-hot softmax", np.argmax(softmax(data["x_test"] @ weights + bias), axis=1), data["y_test"])
        plot_loss("One-hot softmax", losses, args.plot)
        print("Top neutral terms:", [vocabulary[i] for i in np.argsort(weights[:, 0])[-20:][::-1]])

    if args.model in ("mlp", "all"):
        input_weights, output_weights, losses = train_mlp(data["x_train"], data["y_train"], args.mlp_iterations, 0.01, args.hidden_size, args.seed)
        hidden = np.maximum(data["x_test"] @ input_weights, 0)
        report("MLP", np.argmax(softmax(hidden @ output_weights), axis=1), data["y_test"])
        plot_loss("MLP", losses, args.plot)

    if args.model == "word2vec":
        run_word2vec(reviews, ratings, args)


def plot_loss(title: str, losses: list[float], enabled: bool) -> None:
    if not enabled:
        return
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()


def run_word2vec(reviews: list[str], ratings: list[str], args: argparse.Namespace) -> None:
    try:
        import gensim.downloader as api
    except ImportError as error:
        raise SystemExit("Word2Vec requires gensim; install it with `pip install gensim`.") from error
    print("Loading word2vec-google-news-300 (this may download several GB on first use)...")
    model = api.load("word2vec-google-news-300")
    tokenized = [tokenize(review) for review in reviews]
    embeddings = np.array([sum((model[token] for token in tokens if token in model), np.zeros(300)) for tokens in tokenized]) / 2
    data = split_data(embeddings, ratings, args.seed)
    weights, bias, losses = train_softmax(data["x_train"], data["y_train"], args.embedding_iterations, 0.01, args.seed)
    report("Word2Vec softmax", np.argmax(softmax(data["x_test"] @ weights + bias), axis=1), data["y_test"])
    plot_loss("Word2Vec softmax", losses, args.plot)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Compiled_Reviews.txt"), help="Path to the tab-separated review data")
    parser.add_argument("--model", choices=("softmax", "mlp", "word2vec", "all"), default="all")
    parser.add_argument("--vocabulary-size", type=int, default=4804)
    parser.add_argument("--hidden-size", type=int, default=2)
    parser.add_argument("--softmax-iterations", type=int, default=1200)
    parser.add_argument("--mlp-iterations", type=int, default=300)
    parser.add_argument("--embedding-iterations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--plot", action="store_true", help="Display loss plots")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
