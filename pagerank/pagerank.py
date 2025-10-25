import os
import re
import sys
import random

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks_sampling = sample_pagerank(corpus, DAMPING, SAMPLES)
    print("PageRank Results (Sampling):")
    for page in sorted(ranks_sampling):
        print(f"  {page}: {ranks_sampling[page]:.4f}")
    ranks_iter = iterate_pagerank(corpus, DAMPING)
    print("\nPageRank Results (Iteration):")
    for page in sorted(ranks_iter):
        print(f"  {page}: {ranks_iter[page]:.4f}")


def crawl(directory):
    pages = {}
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename), encoding="utf8") as f:
            contents = f.read()
            links = re.findall(r'<a\s+(?:[^>]*?)href="([^"]*)"', contents)
            pages[filename] = set(links) - {filename}
    for filename in pages:
        pages[filename] = {link for link in pages[filename] if link in pages}
    return pages


def transition_model(corpus, page, damping_factor):
    n = len(corpus)
    probs = {p: 0 for p in corpus}
    links = corpus[page]

    if len(links) == 0:
        for p in probs:
            probs[p] = 1 / n
        return probs

    for p in probs:
        probs[p] = (1 - damping_factor) / n
    for link in links:
        probs[link] += damping_factor / len(links)
    return probs


def sample_pagerank(corpus, damping_factor, n):
    pages = list(corpus.keys())
    current = random.choice(pages)
    counts = {p: 0 for p in pages}
    counts[current] += 1

    for _ in range(1, n):
        probs = transition_model(corpus, current, damping_factor)
        current = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
        counts[current] += 1

    return {p: counts[p] / n for p in pages}


def iterate_pagerank(corpus, damping_factor, tol=1e-4, max_iter=1000):
    n = len(corpus)
    pages = list(corpus.keys())
    ranks = {p: 1 / n for p in pages}

    for _ in range(max_iter):
        new_ranks = {}
        dangling_rank = sum(ranks[p] for p in pages if len(corpus[p]) == 0)

        for p in pages:
            rank = (1 - damping_factor) / n
            rank += damping_factor * (dangling_rank / n)
            for q in pages:
                if p in corpus[q] and len(corpus[q]) > 0:
                    rank += damping_factor * (ranks[q] / len(corpus[q]))
            new_ranks[p] = rank

        deltas = [abs(new_ranks[p] - ranks[p]) for p in pages]
        ranks = new_ranks
        if all(delta < tol for delta in deltas):
            break

    s = sum(ranks.values())
    return {p: ranks[p] / s for p in pages}


if __name__ == "__main__":
    main()
