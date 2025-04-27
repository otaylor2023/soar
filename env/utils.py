def encode_one_hot(idx, total):
    one_hot = [0] * total
    if idx is not None and 0 <= idx < total:
        one_hot[idx] = 1
    return one_hot