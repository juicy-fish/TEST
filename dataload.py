import torch
import random
import numpy as np
import string


def generate_random_name():
    return ''.join(random.choices(string.ascii_uppercase, k=5))


def generate_random_date():
    y = random.randint(2015, 2025)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def hash_embed(s, dim=64):
    # Simulate a hash with fixed random projection
    random.seed(hash(s))
    return torch.tensor([random.random() for _ in range(dim)], dtype=torch.float32)


def generate_sample(dim_r=64, dim_f=32):
    name = generate_random_name()
    date = generate_random_date()
    identity_str = name + "_" + date

    # Robust watermark: simulate hash
    W_r = hash_embed(identity_str, dim=dim_r)

    # Fragile watermark: use name+date encoded as float (e.g., ASCII)
    s = name + date
    W_f = torch.tensor([ord(c) / 127.0 for c in s.ljust(dim_f)], dtype=torch.float32)[:dim_f]

    return W_r, W_f
