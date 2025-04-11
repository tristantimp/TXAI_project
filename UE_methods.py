import torch
import numpy as np

def mean_token_entropy(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropies = -(probs * log_probs).sum(dim=-1)
    return entropies.mean().item()

def max_sequence_probability(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1).values
    return max_probs.mean().item()

def normalized_entropy(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    max_entropy = np.log(logits.shape[-1])
    return (entropy / max_entropy).mean().item()