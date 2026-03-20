import torch

def accuracy(similarity, labels, threshold=0.5):
    sim = similarity.view(-1)
    lbs = labels.view(-1).float()
    preds = (sim > threshold).float()
    return (preds == lbs).sum().item() / lbs.numel()


def contrastive_similarity_loss(similarity, labels, margin=0.4):
    s = similarity.view(-1)
    y = labels.view(-1).float()

    pos = y * 0.5 * (1 - s)**2
    neg = (1 - y) * 0.5 * torch.clamp(s - margin, min=0)**2

    return (pos + neg).mean()
