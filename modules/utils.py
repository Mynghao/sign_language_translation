import torch
import numpy as np
from collections import Counter
import string
import random

def build_vocab(labels_dict, max_vocab=8000, lowercase=True):
    """
    labels_dict: { sid: [word1, word2, ...], ... }
    max_vocab: total vocab size INCLUDING special tokens

    Returns:
        stoi, itoi, PAD_ID, BOS_ID, EOS_ID, UNK_ID
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    counter = Counter()

    for words in labels_dict.values():
        if lowercase:
            words = clean_text(words)
            words = [((w.lower()).translate(str.maketrans('', '', string.punctuation))).strip() for w in words]
        counter.update(words)

    # reserve space for specials
    specials = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    num_specials = len(specials)

    if max_vocab <= num_specials:
        raise ValueError("max_vocab must be > number of special tokens")

    # take most frequent tokens so that total size = max_vocab
    most_common = [
        w for (w, _) in counter.most_common(max_vocab - num_specials)
    ]

    itoi = specials + most_common
    stoi = {tok: i for i, tok in enumerate(itoi)}

    PAD_ID = stoi[PAD_TOKEN]
    BOS_ID = stoi[BOS_TOKEN]
    EOS_ID = stoi[EOS_TOKEN]
    UNK_ID = stoi[UNK_TOKEN]

    return stoi, itoi, PAD_ID, BOS_ID, EOS_ID, UNK_ID

def encode_sentence(words, stoi, BOS_ID, EOS_ID, UNK_ID, lowercase=True):
    """
    words: list of str
    Any word not in `stoi` is mapped to UNK_ID.
    """
    ids = [BOS_ID]
    words = clean_text(words)
    for w in words:
        if lowercase:
            w = w.lower()
            w = w.translate(str.maketrans('', '', string.punctuation))
            w = w.strip()
            
        ids.append(stoi.get(w, UNK_ID))
    ids.append(EOS_ID)
    return ids

def to_TJC(keypoints_flat):
    """
    keypoints_flat: array-like of shape (T, 3*J)
        where each row = [x0, y0, c0, x1, y1, c1, ...]
    Returns:
        x: torch.FloatTensor of shape (T, J, 3)
    """
    arr = np.asarray(keypoints_flat, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)  # (1, 3*J)
    T, F = arr.shape

    assert F % 3 == 0, f"Expected feature dimension divisible by 3, got {F}"

    J = F // 3
    arr = arr.reshape(T, J, 3)
    x = torch.from_numpy(arr)  # (T, J, 3)
    return x

def clean_text(words):
    """
    words: list of str
    Returns cleaned list of str.
    """
    cleaned = []
    for w in words:
        w = w.replace('\n', ' ').replace('\r', ' ').strip()
        if w:
            cleaned.append(w)
    return cleaned

def normalize_skeleton(x, neck_index=1):
    """
    Normalizes the skeleton data.
    
    Args:
        x: Tensor of shape (T, V, 3). 
           Last dim is [x, y, confidence].
        neck_index: The index of the neck joint in the V dimension. 
                    For OpenPose Body-25, this is 1.
    
    Returns:
        x_norm: Tensor of shape (T, V, 3) with centered/scaled coordinates.
    """
    # 1. Separate coordinates and confidence
    coords = x[..., :2]
    conf = x[..., 2:3]

    # 2. Create a mask for valid joints
    mask = (conf > 0.0).float() 
    
    # 3. Find the Neck (Anchor) at every timeframe
    neck = coords[:, neck_index:neck_index+1, :]

    # 4. Center: Subtract neck coordinates
    coords_centered = coords - neck

    # 5. Scale: Squash pixels to approx [-1, 1] range
    coords_scaled = coords_centered / 256.0

    # 6. Re-apply Mask
    coords_scaled = coords_scaled * mask
    
    x_norm = torch.cat([coords_scaled, conf], dim=-1)
    return x_norm

import random

def augment_skeleton(x):
    """
    x: Tensor (T, V, C) or (C, T, V). Assumes last dim is (x, y, conf)
    Applies random augmentations:
        1. Random Scaling (Zoom in/out)
        2. Random Translation (Shift left/right/up/down)
        3. Gaussian Noise (Jitter)
    """
    
    # 1. Random Scaling (Zoom in/out)
    scale = random.uniform(0.9, 1.1)
    x[..., :2] = x[..., :2] * scale

    # 2. Random Translation (Shift left/right/up/down)
    shift_x = random.uniform(-0.05, 0.05)
    shift_y = random.uniform(-0.05, 0.05)
    x[..., 0] += shift_x
    x[..., 1] += shift_y
    
    # 3. Gaussian Noise (Jitter)
    noise = torch.randn_like(x[..., :2]) * 0.005
    x[..., :2] += noise

    return x
