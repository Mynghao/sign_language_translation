import torch
from torch.utils.data import Dataset
from modules.utils import to_TJC, encode_sentence, normalize_skeleton, augment_skeleton

class SignDataset(Dataset):
    def __init__(self, data_dict, labels_dict, stoi,
                 BOS_ID, EOS_ID, UNK_ID, lowercase=True):
        self.data_dict = data_dict
        self.labels_dict = labels_dict
        self.stoi = stoi
        self.BOS_ID = "<bos>"
        self.EOS_ID = "<eos>"
        self.UNK_ID = "<unk>"
        self.lowercase = lowercase
        self.sentence_ids = sorted(list(data_dict.keys()))

    def __len__(self):
        return len(self.sentence_ids)

    def __getitem__(self, idx):
        sid = self.sentence_ids[idx]
        sample = self.data_dict[sid]

        # --- 1. Load Components ---
        # Each to_TJC returns (T, J, 3)
        pose = to_TJC(sample["pose_keypoints_2d"])       # Body-25 (J=25)
        face = to_TJC(sample["face_keypoints_2d"])       # Face (J=70)
        lh   = to_TJC(sample["hand_left_keypoints_2d"])  # Left Hand (J=21)
        rh   = to_TJC(sample["hand_right_keypoints_2d"]) # Right Hand (J=21)

        # --- 2. Concatenate ---
        # Order matters! pose is first, so Index 1 is the Neck.
        # Shape: (T, 137, 3)
        x = torch.cat([pose, face, lh, rh], dim=1)

        # --- 3. Normalize ---
        x = normalize_skeleton(x, neck_index=1)
        x = augment_skeleton(x)
        
        # --- 4. Reshape for ST-GCN ---
        # Current: (T, V, 3)
        # Target:  (C, T, V, M) with M=1
        
        # Permute to (3, T, V)
        x = x.permute(2, 0, 1).contiguous()
        
        # Add Person dimension M=1 -> (3, T, V, 1)
        x = x.unsqueeze(-1)

        # --- 5. Labels ---
        words = self.labels_dict[sid]
        y_ids = encode_sentence(
            words,
            self.stoi,
            self.BOS_ID,
            self.EOS_ID,
            self.UNK_ID,
            lowercase=self.lowercase
        )
        y_ids = torch.tensor(y_ids, dtype=torch.long)

        return x, y_ids