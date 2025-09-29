import os
import glob
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
import os.path as osp

import torch.serialization
import torch_geometric.data

torch.serialization.add_safe_globals([
    torch_geometric.data.Data,
])

# ====== FaceMesh adjacency (468 nodes) ======
# Diambil dari mediapipe topology (paket resmi / dump manual)
# Untuk demo, kita pakai contoh subset edges. 
# Nanti bisa di-extend pakai list full dari mediapipe.
# FACEMESH_CONNECTIONS = [
#     (0, 1), (1, 2), (2, 3), (3, 4),   # contoh sekitar bibir/mata
#     (61, 146), (146, 91), (91, 181),  # bagian bibir
#     (78, 95), (95, 88), (88, 178),    # bagian bibir atas
#     (191, 80), (80, 81), (81, 82),    # dagu
#     # ... (TODO: masukkan semua edges dari mp.solutions.face_mesh.FACEMESH_TESSELATION)
# ]
from gnn.facemesh_topology import get_facemesh_topology

# load adjacency
FACEMESH_ADJ, FACEMESH_CONNECTIONS = get_facemesh_topology()


def get_class_names(landmark_root, split):
    files = glob.glob(os.path.join(landmark_root, split, "*.npz"))
    classes = set()
    for f in files:
        fname = os.path.basename(f)
        label = fname.split("_")[0]   # "ada_00003.npz" -> "ada"
        classes.add(label)
    classes = sorted(list(classes))
    return classes


class LipReadingDataset(InMemoryDataset):
    def __init__(self, root, split="train", transform=None, pre_transform=None):
        self.split = split
        self.landmark_root = root

        # ambil daftar kelas (subfolder di train/val/test)
        split_dir = osp.join(self.landmark_root, self.split)
        if osp.exists(split_dir):
            self.classes = sorted(os.listdir(split_dir))
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        else:
            self.classes = []
            self.class_to_idx = {}

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f"{self.split}.pt"]

    def process(self):
        split_dir = osp.join(self.root, self.split)
        classes = sorted(os.listdir(split_dir))

        data_list = []
        for cls_idx, cls in enumerate(classes):
            cls_dir = osp.join(split_dir, cls)
            for file in os.listdir(cls_dir):
                if file.endswith(".npz"):
                    path = osp.join(cls_dir, file)
                    sample = np.load(path, allow_pickle=True)
                    landmarks = sample["landmarks"]  # (T, N, 3)
                    T, N, F = landmarks.shape

                    x = torch.tensor(landmarks.reshape(T * N, F), dtype=torch.float)

                    # build edge_index
                    _, FACEMESH_CONNECTIONS = get_facemesh_topology()
                    edge_index = []
                    for t in range(T):
                        offset = t * N
                        for (i, j) in FACEMESH_CONNECTIONS:
                            edge_index.append([offset + i, offset + j])
                            edge_index.append([offset + j, offset + i])
                    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

                    y = torch.tensor([cls_idx], dtype=torch.long)

                    data = Data(x=x, edge_index=edge_index, y=y)
                    data.num_frames = T
                    data.num_landmarks = N
                    data_list.append(data)

        if len(data_list) == 0:
            print(f"[WARNING] No samples found in {split_dir}. Skipping...")
            return

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


