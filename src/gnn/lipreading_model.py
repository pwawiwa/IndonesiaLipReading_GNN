import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphTemporalGNN(nn.Module):
    """
    Pipeline:
      - Input Data: flattened graph with nodes = T * N
        * data.x: [T*N, in_feats]
        * data.edge_index: edges connecting nodes only within same frame
        * data.num_frames, data.num_landmarks : ints
      - Apply L layers of GCN over the whole flattened graph -> node embeddings [T*N, hid]
      - Reshape -> [batch_size, T, N, hid]
      - Frame pooling (mean over N) -> [batch_size, T, hid]
      - Temporal aggregator (GRU) -> final embedding
      - MLP classifier -> logits (num_classes)
    """
    def __init__(
        self,
        in_feats=3,
        gcn_hidden=128,
        gcn_layers=2,
        gru_hidden=128,
        gru_layers=1,
        num_classes=10,
        dropout=0.3,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.gcn_hidden = gcn_hidden
        self.gcn_layers = gcn_layers
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.num_classes = num_classes
        self.dropout = dropout

        # GCN stack
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_feats, gcn_hidden))
        for _ in range(gcn_layers - 1):
            self.convs.append(GCNConv(gcn_hidden, gcn_hidden))

        # node-level MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(gcn_hidden, gcn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, num_classes)
        )

    def forward(self, data):
        """
        data: batched PyG Data
          - data.x: [total_nodes_in_batch, in_feats]
          - data.edge_index: adjusted edge_index by DataLoader batching
          - data.num_frames: int or tensor per sample (we expect attribute on Data object)
          - data.num_landmarks: int (fixed across dataset)
        """
        x, edge_index = data.x, data.edge_index

        # 1) GCN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.node_mlp(x)

        # 2) Parse meta info
        if hasattr(data, "num_landmarks"):
            if isinstance(data.num_landmarks, torch.Tensor):
                # if tensor with one element
                if data.num_landmarks.numel() == 1:
                    N = int(data.num_landmarks.item())
                else:
                    # fallback: take first element if accidentally stored as vector
                    N = int(data.num_landmarks[0].item())
            else:
                N = int(data.num_landmarks)
        else:
            raise ValueError("Data must have attribute num_landmarks (int).")

        batch = data.batch  # [total_nodes] -> sample idx
        batch_size = int(batch.max().item() + 1)

        # 3) Count nodes per sample
        nodes_per_sample = torch.zeros(batch_size, dtype=torch.long, device=batch.device)
        for i in range(batch_size):
            nodes_per_sample[i] = (batch == i).sum()

        if not torch.all(nodes_per_sample % N == 0):
            raise ValueError("Number of nodes per sample not divisible by num_landmarks N.")

        Ts = (nodes_per_sample // N).cpu().numpy().tolist()

        # 4) Split by sample and do frame pooling
        frame_embeddings_list = []
        ptr = 0
        for sample_idx in range(batch_size):
            num_nodes = int(nodes_per_sample[sample_idx].item())
            T = num_nodes // N
            sample_nodes = x[ptr: ptr + num_nodes]  # [T*N, hid]
            ptr += num_nodes

            sample_nodes = sample_nodes.view(T, N, -1)  # [T, N, hid]
            frame_emb = sample_nodes.mean(dim=1)       # [T, hid]
            frame_embeddings_list.append(frame_emb)

        # 5) Pad to max_T
        max_T = max([fe.shape[0] for fe in frame_embeddings_list])
        hid = frame_embeddings_list[0].shape[1]
        seq_tensor = torch.zeros(batch_size, max_T, hid, device=x.device)
        seq_lengths = []
        for i, fe in enumerate(frame_embeddings_list):
            T = fe.shape[0]
            seq_tensor[i, :T] = fe
            seq_lengths.append(T)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long, device=x.device)

        # 6) Pack for GRU
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_tensor, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        last_hidden = h_n[-1]  # [batch_size, gru_hidden]

        # 7) Classify
        logits = self.classifier(last_hidden)
        return logits
