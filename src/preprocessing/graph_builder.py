import torch
from torch_geometric.data import Data
from pathlib import Path
import numpy as np
import mediapipe as mp
from tqdm import tqdm


class GraphBuilderST:
    def __init__(self, facemesh_dir, graph_save_dir):
        self.facemesh_dir = Path(facemesh_dir)
        self.graph_save_dir = Path(graph_save_dir)
        self.graph_save_dir.mkdir(parents=True, exist_ok=True)
        self.spatial_edges = self._get_facemesh_connections()

    def _get_facemesh_connections(self):
        """Get MediaPipe FaceMesh adjacency list (spatial edges)."""
        mp_face_mesh = mp.solutions.face_mesh
        edges = set()
        for conn in mp_face_mesh.FACEMESH_TESSELATION:
            edges.add(tuple(conn))
        edges = list(edges)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()  # shape [2, E]

    def _add_temporal_edges(self, T, num_nodes):
        """Connect the same landmark across consecutive frames."""
        temporal_edges = []
        for t in range(T - 1):
            for n in range(num_nodes):
                src = t * num_nodes + n
                dst = (t + 1) * num_nodes + n
                temporal_edges.append((src, dst))
                temporal_edges.append((dst, src))  # bidirectional
        return torch.tensor(temporal_edges, dtype=torch.long).t().contiguous()

    def _repeat_spatial_edges_over_time(self, T, num_nodes, spatial_edges):
        """Duplicate spatial edges for each time frame with index offset."""
        repeated = []
        for t in range(T):
            offset = t * num_nodes
            for e0, e1 in spatial_edges.t().tolist():
                repeated.append((e0 + offset, e1 + offset))
                repeated.append((e1 + offset, e0 + offset))  # make bidirectional
        return torch.tensor(repeated, dtype=torch.long).t().contiguous()

    def process_split(self, split):
        input_file = self.facemesh_dir / f"{split}.pt"
        output_file = self.graph_save_dir / f"{split}_st_graphs.pt"
        data_list = []

        dataset = torch.load(input_file, weights_only=False)
        print(f"📂 Building spatio-temporal graphs for {len(dataset)} {split} videos")

        for sample in tqdm(dataset):
            landmarks = sample["landmarks"]  # (T, 468, 3)
            video_name = sample["video"]

            T, num_nodes, _ = landmarks.shape
            node_features = torch.tensor(
                landmarks.reshape(T * num_nodes, 3),
                dtype=torch.float32
            )

            # Build edges
            spatial_edges = self._repeat_spatial_edges_over_time(T, num_nodes, self.spatial_edges)
            temporal_edges = self._add_temporal_edges(T, num_nodes)
            edge_index = torch.cat([spatial_edges, temporal_edges], dim=1)

            data = Data(
                x=node_features,        # shape [T*468, 3]
                edge_index=edge_index,  # shape [2, E_total]
                num_frames=T,
                video_name=video_name
            )
            data_list.append(data)

        torch.save(data_list, output_file)
        print(f"✅ Saved {split} spatio-temporal graphs to {output_file}")

    def run_all(self):
        for split in ["train", "val", "test"]:
            self.process_split(split)


def main():
    builder = GraphBuilderST(
        facemesh_dir="data/processed",
        graph_save_dir="data/graphs"
    )
    builder.run_all()


if __name__ == "__main__":
    main()
