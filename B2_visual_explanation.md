# B2 Feature Engineering: Visual Explanation

## What B2 Computes

### Feature 0: Distance from Anchor Node
```
For each node i at frame t:
  distance[i, t] = ||coords[i, t] - anchor_coords[t]||
  
Where:
  - anchor = Node 0 (for mouth partition)
  - Measures: How far each landmark is from the anchor point
  - Range: [0, +∞)
  - Captures: Overall mouth shape and size
```

### Feature 1: Angle Between Consecutive Nodes
```
For each node i at frame t:
  vec1 = coords[i, t] - coords[i-1, t]  (vector from previous node)
  vec2 = coords[i+1, t] - coords[i, t]  (vector to next node)
  angle[i, t] = arccos(dot(vec1, vec2) / (||vec1|| * ||vec2||))
  
Where:
  - Measures: Local curvature/bending at each node
  - Range: [0, π] radians (0° to 180°)
  - Captures: Lip shape curvature (rounded vs stretched)
```

## How Features Are Concatenated in Models

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT TO MODEL                           │
│         (frames, n_nodes, total_features)                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Concatenated along feature dimension
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  B0 Features (2):  [x, y]                                   │
│  ─────────────────────────────────────────────────────────  │
│  B1 Features (3):  [vx, vy, speed]                          │
│  ─────────────────────────────────────────────────────────  │
│  B2 Features (2):  [distance, angle]  ← B2 adds this       │
│  ─────────────────────────────────────────────────────────  │
│  B3 Features (4):  [AU25, AU26, AU12, AU27]                 │
│  ─────────────────────────────────────────────────────────  │
│  TOTAL: 11 features per node                                │
└─────────────────────────────────────────────────────────────┘
```

## Feature Flow in Training

```
Training with B2:
  ┌─────────────┐
  │   B0 File   │──┐
  └─────────────┘  │
                   ├──► Concatenate ──► Model Input (7 features)
  ┌─────────────┐  │
  │   B1 File   │──┤
  └─────────────┘  │
                   │
  ┌─────────────┐  │
  │   B2 File   │──┘  ← B2 adds 2 features here
  └─────────────┘
```

## What Information B2 Provides

### 1. Spatial Relationships
```
Anchor Node (Node 0)
    │
    ├──► Distance to Node 1
    ├──► Distance to Node 2
    ├──► Distance to Node 3
    └──► ... (all 277 nodes)
    
When mouth opens: distances increase
When mouth closes: distances decrease
```

### 2. Local Curvature
```
Node i-1 ──► Node i ──► Node i+1
            │
            └──► Angle measures how "bent" the path is
            
Small angle (0°):  Straight line
Large angle (180°): Sharp turn
```

## Example: How B2 Helps Distinguish Visemes

### Scenario: "O" vs "U" sounds

**Without B2:**
- B0: Similar positions
- B1: Similar motion patterns
- **Problem**: Model struggles to distinguish

**With B2:**
- B0: Similar positions ✓
- B1: Similar motion patterns ✓
- **B2 Distance**: Different mouth opening sizes
- **B2 Angle**: Different lip curvature patterns
- **Result**: Model can better separate these visemes

## Technical Implementation

### Distance Computation
```python
# Vectorized computation for all nodes at once
anchor_coords = coords[:, anchor_idx:anchor_idx+1, :]  # (frames, 1, 2)
distances = torch.norm(
    coords.unsqueeze(2) - anchor_coords.unsqueeze(1), 
    dim=3
)  # (frames, n_nodes, 1)
```

### Angle Computation
```python
# Compute vectors between consecutive nodes
coords_prev = torch.cat([coords[:, 0:1, :], coords[:, :-1, :]], dim=1)
coords_next = torch.cat([coords[:, 1:, :], coords[:, -1:, :]], dim=1)
vec1 = coords - coords_prev
vec2 = coords_next - coords

# Compute angle using dot product
cos_angles = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
angles = arccos(clamp(cos_angles, -1, 1))  # (frames, n_nodes)
```

## Summary

**B2 adds geometric understanding:**
- **Distance**: "How far is each point from the anchor?"
- **Angle**: "How curved is the path at each point?"

**When combined with B0+B1:**
- B0: Where are the landmarks? (position)
- B1: How are they moving? (motion)
- B2: What's their geometric relationship? (geometry)

**Result**: Model gets richer feature representation → Better viseme classification


