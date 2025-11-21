# Memory Consumption Analysis

## Top Memory Consumers (batch_size=32)

### 1. **GAT Attention Weights: 15.62 GB** ⚠️ **BIGGEST CONSUMER**
- **Why so large**: GAT computes attention weights for every node-edge pair
- **Calculation**: `batch_size × seq_len × num_nodes × num_edges × num_heads × 4 bytes`
- **Current**: 32 × 100 × 468 × 1400 × 2 × 4 = **15.62 GB**
- **Problem**: Scales quadratically with graph size (nodes × edges)
- **Impact**: This is 90% of total memory!

### 2. **GAT Activations: 1.43 GB** ⚠️ **VERY LARGE**
- **Why**: Intermediate activations during GAT forward pass
- **Calculation**: `batch_size × seq_len × num_nodes × spatial_dim × num_heads × 4 bytes`
- **Current**: 32 × 100 × 468 × 128 × 2 × 4 = **1.43 GB**

### 3. **Input Data: 0.23 GB**
- **Why**: Stored sequence data `[batch, seq, nodes, features]`
- **Calculation**: 32 × 100 × 468 × 42 × 4 = **0.23 GB**

### 4. **LSTM: 3.12 MB** ✓ Small
### 5. **Model Parameters: 2.92 MB** ✓ Small
### 6. **Gradients + Optimizer: 8.75 MB** ✓ Small

**Total Estimated: ~17.30 GB**

## Key Insight

**The GAT (Graph Attention Network) attention mechanism is the problem!**

- Attention weights scale as: `O(nodes × edges × heads × batch × seq_len)`
- With 468 nodes and 1400 edges, this creates a **huge attention matrix**
- Each timestep processes this for all nodes and edges
- Multiplied by sequence length (100) and batch size (32), it explodes

## Solutions to Reduce Memory

### Option 1: Reduce Attention Heads (Easiest)
- Current: `num_heads=2`
- Change to: `num_heads=1`
- **Savings**: ~50% reduction in attention weights = **~7.8 GB saved**

### Option 2: Use Regular GCN Instead of GAT (Most Effective)
- Replace GAT with standard GCN (no attention weights)
- **Savings**: Eliminates 15.62 GB attention weights entirely
- **Trade-off**: Loses attention mechanism, but GCN still learns spatial patterns

### Option 3: Reduce Sequence Length
- Process shorter sequences or use chunking
- **Savings**: Linear reduction with seq_len
- **Trade-off**: May lose long-term temporal dependencies

### Option 4: Gradient Checkpointing
- Trade computation for memory
- **Savings**: ~50% memory reduction
- **Trade-off**: Slower training (2x forward passes)

### Option 5: Reduce Graph Size
- Use fewer landmarks (e.g., only lip region instead of full face)
- **Savings**: Quadratic reduction (nodes² × edges)
- **Trade-off**: Less spatial information

## Recommended Solution

**Use Regular GCN instead of GAT** for spatial processing:
- Eliminates 15.62 GB attention weights
- Still learns spatial patterns through graph convolutions
- Much more memory efficient
- Can keep full model complexity otherwise

This would reduce total memory from **~17.3 GB to ~1.7 GB** (90% reduction!)

