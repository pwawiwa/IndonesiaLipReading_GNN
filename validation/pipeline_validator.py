"""
Comprehensive Pipeline Validator - Brutal Checking of Every Variable and Number

This script performs exhaustive validation of the entire pipeline:
1. Data Extraction (landmarks, adjacency, speech masks)
2. Feature Engineering (B0-B3 features, shapes, dtypes, ranges)
3. Data Loading (dataset, dataloader, collate function)
4. Model (initialization, forward pass, outputs)
5. Training (loss, gradients, metrics)

All checks produce a PASS/FAIL result with detailed diagnostics.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import traceback
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import setup_logger, load_config
from training import get_dataloader
from models import get_model
from utils.model_loader import filter_model_config


class ValidationResult:
    """Container for validation check results."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.failed = False
        self.errors = []
        self.warnings = []
        self.details = {}
    
    def add_error(self, msg: str):
        self.failed = True
        self.errors.append(msg)
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    def add_detail(self, key: str, value: Any):
        self.details[key] = value
    
    def set_passed(self):
        self.passed = True
        self.failed = False
    
    def get_status(self) -> str:
        if self.failed:
            return "❌ FAILED"
        elif self.passed:
            return "✅ PASSED"
        else:
            return "⚠️  UNKNOWN"


class PipelineValidator:
    """Comprehensive pipeline validator."""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger('PipelineValidator')
        self.results: Dict[str, ValidationResult] = {}
        self.check_counter = 0
    
    def _check(self, name: str, func: callable) -> ValidationResult:
        """Run a validation check and store result."""
        self.check_counter += 1
        result = ValidationResult(f"{self.check_counter:03d}. {name}")
        self.results[name] = result
        
        try:
            func(result)
            if not result.failed:
                result.set_passed()
        except Exception as e:
            result.add_error(f"Exception during check: {str(e)}\n{traceback.format_exc()}")
        
        return result
    
    def check_tensor_properties(
        self,
        result: ValidationResult,
        tensor: torch.Tensor,
        name: str,
        expected_shape: Optional[Tuple] = None,
        expected_dtype: Optional[torch.dtype] = None,
        check_finite: bool = True,
        check_nan: bool = True,
        check_inf: bool = True,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        allow_empty: bool = False
    ):
        """Check tensor properties comprehensively."""
        # Check if tensor exists
        if tensor is None:
            result.add_error(f"{name}: Tensor is None")
            return
        
        # Check if empty
        if tensor.numel() == 0:
            if not allow_empty:
                result.add_error(f"{name}: Tensor is empty")
            else:
                result.add_warning(f"{name}: Tensor is empty (allowed)")
            return
        
        # Check shape
        if expected_shape is not None:
            if tensor.shape != expected_shape:
                result.add_error(
                    f"{name}: Shape mismatch. Expected {expected_shape}, got {tensor.shape}"
                )
            else:
                result.add_detail(f"{name}_shape", tensor.shape)
        
        # Check dtype
        if expected_dtype is not None:
            if tensor.dtype != expected_dtype:
                result.add_error(
                    f"{name}: Dtype mismatch. Expected {expected_dtype}, got {tensor.dtype}"
                )
            else:
                result.add_detail(f"{name}_dtype", str(tensor.dtype))
        
        # Check for NaN
        if check_nan:
            nan_count = torch.isnan(tensor).sum().item()
            if nan_count > 0:
                result.add_error(f"{name}: Contains {nan_count} NaN values")
            else:
                result.add_detail(f"{name}_nan_count", 0)
        
        # Check for Inf
        if check_inf:
            inf_count = torch.isinf(tensor).sum().item()
            if inf_count > 0:
                result.add_error(f"{name}: Contains {inf_count} Inf values")
            else:
                result.add_detail(f"{name}_inf_count", 0)
        
        # Check finite
        if check_finite:
            finite_count = torch.isfinite(tensor).sum().item()
            total_count = tensor.numel()
            if finite_count < total_count:
                result.add_error(
                    f"{name}: Only {finite_count}/{total_count} values are finite"
                )
            else:
                result.add_detail(f"{name}_finite_count", finite_count)
        
        # Check value ranges
        if min_val is not None or max_val is not None:
            if check_finite and torch.isfinite(tensor).all():
                min_actual = tensor.min().item()
                max_actual = tensor.max().item()
                
                if min_val is not None and min_actual < min_val:
                    result.add_error(
                        f"{name}: Min value {min_actual:.6f} < expected {min_val:.6f}"
                    )
                
                if max_val is not None and max_actual > max_val:
                    result.add_error(
                        f"{name}: Max value {max_actual:.6f} > expected {max_val:.6f}"
                    )
                
                result.add_detail(f"{name}_range", (min_actual, max_actual))
    
    def validate_extraction_file(
        self,
        extraction_file: str,
        partition: str,
        expected_n_nodes: Optional[int] = None
    ):
        """Validate extracted landmarks file."""
        
        def _check_extraction(result: ValidationResult):
            try:
                data = torch.load(extraction_file, map_location='cpu', weights_only=False)
            except Exception as e:
                result.add_error(f"Failed to load extraction file: {e}")
                return
            
            # Check required keys
            required_keys = ['partition', 'split', 'adjacency', 'videos', 'word_to_label']
            for key in required_keys:
                if key not in data:
                    result.add_error(f"Missing required key: {key}")
            
            if result.failed:
                return
            
            # Check partition matches
            if data['partition'] != partition:
                result.add_error(
                    f"Partition mismatch: expected {partition}, got {data['partition']}"
                )
            
            result.add_detail('partition', data['partition'])
            result.add_detail('split', data['split'])
            result.add_detail('num_videos', len(data['videos']))
            result.add_detail('num_classes', len(data['word_to_label']))
            
            # Check adjacency matrix
            adj = data['adjacency']
            self.check_tensor_properties(
                result, adj, "adjacency",
                expected_dtype=torch.float32,
                check_finite=True,
                check_nan=True,
                check_inf=True
            )
            
            if adj is not None and not result.failed:
                # Check adjacency shape
                n_nodes_adj = adj.shape[0]
                if adj.shape != (n_nodes_adj, n_nodes_adj):
                    result.add_error(
                        f"Adjacency shape invalid: {adj.shape}, expected square matrix"
                    )
                
                # Check symmetry (should be symmetric for undirected graph)
                if not torch.allclose(adj, adj.T, atol=1e-6):
                    result.add_warning("Adjacency matrix is not symmetric")
                
                # Check diagonal (self-loops should be 0 or 1)
                diagonal = torch.diag(adj)
                if not torch.allclose(diagonal, torch.zeros_like(diagonal)) and \
                   not torch.allclose(diagonal, torch.ones_like(diagonal)):
                    result.add_warning(
                        f"Adjacency diagonal not all 0 or 1: min={diagonal.min():.3f}, max={diagonal.max():.3f}"
                    )
                
                # Check expected node count
                if expected_n_nodes is not None and n_nodes_adj != expected_n_nodes:
                    result.add_error(
                        f"Node count mismatch: expected {expected_n_nodes}, got {n_nodes_adj}"
                    )
                
                result.add_detail('n_nodes', n_nodes_adj)
                result.add_detail('n_edges', int((adj.sum() - adj.trace()).item() // 2))
            
            # Check videos
            if 'videos' in data and len(data['videos']) > 0:
                sample_vid_id = list(data['videos'].keys())[0]
                sample_video = data['videos'][sample_vid_id]
                
                # Check video structure
                required_video_keys = ['landmarks', 'speech_mask', 'word', 'label']
                for key in required_video_keys:
                    if key not in sample_video:
                        result.add_error(f"Video missing required key: {key}")
                
                if not result.failed:
                    # Check landmarks
                    landmarks = sample_video['landmarks']
                    self.check_tensor_properties(
                        result, landmarks, "landmarks",
                        expected_dtype=torch.float32,
                        check_finite=True,
                        check_nan=True,
                        check_inf=True,
                        min_val=-10.0,  # Allow some negative after normalization
                        max_val=10.0
                    )
                    
                    if landmarks is not None and not result.failed:
                        if len(landmarks.shape) != 3:
                            result.add_error(
                                f"Landmarks shape invalid: {landmarks.shape}, expected (frames, n_nodes, 2)"
                            )
                        else:
                            frames, n_nodes, coords = landmarks.shape
                            if coords != 2:
                                result.add_error(
                                    f"Landmarks coordinate dimension invalid: {coords}, expected 2"
                                )
                            result.add_detail('sample_frames', frames)
                            result.add_detail('sample_n_nodes', n_nodes)
                    
                    # Check speech mask
                    speech_mask = sample_video['speech_mask']
                    self.check_tensor_properties(
                        result, speech_mask, "speech_mask",
                        expected_dtype=torch.float32,
                        check_finite=True,
                        check_nan=True,
                        check_inf=True,
                        min_val=0.0,
                        max_val=1.0
                    )
                    
                    if speech_mask is not None and not result.failed:
                        # Check binary values
                        unique_vals = torch.unique(speech_mask)
                        if not torch.allclose(unique_vals, torch.tensor([0.0, 1.0]), atol=1e-6) and \
                           not torch.allclose(unique_vals, torch.tensor([0.0]), atol=1e-6) and \
                           not torch.allclose(unique_vals, torch.tensor([1.0]), atol=1e-6):
                            result.add_error(
                                f"Speech mask not binary: unique values {unique_vals.tolist()}"
                            )
                        
                        # Check length matches frames
                        if landmarks is not None:
                            if speech_mask.shape[0] != landmarks.shape[0]:
                                result.add_error(
                                    f"Speech mask length {speech_mask.shape[0]} != landmarks frames {landmarks.shape[0]}"
                                )
                        
                        result.add_detail('speech_mask_length', speech_mask.shape[0])
                        result.add_detail('speech_frames', speech_mask.sum().item())
                    
                    # Check label
                    label = sample_video['label']
                    if not isinstance(label, (int, torch.int64, torch.int32)):
                        result.add_error(f"Label type invalid: {type(label)}, expected int")
                    else:
                        if label < 0 or label >= len(data['word_to_label']):
                            result.add_error(
                                f"Label {label} out of range [0, {len(data['word_to_label'])-1}]"
                            )
                        result.add_detail('sample_label', label)
                
                # Validate all videos
                video_errors = 0
                for vid_id, video in list(data['videos'].items())[:10]:  # Check first 10
                    if 'landmarks' not in video or video['landmarks'] is None:
                        video_errors += 1
                        continue
                    
                    vid_landmarks = video['landmarks']
                    if vid_landmarks.shape[1] != n_nodes_adj:
                        video_errors += 1
                        result.add_error(
                            f"Video {vid_id}: landmarks nodes {vid_landmarks.shape[1]} != adjacency nodes {n_nodes_adj}"
                        )
                
                if video_errors > 0:
                    result.add_error(f"{video_errors} videos have shape mismatches")
                else:
                    result.add_detail('videos_validated', min(10, len(data['videos'])))
        
        self._check(f"Extraction File: {Path(extraction_file).name}", _check_extraction)
    
    def validate_feature_file(
        self,
        feature_file: str,
        feature_level: str,
        expected_n_features: Optional[int] = None
    ):
        """Validate feature engineering file."""
        
        def _check_features(result: ValidationResult):
            try:
                data = torch.load(feature_file, map_location='cpu', weights_only=False)
            except Exception as e:
                result.add_error(f"Failed to load feature file: {e}")
                return
            
            # Check required keys
            required_keys = ['partition', 'split', 'adjacency', 'videos', 'meta']
            for key in required_keys:
                if key not in data:
                    result.add_error(f"Missing required key: {key}")
            
            if result.failed:
                return
            
            # Check feature level in meta
            if 'meta' in data:
                meta = data['meta']
                if 'featureset' in meta:
                    if meta['featureset'] != feature_level:
                        result.add_error(
                            f"Feature level mismatch: expected {feature_level}, got {meta['featureset']}"
                        )
                    result.add_detail('featureset', meta['featureset'])
                
                # Check expected feature count
                if 'n_features' in meta:
                    n_features_meta = meta['n_features']
                    result.add_detail('n_features_meta', n_features_meta)
                    
                    if expected_n_features is not None:
                        if n_features_meta != expected_n_features:
                            result.add_error(
                                f"Feature count mismatch: expected {expected_n_features}, got {n_features_meta}"
                            )
            
            # Check adjacency (should match extraction)
            adj = data['adjacency']
            self.check_tensor_properties(
                result, adj, "adjacency",
                expected_dtype=torch.float32,
                check_finite=True,
                check_nan=True,
                check_inf=True
            )
            
            # Check videos
            if 'videos' in data and len(data['videos']) > 0:
                sample_vid_id = list(data['videos'].keys())[0]
                sample_video = data['videos'][sample_vid_id]
                
                # Check features
                features = sample_video['features']
                self.check_tensor_properties(
                    result, features, "features",
                    expected_dtype=torch.float32,  # Should be float32 in memory
                    check_finite=True,
                    check_nan=True,
                    check_inf=True
                )
                
                if features is not None and not result.failed:
                    if len(features.shape) != 3:
                        result.add_error(
                            f"Features shape invalid: {features.shape}, expected (frames, n_nodes, n_features)"
                        )
                    else:
                        frames, n_nodes, n_features = features.shape
                        result.add_detail('sample_frames', frames)
                        result.add_detail('sample_n_nodes', n_nodes)
                        result.add_detail('sample_n_features', n_features)
                        
                        # Check feature count matches meta
                        if 'meta' in data and 'n_features' in data['meta']:
                            if n_features != data['meta']['n_features']:
                                result.add_error(
                                    f"Feature count mismatch: sample has {n_features}, meta says {data['meta']['n_features']}"
                                )
                        
                        # Check against expected
                        if expected_n_features is not None:
                            if n_features != expected_n_features:
                                result.add_error(
                                    f"Feature count mismatch: expected {expected_n_features}, got {n_features}"
                                )
                
                # Check speech mask
                speech_mask = sample_video['speech_mask']
                self.check_tensor_properties(
                    result, speech_mask, "speech_mask",
                    expected_dtype=torch.float32,
                    check_finite=True,
                    check_nan=True,
                    check_inf=True,
                    min_val=0.0,
                    max_val=1.0
                )
                
                # Validate all videos
                video_errors = 0
                for vid_id, video in list(data['videos'].items())[:10]:  # Check first 10
                    if 'features' not in video or video['features'] is None:
                        video_errors += 1
                        continue
                    
                    vid_features = video['features']
                    if len(vid_features.shape) != 3:
                        video_errors += 1
                        result.add_error(f"Video {vid_id}: invalid feature shape {vid_features.shape}")
                    
                    if vid_features.shape[-1] != n_features:
                        video_errors += 1
                        result.add_error(
                            f"Video {vid_id}: feature dim {vid_features.shape[-1]} != expected {n_features}"
                        )
                
                if video_errors > 0:
                    result.add_error(f"{video_errors} videos have feature issues")
                else:
                    result.add_detail('videos_validated', min(10, len(data['videos'])))
        
        self._check(f"Feature File: {Path(feature_file).name} ({feature_level})", _check_features)
    
    def validate_dataloader(
        self,
        feature_file: str,
        batch_size: int = 2,
        feature_level: Optional[str] = None,
        feature_dir: Optional[str] = None,
        expected_total_features: Optional[int] = None
    ):
        """Validate dataloader output."""
        
        def _check_dataloader(result: ValidationResult):
            try:
                dataloader = get_dataloader(
                    feature_file,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,  # Use 0 for validation
                    feature_level=feature_level,
                    feature_dir=feature_dir
                )
            except Exception as e:
                result.add_error(f"Failed to create dataloader: {e}")
                return
            
            dataset = dataloader.dataset
            
            # Check dataset properties
            num_classes = dataset.get_num_classes()
            num_nodes = dataset.get_num_nodes()
            feature_dim = dataset.get_feature_dim()
            
            result.add_detail('num_classes', num_classes)
            result.add_detail('num_nodes', num_nodes)
            result.add_detail('feature_dim', feature_dim)
            result.add_detail('dataset_size', len(dataset))
            
            if num_classes <= 0:
                result.add_error(f"Invalid num_classes: {num_classes}")
            
            if num_nodes <= 0:
                result.add_error(f"Invalid num_nodes: {num_nodes}")
            
            if feature_dim <= 0:
                result.add_error(f"Invalid feature_dim: {feature_dim}")
            
            # Check feature dimension matches expected (for incremental loading)
            # This is the total concatenated features, not per-level
            if expected_total_features is not None:
                if feature_dim != expected_total_features:
                    result.add_warning(
                        f"Feature dim {feature_dim} != expected total {expected_total_features} "
                        f"(incremental loading)"
                    )
            
            # Get a batch
            try:
                batch = next(iter(dataloader))
                features, speech_mask, adj, labels = batch
            except Exception as e:
                result.add_error(f"Failed to get batch: {e}")
                return
            
            # Check batch shapes and types
            self.check_tensor_properties(
                result, features, "batch_features",
                expected_dtype=torch.float32,
                check_finite=True,
                check_nan=True,
                check_inf=True
            )
            
            if features is not None and not result.failed:
                if len(features.shape) != 4:
                    result.add_error(
                        f"Batch features shape invalid: {features.shape}, expected (batch, frames, nodes, features)"
                    )
                else:
                    batch_size_actual, max_frames, n_nodes_actual, n_features_actual = features.shape
                    result.add_detail('batch_size', batch_size_actual)
                    result.add_detail('max_frames', max_frames)
                    result.add_detail('batch_n_nodes', n_nodes_actual)
                    result.add_detail('batch_n_features', n_features_actual)
                    
                    if n_nodes_actual != num_nodes:
                        result.add_error(
                            f"Batch nodes {n_nodes_actual} != dataset nodes {num_nodes}"
                        )
                    
                    if n_features_actual != feature_dim:
                        result.add_error(
                            f"Batch features {n_features_actual} != dataset features {feature_dim}"
                        )
            
            self.check_tensor_properties(
                result, speech_mask, "batch_speech_mask",
                expected_dtype=torch.float32,
                check_finite=True,
                check_nan=True,
                check_inf=True,
                min_val=0.0,
                max_val=1.0
            )
            
            if speech_mask is not None and not result.failed:
                if len(speech_mask.shape) != 2:
                    result.add_error(
                        f"Batch speech mask shape invalid: {speech_mask.shape}, expected (batch, frames)"
                    )
                else:
                    if speech_mask.shape[0] != batch_size_actual:
                        result.add_error(
                            f"Speech mask batch size {speech_mask.shape[0]} != features batch size {batch_size_actual}"
                        )
                    if speech_mask.shape[1] != max_frames:
                        result.add_error(
                            f"Speech mask frames {speech_mask.shape[1]} != features frames {max_frames}"
                        )
            
            self.check_tensor_properties(
                result, adj, "batch_adjacency",
                expected_dtype=torch.float32,
                check_finite=True,
                check_nan=True,
                check_inf=True
            )
            
            if adj is not None and not result.failed:
                if adj.shape != (num_nodes, num_nodes):
                    result.add_error(
                        f"Adjacency shape {adj.shape} != expected ({num_nodes}, {num_nodes})"
                    )
            
            self.check_tensor_properties(
                result, labels, "batch_labels",
                expected_dtype=torch.long,
                check_finite=False,  # Long tensors don't need finite check
                check_nan=False,
                check_inf=False
            )
            
            if labels is not None and not result.failed:
                if len(labels.shape) != 1:
                    result.add_error(f"Labels shape invalid: {labels.shape}, expected (batch,)")
                else:
                    if labels.shape[0] != batch_size_actual:
                        result.add_error(
                            f"Labels batch size {labels.shape[0]} != features batch size {batch_size_actual}"
                        )
                    
                    # Check label range
                    if labels.min().item() < 0 or labels.max().item() >= num_classes:
                        result.add_error(
                            f"Labels out of range: min={labels.min().item()}, max={labels.max().item()}, num_classes={num_classes}"
                        )
                    
                    result.add_detail('label_range', (labels.min().item(), labels.max().item()))
        
        self._check(f"DataLoader: {Path(feature_file).name}", _check_dataloader)
    
    def validate_model(
        self,
        model_name: str,
        model_config: Dict,
        sample_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        dataset=None,
        device: str = 'cuda'
    ):
        """Validate model forward pass."""
        
        def _check_model(result: ValidationResult):
            try:
                # Extract in_features and num_classes from dataset or batch
                if dataset is not None:
                    in_features = dataset.get_feature_dim()
                    num_classes = dataset.get_num_classes()
                    num_nodes = dataset.get_num_nodes()
                else:
                    # Fallback: extract from batch (less reliable)
                    features, _, _, labels = sample_batch
                    in_features = features.shape[-1]  # Last dimension is feature dim
                    num_classes = labels.max().item() + 1  # Infer from labels (may be incomplete)
                    num_nodes = features.shape[2]  # Third dimension is nodes
                    result.add_warning("Using batch to infer dimensions (dataset preferred)")
                
                # Build model with required parameters
                filtered_params = filter_model_config(model_name, model_config['params'])
                filtered_params['in_features'] = in_features
                filtered_params['num_classes'] = num_classes
                if model_name in ['adaptive_gcn', 'adaptive_gcn_lstm', 'adaptive_gcn_lstm_mamba']:
                    filtered_params['num_nodes'] = num_nodes
                
                model = get_model(model_name, **filtered_params)
                model = model.to(device)
            except Exception as e:
                result.add_error(f"Failed to build model: {e}\n{traceback.format_exc()}")
                return
            
            result.add_detail('model_name', model_name)
            result.add_detail('in_features', in_features)
            result.add_detail('num_classes', num_classes)
            if model_name in ['adaptive_gcn', 'adaptive_gcn_lstm', 'adaptive_gcn_lstm_mamba']:
                result.add_detail('num_nodes', num_nodes)
            result.add_detail('num_parameters', model.count_parameters())
            
            # Get sample batch
            features, speech_mask, adj, labels = sample_batch
            features = features.to(device)
            speech_mask = speech_mask.to(device)
            adj = adj.to(device)
            
            # Forward pass
            try:
                model.eval()
                with torch.no_grad():
                    outputs = model(features, adj, speech_mask)
            except Exception as e:
                result.add_error(f"Forward pass failed: {e}\n{traceback.format_exc()}")
                return
            
            # Check output
            self.check_tensor_properties(
                result, outputs, "model_outputs",
                expected_dtype=torch.float32,
                check_finite=True,
                check_nan=True,
                check_inf=True
            )
            
            if outputs is not None and not result.failed:
                if len(outputs.shape) != 2:
                    result.add_error(
                        f"Output shape invalid: {outputs.shape}, expected (batch, num_classes)"
                    )
                else:
                    batch_size, num_classes = outputs.shape
                    result.add_detail('output_batch_size', batch_size)
                    result.add_detail('output_num_classes', num_classes)
                    
                    if batch_size != features.shape[0]:
                        result.add_error(
                            f"Output batch size {batch_size} != input batch size {features.shape[0]}"
                        )
                    
                    # Check output range (logits can be any value, but check for extreme values)
                    output_min = outputs.min().item()
                    output_max = outputs.max().item()
                    result.add_detail('output_range', (output_min, output_max))
                    
                    if abs(output_min) > 100 or abs(output_max) > 100:
                        result.add_warning(
                            f"Output logits have extreme values: min={output_min:.3f}, max={output_max:.3f}"
                        )
            
            # Test backward pass
            try:
                model.train()
                features.requires_grad_(True)
                outputs = model(features, adj, speech_mask)
                loss = torch.nn.functional.cross_entropy(outputs, labels.to(device))
                loss.backward()
                
                # Check gradients
                grad_norm = 0.0
                param_count = 0
                nan_grad_count = 0
                inf_grad_count = 0
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_count += 1
                        grad_norm += param.grad.data.norm(2).item() ** 2
                        nan_grad_count += torch.isnan(param.grad).sum().item()
                        inf_grad_count += torch.isinf(param.grad).sum().item()
                
                grad_norm = grad_norm ** 0.5
                
                result.add_detail('gradient_norm', grad_norm)
                result.add_detail('params_with_grad', param_count)
                result.add_detail('nan_gradients', nan_grad_count)
                result.add_detail('inf_gradients', inf_grad_count)
                
                if nan_grad_count > 0:
                    result.add_error(f"Model has {nan_grad_count} NaN gradients")
                
                if inf_grad_count > 0:
                    result.add_error(f"Model has {inf_grad_count} Inf gradients")
                
                if grad_norm == 0.0:
                    result.add_warning("Gradient norm is zero (no learning)")
                elif grad_norm > 1000:
                    result.add_warning(f"Gradient norm is very large: {grad_norm:.3f} (may cause instability)")
                
                result.add_detail('loss_value', loss.item())
                
            except Exception as e:
                result.add_error(f"Backward pass failed: {e}\n{traceback.format_exc()}")
        
        self._check(f"Model: {model_name}", _check_model)
    
    def validate_training_step(
        self,
        model: torch.nn.Module,
        dataloader,
        device: str = 'cuda'
    ):
        """Validate a single training step."""
        
        def _check_training(result: ValidationResult):
            try:
                model.train()
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                
                # Get a batch
                batch = next(iter(dataloader))
                features, speech_mask, adj, labels = batch
                features = features.to(device)
                speech_mask = speech_mask.to(device)
                adj = adj.to(device)
                labels = labels.to(device)
                
                # Forward
                optimizer.zero_grad()
                outputs = model(features, adj, speech_mask)
                loss = criterion(outputs, labels)
                
                # Check loss
                loss_val = loss.item()
                result.add_detail('loss_value', loss_val)
                
                if not np.isfinite(loss_val):
                    result.add_error(f"Loss is not finite: {loss_val}")
                elif loss_val < 0:
                    result.add_error(f"Loss is negative: {loss_val}")
                elif loss_val > 100:
                    result.add_warning(f"Loss is very large: {loss_val}")
                
                # Backward
                loss.backward()
                
                # Check gradients
                total_grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                
                result.add_detail('total_grad_norm', total_grad_norm)
                
                if not np.isfinite(total_grad_norm):
                    result.add_error(f"Gradient norm is not finite: {total_grad_norm}")
                elif total_grad_norm == 0:
                    result.add_warning("Gradient norm is zero")
                elif total_grad_norm > 100:
                    result.add_warning(f"Gradient norm is large: {total_grad_norm}")
                
                # Optimizer step
                optimizer.step()
                
                # Check accuracy
                with torch.no_grad():
                    _, predicted = outputs.max(1)
                    correct = predicted.eq(labels).sum().item()
                    accuracy = 100.0 * correct / labels.size(0)
                
                result.add_detail('accuracy', accuracy)
                
                if accuracy < 0 or accuracy > 100:
                    result.add_error(f"Accuracy out of range: {accuracy}")
                
            except Exception as e:
                result.add_error(f"Training step failed: {e}\n{traceback.format_exc()}")
        
        self._check("Training Step", _check_training)
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 80)
        print("PIPELINE VALIDATION SUMMARY")
        print("=" * 80)
        
        passed = 0
        failed = 0
        warnings = 0
        
        for name, result in sorted(self.results.items()):
            status = result.get_status()
            print(f"\n{status} - {result.name}")
            
            if result.failed:
                failed += 1
                print("  ERRORS:")
                for error in result.errors:
                    print(f"    ❌ {error}")
            else:
                passed += 1
            
            if result.warnings:
                warnings += len(result.warnings)
                print("  WARNINGS:")
                for warning in result.warnings:
                    print(f"    ⚠️  {warning}")
            
            if result.details:
                print("  DETAILS:")
                for key, value in result.details.items():
                    print(f"    • {key}: {value}")
        
        print("\n" + "=" * 80)
        print(f"TOTAL CHECKS: {len(self.results)}")
        print(f"✅ PASSED: {passed}")
        print(f"❌ FAILED: {failed}")
        print(f"⚠️  WARNINGS: {warnings}")
        print("=" * 80)
        
        return failed == 0
    
    def save_report(self, output_file: str):
        """Save validation report to file."""
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PIPELINE VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for name, result in sorted(self.results.items()):
                status = result.get_status()
                f.write(f"{status} - {result.name}\n")
                f.write("-" * 80 + "\n")
                
                if result.errors:
                    f.write("ERRORS:\n")
                    for error in result.errors:
                        f.write(f"  ❌ {error}\n")
                
                if result.warnings:
                    f.write("WARNINGS:\n")
                    for warning in result.warnings:
                        f.write(f"  ⚠️  {warning}\n")
                
                if result.details:
                    f.write("DETAILS:\n")
                    for key, value in result.details.items():
                        f.write(f"  • {key}: {value}\n")
                
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"TOTAL CHECKS: {len(self.results)}\n")
            f.write(f"✅ PASSED: {sum(1 for r in self.results.values() if r.passed)}\n")
            f.write(f"❌ FAILED: {sum(1 for r in self.results.values() if r.failed)}\n")
            f.write("=" * 80 + "\n")


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate entire pipeline')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config YAML file')
    parser.add_argument('--extraction-file', type=str, default=None,
                        help='Path to extraction file to validate')
    parser.add_argument('--feature-file', type=str, default=None,
                        help='Path to feature file to validate')
    parser.add_argument('--output', type=str, default='validation_report.txt',
                        help='Output report file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    partition = config['data']['partition']
    feature_level = config['data']['feature_level']
    model_name = config['model']['name']
    
    # Setup validator
    logger = setup_logger('PipelineValidator')
    validator = PipelineValidator(logger)
    
    # Expected feature counts per level (individual files)
    # When loading incrementally, total = sum of all levels up to target
    feature_counts_per_level = {'B0': 2, 'B1': 3, 'B2': 2, 'B3': 4}
    expected_n_features = feature_counts_per_level.get(feature_level, None)
    
    # Total expected features when loading incrementally
    level_map = {'B0': 0, 'B1': 1, 'B2': 2, 'B3': 3}
    target_level = level_map.get(feature_level, 0)
    total_expected_features = sum(feature_counts_per_level[f'B{i}'] for i in range(target_level + 1))
    
    # Validate extraction file if provided
    if args.extraction_file:
        # Get expected node count from partition
        from preprocessing.mediapipe_nodes import get_partition_nodes
        nodes = get_partition_nodes(partition)
        expected_n_nodes = len(nodes)
        
        validator.validate_extraction_file(
            args.extraction_file,
            partition,
            expected_n_nodes
        )
    
    # Validate feature file if provided
    if args.feature_file:
        validator.validate_feature_file(
            args.feature_file,
            feature_level,
            expected_n_features
        )
    
    # Validate dataloader
    feature_dir = Path(config['data']['feature_dir'])
    train_file = f"{partition}_train.pt"
    
    validator.validate_dataloader(
        train_file,
        batch_size=2,
        feature_level=feature_level,
        feature_dir=str(feature_dir),
        expected_total_features=total_expected_features
    )
    
    # Get a batch for model validation
    try:
        dataloader = get_dataloader(
            train_file,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            feature_level=feature_level,
            feature_dir=str(feature_dir)
        )
        sample_batch = next(iter(dataloader))
    except Exception as e:
        logger.error(f"Failed to get sample batch: {e}")
        sample_batch = None
    
    if sample_batch is not None:
        # Validate model (pass dataset for accurate dimensions)
        dataset = dataloader.dataset
        validator.validate_model(
            model_name,
            config['model'],
            sample_batch,
            dataset=dataset,
            device=args.device
        )
        
        # Validate training step
        try:
            filtered_params = filter_model_config(model_name, config['model']['params'])
            dataset = dataloader.dataset
            filtered_params['in_features'] = dataset.get_feature_dim()
            filtered_params['num_classes'] = dataset.get_num_classes()
            if model_name in ['adaptive_gcn', 'adaptive_gcn_lstm', 'adaptive_gcn_lstm_mamba']:
                filtered_params['num_nodes'] = dataset.get_num_nodes()
            
            model = get_model(model_name, **filtered_params)
            model = model.to(args.device)
            
            validator.validate_training_step(
                model,
                dataloader,
                device=args.device
            )
        except Exception as e:
            logger.error(f"Failed to validate training step: {e}")
    
    # Print summary
    all_passed = validator.print_summary()
    
    # Save report
    validator.save_report(args.output)
    logger.info(f"Validation report saved to {args.output}")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

