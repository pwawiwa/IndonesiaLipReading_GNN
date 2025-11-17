"""
Comprehensive debugging script - runs all analysis tools
"""
import argparse
import sys
from pathlib import Path
import json
import logging
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.debug.visualize_facemesh import visualize_from_pt
from src.debug.enhanced_evaluation import evaluate_model
from src.debug.model_explainability import explain_predictions
from src.dataset.dataset import LipReadingDataset

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_comprehensive_debug(
    checkpoint_path: str,
    train_pt: str,
    val_pt: str,
    test_pt: str,
    output_dir: str = 'debug_outputs/comprehensive',
    num_facemesh_samples: int = 10,
    num_explanation_samples: int = 10,
    input_dim: int = 31
):
    """
    Run comprehensive debugging analysis
    
    Args:
        checkpoint_path: Path to model checkpoint
        train_pt: Path to train.pt
        val_pt: Path to val.pt
        test_pt: Path to test.pt
        output_dir: Output directory
        num_facemesh_samples: Number of samples for facemesh visualization
        num_explanation_samples: Number of samples for explanation
        input_dim: Input dimension
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / f"debug_session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("COMPREHENSIVE MODEL DEBUGGING")
    logger.info("="*70)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output: {session_dir}")
    logger.info("="*70 + "\n")
    
    # Load label map
    logger.info("📦 Loading dataset to get label map...")
    train_dataset = LipReadingDataset(train_pt)
    label_map = train_dataset.label_map
    num_classes = train_dataset.num_classes
    
    config = {
        'input_dim': input_dim,
        'num_classes': num_classes,
        'spatial_dim': 256,
        'temporal_dim': 256,
        'spatial_layers': 3,
        'temporal_layers': 2,
        'dropout': 0.5
    }
    
    results_summary = {
        'timestamp': timestamp,
        'checkpoint': str(checkpoint_path),
        'config': config,
        'label_map': label_map
    }
    
    # 1. Facemesh Visualization
    logger.info("\n" + "="*70)
    logger.info("STEP 1: FACE MESH VISUALIZATION")
    logger.info("="*70)
    facemesh_dir = session_dir / "facemesh_visualization"
    try:
        visualize_from_pt(test_pt, facemesh_dir, num_samples=num_facemesh_samples)
        results_summary['facemesh_visualization'] = {'status': 'success', 'output': str(facemesh_dir)}
        logger.info("✅ Facemesh visualization complete")
    except Exception as e:
        logger.error(f"❌ Facemesh visualization failed: {e}")
        results_summary['facemesh_visualization'] = {'status': 'failed', 'error': str(e)}
    
    # 2. Enhanced Evaluation
    logger.info("\n" + "="*70)
    logger.info("STEP 2: ENHANCED EVALUATION")
    logger.info("="*70)
    eval_dir = session_dir / "evaluation"
    try:
        eval_results = evaluate_model(
            checkpoint_path, config, test_pt, eval_dir, label_map
        )
        results_summary['evaluation'] = {
            'status': 'success',
            'output': str(eval_dir),
            'overall_accuracy': eval_results['overall']['accuracy'],
            'f1_macro': eval_results['overall']['f1_macro'],
            'error_rate': eval_results['error_analysis']['error_rate']
        }
        logger.info("✅ Enhanced evaluation complete")
    except Exception as e:
        logger.error(f"❌ Enhanced evaluation failed: {e}")
        results_summary['evaluation'] = {'status': 'failed', 'error': str(e)}
    
    # 3. Model Explainability
    logger.info("\n" + "="*70)
    logger.info("STEP 3: MODEL EXPLAINABILITY")
    logger.info("="*70)
    explain_dir = session_dir / "explanations"
    try:
        explain_results = explain_predictions(
            checkpoint_path, config, test_pt, explain_dir,
            num_samples=num_explanation_samples, label_map=label_map
        )
        results_summary['explainability'] = {
            'status': 'success',
            'output': str(explain_dir),
            'num_samples_analyzed': len(explain_results)
        }
        logger.info("✅ Model explainability complete")
    except Exception as e:
        logger.error(f"❌ Model explainability failed: {e}")
        results_summary['explainability'] = {'status': 'failed', 'error': str(e)}
    
    # Save summary
    summary_path = session_dir / "debug_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("DEBUGGING COMPLETE")
    logger.info("="*70)
    logger.info(f"Results saved to: {session_dir}")
    logger.info(f"Summary: {summary_path}")
    
    # Print key findings
    if 'evaluation' in results_summary and results_summary['evaluation']['status'] == 'success':
        logger.info("\n📊 KEY FINDINGS:")
        logger.info(f"   Overall Accuracy: {results_summary['evaluation']['overall_accuracy']:.2%}")
        logger.info(f"   F1 Macro: {results_summary['evaluation']['f1_macro']:.4f}")
        logger.info(f"   Error Rate: {results_summary['evaluation']['error_rate']:.2%}")
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(description='Comprehensive model debugging')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--train_pt', type=str, required=True,
                       help='Path to train.pt file')
    parser.add_argument('--val_pt', type=str, required=True,
                       help='Path to val.pt file')
    parser.add_argument('--test_pt', type=str, required=True,
                       help='Path to test.pt file')
    parser.add_argument('--output', type=str, default='debug_outputs/comprehensive',
                       help='Output directory')
    parser.add_argument('--num_facemesh_samples', type=int, default=10,
                       help='Number of samples for facemesh visualization')
    parser.add_argument('--num_explanation_samples', type=int, default=10,
                       help='Number of samples for explanation')
    parser.add_argument('--input_dim', type=int, default=31,
                       help='Input dimension')
    
    args = parser.parse_args()
    
    run_comprehensive_debug(
        args.checkpoint,
        args.train_pt,
        args.val_pt,
        args.test_pt,
        args.output,
        args.num_facemesh_samples,
        args.num_explanation_samples,
        args.input_dim
    )


if __name__ == '__main__':
    main()

