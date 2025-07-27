"""
Plotting utilities for NutriSage nutrition grade prediction.
Provides visualization for model evaluation and training results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger
import typer
import pickle

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, MODELS_DIR

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 15,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot feature importance from XGBoost model.
    
    Args:
        feature_importance: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        save_path: Path to save the plot
    """
    logger.info(f"Creating feature importance plot (top {top_n} features)")
    
    # Get top N features
    top_features = feature_importance.head(top_n)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Horizontal bar plot
    bars = plt.barh(
        range(len(top_features)),
        top_features['importance'],
        color=sns.color_palette("viridis", len(top_features))
    )
    
    # Customize the plot
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importance (XGBoost)')
    plt.gca().invert_yaxis()  # Show most important at top
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        plt.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height()/2,
            f'{importance:.4f}',
            va='center',
            fontsize=10
        )
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes (e.g., ['A', 'B', 'C', 'D'])
        save_path: Path to save the plot
    """
    logger.info("Creating confusion matrix plot")
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix - Nutrition Grade Prediction')
    plt.xlabel('Predicted Grade')
    plt.ylabel('True Grade')
    
    # Add accuracy text
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    plt.text(
        0.5, -0.1,
        f'Overall Accuracy: {accuracy:.3f}',
        ha='center',
        va='center',
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    train_scores: List[float],
    val_scores: List[float],
    metric_name: str = 'Accuracy',
    save_path: Optional[Path] = None
) -> None:
    """
    Plot training curves (accuracy/loss over epochs).
    
    Args:
        train_scores: Training scores for each epoch
        val_scores: Validation scores for each epoch
        metric_name: Name of the metric being plotted
        save_path: Path to save the plot
    """
    logger.info(f"Creating training curves plot for {metric_name}")
    
    epochs = range(1, len(train_scores) + 1)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, train_scores, 'b-', label=f'Training {metric_name}', linewidth=2)
    plt.plot(epochs, val_scores, 'r-', label=f'Validation {metric_name}', linewidth=2)
    
    plt.title(f'Training Curves - {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add final values annotation
    plt.text(
        0.02, 0.98,
        f'Final Training {metric_name}: {train_scores[-1]:.4f}\nFinal Validation {metric_name}: {val_scores[-1]:.4f}',
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def load_model_and_metadata(model_path: Path, metadata_path: Path) -> tuple:
    """
    Load trained model and metadata for plotting.
    
    Args:
        model_path: Path to the trained model
        metadata_path: Path to the model metadata
        
    Returns:
        Tuple of (model, metadata)
    """
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return model, metadata


def create_model_evaluation_plots(
    model_path: Path = MODELS_DIR / "nutrition_grade_model.pkl",
    metadata_path: Path = MODELS_DIR / "model_metadata.pkl",
    test_predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    output_dir: Path = FIGURES_DIR
) -> None:
    """
    Create comprehensive model evaluation plots.
    
    Args:
        model_path: Path to trained model
        metadata_path: Path to model metadata
        test_predictions_path: Path to test predictions
        output_dir: Directory to save plots
    """
    logger.info("Creating comprehensive model evaluation plots")
    
    try:
        # Load model and metadata
        model, metadata = load_model_and_metadata(model_path, metadata_path)
        
        # Create feature importance plot
        if 'feature_importance' in metadata:
            feature_importance_df = pd.DataFrame(metadata['feature_importance'])
            plot_feature_importance(
                feature_importance_df,
                top_n=15,
                save_path=output_dir / "feature_importance.png"
            )
        
        # Create confusion matrix if test predictions exist
        if test_predictions_path.exists():
            test_results = pd.read_csv(test_predictions_path)
            
            # Get class names from metadata
            class_names = metadata.get('classes', [])
            if metadata.get('label_encoder'):
                class_names = metadata['label_encoder'].classes_
            
            # Create confusion matrix plot
            # Note: This would need actual y_true and y_pred from test data
            # For now, we'll create a placeholder
            logger.info("Test predictions found, but confusion matrix requires actual test labels")
        
        logger.success("Model evaluation plots created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating model evaluation plots: {str(e)}")
        raise


# Command-line interface
app = typer.Typer()


@app.command()
def main(
    plot_type: str = typer.Option("all", "--type", "-t", help="Type of plot: 'feature_importance', 'confusion_matrix', 'training_curves', or 'all'"),
    model_path: Path = typer.Option(MODELS_DIR / "nutrition_grade_model.pkl", "--model", "-m", help="Path to trained model"),
    metadata_path: Path = typer.Option(MODELS_DIR / "model_metadata.pkl", "--metadata", help="Path to model metadata"),
    output_dir: Path = typer.Option(FIGURES_DIR, "--output", "-o", help="Output directory for plots"),
    top_features: int = typer.Option(15, "--top-features", help="Number of top features to show"),
    save_plots: bool = typer.Option(True, "--save/--no-save", help="Save plots to files")
):
    """
    Generate plots for model evaluation and training results.
    
    Examples:
        # Generate all plots
        python -m src.plots
        
        # Generate only feature importance plot
        python -m src.plots --type feature_importance
        
        # Generate plots with custom paths
        python -m src.plots --model models/my_model.pkl --output reports/figures
    """
    logger.info("Starting plot generation...")
    
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if plot_type == "all" or plot_type == "feature_importance":
            # Load model and metadata for feature importance
            if model_path.exists() and metadata_path.exists():
                model, metadata = load_model_and_metadata(model_path, metadata_path)
                
                if 'feature_importance' in metadata:
                    feature_importance_df = pd.DataFrame(metadata['feature_importance'])
                    save_path = output_dir / "feature_importance.png" if save_plots else None
                    plot_feature_importance(feature_importance_df, top_features, save_path)
                else:
                    logger.warning("No feature importance data found in metadata")
            else:
                logger.warning("Model or metadata file not found, skipping feature importance plot")
        
        if plot_type == "all" or plot_type == "confusion_matrix":
            logger.info("Confusion matrix requires test predictions and true labels")
            logger.info("This plot should be generated during model evaluation")
        
        if plot_type == "all" or plot_type == "training_curves":
            logger.info("Training curves require training history data")
            logger.info("This plot should be generated during model training")
        
        logger.success("Plot generation completed!")
        
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
        raise


if __name__ == "__main__":
    app()
