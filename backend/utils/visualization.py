"""
Visualization utilities for MicroBias Watermarker.
This module provides functions for visualizing and analyzing bias detection results,
comparing biased vs neutral images, and evaluating steganographic embedding.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import List, Tuple, Dict, Optional, Any
import os
from pathlib import Path
import json
import logging
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix, classification_report
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
class BiasVisualizer:
    """
    Visualizer for bias detection results and analysis.
    """
    def __init__(self, output_dir: str = "outputs/visualizations"):
        """
        Initialize visualizer.
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Color scheme
        self.colors = {
            'neutral': '#4A90E2',  # Blue
            'biased': '#E74C3C',   # Red
            'confidence': '#2ECC71',  # Green
            'uncertainty': '#F39C12',  # Orange
        }
    def visualize_detection_results(
        self,
        images: List[np.ndarray],
        detected_bits: List[int],
        confidences: List[float],
        ground_truth: Optional[List[int]] = None,
        prompts: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        max_images: int = 12,
    ) -> None:
        """
        Visualize bias detection results for multiple images.
        Args:
            images: List of input images (numpy arrays)
            detected_bits: List of detected bias flags
            confidences: List of confidence scores
            ground_truth: Optional list of ground truth bias flags
            prompts: Optional list of text prompts
            save_path: Optional path to save visualization
            max_images: Maximum number of images to display
        """
        # Limit number of images
        num_images = min(len(images), max_images)
        images = images[:num_images]
        detected_bits = detected_bits[:num_images]
        confidences = confidences[:num_images]
        if ground_truth:
            ground_truth = ground_truth[:num_images]
        if prompts:
            prompts = prompts[:num_images]
        # Calculate grid size
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        for i, (img, bit, conf) in enumerate(zip(images, detected_bits, confidences)):
            ax = axes[i] if num_images > 1 else axes[0]
            # Display image
            if img.max() <= 1.0:  # If normalized to [0,1]
                img = (img * 255).astype(np.uint8)
            ax.imshow(img)
            # Determine color based on bias
            bias_color = self.colors['biased'] if bit == 1 else self.colors['neutral']
            # Add title with detection result
            title = f"Detected: bias={bit}\nConfidence: {conf:.3f}"
            if ground_truth:
                correct = "✓" if bit == ground_truth[i] else "✗"
                title += f"\nGround Truth: {ground_truth[i]} {correct}"
                if bit == ground_truth[i]:
                    border_color = self.colors['confidence']
                else:
                    border_color = self.colors['uncertainty']
            else:
                border_color = bias_color
            # Add prompt if available
            if prompts and i < len(prompts):
                # Truncate long prompts
                prompt_text = prompts[i][:50] + "..." if len(prompts[i]) > 50 else prompts[i]
                title += f"\nPrompt: {prompt_text}"
            ax.set_title(title, color=bias_color, fontweight='bold')
            ax.set_xlabel(f"Confidence: {conf:.2f}")
            ax.set_ylabel(f"Bias: {bit}")
            # Add colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])
        # Hide unused subplots
        for i in range(num_images, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.suptitle("Bias Detection Results", fontsize=16, y=1.02)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved detection visualization to {save_path}")
        else:
            save_path = self.output_dir / "detection_results.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    def plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        labels: List[str] = ["Neutral (0)", "Biased (1)"],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot confusion matrix for bias detection.
        Args:
            y_true: Ground truth bias labels
            y_pred: Predicted bias labels
            labels: Class labels
            save_path: Optional path to save visualization
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            square=True,
        )
        plt.title('Confusion Matrix - Bias Detection', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        # Add metrics text
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics_text = f"Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}"
        plt.text(1.3, 0.5, metrics_text, transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        else:
            save_path = self.output_dir / "confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    def plot_confidence_distribution(
        self,
        confidences_correct: List[float],
        confidences_incorrect: List[float],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot confidence distribution for correct vs incorrect predictions.
        Args:
            confidences_correct: Confidence scores for correct predictions
            confidences_incorrect: Confidence scores for incorrect predictions
            save_path: Optional path to save visualization
        """
        plt.figure(figsize=(10, 6))
        # Plot histograms
        plt.hist(
            confidences_correct,
            bins=20,
            alpha=0.7,
            color=self.colors['confidence'],
            label='Correct Predictions',
            density=True,
        )
        plt.hist(
            confidences_incorrect,
            bins=20,
            alpha=0.7,
            color=self.colors['uncertainty'],
            label='Incorrect Predictions',
            density=True,
        )
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Confidence Distribution by Prediction Accuracy', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        # Add statistics
        if confidences_correct:
            mean_correct = np.mean(confidences_correct)
            plt.axvline(mean_correct, color=self.colors['confidence'], linestyle='--',
                       label=f'Mean Correct: {mean_correct:.3f}')
        if confidences_incorrect:
            mean_incorrect = np.mean(confidences_incorrect)
            plt.axvline(mean_incorrect, color=self.colors['uncertainty'], linestyle='--',
                       label=f'Mean Incorrect: {mean_incorrect:.3f}')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confidence distribution to {save_path}")
        else:
            save_path = self.output_dir / "confidence_distribution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    def visualize_dct_blocks(
        self,
        image: np.ndarray,
        block_size: int = 8,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize DCT block analysis on an image.
        Args:
            image: Input image (numpy array)
            block_size: Size of DCT blocks
            save_path: Optional path to save visualization
        """
        # Convert to grayscale for DCT visualization
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        # Calculate number of blocks
        h, w = gray.shape
        num_blocks_h = h // block_size
        num_blocks_w = w // block_size
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        # Grayscale with block grid
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale with 8x8 Blocks')
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        # Add block grid
        for i in range(1, num_blocks_h):
            axes[0, 1].axhline(i * block_size - 0.5, color='red', linewidth=0.5, alpha=0.5)
        for j in range(1, num_blocks_w):
            axes[0, 1].axvline(j * block_size - 0.5, color='red', linewidth=0.5, alpha=0.5)
        # DCT coefficients (first few blocks)
        axes[1, 0].set_title('DCT Coefficients (Sample Blocks)')
        axes[1, 0].axis('off')
        # Process a few blocks and show their DCT
        sample_blocks = []
        for i in range(min(3, num_blocks_h)):
            for j in range(min(3, num_blocks_w)):
                block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                dct_block = cv2.dct(block.astype(np.float32))
                sample_blocks.append(dct_block)
        # Show DCT magnitude
        combined_dct = np.zeros((block_size * 3, block_size * 3))
        for idx, dct_block in enumerate(sample_blocks[:9]):
            i, j = idx // 3, idx % 3
            combined_dct[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = np.abs(dct_block)
        im = axes[1, 0].imshow(combined_dct, cmap='hot')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        # Frequency analysis
        axes[1, 1].set_title('Frequency Analysis')
        axes[1, 1].axis('off')
        # Compute global DCT
        global_dct = cv2.dct(gray.astype(np.float32))
        global_dct_mag = np.abs(global_dct)
        global_dct_mag[0, 0] = 0  # Remove DC component for better visualization
        im = axes[1, 1].imshow(np.log1p(global_dct_mag), cmap='viridis')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved DCT visualization to {save_path}")
        else:
            save_path = self.output_dir / "dct_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    def plot_training_metrics(
        self,
        metrics_history: Dict[str, List[float]],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot training metrics over epochs.
        Args:
            metrics_history: Dictionary of metric names to values over time
            save_path: Optional path to save visualization
        """
        num_metrics = len(metrics_history)
        cols = min(3, num_metrics)
        rows = (num_metrics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        for i, (metric_name, values) in enumerate(metrics_history.items()):
            ax = axes[i] if num_metrics > 1 else axes[0]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, marker='o', linewidth=2, markersize=4)
            # Add trend line
            if len(values) > 2:
                z = np.polyfit(epochs, values, 1)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), "--", alpha=0.7, color='red')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} Over Time')
            ax.grid(True, alpha=0.3)
        # Hide unused subplots
        for i in range(num_metrics, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.suptitle('Training Metrics', fontsize=16, y=1.02)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training metrics to {save_path}")
        else:
            save_path = self.output_dir / "training_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    def create_demo_grid(
        self,
        neutral_images: List[np.ndarray],
        biased_images: List[np.ndarray],
        prompts: List[str],
        save_path: Optional[str] = None,
        max_pairs: int = 5,
    ) -> None:
        """
        Create side-by-side comparison grid of neutral vs biased images.
        Args:
            neutral_images: List of bias=0 images
            biased_images: List of bias=1 images
            prompts: List of text prompts
            save_path: Optional path to save visualization
            max_pairs: Maximum number of image pairs to display
        """
        num_pairs = min(len(neutral_images), len(biased_images), max_pairs)
        fig, axes = plt.subplots(num_pairs, 2, figsize=(12, 4 * num_pairs))
        if num_pairs == 1:
            axes = axes.reshape(1, -1)
        for i in range(num_pairs):
            # Neutral image
            if neutral_images[i].max() <= 1.0:
                neutral_img = (neutral_images[i] * 255).astype(np.uint8)
            else:
                neutral_img = neutral_images[i]
            axes[i, 0].imshow(neutral_img)
            axes[i, 0].set_title(f'Neutral (bias=0)', color=self.colors['neutral'], fontweight='bold')
            # Biased image
            if biased_images[i].max() <= 1.0:
                biased_img = (biased_images[i] * 255).astype(np.uint8)
            else:
                biased_img = biased_images[i]
            axes[i, 1].imshow(biased_img)
            axes[i, 1].set_title(f'Bias=1 Embedded', color=self.colors['biased'], fontweight='bold')
            # Add prompt as subtitle
            if i < len(prompts):
                prompt_text = prompts[i][:60] + "..." if len(prompts[i]) > 60 else prompts[i]
                fig.text(0.5, 1.0 - (i + 0.5) / num_pairs - 0.05,
                        f'Prompt: {prompt_text}',
                        ha='center', fontsize=10, style='italic')
            # Style both images
            for j in range(2):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].spines['top'].set_linewidth(2)
                axes[i, j].spines['right'].set_linewidth(2)
                axes[i, j].spines['bottom'].set_linewidth(2)
                axes[i, j].spines['left'].set_linewidth(2)
                if j == 0:
                    axes[i, j].spines['bottom'].set_color(self.colors['neutral'])
                    axes[i, j].spines['top'].set_color(self.colors['neutral'])
                    axes[i, j].spines['left'].set_color(self.colors['neutral'])
                    axes[i, j].spines['right'].set_color(self.colors['neutral'])
                else:
                    axes[i, j].spines['bottom'].set_color(self.colors['biased'])
                    axes[i, j].spines['top'].set_color(self.colors['biased'])
                    axes[i, j].spines['left'].set_color(self.colors['biased'])
                    axes[i, j].spines['right'].set_color(self.colors['biased'])
        plt.tight_layout()
        plt.suptitle('MicroBias Watermarker: Neutral vs Biased Image Comparison',
                    fontsize=16, y=1.02)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved demo grid to {save_path}")
        else:
            save_path = self.output_dir / "demo_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    def generate_report(
        self,
        results: Dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Generate a comprehensive analysis report.
        Args:
            results: Dictionary containing evaluation results
            output_dir: Optional output directory for report
        Returns:
            Path to generated report file
        """
        if output_dir is None:
            output_dir = self.output_dir
        # Create report HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MicroBias Watermarker Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
                .metric {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; flex: 1; min-width: 200px; }}
                .metric h3 {{ margin: 0 0 10px 0; color: #2c3e50; }}
                .section {{ margin: 30px 0; }}
                .image {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MicroBias Watermarker Analysis Report</h1>
                <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="metrics">
                    <div class="metric">
                        <h3>Accuracy</h3>
                        <p style="font-size: 2em; color: #27ae60;">{results.get('accuracy', 'N/A')}</p>
                    </div>
                    <div class="metric">
                        <h3>Precision</h3>
                        <p style="font-size: 2em; color: #3498db;">{results.get('precision', 'N/A')}</p>
                    </div>
                    <div class="metric">
                        <h3>Recall</h3>
                        <p style="font-size: 2em; color: #e74c3c;">{results.get('recall', 'N/A')}</p>
                    </div>
                    <div class="metric">
                        <h3>F1 Score</h3>
                        <p style="font-size: 2em; color: #f39c12;">{results.get('f1_score', 'N/A')}</p>
                    </div>
                </div>
            </div>
            <div class="section">
                <h2>Visualizations</h2>
                <h3>Confusion Matrix</h3>
                <img src="confusion_matrix.png" class="image" alt="Confusion Matrix">
                <h3>Detection Results</h3>
                <img src="detection_results.png" class="image" alt="Detection Results">
                <h3>Confidence Distribution</h3>
                <img src="confidence_distribution.png" class="image" alt="Confidence Distribution">
            </div>
            <div class="section">
                <h2>Detailed Analysis</h2>
                <p>Total samples processed: {results.get('total_samples', 'N/A')}</p>
                <p>Correct predictions: {results.get('correct_predictions', 'N/A')}</p>
                <p>Average confidence: {results.get('avg_confidence', 'N/A')}</p>
                <p>False positive rate: {results.get('false_positive_rate', 'N/A')}</p>
                <p>False negative rate: {results.get('false_negative_rate', 'N/A')}</p>
            </div>
        </body>
        </html>
        """
        # Save report
        import datetime
        report_path = Path(output_dir) / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        logger.info(f"Generated analysis report at {report_path}")
        return str(report_path)
def create_comparison_visualization(
    original_images: List[torch.Tensor],
    embedded_images: List[torch.Tensor],
    detected_bits: List[int],
    confidences: List[float],
    save_path: str = "outputs/comparison.png",
) -> None:
    """
    Create a comparison visualization showing original vs embedded images.
    Args:
        original_images: List of original image tensors
        embedded_images: List of embedded image tensors
        detected_bits: List of detected bias flags
        confidences: List of confidence scores
        save_path: Path to save visualization
    """
    num_images = min(len(original_images), 4)  # Limit to 4 pairs
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    for i in range(num_images):
        # Original image
        orig_img = original_images[i].permute(1, 2, 0).cpu().numpy()
        orig_img = np.clip(orig_img, 0, 1)
        # Embedded image
        emb_img = embedded_images[i].permute(1, 2, 0).cpu().numpy()
        emb_img = np.clip(emb_img, 0, 1)
        # Difference image
        diff_img = np.abs(orig_img - emb_img)
        diff_img = np.clip(diff_img * 10, 0, 1)  # Amplify differences for visibility
        # Display images
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(emb_img)
        axes[i, 1].set_title(f'Embedded (bias={detected_bits[i]})')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(diff_img, cmap='hot')
        axes[i, 2].set_title(f'Differences (conf={confidences[i]:.3f})')
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
if __name__ == "__main__":
    # Test visualization utilities
    print("Testing visualization utilities...")
    # Create sample data
    visualizer = BiasVisualizer()
    # Generate random test images
    test_images = [np.random.rand(256, 256, 3) for _ in range(8)]
    test_bits = [np.random.randint(0, 2) for _ in range(8)]
    test_confidences = [np.random.uniform(0.7, 1.0) for _ in range(8)]
    test_prompts = [f"Test prompt {i}" for i in range(8)]
    # Test detection visualization
    print("Testing detection visualization...")
    visualizer.visualize_detection_results(
        test_images, test_bits, test_confidences, prompts=test_prompts
    )
    # Test confusion matrix
    print("Testing confusion matrix...")
    y_true = [0, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0, 0, 1]
    visualizer.plot_confusion_matrix(y_true, y_pred)
    print("Visualization utilities test completed successfully!")