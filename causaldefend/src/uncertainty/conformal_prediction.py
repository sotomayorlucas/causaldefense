"""
Conformal Prediction for Uncertainty Quantification

Implements split conformal prediction and adaptive variants for
calibrated uncertainty estimates with finite-sample coverage guarantees.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import train_test_split


class SplitConformalPredictor:
    """
    Split Conformal Prediction for classification.
    
    Provides prediction sets with guaranteed coverage:
    P(y ∈ C(x)) ≥ 1 - α
    
    Implements equations (27)-(29) from the paper.
    """
    
    def __init__(
        self,
        model: nn.Module,
        significance_level: float = 0.05
    ) -> None:
        """
        Initialize split conformal predictor.
        
        Args:
            model: Trained prediction model (outputs probabilities)
            significance_level: Target miscoverage rate α
        """
        self.model = model
        self.alpha = significance_level
        self.q_hat: Optional[float] = None  # Calibrated quantile
        self.calibration_scores: Optional[np.ndarray] = None
    
    def calibrate(
        self,
        calibration_data: List[Tuple[torch.Tensor, int]]
    ) -> None:
        """
        Calibrate on held-out calibration set.
        
        Computes non-conformity scores:
        S_i = 1 - f_θ(x_i)[y_i]
        
        Then finds quantile q̂ such that coverage ≥ 1-α.
        
        Args:
            calibration_data: List of (input, label) tuples
        """
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for x, y in calibration_data:
                # Get predicted probabilities
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1)
                
                # Non-conformity score: 1 - P(true class)
                score = 1 - probs[0, y].item()
                scores.append(score)
        
        self.calibration_scores = np.array(scores)
        
        # Compute quantile
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(self.calibration_scores, level)
    
    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = False
    ) -> Union[List[int], Tuple[List[int], torch.Tensor]]:
        """
        Compute prediction set C(x).
        
        C(x) = {y : 1 - f_θ(x)[y] ≤ q̂}
        
        Args:
            x: Input features
            return_probs: If True, also return predicted probabilities
            
        Returns:
            Prediction set (list of class labels)
            Predicted probabilities (optional)
        """
        if self.q_hat is None:
            raise ValueError("Predictor not calibrated. Call calibrate() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            
            # Prediction set: classes with score ≤ q̂
            scores = 1 - probs[0]
            prediction_set = [
                i for i, score in enumerate(scores)
                if score <= self.q_hat
            ]
        
        if return_probs:
            return prediction_set, probs[0]
        else:
            return prediction_set
    
    def get_confidence_interval(
        self,
        x: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for probability of positive class.
        
        Equation (31): [max(0, p - ε), min(1, p + ε)]
        
        Args:
            x: Input features
            
        Returns:
            (lower_bound, upper_bound)
        """
        self.model.eval()
        
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            
            # For binary classification, use positive class probability
            if probs.size(-1) == 2:
                p = probs[0, 1].item()
            else:
                p = probs[0].max().item()
        
        # Epsilon based on calibration quantile
        epsilon = self.q_hat if self.q_hat is not None else 0.1
        
        lower = max(0.0, p - epsilon)
        upper = min(1.0, p + epsilon)
        
        return lower, upper
    
    def should_escalate(
        self,
        x: torch.Tensor,
        escalation_threshold: float = 0.8
    ) -> bool:
        """
        Determine if prediction should be escalated to human analyst.
        
        Escalate if prediction set is ambiguous (contains multiple classes).
        
        Args:
            x: Input features
            escalation_threshold: Confidence threshold for auto-response
            
        Returns:
            True if should escalate to analyst
        """
        prediction_set, probs = self.predict(x, return_probs=True)
        
        # Escalate if prediction set contains multiple classes
        if len(prediction_set) > 1:
            return True
        
        # Escalate if confidence is low
        max_prob = probs.max().item()
        if max_prob < escalation_threshold:
            return True
        
        return False


class AdaptiveConformalPredictor(SplitConformalPredictor):
    """
    Adaptive Conformal Prediction for handling concept drift.
    
    Updates calibration quantile using rolling window.
    Equation (30): q̂_t = Quantile({S_i : i ∈ [t-w, t]}, 1-α)
    """
    
    def __init__(
        self,
        model: nn.Module,
        significance_level: float = 0.05,
        window_size: int = 1000
    ) -> None:
        """
        Initialize adaptive conformal predictor.
        
        Args:
            model: Trained prediction model
            significance_level: Target miscoverage rate α
            window_size: Size of rolling window for calibration
        """
        super().__init__(model, significance_level)
        self.window_size = window_size
        self.score_buffer: List[float] = []
    
    def update(
        self,
        x: torch.Tensor,
        y: int
    ) -> None:
        """
        Update calibration with new sample.
        
        Maintains rolling window of recent scores.
        
        Args:
            x: Input features
            y: True label
        """
        self.model.eval()
        
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            
            # Compute non-conformity score
            score = 1 - probs[0, y].item()
        
        # Add to buffer
        self.score_buffer.append(score)
        
        # Maintain window size
        if len(self.score_buffer) > self.window_size:
            self.score_buffer.pop(0)
        
        # Update quantile
        if len(self.score_buffer) >= 10:  # Minimum samples for calibration
            n = len(self.score_buffer)
            level = np.ceil((n + 1) * (1 - self.alpha)) / n
            self.q_hat = np.quantile(self.score_buffer, level)
    
    def rolling_window_calibration(
        self,
        recent_data: List[Tuple[torch.Tensor, int]]
    ) -> None:
        """
        Recalibrate using recent data window.
        
        Args:
            recent_data: Recent (input, label) samples
        """
        # Clear buffer
        self.score_buffer = []
        
        # Add recent samples
        for x, y in recent_data[-self.window_size:]:
            self.update(x, y)


class UncertaintyQuantifier:
    """
    Complete uncertainty quantification system.
    
    Combines conformal prediction with confidence calibration
    for production deployment.
    """
    
    def __init__(
        self,
        model: nn.Module,
        significance_level: float = 0.05,
        window_size: int = 1000,
        use_adaptive: bool = True
    ) -> None:
        """
        Initialize uncertainty quantifier.
        
        Args:
            model: Trained prediction model
            significance_level: Target miscoverage rate α
            window_size: Rolling window size for adaptive calibration
            use_adaptive: Whether to use adaptive conformal prediction
        """
        if use_adaptive:
            self.predictor = AdaptiveConformalPredictor(
                model,
                significance_level,
                window_size
            )
        else:
            self.predictor = SplitConformalPredictor(
                model,
                significance_level
            )
        
        self.use_adaptive = use_adaptive
    
    def calibrate(
        self,
        calibration_data: List[Tuple[torch.Tensor, int]]
    ) -> None:
        """Calibrate predictor"""
        self.predictor.calibrate(calibration_data)
    
    def classify_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Dict[str, Union[int, float, List[int], Tuple[float, float]]]:
        """
        Classify with comprehensive uncertainty information.
        
        Args:
            x: Input features
            
        Returns:
            Dictionary containing:
            - prediction: Most likely class
            - confidence: Confidence score
            - prediction_set: Set of plausible classes
            - confidence_interval: Confidence interval
            - should_escalate: Whether to escalate to analyst
        """
        # Get prediction set and probabilities
        prediction_set, probs = self.predictor.predict(x, return_probs=True)
        
        # Most likely class
        prediction = probs.argmax().item()
        confidence = probs.max().item()
        
        # Confidence interval
        conf_interval = self.predictor.get_confidence_interval(x)
        
        # Escalation decision
        should_escalate = self.predictor.should_escalate(x)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'prediction_set': prediction_set,
            'confidence_interval': conf_interval,
            'should_escalate': should_escalate,
            'set_size': len(prediction_set)
        }
    
    def update(
        self,
        x: torch.Tensor,
        y: int
    ) -> None:
        """
        Update with new labeled sample (for adaptive predictor).
        
        Args:
            x: Input features
            y: True label
        """
        if self.use_adaptive:
            self.predictor.update(x, y)
    
    def evaluate_coverage(
        self,
        test_data: List[Tuple[torch.Tensor, int]]
    ) -> Dict[str, float]:
        """
        Evaluate empirical coverage on test data.
        
        Should satisfy: coverage ≥ 1 - α
        
        Args:
            test_data: Test (input, label) samples
            
        Returns:
            Dictionary with coverage metrics
        """
        correct = 0
        total = 0
        set_sizes = []
        
        for x, y in test_data:
            prediction_set = self.predictor.predict(x)
            
            # Check if true label in prediction set
            if y in prediction_set:
                correct += 1
            
            total += 1
            set_sizes.append(len(prediction_set))
        
        coverage = correct / total if total > 0 else 0.0
        avg_set_size = np.mean(set_sizes) if set_sizes else 0.0
        
        return {
            'coverage': coverage,
            'target_coverage': 1 - self.predictor.alpha,
            'avg_set_size': avg_set_size,
            'n_samples': total
        }
    
    def plot_calibration(
        self,
        test_data: List[Tuple[torch.Tensor, int]],
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate calibration plot data.
        
        Args:
            test_data: Test samples
            n_bins: Number of confidence bins
            
        Returns:
            bin_confidences: Average confidence per bin
            bin_accuracies: Average accuracy per bin
            ece: Expected Calibration Error
        """
        confidences = []
        predictions = []
        labels = []
        
        # Collect predictions
        for x, y in test_data:
            _, probs = self.predictor.predict(x, return_probs=True)
            
            max_prob = probs.max().item()
            pred = probs.argmax().item()
            
            confidences.append(max_prob)
            predictions.append(pred)
            labels.append(y)
        
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Compute binned statistics
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1
        
        bin_confidences = []
        bin_accuracies = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_conf = confidences[mask].mean()
                bin_acc = (predictions[mask] == labels[mask]).mean()
                
                bin_confidences.append(bin_conf)
                bin_accuracies.append(bin_acc)
            else:
                bin_confidences.append(0.0)
                bin_accuracies.append(0.0)
        
        bin_confidences = np.array(bin_confidences)
        bin_accuracies = np.array(bin_accuracies)
        
        # Expected Calibration Error (ECE)
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                weight = mask.sum() / len(confidences)
                ece += weight * abs(bin_accuracies[i] - bin_confidences[i])
        
        return bin_confidences, bin_accuracies, ece
