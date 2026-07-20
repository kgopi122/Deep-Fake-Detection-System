from __future__ import annotations

import cv2
import numpy as np
import scipy.fft
from scipy import ndimage
from scipy.stats import entropy
from skimage import feature, measure
import onnxruntime as ort
import os
from pathlib import Path

from .base import Evaluator
from ..types import EvaluatorResult, Signal


class ExcellentDetector(Evaluator):
    name = "Excellent AI Detection"

    def __init__(self, config):
        self._config = config
        model_dir = Path("models")
        self._models = {
            'general': self._load_onnx_model(model_dir / "model.onnx"),
            'face': self._load_onnx_model(model_dir / "face.onnx"),
            'artifacts': self._load_onnx_model(model_dir / "artifacts.onnx"),
            'noise': self._load_onnx_model(model_dir / "noise.onnx")
        }

    def evaluate(self, image_rgb: np.ndarray, image_path: str) -> EvaluatorResult:
        signals = []
        
        # NEW APPROACH: Score REAL characteristics (lower = more real)
        
        # 1. Natural Image Authenticity Score
        authenticity_score = self._calculate_authenticity_score(image_rgb)
        signals.append(Signal("authenticity", authenticity_score, {"method": "natural_features"}))
        
        # 2. Camera Sensor Fingerprint
        sensor_score = self._detect_sensor_fingerprint(image_rgb)
        signals.append(Signal("sensor_fingerprint", sensor_score, {"method": "prnu_analysis"}))
        
        # 3. Natural Lighting Analysis
        lighting_score = self._analyze_natural_lighting(image_rgb)
        signals.append(Signal("natural_lighting", lighting_score, {"method": "lighting_physics"}))
        
        # 4. Organic Texture Patterns
        texture_score = self._analyze_organic_textures(image_rgb)
        signals.append(Signal("organic_textures", texture_score, {"method": "texture_analysis"}))
        
        # 5. ONNX Models (only if confident)
        model_score = self._run_conservative_models(image_rgb)
        signals.append(Signal("model_consensus", model_score, {"method": "onnx_ensemble"}))
        
        # Calculate final score with new logic
        final_score = self._calculate_excellent_score(signals, image_rgb)
        
        return EvaluatorResult(
            evaluator=self.name,
            score_0_100=float(np.clip(final_score, 0, 100)),
            signals=signals
        )

    def _calculate_authenticity_score(self, image_rgb: np.ndarray) -> float:
        """Calculate how authentic/natural the image appears"""
        authenticity_indicators = []
        
        # 1. EXIF/Metadata presence (if available)
        # This would need to be passed from metadata evaluator
        
        # 2. Natural imperfections
        imperfections = self._detect_natural_imperfections(image_rgb)
        authenticity_indicators.append(imperfections)
        
        # 3. Realistic color distribution
        color_realism = self._analyze_color_realism(image_rgb)
        authenticity_indicators.append(color_realism)
        
        # 4. Natural edge characteristics
        edge_realism = self._analyze_edge_realism(image_rgb)
        authenticity_indicators.append(edge_realism)
        
        # Average authenticity (lower = more authentic)
        avg_authenticity = np.mean(authenticity_indicators)
        
        # Convert to AI probability (invert)
        return 100.0 - avg_authenticity

    def _detect_natural_imperfections(self, image_rgb: np.ndarray) -> float:
        """Detect natural imperfections that AI often lacks"""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Dust spots and sensor artifacts
        # Apply morphological operations to find small bright/dark spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        dust_spots = np.sum(tophat > 10) + np.sum(blackhat > 10)
        dust_score = min(100.0, dust_spots / 100.0)  # Normalize
        
        # 2. Film grain / sensor noise
        # High-frequency noise analysis
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        noise = np.abs(gray.astype(np.float32) - blur.astype(np.float32))
        noise_variance = np.var(noise)
        
        # Natural images have noise variance 1-20, AI images often < 0.5
        if noise_variance > 1.0:
            noise_score = 100.0  # Very natural
        elif noise_variance > 0.3:
            noise_score = 70.0   # Somewhat natural
        else:
            noise_score = 10.0   # Suspiciously clean
        
        # 3. Chromatic aberration
        r_channel = image_rgb[:, :, 0]
        b_channel = image_rgb[:, :, 2]
        
        # Find edges in both channels
        r_edges = cv2.Canny(r_channel, 50, 150)
        b_edges = cv2.Canny(b_channel, 50, 150)
        
        # Calculate misalignment (natural cameras have slight CA)
        correlation = cv2.matchTemplate(r_edges.astype(np.float32), b_edges.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        max_corr = np.max(correlation) if correlation.size > 0 else 1.0
        
        if max_corr < 0.95:  # Some misalignment = natural
            ca_score = 90.0
        elif max_corr < 0.98:
            ca_score = 60.0
        else:  # Perfect alignment = suspicious
            ca_score = 20.0
        
        return np.mean([dust_score, noise_score, ca_score])

    def _analyze_color_realism(self, image_rgb: np.ndarray) -> float:
        """Analyze if colors appear natural/realistic"""
        
        # 1. Color temperature consistency
        # Convert to LAB color space
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        a_channel = lab[:, :, 1].astype(np.float32) - 128  # Green-Red
        b_channel = lab[:, :, 2].astype(np.float32) - 128  # Blue-Yellow
        
        # Natural images have consistent color temperature
        a_std = np.std(a_channel)
        b_std = np.std(b_channel)
        
        # Natural range: 10-40, AI often too consistent (< 8) or too varied (> 50)
        if 10 <= a_std <= 40 and 10 <= b_std <= 40:
            temp_score = 95.0
        elif 8 <= a_std <= 50 and 8 <= b_std <= 50:
            temp_score = 70.0
        else:
            temp_score = 30.0
        
        # 2. Skin tone realism (if faces present)
        skin_score = self._analyze_skin_tones(image_rgb)
        
        # 3. Shadow color realism
        shadow_score = self._analyze_shadow_colors(image_rgb)
        
        return np.mean([temp_score, skin_score, shadow_score])

    def _analyze_skin_tones(self, image_rgb: np.ndarray) -> float:
        """Analyze skin tone realism"""
        try:
            # Simple skin detection using color ranges
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            
            # Skin color range in HSV
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            if np.sum(skin_mask) < 1000:  # No significant skin area
                return 80.0  # Neutral score
            
            # Extract skin pixels
            skin_pixels = image_rgb[skin_mask > 0]
            
            if len(skin_pixels) == 0:
                return 80.0
            
            # Analyze skin color distribution
            skin_mean = np.mean(skin_pixels, axis=0)
            skin_std = np.std(skin_pixels, axis=0)
            
            # Natural skin has specific R:G:B ratios and variation
            r, g, b = skin_mean
            
            # Natural skin: R > G > B, with R/G ratio 1.1-1.4
            if r > g > b and 1.1 <= r/g <= 1.4:
                ratio_score = 90.0
            else:
                ratio_score = 40.0
            
            # Natural skin has moderate variation (std 8-25)
            avg_std = np.mean(skin_std)
            if 8 <= avg_std <= 25:
                var_score = 90.0
            else:
                var_score = 50.0
            
            return np.mean([ratio_score, var_score])
            
        except:
            return 80.0  # Neutral if analysis fails

    def _analyze_shadow_colors(self, image_rgb: np.ndarray) -> float:
        """Analyze shadow color realism"""
        # Convert to LAB
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Find shadow areas (low luminance)
        shadow_mask = l_channel < np.percentile(l_channel, 25)
        
        if np.sum(shadow_mask) < 100:
            return 80.0  # No shadows to analyze
        
        # Extract shadow colors
        shadow_pixels = image_rgb[shadow_mask]
        
        # Natural shadows are slightly blue-shifted
        shadow_mean = np.mean(shadow_pixels, axis=0)
        r, g, b = shadow_mean
        
        # Natural shadows: B >= G >= R (blue shift)
        if b >= g >= r:
            return 90.0
        elif b > r:  # At least some blue shift
            return 70.0
        else:
            return 40.0  # Unnatural shadow colors

    def _analyze_edge_realism(self, image_rgb: np.ndarray) -> float:
        """Analyze edge characteristics for realism"""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Edge sharpness distribution
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge gradient magnitudes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Natural images have varied edge sharpness
        edge_pixels = gradient_magnitude[edges > 0]
        
        if len(edge_pixels) == 0:
            return 50.0
        
        edge_std = np.std(edge_pixels)
        edge_mean = np.mean(edge_pixels)
        
        # Natural edges: varied sharpness (high std), moderate mean
        if edge_std > 15 and 10 < edge_mean < 50:
            return 95.0
        elif edge_std > 8:
            return 70.0
        else:
            return 30.0  # Too uniform = suspicious

    def _detect_sensor_fingerprint(self, image_rgb: np.ndarray) -> float:
        """Detect camera sensor fingerprint (PRNU)"""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
        
        # Extract noise residual
        denoised = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float64)
        noise_residual = gray - denoised
        
        # Analyze noise characteristics
        noise_std = np.std(noise_residual)
        noise_mean = np.abs(np.mean(noise_residual))
        
        # Camera sensor noise characteristics
        if 1.5 <= noise_std <= 8.0 and noise_mean < 1.0:
            sensor_score = 5.0   # Strong camera signature
        elif 0.8 <= noise_std <= 12.0:
            sensor_score = 25.0  # Moderate camera signature
        else:
            sensor_score = 80.0  # No clear sensor signature
        
        return sensor_score

    def _analyze_natural_lighting(self, image_rgb: np.ndarray) -> float:
        """Analyze lighting for natural characteristics"""
        # Convert to LAB for better lighting analysis
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float64)
        
        # 1. Lighting gradient analysis
        grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Natural lighting has smooth gradients
        grad_std = np.std(gradient_magnitude)
        grad_mean = np.mean(gradient_magnitude)
        
        # 2. Dynamic range
        dynamic_range = np.max(l_channel) - np.min(l_channel)
        
        # Natural images: good dynamic range (> 100), smooth gradients
        if dynamic_range > 100 and 2 < grad_mean < 15 and grad_std < 20:
            return 10.0  # Very natural lighting
        elif dynamic_range > 50:
            return 30.0  # Moderate natural lighting
        else:
            return 70.0  # Suspicious lighting

    def _analyze_organic_textures(self, image_rgb: np.ndarray) -> float:
        """Analyze texture patterns for organic characteristics"""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Local Binary Pattern analysis
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Calculate LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        # Natural textures have specific LBP characteristics
        # High entropy = varied textures = natural
        lbp_entropy = entropy(hist + 1e-7)
        
        if lbp_entropy > 3.5:
            return 15.0  # Very natural textures
        elif lbp_entropy > 2.5:
            return 35.0  # Moderate natural textures
        else:
            return 75.0  # Suspicious uniformity

    def _run_conservative_models(self, image_rgb: np.ndarray) -> float:
        """Run ONNX models with conservative interpretation"""
        model_scores = []
        
        for model_name, model in self._models.items():
            if model is not None:
                try:
                    score = self._run_single_model(model, image_rgb)
                    # Only trust very confident predictions
                    if score > 80 or score < 20:
                        model_scores.append(score)
                except:
                    continue
        
        if len(model_scores) == 0:
            return 50.0  # Neutral if no confident predictions
        
        return np.mean(model_scores)

    def _run_single_model(self, model, image: np.ndarray) -> float:
        """Run single ONNX model"""
        processed = self._preprocess_for_onnx(image)
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: processed})
        
        output = outputs[0]
        if len(output.shape) > 1 and output.shape[-1] == 2:
            prob = float(output[0][1])
        else:
            prob = float(1.0 / (1.0 + np.exp(-output[0])))
        
        return prob * 100

    def _preprocess_for_onnx(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model"""
        resized = cv2.resize(image, (224, 224))
        normalized = resized.astype(np.float32) / 255.0
        
        if len(normalized.shape) == 3:
            chw = np.transpose(normalized, (2, 0, 1))
        else:
            chw = np.expand_dims(normalized, axis=0)
        
        return np.expand_dims(chw, axis=0)

    def _calculate_excellent_score(self, signals: list, image_rgb: np.ndarray) -> float:
        """Calculate final score with excellent logic"""
        
        # Extract individual scores
        authenticity = next((s.score_0_100 for s in signals if s.name == "authenticity"), 50.0)
        sensor = next((s.score_0_100 for s in signals if s.name == "sensor_fingerprint"), 50.0)
        lighting = next((s.score_0_100 for s in signals if s.name == "natural_lighting"), 50.0)
        texture = next((s.score_0_100 for s in signals if s.name == "organic_textures"), 50.0)
        models = next((s.score_0_100 for s in signals if s.name == "model_consensus"), 50.0)
        
        # REAL image indicators (lower scores = more real)
        real_indicators = [authenticity, sensor, lighting, texture]
        real_score = np.mean(real_indicators)
        
        # If multiple strong REAL indicators, heavily bias toward real
        strong_real_count = sum(1 for score in real_indicators if score < 30)
        if strong_real_count >= 3:
            return max(5.0, real_score * 0.5)  # Very likely real
        elif strong_real_count >= 2:
            return max(10.0, real_score * 0.7)  # Likely real
        
        # If models are very confident about AI, trust them
        if models > 85:
            return min(90.0, (real_score + models) / 2)
        
        # Default: weighted combination
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # authenticity, sensor, lighting, texture, models
        all_scores = [authenticity, sensor, lighting, texture, models]
        
        final_score = sum(score * weight for score, weight in zip(all_scores, weights))
        
        return np.clip(final_score, 0, 100)

    def _load_onnx_model(self, model_path):
        """Load ONNX model"""
        try:
            if os.path.exists(model_path):
                return ort.InferenceSession(str(model_path))
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
        return None