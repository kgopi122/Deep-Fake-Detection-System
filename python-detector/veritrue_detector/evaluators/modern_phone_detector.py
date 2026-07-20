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


class ModernPhoneDetector(Evaluator):
    name = "Modern Phone Camera Detector"

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
        
        # 1. Detect modern phone camera characteristics
        phone_score = self._detect_phone_camera(image_rgb)
        signals.append(Signal("phone_camera", phone_score, {"method": "computational_photography"}))
        
        # 2. HDR processing detection
        hdr_score = self._detect_hdr_processing(image_rgb)
        signals.append(Signal("hdr_processing", hdr_score, {"method": "tone_mapping"}))
        
        # 3. AI enhancement detection (phone AI vs generative AI)
        enhancement_score = self._detect_phone_ai_enhancement(image_rgb)
        signals.append(Signal("phone_ai_enhancement", enhancement_score, {"method": "enhancement_analysis"}))
        
        # 4. Generative AI artifacts (the real target)
        generative_score = self._detect_generative_ai(image_rgb)
        signals.append(Signal("generative_ai", generative_score, {"method": "gan_detection"}))
        
        # 5. ONNX models with phone-aware interpretation
        model_score = self._run_phone_aware_models(image_rgb)
        signals.append(Signal("model_consensus", model_score, {"method": "onnx_ensemble"}))
        
        # Calculate final score with phone-aware logic
        final_score = self._calculate_phone_aware_score(signals, image_rgb)
        
        return EvaluatorResult(
            evaluator=self.name,
            score_0_100=float(np.clip(final_score, 0, 100)),
            signals=signals
        )

    def _detect_phone_camera(self, image_rgb: np.ndarray) -> float:
        """Detect modern phone camera characteristics"""
        phone_indicators = []
        
        # 1. Aspect ratio (phones often use 4:3, 16:9, or other mobile ratios)
        h, w, _ = image_rgb.shape
        aspect_ratio = w / h
        mobile_ratios = [4/3, 3/4, 16/9, 9/16, 1.0, 3/2, 2/3]
        
        is_mobile_ratio = any(abs(aspect_ratio - ratio) < 0.1 for ratio in mobile_ratios)
        phone_indicators.append(90.0 if is_mobile_ratio else 30.0)
        
        # 2. Resolution patterns (common phone resolutions)
        total_pixels = h * w
        common_phone_pixels = [
            12000000,  # 12MP (4000x3000)
            8000000,   # 8MP (3264x2448)
            5000000,   # 5MP (2592x1944)
            2000000,   # 2MP (1600x1200)
        ]
        
        is_phone_resolution = any(abs(total_pixels - pixels) < 1000000 for pixels in common_phone_pixels)
        phone_indicators.append(80.0 if is_phone_resolution else 40.0)
        
        # 3. Edge enhancement (phones heavily sharpen)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Phone cameras: high edge enhancement (laplacian_var > 500)
        if laplacian_var > 800:
            phone_indicators.append(85.0)  # Strong phone signature
        elif laplacian_var > 400:
            phone_indicators.append(65.0)  # Moderate phone signature
        else:
            phone_indicators.append(25.0)  # Low enhancement
        
        # 4. Color saturation (phones boost saturation)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation)
        
        # Phone cameras: high saturation (> 100)
        if avg_saturation > 120:
            phone_indicators.append(90.0)  # Very phone-like
        elif avg_saturation > 80:
            phone_indicators.append(70.0)  # Somewhat phone-like
        else:
            phone_indicators.append(30.0)  # Low saturation
        
        return np.mean(phone_indicators)

    def _detect_hdr_processing(self, image_rgb: np.ndarray) -> float:
        """Detect HDR tone mapping characteristics"""
        # Convert to LAB for luminance analysis
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float64)
        
        # 1. Tone mapping curve analysis
        # HDR creates characteristic S-curve in luminance histogram
        hist, bins = np.histogram(l_channel, bins=256, range=(0, 255))
        hist = hist.astype(float) / np.sum(hist)
        
        # Look for HDR signature: filled shadows and highlights
        shadow_fill = np.sum(hist[0:64])    # Shadows (0-25%)
        highlight_fill = np.sum(hist[192:256])  # Highlights (75-100%)
        
        # HDR typically has more filled shadows and compressed highlights
        if shadow_fill > 0.15 and highlight_fill < 0.05:
            hdr_score = 85.0  # Strong HDR signature
        elif shadow_fill > 0.10:
            hdr_score = 65.0  # Moderate HDR
        else:
            hdr_score = 25.0  # No HDR
        
        # 2. Local contrast analysis
        # HDR reduces local contrast while maintaining global contrast
        kernel = np.ones((5,5), np.float32) / 25
        local_mean = cv2.filter2D(l_channel, -1, kernel)
        local_contrast = np.std(l_channel - local_mean)
        global_contrast = np.std(l_channel)
        
        if global_contrast > 0:
            contrast_ratio = local_contrast / global_contrast
            # HDR: low local contrast relative to global
            if contrast_ratio < 0.3:
                contrast_score = 80.0  # HDR-like
            elif contrast_ratio < 0.5:
                contrast_score = 60.0  # Moderate processing
            else:
                contrast_score = 30.0  # Natural contrast
        else:
            contrast_score = 50.0
        
        return (hdr_score + contrast_score) / 2

    def _detect_phone_ai_enhancement(self, image_rgb: np.ndarray) -> float:
        """Detect phone AI enhancement (different from generative AI)"""
        # Phone AI enhancement characteristics:
        # 1. Noise reduction while preserving details
        # 2. Selective sharpening
        # 3. Color enhancement
        # 4. BUT maintains photographic realism
        
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Noise reduction analysis
        # Apply slight blur and check difference
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)
        noise_diff = np.mean(np.abs(gray.astype(np.float32) - blurred.astype(np.float32)))
        
        # Phone AI: very clean (low noise) but not artificially smooth
        if 0.5 < noise_diff < 2.0:
            noise_score = 80.0  # Phone AI signature
        elif noise_diff < 0.3:
            noise_score = 40.0  # Too smooth (suspicious)
        else:
            noise_score = 60.0  # Natural noise levels
        
        # 2. Detail preservation check
        # High-frequency content analysis
        high_freq = cv2.filter2D(gray, cv2.CV_64F, np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]))
        detail_strength = np.std(high_freq)
        
        # Phone AI: preserves details while reducing noise
        if detail_strength > 15:
            detail_score = 85.0  # Good detail preservation
        elif detail_strength > 8:
            detail_score = 65.0  # Moderate details
        else:
            detail_score = 30.0  # Loss of details
        
        return (noise_score + detail_score) / 2

    def _detect_generative_ai(self, image_rgb: np.ndarray) -> float:
        """Detect actual generative AI (GANs, diffusion models)"""
        # Focus on artifacts that generative AI creates but phone processing doesn't
        
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
        
        # 1. Frequency domain analysis for GAN artifacts
        windowed = gray * np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
        fft_result = scipy.fft.fft2(windowed)
        magnitude = np.abs(scipy.fft.fftshift(fft_result))
        log_magnitude = np.log(magnitude + 1e-10)
        
        # Look for checkerboard patterns (GAN upsampling)
        h, w = log_magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Check quarter frequencies for GAN artifacts
        artifact_score = 0.0
        if h > 100 and w > 100:  # Only for reasonable sizes
            quarter_y = center_y + h // 4
            quarter_x = center_x + w // 4
            
            if quarter_y < h - 5 and quarter_x < w - 5:
                quarter_region = log_magnitude[quarter_y-5:quarter_y+5, quarter_x-5:quarter_x+5]
                quarter_peak = np.max(quarter_region)
                surround_mean = np.mean(log_magnitude[center_y-25:center_y+25, center_x-25:center_x+25])
                
                if quarter_peak > surround_mean + 3.0:  # Strong artifact
                    artifact_score = 70.0
                elif quarter_peak > surround_mean + 2.0:  # Moderate artifact
                    artifact_score = 45.0
        
        # 2. Unnatural smoothness (different from phone processing)
        # Generative AI creates unnaturally smooth regions
        # Phone processing maintains texture while reducing noise
        
        # Calculate local variance in small patches
        patch_size = 8
        variances = []
        for y in range(0, gray.shape[0] - patch_size, patch_size):
            for x in range(0, gray.shape[1] - patch_size, patch_size):
                patch = gray[y:y+patch_size, x:x+patch_size]
                variances.append(np.var(patch))
        
        if len(variances) > 0:
            var_of_vars = np.var(variances)
            # Generative AI: very low variance of variances (too uniform)
            # Phone processing: maintains natural variation
            if var_of_vars < 50:
                smoothness_score = 60.0  # Suspicious uniformity
            elif var_of_vars < 200:
                smoothness_score = 35.0  # Moderate uniformity
            else:
                smoothness_score = 15.0  # Natural variation
        else:
            smoothness_score = 25.0
        
        return max(artifact_score, smoothness_score)

    def _run_phone_aware_models(self, image_rgb: np.ndarray) -> float:
        """Run ONNX models with phone-aware interpretation"""
        model_scores = []
        
        for model_name, model in self._models.items():
            if model is not None:
                try:
                    score = self._run_single_model(model, image_rgb)
                    # Adjust model scores for phone processing
                    # Models might flag HDR/enhanced photos as AI
                    if 40 <= score <= 70:  # Uncertain range - likely phone processing
                        score = score * 0.6  # Reduce suspicion
                    model_scores.append(score)
                except:
                    continue
        
        if len(model_scores) == 0:
            return 25.0  # Default to likely real
        
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

    def _calculate_phone_aware_score(self, signals: list, image_rgb: np.ndarray) -> float:
        """Calculate final score with phone-aware logic"""
        
        # Extract scores
        phone_camera = next((s.score_0_100 for s in signals if s.name == "phone_camera"), 50.0)
        hdr_processing = next((s.score_0_100 for s in signals if s.name == "hdr_processing"), 50.0)
        phone_ai = next((s.score_0_100 for s in signals if s.name == "phone_ai_enhancement"), 50.0)
        generative_ai = next((s.score_0_100 for s in signals if s.name == "generative_ai"), 50.0)
        models = next((s.score_0_100 for s in signals if s.name == "model_consensus"), 50.0)
        
        # Phone detection logic
        phone_indicators = [phone_camera, hdr_processing, phone_ai]
        avg_phone_score = np.mean(phone_indicators)
        
        # If strong phone indicators, heavily bias toward REAL
        strong_phone_indicators = sum(1 for score in phone_indicators if score > 70)
        
        if strong_phone_indicators >= 2:
            # Definitely a phone camera - return very low AI score
            base_score = max(5.0, (100 - avg_phone_score) * 0.3)
            
            # Only increase if generative AI score is very high
            if generative_ai > 80:
                return min(base_score + 20, 40.0)
            else:
                return base_score
                
        elif strong_phone_indicators >= 1:
            # Likely phone camera - bias toward real
            base_score = max(10.0, (100 - avg_phone_score) * 0.5)
            
            # Moderate adjustment for generative AI
            if generative_ai > 70:
                return min(base_score + 25, 50.0)
            else:
                return base_score
        
        else:
            # Not clearly a phone - use generative AI detection
            if generative_ai > 60:
                return min(85.0, generative_ai + 10)
            elif models > 75:
                return min(80.0, models)
            else:
                return max(15.0, (generative_ai + models) / 2)

    def _load_onnx_model(self, model_path):
        """Load ONNX model"""
        try:
            if os.path.exists(model_path):
                return ort.InferenceSession(str(model_path))
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
        return None