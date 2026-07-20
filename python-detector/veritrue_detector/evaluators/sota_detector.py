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


class StateOfTheArtDetector(Evaluator):
    name = "State-of-the-Art AI Detection"

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
        
        # Pre-analysis: Detect image type and adjust thresholds
        image_context = self._analyze_image_context(image_rgb)
        
        # 1. GAN Fingerprint Detection (adjusted for context)
        gan_score = self._detect_gan_fingerprints(image_rgb, image_context)
        signals.append(Signal("gan_fingerprints", gan_score, {"method": "spectral_analysis", "context": image_context}))
        
        # 2. Pixel Co-occurrence Analysis (context-aware)
        cooccurrence_score = self._analyze_pixel_cooccurrence(image_rgb, image_context)
        signals.append(Signal("pixel_cooccurrence", cooccurrence_score, {"method": "glcm_analysis"}))
        
        # 3. Compression Inconsistency Detection
        compression_score = self._detect_compression_inconsistencies(image_rgb)
        signals.append(Signal("compression_inconsistency", compression_score, {"method": "jpeg_analysis"}))
        
        # 4. Neural Network Artifacts (skip for portraits with uniform backgrounds)
        if not image_context.get('uniform_background', False):
            nn_artifacts_score = self._detect_nn_artifacts(image_rgb)
        else:
            nn_artifacts_score = 25.0  # Assume natural for uniform backgrounds
        signals.append(Signal("nn_artifacts", nn_artifacts_score, {"method": "upsampling_detection"}))
        
        # 5. Statistical Anomaly Detection (adjusted)
        statistical_score = self._statistical_anomaly_detection(image_rgb, image_context)
        signals.append(Signal("statistical_anomalies", statistical_score, {"method": "benford_law"}))
        
        # 6. ONNX Model Ensemble (if available)
        model_score = self._run_model_ensemble(image_rgb)
        signals.append(Signal("model_ensemble", model_score, {"method": "onnx_models"}))
        
        # Context-aware scoring
        final_score = self._calculate_final_score(signals, image_context)
        
        return EvaluatorResult(
            evaluator=self.name,
            score_0_100=float(np.clip(final_score, 0, 100)),
            signals=signals
        )

    def _analyze_image_context(self, image_rgb: np.ndarray) -> dict:
        """Analyze image context to adjust detection thresholds"""
        h, w, _ = image_rgb.shape
        context = {}
        
        # Check for uniform background
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Sample edges to detect uniform background
        edge_pixels = np.concatenate([
            gray[0, :],      # top edge
            gray[-1, :],     # bottom edge
            gray[:, 0],      # left edge
            gray[:, -1]      # right edge
        ])
        
        edge_std = np.std(edge_pixels)
        edge_mean = np.mean(edge_pixels)
        
        # Uniform background detection
        context['uniform_background'] = edge_std < 10 and (edge_mean > 240 or edge_mean < 15)
        context['white_background'] = edge_std < 10 and edge_mean > 240
        
        # Natural camera indicators
        context['natural_noise'] = self._detect_natural_noise(gray)
        context['camera_artifacts'] = self._detect_camera_artifacts(image_rgb)
        context['natural_lighting'] = self._detect_natural_lighting(image_rgb)
        
        # Face detection for portrait classification
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            context['has_face'] = len(faces) > 0
            if len(faces) > 0:
                face_area = faces[0][2] * faces[0][3]  # width * height
                image_area = h * w
                context['face_ratio'] = face_area / image_area
                context['is_portrait'] = context['face_ratio'] > 0.1  # Face takes >10% of image
            else:
                context['face_ratio'] = 0.0
                context['is_portrait'] = False
        except:
            context['has_face'] = False
            context['is_portrait'] = False
            context['face_ratio'] = 0.0
        
        # Professional photo indicators
        context['likely_professional'] = (
            context.get('uniform_background', False) and 
            context.get('is_portrait', False)
        )
        
        # Natural camera image indicators
        natural_indicators = sum([
            context.get('natural_noise', False),
            context.get('camera_artifacts', False),
            context.get('natural_lighting', False)
        ])
        context['likely_camera'] = natural_indicators >= 2
        
        return context
    def _detect_natural_noise(self, gray: np.ndarray) -> bool:
        """Detect natural sensor noise patterns"""
        # Apply median filter to remove noise
        denoised = cv2.medianBlur(gray, 3)
        noise = gray.astype(np.float32) - denoised.astype(np.float32)
        
        # Natural noise has specific characteristics
        noise_std = np.std(noise)
        noise_mean = np.abs(np.mean(noise))
        
        # Camera sensors produce noise with std between 2-15
        # AI images often have very low noise (< 1) or artificial patterns
        return 2.0 < noise_std < 15.0 and noise_mean < 2.0
    
    def _detect_camera_artifacts(self, image_rgb: np.ndarray) -> bool:
        """Detect natural camera artifacts like chromatic aberration, vignetting"""
        h, w, _ = image_rgb.shape
        
        # Check for vignetting (natural darkening at edges)
        center_brightness = np.mean(image_rgb[h//3:2*h//3, w//3:2*w//3])
        corner_brightness = np.mean([
            np.mean(image_rgb[0:h//4, 0:w//4]),
            np.mean(image_rgb[0:h//4, 3*w//4:w]),
            np.mean(image_rgb[3*h//4:h, 0:w//4]),
            np.mean(image_rgb[3*h//4:h, 3*w//4:w])
        ])
        
        # Natural vignetting: center 5-20% brighter than corners
        vignetting_ratio = center_brightness / (corner_brightness + 1e-6)
        has_vignetting = 1.05 < vignetting_ratio < 1.25
        
        # Check for chromatic aberration (slight color fringing)
        r_channel = image_rgb[:, :, 0]
        g_channel = image_rgb[:, :, 1]
        b_channel = image_rgb[:, :, 2]
        
        # Calculate channel alignment (chromatic aberration causes slight misalignment)
        r_edges = cv2.Canny(r_channel, 50, 150)
        g_edges = cv2.Canny(g_channel, 50, 150)
        
        # Simple correlation check
        correlation = cv2.matchTemplate(r_edges.astype(np.float32), g_edges.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        max_corr = np.max(correlation)
        
        # Natural images have slight misalignment (correlation < 0.98)
        # AI images often have perfect alignment (correlation > 0.99)
        has_chromatic_aberration = max_corr < 0.98
        
        return has_vignetting or has_chromatic_aberration
    
    def _detect_natural_lighting(self, image_rgb: np.ndarray) -> bool:
        """Detect natural lighting patterns"""
        # Convert to LAB color space for better lighting analysis
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Natural lighting has gradual transitions
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(l_channel, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Natural images have moderate gradient variance
        # AI images often have either too smooth or too sharp transitions
        grad_std = np.std(gradient_magnitude)
        grad_mean = np.mean(gradient_magnitude)
        
        # Natural lighting characteristics
        return 5.0 < grad_std < 50.0 and 2.0 < grad_mean < 30.0
    def _detect_gan_fingerprints(self, image_rgb: np.ndarray, context: dict = None) -> float:
        """Detect GAN-specific artifacts in frequency domain (context-aware)"""
        if context is None:
            context = {}
        
        # Strong bias toward REAL for camera images
        if context.get('likely_camera', False):
            return 5.0  # Very low AI probability for camera images
            
        # Skip frequency analysis for professional portraits with uniform backgrounds
        if context.get('likely_professional', False):
            return 15.0  # Low AI probability for professional photos
        
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
        
        # 2D FFT with proper windowing
        windowed = gray * np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
        fft_result = scipy.fft.fft2(windowed)
        magnitude = np.abs(scipy.fft.fftshift(fft_result))
        
        # Log magnitude spectrum
        log_magnitude = np.log(magnitude + 1e-10)
        
        # Detect checkerboard artifacts (GAN upsampling signature)
        h, w = log_magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Check for periodic patterns at specific frequencies
        checkerboard_score = 0.0
        
        # Look for peaks at Nyquist/2 frequencies (common GAN artifact)
        quarter_freq_y = center_y + h // 4
        quarter_freq_x = center_x + w // 4
        
        # Sample regions around quarter frequencies
        region_size = 5
        if quarter_freq_y < h - region_size and quarter_freq_x < w - region_size:
            quarter_region = log_magnitude[quarter_freq_y-region_size:quarter_freq_y+region_size,
                                         quarter_freq_x-region_size:quarter_freq_x+region_size]
            quarter_peak = np.max(quarter_region)
            
            # Compare with surrounding area
            surround_mean = np.mean(log_magnitude[center_y-50:center_y+50, center_x-50:center_x+50])
            
            # More strict threshold for detecting artifacts
            if quarter_peak > surround_mean + 3.0:  # Increased from 2.0
                checkerboard_score += 40.0  # Increased penalty
        
        # Detect unnatural frequency roll-off (AI images often have abrupt cutoffs)
        radial_profile = self._get_radial_profile(log_magnitude)
        rolloff_score = self._analyze_frequency_rolloff(radial_profile)
        
        # Combine scores (higher = more likely AI)
        total_score = checkerboard_score + rolloff_score
        
        # Reduce score for portraits with faces
        if context.get('is_portrait', False):
            total_score *= 0.5  # 50% reduction for portraits
            
        return float(np.clip(total_score, 0, 100))

    def _analyze_pixel_cooccurrence(self, image_rgb: np.ndarray, context: dict = None) -> float:
        """Analyze pixel co-occurrence patterns (GLCM) with context awareness"""
        if context is None:
            context = {}
        
        # Strong bias toward REAL for camera images
        if context.get('likely_camera', False):
            return 10.0  # Very low AI probability for camera images
            
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # For professional portraits, focus on face region if available
        if context.get('is_portrait', False) and context.get('has_face', False):
            face_region = self._extract_face_region(image_rgb)
            if face_region is not None:
                gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        
        # Compute GLCM (Gray Level Co-occurrence Matrix)
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
        features = []
        
        for distance in distances:
            for angle in angles:
                glcm = feature.graycomatrix(gray, [distance], [angle], levels=256, symmetric=True, normed=True)
                for prop in properties:
                    features.append(feature.graycoprops(glcm, prop)[0, 0])
        
        # AI images often have unnatural texture uniformity
        homogeneity_mean = np.mean([f for i, f in enumerate(features) if i % 4 == 2])  # homogeneity values
        energy_mean = np.mean([f for i, f in enumerate(features) if i % 4 == 3])      # energy values
        
        # Much more strict thresholds - only flag extreme cases
        if context.get('is_portrait', False):
            # Portraits naturally have smoother skin, so very high thresholds
            if homogeneity_mean > 0.95 and energy_mean > 0.5:
                return 60.0  # Reduced from 75.0
            else:
                return 10.0  # Much lower baseline
        else:
            # Very strict thresholds for non-portraits
            if homogeneity_mean > 0.92 and energy_mean > 0.45:
                return 70.0  # Only extreme cases
            elif homogeneity_mean > 0.85 and energy_mean > 0.35:
                return 40.0  # Moderate cases
            else:
                return 15.0  # Lower baseline

    def _detect_compression_inconsistencies(self, image_rgb: np.ndarray) -> float:
        """Detect JPEG compression inconsistencies - INVERTED LOGIC"""
        # Convert to YUV for better compression analysis
        yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
        y_channel = yuv[:, :, 0].astype(np.float64)
        
        # Analyze 8x8 DCT blocks
        h, w = y_channel.shape
        block_variances = []
        
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block = y_channel[y:y+8, x:x+8]
                dct_block = cv2.dct(block)
                
                # Check AC coefficient distribution
                ac_coeffs = dct_block.flatten()[1:]  # Skip DC component
                block_variances.append(np.var(ac_coeffs))
        
        if len(block_variances) == 0:
            return 50.0
        
        # CORRECTED LOGIC: Natural images have MORE variance in compression
        variance_of_variances = np.var(block_variances)
        mean_variance = np.mean(block_variances)
        
        # High variance = natural compression = LOW AI score
        # Low variance = artificial uniformity = HIGH AI score
        if variance_of_variances > 1000 and mean_variance > 100:
            return 15.0  # Natural compression patterns
        elif variance_of_variances > 500:
            return 35.0  # Moderate natural patterns
        else:
            return 70.0  # Suspicious uniformity

    def _detect_nn_artifacts(self, image_rgb: np.ndarray) -> float:
        """Detect neural network upsampling artifacts"""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
        
        # Detect grid-like patterns from upsampling
        # Apply Laplacian to enhance edges
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Look for periodic patterns in the Laplacian
        fft_laplacian = scipy.fft.fft2(laplacian)
        magnitude = np.abs(scipy.fft.fftshift(fft_laplacian))
        
        # Check for peaks at regular intervals (upsampling artifacts)
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Sample at regular intervals from center
        artifact_score = 0.0
        for scale in [2, 4, 8]:  # Common upsampling factors
            if center_y + h//scale < h and center_x + w//scale < w:
                peak_val = magnitude[center_y + h//scale, center_x + w//scale]
                baseline = np.mean(magnitude[center_y-10:center_y+10, center_x-10:center_x+10])
                
                if peak_val > baseline * 3:  # Significant peak
                    artifact_score += 20.0
        
        return float(np.clip(artifact_score, 0, 100))

    def _statistical_anomaly_detection(self, image_rgb: np.ndarray, context: dict = None) -> float:
        """Detect statistical anomalies using Benford's Law with context awareness"""
        if context is None:
            context = {}
            
        # Skip Benford's law for uniform background images (not applicable)
        if context.get('uniform_background', False):
            return 20.0  # Low AI probability for uniform backgrounds
            
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Benford's Law test on pixel values
        pixel_values = gray.flatten()
        first_digits = []
        
        for val in pixel_values:
            if val > 0:
                first_digit = int(str(val)[0])
                first_digits.append(first_digit)
        
        if len(first_digits) == 0:
            return 50.0
        
        # Expected Benford distribution
        benford_expected = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
        
        # Observed distribution
        digit_counts = np.bincount(first_digits, minlength=10)[1:10]  # digits 1-9
        observed_freq = digit_counts / np.sum(digit_counts)
        
        # Chi-square test
        chi_square = np.sum((observed_freq - benford_expected)**2 / benford_expected)
        
        # Adjust thresholds for portraits
        if context.get('is_portrait', False):
            # Portraits may naturally deviate from Benford's law due to skin tones
            if chi_square > 0.15:  # Increased threshold
                return 60.0  # Reduced score
            elif chi_square > 0.08:
                return 40.0
            else:
                return 15.0
        else:
            # Original thresholds for non-portraits
            if chi_square > 0.1:
                return 75.0
            elif chi_square > 0.05:
                return 55.0
            else:
                return 25.0

    def _run_model_ensemble(self, image_rgb: np.ndarray) -> float:
        """Run ensemble of ONNX models with proper preprocessing"""
        scores = []
        
        for model_name, model in self._models.items():
            if model is not None:
                try:
                    if model_name == 'face':
                        # Extract face region for face model
                        face_region = self._extract_face_region(image_rgb)
                        if face_region is not None:
                            score = self._run_single_model(model, face_region)
                            scores.append(score)
                    else:
                        score = self._run_single_model(model, image_rgb)
                        scores.append(score)
                except Exception:
                    continue
        
        if len(scores) == 0:
            return 50.0
        
        # Weighted average of model predictions
        return float(np.mean(scores))

    def _run_single_model(self, model, image: np.ndarray) -> float:
        """Run single ONNX model with proper preprocessing"""
        # Preprocess image
        processed = self._preprocess_for_onnx(image)
        
        # Run inference
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: processed})
        
        # Extract probability
        output = outputs[0]
        if len(output.shape) > 1 and output.shape[-1] == 2:
            # Binary classification - return fake probability
            prob = float(output[0][1])
        else:
            # Single output - apply sigmoid
            prob = float(1.0 / (1.0 + np.exp(-output[0])))
        
        return prob * 100

    def _preprocess_for_onnx(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model"""
        # Resize to 224x224
        resized = cv2.resize(image, (224, 224))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to CHW format
        if len(normalized.shape) == 3:
            chw = np.transpose(normalized, (2, 0, 1))
        else:
            chw = np.expand_dims(normalized, axis=0)
        
        # Add batch dimension
        return np.expand_dims(chw, axis=0)

    def _calculate_final_score(self, signals: list, context: dict = None) -> float:
        """Calculate final score with context-aware weighting"""
        if context is None:
            context = {}
        
        # MAJOR bias toward REAL for camera images
        if context.get('likely_camera', False):
            return 8.0  # Almost certainly real for camera images
            
        # Adjust weights based on image context
        if context.get('likely_professional', False):
            # Professional portraits - rely more on models, less on forensics
            weights = {
                'gan_fingerprints': 0.10,      # Much reduced for professional photos
                'pixel_cooccurrence': 0.05,    # Much reduced for portraits
                'compression_inconsistency': 0.15,  # Still somewhat important
                'nn_artifacts': 0.05,          # Very low for uniform backgrounds
                'statistical_anomalies': 0.05, # Not applicable to uniform backgrounds
                'model_ensemble': 0.60         # Heavy reliance on trained models
            }
        else:
            # Regular images - more balanced but still conservative
            weights = {
                'gan_fingerprints': 0.25,      # Reduced from 0.35
                'pixel_cooccurrence': 0.15,    # Reduced from 0.20
                'compression_inconsistency': 0.20,  # Increased (corrected logic)
                'nn_artifacts': 0.15,          # Same
                'statistical_anomalies': 0.10, # Same
                'model_ensemble': 0.15         # Increased from 0.05
            }
        
        total_score = 0.0
        total_weight = 0.0
        
        for signal in signals:
            if signal.name in weights:
                weight = weights[signal.name]
                total_score += signal.score_0_100 * weight
                total_weight += weight
        
        if total_weight == 0:
            return 50.0
        
        final_score = total_score / total_weight
        
        # Context-based adjustments - MORE CONSERVATIVE
        if context.get('likely_professional', False):
            # Professional portraits are much less likely to be AI-generated
            final_score *= 0.4  # 60% reduction (was 40%)
        
        if context.get('is_portrait', False):
            # Any portrait gets reduction
            final_score *= 0.7  # 30% reduction
            
        # Apply logic corrections - MORE CONSERVATIVE
        high_ai_signals = sum(1 for s in signals if s.score_0_100 > 80)  # Increased threshold
        if high_ai_signals >= 4:  # Need more signals
            final_score = min(90.0, final_score * 1.1)  # Reduced boost
        
        low_ai_signals = sum(1 for s in signals if s.score_0_100 < 25)  # Increased threshold
        if low_ai_signals >= 2:  # Need fewer signals
            final_score = max(5.0, final_score * 0.6)  # Stronger reduction
        
        return final_score

    # Helper methods
    def _get_radial_profile(self, image: np.ndarray) -> np.ndarray:
        """Get radial frequency profile"""
        h, w = image.shape
        center_y, center_x = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)
        
        # Calculate mean value at each radius
        tbin = np.bincount(r.ravel(), image.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / (nr + 1e-10)
        
        return radialprofile

    def _analyze_frequency_rolloff(self, radial_profile: np.ndarray) -> float:
        """Analyze frequency rolloff characteristics"""
        if len(radial_profile) < 10:
            return 50.0
        
        # Look for unnatural sharp cutoffs
        diff = np.diff(radial_profile)
        sharp_drops = np.sum(diff < -0.5)  # Sharp negative transitions
        
        # AI images often have abrupt frequency cutoffs
        if sharp_drops > 3:
            return 60.0
        else:
            return 20.0

    def _extract_face_region(self, image_rgb: np.ndarray):
        """Extract face region using Haar cascade"""
        try:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                return image_rgb[y:y+h, x:x+w]
        except:
            pass
        return None

    def _load_onnx_model(self, model_path):
        """Load ONNX model"""
        try:
            if os.path.exists(model_path):
                return ort.InferenceSession(str(model_path))
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
        return None