from __future__ import annotations

import cv2
import numpy as np
import pywt
import onnxruntime as ort
import os
from pathlib import Path

from .base import Evaluator
from ..types import EvaluatorResult, Signal


class AdvancedEnsembleEvaluator(Evaluator):
    name = "Advanced Multi-Modal Ensemble"

    def __init__(self, config):
        self._config = config
        # Load existing ONNX models
        model_dir = Path("models")
        self._face_model = self._load_onnx_model(model_dir / "face.onnx")
        self._artifacts_model = self._load_onnx_model(model_dir / "artifacts.onnx") 
        self._noise_model = self._load_onnx_model(model_dir / "noise.onnx")
        self._general_model = self._load_onnx_model(model_dir / "model.onnx")

    def evaluate(self, image_rgb: np.ndarray, image_path: str) -> EvaluatorResult:
        signals = []
        
        # 1. General model analysis (model.onnx)
        general_score = self._run_general_model(image_rgb)
        signals.append(Signal("general_model", general_score, {"model": "model.onnx"}))
        
        # 2. Face-specific analysis (face.onnx)
        face_score = self._run_face_model(image_rgb)
        signals.append(Signal("face_analysis", face_score, {"model": "face.onnx"}))
        
        # 3. Artifacts detection (artifacts.onnx)
        artifacts_score = self._run_artifacts_model(image_rgb)
        signals.append(Signal("artifacts_analysis", artifacts_score, {"model": "artifacts.onnx"}))
        
        # 4. Noise analysis (noise.onnx)
        noise_score = self._run_noise_model(image_rgb)
        signals.append(Signal("noise_analysis", noise_score, {"model": "noise.onnx"}))
        
        # 5. Advanced frequency analysis (fallback)
        freq_score = self._advanced_frequency_analysis(image_rgb)
        signals.append(Signal("frequency_analysis", freq_score, {"method": "wavelet_dct"}))
        
        # 6. Geometric consistency (fallback)
        geometry_score = self._face_geometry_analysis(image_rgb)
        signals.append(Signal("geometry_consistency", geometry_score, {"method": "proportions"}))
        
        # Advanced ensemble scoring
        final_score = self._ensemble_decision(signals)
        
        return EvaluatorResult(
            evaluator=self.name,
            score_0_100=float(np.clip(final_score, 0, 100)),
            signals=signals
        )

    def _run_general_model(self, image_rgb: np.ndarray) -> float:
        """Run the general model.onnx for deepfake detection"""
        if self._general_model is None:
            return 50.0
        
        try:
            # Preprocess image (224x224, normalized)
            processed = self._preprocess_image(image_rgb, (224, 224))
            
            # Run inference
            inputs = {self._general_model.get_inputs()[0].name: processed}
            outputs = self._general_model.run(None, inputs)
            
            # Convert output to probability
            prob = self._extract_probability(outputs[0])
            return float(prob * 100)
            
        except Exception as e:
            return 50.0
    
    def _run_face_model(self, image_rgb: np.ndarray) -> float:
        """Run face.onnx for face-specific deepfake detection"""
        if self._face_model is None:
            return 50.0
            
        try:
            # Extract face region first
            face_region = self._extract_face_region(image_rgb)
            if face_region is None:
                return 50.0
                
            processed = self._preprocess_image(face_region, (224, 224))
            inputs = {self._face_model.get_inputs()[0].name: processed}
            outputs = self._face_model.run(None, inputs)
            
            prob = self._extract_probability(outputs[0])
            return float(prob * 100)
            
        except Exception:
            return 50.0
    
    def _run_artifacts_model(self, image_rgb: np.ndarray) -> float:
        """Run artifacts.onnx for artifact detection"""
        if self._artifacts_model is None:
            return 50.0
            
        try:
            processed = self._preprocess_image(image_rgb, (224, 224))
            inputs = {self._artifacts_model.get_inputs()[0].name: processed}
            outputs = self._artifacts_model.run(None, inputs)
            
            prob = self._extract_probability(outputs[0])
            return float(prob * 100)
            
        except Exception:
            return 50.0
    
    def _run_noise_model(self, image_rgb: np.ndarray) -> float:
        """Run noise.onnx for noise pattern analysis"""
        if self._noise_model is None:
            return self._fallback_noise_analysis(image_rgb)
            
        try:
            # Extract noise residual
            noise_residual = self._extract_noise_residual(image_rgb)
            processed = self._preprocess_image(noise_residual, (256, 256), single_channel=True)
            
            inputs = {self._noise_model.get_inputs()[0].name: processed}
            outputs = self._noise_model.run(None, inputs)
            
            prob = self._extract_probability(outputs[0])
            return float(prob * 100)
            
        except Exception:
            return self._fallback_noise_analysis(image_rgb)

    def _advanced_frequency_analysis(self, image_rgb: np.ndarray) -> float:
        """Multi-scale frequency domain analysis"""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Wavelet decomposition (4 levels)
        coeffs = pywt.wavedec2(gray, 'db4', levels=4)
        
        # DCT analysis on 8x8 blocks
        dct_score = self._dct_block_analysis(gray)
        
        # High-frequency artifact detection
        wavelet_score = self._wavelet_artifact_detection(coeffs)
        
        # Combine frequency domain scores
        freq_score = (dct_score * 0.6) + (wavelet_score * 0.4)
        
        return float(np.clip(freq_score, 0, 100))

    def _face_geometry_analysis(self, image_rgb: np.ndarray) -> float:
        """3D face geometry consistency check"""
        # Extract 68-point facial landmarks
        landmarks = self._extract_facial_landmarks(image_rgb)
        if landmarks is None:
            return 50.0
            
        # Check geometric ratios and proportions
        geometry_score = self._validate_face_proportions(landmarks)
        
        # 3D consistency check
        depth_consistency = self._check_depth_consistency(image_rgb, landmarks)
        
        return float((geometry_score * 0.7) + (depth_consistency * 0.3))

    def _advanced_noise_analysis(self, image_rgb: np.ndarray) -> float:
        """PRNU and sensor noise pattern analysis"""
        if self._noise_model is None:
            return self._fallback_noise_analysis(image_rgb)
            
        # Extract noise residual
        noise_residual = self._extract_noise_residual(image_rgb)
        
        # Run noise pattern ONNX model
        noise_features = self._preprocess_for_model(noise_residual, (256, 256))
        prediction = self._run_onnx_inference(self._noise_model, noise_features)
        
        return float(prediction[0] * 100)

    def _compression_analysis(self, image_rgb: np.ndarray) -> float:
        """Advanced JPEG compression artifact analysis"""
        # Convert to YUV for better compression analysis
        yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
        y_channel = yuv[:, :, 0]
        
        # DCT coefficient analysis
        dct_artifacts = self._detect_dct_artifacts(y_channel)
        
        # Quantization table estimation
        quantization_score = self._estimate_quantization_artifacts(y_channel)
        
        return float((dct_artifacts * 0.6) + (quantization_score * 0.4))

    def _biological_analysis(self, image_rgb: np.ndarray) -> float:
        """Biological plausibility analysis"""
        # Skin texture analysis
        skin_score = self._analyze_skin_texture(image_rgb)
        
        # Eye reflection consistency
        eye_score = self._analyze_eye_reflections(image_rgb)
        
        # Micro-expression detection
        expression_score = self._detect_micro_expressions(image_rgb)
        
        bio_score = (skin_score * 0.4) + (eye_score * 0.3) + (expression_score * 0.3)
        return float(np.clip(bio_score, 0, 100))

    def _ensemble_decision(self, signals: list) -> float:
        """Advanced ensemble scoring with model-based weighting"""
        # Model-based weights (prioritize ONNX models)
        weights = {
            "general_model": 0.30,      # Primary general detector
            "face_analysis": 0.25,      # Face-specific detector
            "artifacts_analysis": 0.20, # Artifacts detector
            "noise_analysis": 0.15,     # Noise detector
            "frequency_analysis": 0.07, # Frequency fallback
            "geometry_consistency": 0.03 # Geometry fallback
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for signal in signals:
            if signal.name in weights:
                weight = weights[signal.name]
                # Boost weight for confident predictions
                confidence = abs(signal.score_0_100 - 50.0) / 50.0
                adjusted_weight = weight * (0.5 + 0.5 * confidence)
                
                total_score += signal.score_0_100 * adjusted_weight
                total_weight += adjusted_weight
        
        if total_weight == 0:
            return 50.0
            
        final_score = total_score / total_weight
        
        # Apply consistency boost
        high_confidence_signals = sum(1 for s in signals if s.score_0_100 > 75 or s.score_0_100 < 25)
        if high_confidence_signals >= 3:
            if final_score > 50:
                final_score = min(95.0, final_score * 1.1)
            else:
                final_score = max(5.0, final_score * 0.9)
        
        return final_score

    def _calculate_signal_confidence(self, signal: Signal) -> float:
        """Calculate confidence in individual signal"""
        # Higher confidence for extreme scores
        deviation_from_neutral = abs(signal.score_0_100 - 50.0)
        confidence = min(1.0, deviation_from_neutral / 40.0)
        return max(0.1, confidence)  # Minimum 10% confidence

    def _apply_consistency_checks(self, signals: list, base_score: float) -> float:
        """Apply cross-signal consistency checks"""
        signal_scores = [s.score_0_100 for s in signals]
        
        # Check for conflicting signals
        score_variance = np.var(signal_scores)
        if score_variance > 1000:  # High disagreement
            # Move toward neutral when signals conflict
            base_score = (base_score * 0.7) + (50.0 * 0.3)
        
        # Boost confidence when multiple signals agree
        extreme_signals = sum(1 for score in signal_scores if score > 70 or score < 30)
        if extreme_signals >= 3:
            if base_score > 50:
                base_score = min(95.0, base_score * 1.1)
            else:
                base_score = max(5.0, base_score * 0.9)
        
        return base_score

    # Helper methods (simplified implementations)
    def _load_onnx_model(self, model_path):
        """Load ONNX model"""
        try:
            if os.path.exists(model_path):
                return ort.InferenceSession(str(model_path))
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
        return None

    def _extract_aligned_face(self, image_rgb: np.ndarray):
        """Extract and align face region"""
        # Placeholder - implement with dlib or MediaPipe
        return image_rgb  # Simplified

    def _preprocess_image(self, image: np.ndarray, target_size: tuple, single_channel: bool = False) -> np.ndarray:
        """Preprocess image for ONNX model input"""
        if len(image.shape) == 3 and not single_channel:
            # RGB image
            resized = cv2.resize(image, target_size)
            normalized = resized.astype(np.float32) / 255.0
            # Convert to CHW format
            chw = np.transpose(normalized, (2, 0, 1))
            return np.expand_dims(chw, axis=0)
        else:
            # Grayscale or single channel
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(image, target_size)
            normalized = resized.astype(np.float32) / 255.0
            return np.expand_dims(np.expand_dims(normalized, axis=0), axis=0)
    
    def _extract_probability(self, output) -> float:
        """Extract probability from model output"""
        if isinstance(output, np.ndarray):
            if output.shape[-1] == 1:
                # Single output (sigmoid)
                return float(1.0 / (1.0 + np.exp(-output[0])))
            elif output.shape[-1] == 2:
                # Binary classification (softmax)
                exp_vals = np.exp(output[0] - np.max(output[0]))
                softmax = exp_vals / np.sum(exp_vals)
                return float(softmax[1])  # Fake class probability
        return 0.5
    
    def _extract_face_region(self, image_rgb: np.ndarray):
        """Extract face region using simple detection"""
        # Simple face detection fallback
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Take first face
            return image_rgb[y:y+h, x:x+w]
        return image_rgb  # Return full image if no face detected

    def _dct_block_analysis(self, gray_image: np.ndarray) -> float:
        """DCT-based artifact detection"""
        h, w = gray_image.shape
        artifacts = 0
        total_blocks = 0
        
        for y in range(0, h-8, 8):
            for x in range(0, w-8, 8):
                block = gray_image[y:y+8, x:x+8].astype(np.float32)
                dct_block = cv2.dct(block)
                
                # Check for artificial patterns in DCT coefficients
                if self._has_artificial_dct_pattern(dct_block):
                    artifacts += 1
                total_blocks += 1
        
        if total_blocks == 0:
            return 50.0
        
        artifact_ratio = artifacts / total_blocks
        return min(100.0, artifact_ratio * 200.0)

    def _has_artificial_dct_pattern(self, dct_block: np.ndarray) -> bool:
        """Detect artificial patterns in DCT coefficients"""
        # Check for unnatural coefficient distributions
        high_freq_energy = np.sum(np.abs(dct_block[4:, 4:]))
        total_energy = np.sum(np.abs(dct_block))
        
        if total_energy == 0:
            return False
            
        hf_ratio = high_freq_energy / total_energy
        return hf_ratio < 0.05 or hf_ratio > 0.8  # Unnatural ratios

    def _wavelet_artifact_detection(self, coeffs) -> float:
        """Detect artifacts in wavelet domain"""
        # Analyze high-frequency subbands for artificial patterns
        score = 50.0
        
        for level in range(1, len(coeffs)):
            if len(coeffs[level]) == 3:  # (LH, HL, HH)
                lh, hl, hh = coeffs[level]
                
                # Check for unnatural sparsity
                sparsity = self._calculate_sparsity(hh)
                if sparsity > 0.95:  # Too sparse (AI-generated)
                    score += 15.0
                elif sparsity < 0.7:  # Too dense (natural)
                    score -= 10.0
        
        return float(np.clip(score, 0, 100))

    def _calculate_sparsity(self, coeffs: np.ndarray) -> float:
        """Calculate coefficient sparsity"""
        threshold = 0.01 * np.max(np.abs(coeffs))
        sparse_count = np.sum(np.abs(coeffs) < threshold)
        return sparse_count / coeffs.size

    def _extract_facial_landmarks(self, image_rgb: np.ndarray):
        """Extract 68-point facial landmarks"""
        # Placeholder - implement with dlib
        return None

    def _validate_face_proportions(self, landmarks) -> float:
        """Validate facial proportions"""
        # Placeholder for geometric validation
        return 50.0

    def _check_depth_consistency(self, image_rgb: np.ndarray, landmarks) -> float:
        """Check 3D depth consistency"""
        # Placeholder for depth analysis
        return 50.0

    def _extract_noise_residual(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract noise residual using advanced denoising"""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Use bilateral filter for better edge preservation
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        residual = gray.astype(np.float32) - denoised.astype(np.float32)
        
        return residual

    def _fallback_noise_analysis(self, image_rgb: np.ndarray) -> float:
        """Fallback noise analysis when model unavailable"""
        residual = self._extract_noise_residual(image_rgb)
        variance = np.var(residual)
        
        # AI images typically have lower noise variance
        if variance < 10.0:
            return 85.0
        elif variance < 25.0:
            return 65.0
        else:
            return 25.0

    def _detect_dct_artifacts(self, y_channel: np.ndarray) -> float:
        """Detect DCT compression artifacts"""
        # Simplified implementation
        return 50.0

    def _estimate_quantization_artifacts(self, y_channel: np.ndarray) -> float:
        """Estimate quantization artifacts"""
        # Simplified implementation
        return 50.0

    def _analyze_skin_texture(self, image_rgb: np.ndarray) -> float:
        """Analyze skin texture naturalness"""
        # Simplified implementation
        return 50.0

    def _analyze_eye_reflections(self, image_rgb: np.ndarray) -> float:
        """Analyze eye reflection consistency"""
        # Simplified implementation
        return 50.0

    def _detect_micro_expressions(self, image_rgb: np.ndarray) -> float:
        """Detect micro-expression authenticity"""
        # Simplified implementation
        return 50.0