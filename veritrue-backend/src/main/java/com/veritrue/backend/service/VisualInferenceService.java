package com.veritrue.backend.service;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

@Service
public class VisualInferenceService {

    private static final Logger log = LoggerFactory.getLogger(VisualInferenceService.class);

    @Value("${veritrue.onnx.enabled:true}")
    private boolean enabled;

    @Value("${veritrue.onnx.modelPath:classpath:models/veritrue_int8.onnx}")
    private String modelPath;

    @Value("${veritrue.onnx.inputWidth:224}")
    private int inputWidth;

    @Value("${veritrue.onnx.inputHeight:224}")
    private int inputHeight;

    @Value("${veritrue.onnx.inputName:input}")
    private String inputName;

    @Value("${veritrue.onnx.outputName:}")
    private String outputName;

    @Value("${veritrue.onnx.fakeIndex:1}")
    private int fakeIndex;

    @Value("${veritrue.onnx.mean:0.485,0.456,0.406}")
    private String meanString;

    @Value("${veritrue.onnx.std:0.229,0.224,0.225}")
    private String stdString;

    private final ForensicEngine forensicEngine;
    private final PixelForensicsService pixelForensicsService;

    private OrtEnvironment env;
    private OrtSession session;
    private boolean initialized = false;

    private float[] mean = new float[]{0.485f, 0.456f, 0.406f};
    private float[] std = new float[]{0.229f, 0.224f, 0.225f};

    public VisualInferenceService(ForensicEngine forensicEngine, PixelForensicsService pixelForensicsService) {
        this.forensicEngine = forensicEngine;
        this.pixelForensicsService = pixelForensicsService;
    }

    @PostConstruct
    public synchronized void init() {
        parseNormalizationConfig();
        if (!enabled) {
            log.info("ONNX inference service is disabled in configuration.");
            return;
        }

        try {
            byte[] modelBytes = loadModelBytes(modelPath);
            if (modelBytes != null && modelBytes.length > 0) {
                env = OrtEnvironment.getEnvironment();
                OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
                session = env.createSession(modelBytes, opts);
                initialized = true;
                log.info("ONNX Runtime initialized successfully. Loaded model from: {}", modelPath);
            } else {
                log.warn("ONNX model file not found at {}. VisualInferenceService will use heuristic forensics fallback.", modelPath);
            }
        } catch (Exception e) {
            log.error("Failed to initialize ONNX session from {}: {}. Falling back to heuristic forensics.", modelPath, e.getMessage(), e);
            initialized = false;
        }
    }

    @PreDestroy
    public synchronized void close() {
        try {
            if (session != null) {
                session.close();
                session = null;
            }
            if (env != null) {
                env.close();
                env = null;
            }
        } catch (Exception e) {
            log.warn("Error closing ONNX Runtime session or environment: {}", e.getMessage());
        }
    }

    private void parseNormalizationConfig() {
        try {
            if (meanString != null && !meanString.isBlank()) {
                String[] parts = meanString.split(",");
                if (parts.length == 3) {
                    for (int i = 0; i < 3; i++) mean[i] = Float.parseFloat(parts[i].trim());
                }
            }
            if (stdString != null && !stdString.isBlank()) {
                String[] parts = stdString.split(",");
                if (parts.length == 3) {
                    for (int i = 0; i < 3; i++) std[i] = Float.parseFloat(parts[i].trim());
                }
            }
        } catch (Exception e) {
            log.warn("Error parsing ONNX mean/std configuration, using standard defaults (ImageNet)");
        }
    }

    private byte[] loadModelBytes(String path) {
        if (path == null || path.isBlank()) return null;
        try {
            if (path.startsWith("classpath:")) {
                String resourcePath = path.substring("classpath:".length());
                if (resourcePath.startsWith("/")) {
                    resourcePath = resourcePath.substring(1);
                }
                ClassPathResource resource = new ClassPathResource(resourcePath);
                if (resource.exists()) {
                    try (InputStream in = resource.getInputStream()) {
                        return in.readAllBytes();
                    }
                }
            } else {
                File f = new File(path);
                if (f.exists() && f.isFile()) {
                    try (InputStream in = new FileInputStream(f)) {
                        return in.readAllBytes();
                    }
                }
            }
        } catch (Exception e) {
            log.warn("Unable to load model bytes from {}: {}", path, e.getMessage());
        }
        return null;
    }

    public Map<String, Object> predict(byte[] fileBytes) {
        if (initialized && session != null && env != null) {
            try {
                return runOnnxInference(fileBytes);
            } catch (Exception e) {
                log.warn("ONNX inference failed: {}. Falling back to heuristic forensics.", e.getMessage(), e);
            }
        }

        return runHeuristicFallback(fileBytes);
    }

    private Map<String, Object> runOnnxInference(byte[] fileBytes) throws Exception {
        BufferedImage original = ImageIO.read(new ByteArrayInputStream(fileBytes));
        if (original == null) {
            throw new IllegalArgumentException("Failed to decode uploaded image for ONNX preprocessing.");
        }

        // Image preprocessing: scale uploaded image to 224x224
        BufferedImage resized = new BufferedImage(inputWidth, inputHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = resized.createGraphics();
        graphics.drawImage(original, 0, 0, inputWidth, inputHeight, null);
        graphics.dispose();

        // Normalize RGB channels: (channel / 255.0 - mean) / std
        float[][][][] inputData = new float[1][3][inputHeight][inputWidth];
        for (int y = 0; y < inputHeight; y++) {
            for (int x = 0; x < inputWidth; x++) {
                int rgb = resized.getRGB(x, y);
                float rNorm = (((rgb >> 16) & 0xFF) / 255.0f - mean[0]) / std[0];
                float gNorm = (((rgb >> 8) & 0xFF) / 255.0f - mean[1]) / std[1];
                float bNorm = ((rgb & 0xFF) / 255.0f - mean[2]) / std[2];

                inputData[0][0][y][x] = rNorm;
                inputData[0][1][y][x] = gNorm;
                inputData[0][2][y][x] = bNorm;
            }
        }

        String inName = (inputName != null && !inputName.isBlank()) ? inputName : "input";
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData)) {
            Map<String, OnnxTensor> inputs = Map.of(inName, inputTensor);
            try (OrtSession.Result results = session.run(inputs)) {
                OnnxValue outputValue = (outputName != null && !outputName.isBlank())
                        ? results.get(outputName).orElse(results.get(0))
                        : results.get(0);

                float[][] outputs = (float[][]) outputValue.getValue();
                float[] logits = outputs[0];

                double fakeProb;
                if (logits.length >= 2) {
                    double maxLogit = Math.max(logits[0], logits[1]);
                    double exp0 = Math.exp(logits[0] - maxLogit);
                    double exp1 = Math.exp(logits[1] - maxLogit);
                    double sumExp = exp0 + exp1;
                    fakeProb = (fakeIndex == 0 ? exp0 : exp1) / sumExp;
                } else if (logits.length == 1) {
                    fakeProb = 1.0 / (1.0 + Math.exp(-logits[0]));
                } else {
                    fakeProb = 0.5;
                }

                double confidence = Math.round(fakeProb * 10000.0) / 100.0;
                boolean isFake = confidence >= 50.0;

                Map<String, Object> res = new HashMap<>();
                res.put("visual_score", confidence);
                res.put("composite_score", confidence);
                res.put("vit_fake_score", confidence);
                res.put("is_fake", isFake);
                res.put("engine", "NATIVE_ONNX_INT8");
                res.put("detected_label_string", isFake ? "AI-Generated Image (ONNX)" : "Authentic Image (ONNX)");
                res.put("status", "Native ONNX Inference Complete");
                return res;
            }
        }
    }

    private Map<String, Object> runHeuristicFallback(byte[] fileBytes) {
        Map<String, Object> result = new HashMap<>();
        try {
            String signature = forensicEngine.detectSoftwareSignature(fileBytes);
            double metadataScore = 10.0;
            String status = "No AI signatures detected in metadata.";
            if (signature.startsWith("AI")) {
                metadataScore = 98.0;
                status = "Deepfake Signature Detected: " + signature;
            } else if (signature.startsWith("Edited")) {
                metadataScore = 50.0;
                status = "Editing Software Detected: " + signature;
            }

            double spectralScore = forensicEngine.performSpectralAnalysis(fileBytes);
            double noiseScore = forensicEngine.analyzeNoise(fileBytes);

            BufferedImage img = ImageIO.read(new ByteArrayInputStream(fileBytes));
            double pixelAiScore = 10.0;
            if (img != null) {
                PixelForensicsService.Result pixelResult = pixelForensicsService.analyze(img);
                pixelAiScore = pixelResult.pixelAiScore();
            }

            double compositeScore = (metadataScore * 0.20) + (spectralScore * 0.25) + (noiseScore * 0.25) + (pixelAiScore * 0.30);
            compositeScore = Math.round(Math.min(100.0, Math.max(0.0, compositeScore)) * 100.0) / 100.0;
            boolean isFake = compositeScore >= 55.0;

            result.put("visual_score", compositeScore);
            result.put("composite_score", compositeScore);
            result.put("vit_fake_score", compositeScore);
            result.put("srm_physics_score", spectralScore);
            result.put("is_fake", isFake);
            result.put("engine", "HEURISTIC_FORENSICS_FALLBACK");
            result.put("detected_label_string", isFake ? "Heuristic AI Indicator: " + status : "Authentic Image (Heuristic)");
            result.put("status", status);
            return result;
        } catch (Exception e) {
            log.error("Error in heuristic fallback analysis: {}", e.getMessage(), e);
            result.put("visual_score", 50.0);
            result.put("composite_score", 50.0);
            result.put("is_fake", false);
            result.put("engine", "HEURISTIC_FORENSICS_FALLBACK");
            result.put("error", e.getMessage());
            return result;
        }
    }
}
