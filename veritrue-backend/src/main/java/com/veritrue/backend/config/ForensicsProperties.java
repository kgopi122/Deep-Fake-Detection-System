package com.veritrue.backend.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "veritrue.forensics")
public class ForensicsProperties {

    /** 0..1: increases heuristic sensitivity when closer to 1 */
    private double sensitivity = 0.65;

    /** Verdict thresholds */
    private double suspiciousThreshold = 45.0;
    private double fakeThreshold = 58.0;

    /** Weights for non-ONNX mode */
    private double weightSignal = 0.40;
    private double weightPixels = 0.45;
    private double weightMetadata = 0.15;

    public double getSensitivity() {
        return sensitivity;
    }

    public void setSensitivity(double sensitivity) {
        this.sensitivity = sensitivity;
    }

    public double getSuspiciousThreshold() {
        return suspiciousThreshold;
    }

    public void setSuspiciousThreshold(double suspiciousThreshold) {
        this.suspiciousThreshold = suspiciousThreshold;
    }

    public double getFakeThreshold() {
        return fakeThreshold;
    }

    public void setFakeThreshold(double fakeThreshold) {
        this.fakeThreshold = fakeThreshold;
    }

    public double getWeightSignal() {
        return weightSignal;
    }

    public void setWeightSignal(double weightSignal) {
        this.weightSignal = weightSignal;
    }

    public double getWeightPixels() {
        return weightPixels;
    }

    public void setWeightPixels(double weightPixels) {
        this.weightPixels = weightPixels;
    }

    public double getWeightMetadata() {
        return weightMetadata;
    }

    public void setWeightMetadata(double weightMetadata) {
        this.weightMetadata = weightMetadata;
    }
}
