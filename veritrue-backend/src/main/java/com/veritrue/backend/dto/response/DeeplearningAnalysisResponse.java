package com.veritrue.backend.dto.response;

public class DeeplearningAnalysisResponse {
    private boolean isFake;
    private double confidenceScore;
    private long processingTimeMs;

    public DeeplearningAnalysisResponse() {
    }

    public DeeplearningAnalysisResponse(boolean isFake, double confidenceScore, long processingTimeMs) {
        this.isFake = isFake;
        this.confidenceScore = confidenceScore;
        this.processingTimeMs = processingTimeMs;
    }

    public boolean getIsFake() {
        return isFake;
    }

    public void setIsFake(boolean fake) {
        this.isFake = fake;
    }

    public double getConfidenceScore() {
        return confidenceScore;
    }

    public void setConfidenceScore(double confidenceScore) {
        this.confidenceScore = confidenceScore;
    }

    public long getProcessingTimeMs() {
        return processingTimeMs;
    }

    public void setProcessingTimeMs(long processingTimeMs) {
        this.processingTimeMs = processingTimeMs;
    }
}
