package com.veritrue.backend.dto.response;

import java.util.List;

public class AnalysisResponse {
    private String id;
    private String type;
    private String fileName;
    private double confidence;
    private boolean isFake;
    private Attribution attribution;
    private Details details;
    private List<Evidence> evidence;
    private Double credibility;
    private OriginAnalysis originAnalysis;

    // Getters and Setters
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }

    public boolean isFake() {
        return isFake;
    }

    public void setFake(boolean fake) {
        isFake = fake;
    }

    public Attribution getAttribution() {
        return attribution;
    }

    public void setAttribution(Attribution attribution) {
        this.attribution = attribution;
    }

    public Details getDetails() {
        return details;
    }

    public void setDetails(Details details) {
        this.details = details;
    }

    public List<Evidence> getEvidence() {
        return evidence;
    }

    public void setEvidence(List<Evidence> evidence) {
        this.evidence = evidence;
    }

    public Double getCredibility() {
        return credibility;
    }

    public void setCredibility(Double credibility) {
        this.credibility = credibility;
    }

    public OriginAnalysis getOriginAnalysis() {
        return originAnalysis;
    }

    public void setOriginAnalysis(OriginAnalysis originAnalysis) {
        this.originAnalysis = originAnalysis;
    }

}
