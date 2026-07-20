package com.veritrue.backend.dto.response;

import java.util.List;

public class OriginAnalysis {
    private String sourceType;
    private List<String> flaggedBy;
    private String detectionLogic;

    public OriginAnalysis() {
    }

    public OriginAnalysis(String sourceType, List<String> flaggedBy, String detectionLogic) {
        this.sourceType = sourceType;
        this.flaggedBy = flaggedBy;
        this.detectionLogic = detectionLogic;
    }

    // Getters and Setters
    public String getSourceType() {
        return sourceType;
    }

    public void setSourceType(String sourceType) {
        this.sourceType = sourceType;
    }

    public List<String> getFlaggedBy() {
        return flaggedBy;
    }

    public void setFlaggedBy(List<String> flaggedBy) {
        this.flaggedBy = flaggedBy;
    }

    public String getDetectionLogic() {
        return detectionLogic;
    }

    public void setDetectionLogic(String detectionLogic) {
        this.detectionLogic = detectionLogic;
    }

}
