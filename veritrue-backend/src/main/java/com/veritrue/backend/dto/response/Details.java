package com.veritrue.backend.dto.response;

public class Details {
    private double compressionArtifacts;
    private double faceInconsistencies;
    private double lightingAnomalies;
    private Metadata metadata;

    // Getters and Setters
    public double getCompressionArtifacts() {
        return compressionArtifacts;
    }

    public void setCompressionArtifacts(double compressionArtifacts) {
        this.compressionArtifacts = compressionArtifacts;
    }

    public double getFaceInconsistencies() {
        return faceInconsistencies;
    }

    public void setFaceInconsistencies(double faceInconsistencies) {
        this.faceInconsistencies = faceInconsistencies;
    }

    public double getLightingAnomalies() {
        return lightingAnomalies;
    }

    public void setLightingAnomalies(double lightingAnomalies) {
        this.lightingAnomalies = lightingAnomalies;
    }

    public Metadata getMetadata() {
        return metadata;
    }

    public void setMetadata(Metadata metadata) {
        this.metadata = metadata;
    }

}
