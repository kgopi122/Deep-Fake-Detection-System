package com.veritrue.backend.dto.response;

public class Metadata {
    private String source;
    private boolean edited;
    private String timestamp;
    private String dimensions;

    public Metadata() {
    }

    public Metadata(String source, boolean edited, String timestamp, String dimensions) {
        this.source = source;
        this.edited = edited;
        this.timestamp = timestamp;
        this.dimensions = dimensions;
    }

    // Getters and Setters
    public String getSource() {
        return source;
    }

    public void setSource(String source) {
        this.source = source;
    }

    public boolean isEdited() {
        return edited;
    }

    public void setEdited(boolean edited) {
        this.edited = edited;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }

    public String getDimensions() {
        return dimensions;
    }

    public void setDimensions(String dimensions) {
        this.dimensions = dimensions;
    }

}
