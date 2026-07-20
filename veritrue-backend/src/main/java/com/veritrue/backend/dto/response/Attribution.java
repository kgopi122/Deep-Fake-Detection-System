package com.veritrue.backend.dto.response;

import java.util.List;

public class Attribution {
    private String generator;
    private List<String> methodology;
    private List<String> verifiedBy;

    public Attribution() {
    }

    public Attribution(String generator, List<String> methodology, List<String> verifiedBy) {
        this.generator = generator;
        this.methodology = methodology;
        this.verifiedBy = verifiedBy;
    }

    // Getters and Setters
    public String getGenerator() {
        return generator;
    }

    public void setGenerator(String generator) {
        this.generator = generator;
    }

    public List<String> getMethodology() {
        return methodology;
    }

    public void setMethodology(List<String> methodology) {
        this.methodology = methodology;
    }

    public List<String> getVerifiedBy() {
        return verifiedBy;
    }

    public void setVerifiedBy(List<String> verifiedBy) {
        this.verifiedBy = verifiedBy;
    }

}
