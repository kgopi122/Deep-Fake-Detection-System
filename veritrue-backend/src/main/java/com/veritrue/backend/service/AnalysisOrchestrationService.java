package com.veritrue.backend.service;

import com.veritrue.backend.dto.response.AnalysisResponse;
import com.veritrue.backend.dto.response.Attribution;
import com.veritrue.backend.dto.response.Details;
import com.veritrue.backend.dto.response.Evidence;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
public class AnalysisOrchestrationService {

    private final VisualInferenceService visualInferenceService;

    public AnalysisOrchestrationService(VisualInferenceService visualInferenceService) {
        this.visualInferenceService = visualInferenceService;
    }

    public AnalysisResponse orchestrateAnalysis(MultipartFile file) throws IOException {
        byte[] fileBytes = file.getBytes();
        Map<String, Object> pred = visualInferenceService.predict(fileBytes);

        double compositeScore = pred.get("composite_score") != null
                ? ((Number) pred.get("composite_score")).doubleValue()
                : 0.0;
        boolean isFake = pred.get("is_fake") instanceof Boolean
                ? (Boolean) pred.get("is_fake")
                : (compositeScore >= 50.0);
        String labelStr = pred.get("detected_label_string") != null
                ? pred.get("detected_label_string").toString()
                : "Analysis Complete";
        String engineName = pred.get("engine") != null
                ? pred.get("engine").toString()
                : "Native Engine";

        AnalysisResponse response = new AnalysisResponse();
        response.setId(UUID.randomUUID().toString());
        response.setType("deepfake");
        response.setFileName(file.getOriginalFilename() != null ? file.getOriginalFilename() : "upload.jpg");
        response.setFake(isFake);
        response.setConfidence(compositeScore);

        Attribution attribution = new Attribution();
        attribution.setGenerator(labelStr);
        attribution.setVerifiedBy(List.of("VeriTrue Native In-Process Engine (" + engineName + ")"));
        response.setAttribution(attribution);

        Details details = new Details();
        details.setFaceInconsistencies(pred.get("vit_fake_score") != null ? ((Number) pred.get("vit_fake_score")).doubleValue() : compositeScore);
        details.setCompressionArtifacts(pred.get("srm_physics_score") != null ? ((Number) pred.get("srm_physics_score")).doubleValue() : 0.0);
        response.setDetails(details);

        List<Evidence> evidenceList = new ArrayList<>();
        Evidence exec = new Evidence();
        exec.setTitle("Executive Verdict");
        exec.setVerdict(isFake ? "DEEPFAKE" : "REAL");
        exec.setSnippet("Composite Score: " + compositeScore + "% (" + labelStr + ")");
        evidenceList.add(exec);
        response.setEvidence(evidenceList);

        return response;
    }
}
