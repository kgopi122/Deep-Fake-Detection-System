package com.veritrue.backend.controller;

import com.veritrue.backend.dto.request.NewsRequest;
import com.veritrue.backend.dto.response.AnalysisResponse;
import com.veritrue.backend.service.AnalysisOrchestrationService;
import com.veritrue.backend.service.NewsVerificationService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/v1/analysis")
public class AnalysisController {

    private final AnalysisOrchestrationService orchestrationService;
    private final NewsVerificationService newsService;

    public AnalysisController(AnalysisOrchestrationService orchestrationService, NewsVerificationService newsService) {
        this.orchestrationService = orchestrationService;
        this.newsService = newsService;
    }

    @PostMapping("/deepfake")
    public ResponseEntity<AnalysisResponse> analyzeDeepfake(@RequestParam("file") MultipartFile file) {
        try {
            AnalysisResponse response = orchestrationService.orchestrateAnalysis(file);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.internalServerError().build();
        }
    }

    @PostMapping("/news")
    public ResponseEntity<AnalysisResponse> verifyNews(@RequestBody NewsRequest request) {
        AnalysisResponse response = newsService.verify(request.getContent());
        return ResponseEntity.ok(response);
    }

}
