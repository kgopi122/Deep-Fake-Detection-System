package com.veritrue.backend.service;

import com.veritrue.backend.dto.response.AnalysisResponse;
import com.veritrue.backend.dto.response.Attribution;
import com.veritrue.backend.dto.response.Details;
import com.veritrue.backend.dto.response.Evidence;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Service
public class AnalysisOrchestrationService {

    private final RestTemplate restTemplate;
    
    @Value("${fastapi.service.url}/api/analyze/pixel")
    private String pythonMicroserviceUrl;


    public AnalysisOrchestrationService() {
        this.restTemplate = new RestTemplate();
    }

    // Using a record for mapping the Python JSON response
    private record PythonAnalysisResponse(
            Boolean is_fake,
            Double composite_score,
            Double vit_fake_score,
            Double srm_physics_score,
            String detected_label_string
    ) {}

    public AnalysisResponse orchestrateAnalysis(MultipartFile file) throws IOException {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new ByteArrayResource(file.getBytes()) {
            @Override
            public String getFilename() {
                return file.getOriginalFilename() != null ? file.getOriginalFilename() : "upload.jpg";
            }
        });

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        ResponseEntity<PythonAnalysisResponse> responseEntity = restTemplate.postForEntity(
                pythonMicroserviceUrl, requestEntity, PythonAnalysisResponse.class);

        PythonAnalysisResponse pyResponse = responseEntity.getBody();

        if (pyResponse == null) {
            throw new RuntimeException("Received empty response from Python microservice");
        }

        AnalysisResponse response = new AnalysisResponse();
        response.setId(UUID.randomUUID().toString());
        response.setType("deepfake");
        response.setFileName(file.getOriginalFilename());
        
        response.setFake(pyResponse.is_fake() != null && pyResponse.is_fake());
        response.setConfidence(pyResponse.composite_score() != null ? pyResponse.composite_score() : 0.0);

        Attribution attribution = new Attribution();
        attribution.setGenerator(pyResponse.detected_label_string());
        attribution.setVerifiedBy(List.of("Python FastAPI Microservice"));
        response.setAttribution(attribution);

        Details details = new Details();
        details.setFaceInconsistencies(pyResponse.vit_fake_score() != null ? pyResponse.vit_fake_score() : 0.0);
        details.setCompressionArtifacts(pyResponse.srm_physics_score() != null ? pyResponse.srm_physics_score() : 0.0);
        response.setDetails(details);

        List<Evidence> evidenceList = new ArrayList<>();
        Evidence exec = new Evidence();
        exec.setTitle("Executive Verdict");
        exec.setVerdict(response.isFake() ? "DEEPFAKE" : "REAL");
        exec.setSnippet("Composite Score: " + response.getConfidence() + "% (" + pyResponse.detected_label_string() + ")");
        evidenceList.add(exec);
        response.setEvidence(evidenceList);

        return response;
    }
}
