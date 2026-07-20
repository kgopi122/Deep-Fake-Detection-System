package com.veritrue.backend.service;

import com.drew.imaging.ImageMetadataReader;
import com.drew.metadata.Directory;
import com.drew.metadata.Metadata;
import com.drew.metadata.Tag;
import org.springframework.stereotype.Service;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

@Service
public class ImageMetadataService {

    public record Result(
            boolean parsed,
            String software,
            String make,
            String model,
            boolean edited,
            boolean aiTagFound,
            double metadataAiScore,
            String summary
    ) {
    }

    public Result analyze(byte[] fileBytes) {
        String software = null;
        String make = null;
        String model = null;
        boolean edited = false;
        boolean aiTagFound = false;
        List<String> hits = new ArrayList<>();

        try {
            Metadata metadata = ImageMetadataReader.readMetadata(new ByteArrayInputStream(fileBytes));
            for (Directory directory : metadata.getDirectories()) {
                for (Tag tag : directory.getTags()) {
                    String tagName = tag.getTagName();
                    String desc = tag.getDescription();
                    if (desc == null) continue;

                    String lower = desc.toLowerCase(Locale.ROOT);

                    if (software == null && (tagName.equalsIgnoreCase("Software") || tagName.equalsIgnoreCase("Creator Tool") || tagName.equalsIgnoreCase("Processing Software"))) {
                        software = desc;
                    }
                    if (make == null && tagName.equalsIgnoreCase("Make")) make = desc;
                    if (model == null && tagName.equalsIgnoreCase("Model")) model = desc;

                    // AI / generator fingerprints often stored in EXIF "Software" or XMP strings
                    if (containsAny(lower,
                            "stable diffusion",
                            "automatic1111",
                            "sd-webui",
                            "midjourney",
                            "dall-e",
                            "openai",
                            "comfyui",
                            "fooocus",
                            "novelai",
                            "invokeai",
                            "diffusion",
                            "generative",
                            "stylegan")) {
                        aiTagFound = true;
                        hits.add(desc);
                    }

                    // Editing fingerprints
                    if (containsAny(lower, "photoshop", "lightroom", "adobe", "gimp", "affinity", "snapseed")) {
                        edited = true;
                    }
                }
            }

            // As a fallback, scan the first chunk of bytes for text signatures.
            // (Some AI tools embed prompt/params as plain text in PNG chunks or XMP blocks.)
            int scanLength = Math.min(fileBytes.length, 128 * 1024);
            String content = new String(fileBytes, 0, scanLength, StandardCharsets.ISO_8859_1).toLowerCase(Locale.ROOT);
            if (!aiTagFound && containsAny(content,
                    "stable diffusion", "automatic1111", "midjourney", "dall-e", "openai", "comfyui", "fooocus", "novelai", "invokeai")) {
                aiTagFound = true;
                hits.add("Embedded generator signature in binary/XMP payload");
            }

            // Important: missing metadata is common (social media, messengers strip EXIF).
            // So the default should be neutral, not "real".
            double score = 50.0;
            if (aiTagFound) score = 98.0;
            else if (edited) score = 60.0;
            else if (make != null || model != null) score = 20.0; // camera EXIF present => evidence for real capture

            StringBuilder summary = new StringBuilder();
            if (make != null || model != null) {
                summary.append("Camera: ").append(make == null ? "Unknown" : make)
                        .append(" ").append(model == null ? "" : model).append(". ");
            }
            if (software != null) summary.append("Software: ").append(software).append(". ");
            if (aiTagFound) summary.append("AI generator signature detected.");
            else if (edited) summary.append("Editing software signature detected.");
            else summary.append("No AI/editing signatures found in metadata.");

            return new Result(true, software, make, model, edited, aiTagFound, score, summary.toString());
        } catch (Exception e) {
            // If parsing fails (e.g., video), return neutral metadata score.
            return new Result(false, null, null, null, false, false, 20.0,
                    "Metadata parse unavailable for this media type.");
        }
    }

    private static boolean containsAny(String haystackLower, String... needlesLower) {
        for (String n : needlesLower) {
            if (haystackLower.contains(n)) return true;
        }
        return false;
    }
}
