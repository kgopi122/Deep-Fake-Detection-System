package com.veritrue.backend.service;

import org.springframework.http.MediaType;

import java.util.Locale;

public final class MediaTypeSniffer {

    public enum Kind {
        IMAGE,
        VIDEO,
        UNKNOWN
    }

    private MediaTypeSniffer() {
    }

    public static Kind sniff(byte[] bytes, String originalFilename, String contentType) {
        if (contentType != null) {
            try {
                MediaType mt = MediaType.parseMediaType(contentType);
                if (mt.getType().equalsIgnoreCase("image")) return Kind.IMAGE;
                if (mt.getType().equalsIgnoreCase("video")) return Kind.VIDEO;
            } catch (Exception ignored) {
            }
        }

        if (originalFilename != null) {
            String lower = originalFilename.toLowerCase(Locale.ROOT);
            if (lower.endsWith(".mp4") || lower.endsWith(".webm") || lower.endsWith(".mov") || lower.endsWith(".mkv")) {
                return Kind.VIDEO;
            }
            if (lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".png") || lower.endsWith(".gif") || lower.endsWith(".bmp") || lower.endsWith(".webp") || lower.endsWith(".tif") || lower.endsWith(".tiff")) {
                return Kind.IMAGE;
            }
        }

        // Magic bytes
        if (bytes != null && bytes.length >= 12) {
            // JPEG
            if ((bytes[0] & 0xFF) == 0xFF && (bytes[1] & 0xFF) == 0xD8) return Kind.IMAGE;
            // PNG
            if ((bytes[0] & 0xFF) == 0x89 && bytes[1] == 0x50 && bytes[2] == 0x4E && bytes[3] == 0x47) return Kind.IMAGE;
            // GIF
            if (bytes[0] == 0x47 && bytes[1] == 0x49 && bytes[2] == 0x46) return Kind.IMAGE;
            // WEBP: RIFF....WEBP
            if (bytes[0] == 0x52 && bytes[1] == 0x49 && bytes[2] == 0x46 && bytes[3] == 0x46
                    && bytes[8] == 0x57 && bytes[9] == 0x45 && bytes[10] == 0x42 && bytes[11] == 0x50) return Kind.IMAGE;
            // MP4: ....ftyp
            if (bytes[4] == 0x66 && bytes[5] == 0x74 && bytes[6] == 0x79 && bytes[7] == 0x70) return Kind.VIDEO;
        }

        return Kind.UNKNOWN;
    }
}
