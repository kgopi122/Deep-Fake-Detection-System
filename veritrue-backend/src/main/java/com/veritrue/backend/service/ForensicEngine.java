package com.veritrue.backend.service;

import org.springframework.stereotype.Service;
import javax.imageio.ImageIO;
import java.io.ByteArrayInputStream;
import java.awt.image.BufferedImage;
import java.nio.charset.StandardCharsets;

@Service
public class ForensicEngine {

    /**
     * 1. Metadata Analysis
     */
    public String detectSoftwareSignature(byte[] fileBytes) {
        try {
            com.drew.metadata.Metadata metadata = com.drew.imaging.ImageMetadataReader.readMetadata(new ByteArrayInputStream(fileBytes));
            for (com.drew.metadata.Directory directory : metadata.getDirectories()) {
                for (com.drew.metadata.Tag tag : directory.getTags()) {
                    String value = tag.getDescription();
                    if (value == null) continue;
                    value = value.toLowerCase();

                    if (value.contains("c2pa")) return "AI: C2PA Provenance";
                    if (value.contains("google deepmind")) return "AI: Google DeepMind";
                    if (value.contains("openai") || value.contains("dall-e")) return "AI: DALL-E";
                    if (value.contains("midjourney") || value.contains("mj_")) return "AI: Midjourney";
                    if (value.contains("stable diffusion") || value.contains("sd-webui")) return "AI: Stable Diffusion";
                    if (value.contains("photoshop") || value.contains("adobe")) return "Edited: Adobe Photoshop";
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "Unknown";
    }

    /**
     * 2. Frequency Domain Analysis (FFT)
     * Detects "Checkerboard Artifacts" caused by GAN upsampling.
     * Uses JTransforms for 2D FFT.
     */
    public double performSpectralAnalysis(byte[] fileBytes) {
        try {
            BufferedImage img = ImageIO.read(new ByteArrayInputStream(fileBytes));
            if (img == null)
                return 50.0;

            // Convert to Grayscale & Resize to power of 2 for optimal FFT (e.g., 512x512)
            // Python script uses full size, but for Java speed/memory we constrain it.
            int N = 512;
            double[][] grayData = getGrayScaleData(img, N, N);

            // JTransforms expects a 1D array of size 2*rows*cols for complex data
            // or 2D array [rows][2*cols]. We use DoubleFFT_2D.
            // Using logic: complexForward(double[][] a)
            org.jtransforms.fft.DoubleFFT_2D fft = new org.jtransforms.fft.DoubleFFT_2D(N, N);

            // Prepare complex input: [row][2*col] -> Real, Imag, Real, Imag...
            double[][] complexData = new double[N][2 * N];
            for (int r = 0; r < N; r++) {
                for (int c = 0; c < N; c++) {
                    complexData[r][2 * c] = grayData[r][c]; // Real part
                    complexData[r][2 * c + 1] = 0.0; // Imaginary part
                }
            }

            fft.complexForward(complexData);

            // Calculate Magnitude Spectrum & Log Scale
            // Also shift low frequencies to center (FFT shift logic is implicit in
            // analysis)
            // For simple "High Frequency" check, we can just iterate the spectrum.

            // We want to detect high-frequency stars/grids.
            // Center (Low Freq) is at (0,0) index in unshifted array usually,
            // but JTransforms standard output puts DC at 0,0.
            // High frequencies are in the middle of the array indices (N/2).

            // Let's implement the Python logic: Log(Abs(F) + epsilon)
            // Then Mask Center (High Pass)

            double sumVal = 0;
            double sumSq = 0;
            int count = 0;
            double[] magnitudes = new double[N * N];

            // To emulate fftshift, we handle indices:
            // Shifted (y, x) corresponds to original indices logic.
            // But easier: High frequencies in unshifted FFT are at indices N/2.
            // So we can just mask the CORNERS (0,0), (0, N), (N, 0), (N, N) which are Low
            // Freqs.

            // Actually, let's just compute magnitude for all pixels and store them.
            // Indices (r, c)
            for (int r = 0; r < N; r++) {
                for (int c = 0; c < N; c++) {
                    double real = complexData[r][2 * c];
                    double imag = complexData[r][2 * c + 1];
                    double mag = Math.sqrt(real * real + imag * imag);
                    double logMag = 20 * Math.log(mag + 1e-10);

                    magnitudes[r * N + c] = logMag;
                }
            }

            // Masking Low Frequencies: In unshifted FFT, low freqs are at corners.
            // We want to analyze the MIDDLE usually (High Freqs in unshifted? No, wait).
            // Unshifted: DC at 0. Nyquist at N/2.
            // So High Frequencies are around N/2.

            // If we want to mask LOW frequencies, we ignore (0,0) and neighbors.
            // Python script: shift -> center is low freq -> mask center.
            // Unshifted equivalent: Mask corners.
            // We will collect stats ONLY from the "High Frequency" regions.
            // High Freq region is roughly the center block of the unshifted array (N/4 to
            // 3N/4).

            for (int r = N / 4; r < 3 * N / 4; r++) {
                for (int c = N / 4; c < 3 * N / 4; c++) {
                    double val = magnitudes[r * N + c];
                    sumVal += val;
                    sumSq += val * val;
                    count++;
                }
            }

            double mean = sumVal / count;
            double stdDev = Math.sqrt((sumSq / count) - (mean * mean));

            // Detect Spikes (Anomalies)
            int anomalies = 0;
            double sensitivity = 4.5; // Decreased sensitivity (was 3.0) to avoid false positives on natural grids
            double threshold = mean + (sensitivity * stdDev);

            for (int r = N / 4; r < 3 * N / 4; r++) {
                for (int c = N / 4; c < 3 * N / 4; c++) {
                    if (magnitudes[r * N + c] > threshold) {
                        anomalies++;
                    }
                }
            }

            // Python score: (anomaly_count / total_pixels) * 10000
            // total_pixels in our high-pass window is count.
            double score = ((double) anomalies / count) * 10000.0;

            // Normalize to 0-100 logic
            // Python: Likely AI if score > 50.
            return Math.min(100.0, score);

        } catch (Exception e) {
            e.printStackTrace();
            return 50.0;
        }
    }

    /**
     * 3. PRNU / Noise Analysis
     * Detects lack of natural sensor noise.
     * Replaces old "Skin Texture" logic with general Noise Analysis.
     */
    public double analyzeNoise(byte[] fileBytes) {
        try {
            BufferedImage img = ImageIO.read(new ByteArrayInputStream(fileBytes));
            if (img == null)
                return 50.0;

            // Resize for performance if needed, but noise is pixel-level.
            // Keep original if reasonable, or cap at 1024.
            img = limitSize(img, 1024);
            int width = img.getWidth();
            int height = img.getHeight();

            // Convert to grayscale matrix
            double[][] gray = new double[height][width];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int rgb = img.getRGB(x, y);
                    int r = (rgb >> 16) & 0xFF;
                    int g = (rgb >> 8) & 0xFF;
                    int b = rgb & 0xFF;
                    gray[y][x] = (0.299 * r) + (0.587 * g) + (0.114 * b);
                }
            }

            // Denoise (Median Filter 3x3 as approximation of Neural Denoising)
            double[][] denoised = performMedianFilter(gray, width, height);

            // Calculate Noise Map (Residuals) and Variance
            double sumVariance = 0;
            double sum = 0;
            long count = 0;
            long zeroNoiseCount = 0; // "Whitespaces" / Perfect smooth detection

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double noise = gray[y][x] - denoised[y][x];
                    if (Math.abs(noise) < 0.5)
                        zeroNoiseCount++;

                    sum += noise;
                    sumVariance += noise * noise;
                    count++;
                }
            }

            // Variance = E[X^2] - (E[X])^2
            double mean = sum / count;
            double variance = (sumVariance / count) - (mean * mean);

            double zeroNoiseRatio = (double) zeroNoiseCount / count;

            // Python Logic Enhanced:
            // "Whitespaces" check: AI leaves perfectly smooth patches.
            // Tuned: Increased to 0.60 to avoid flagging screenshots/digital art as fake.
            if (zeroNoiseRatio > 0.60)
                return 98.0;

            // Tuned: Modern phones often denoise variance down below 10.0
            if (variance < 2.0)
                return 80.0; // Unnaturally smooth
            if (variance < 10.0)
                return 50.0; // Likely aggressive phone denoising
            return 10.0; // Natural or noisy

        } catch (Exception e) {
            return 50.0;
        }
    }

    /**
     * Backward compatibility for Service layer calls
     */
    public double analyzeSkinTexture(byte[] fileBytes) {
        // Redirect to Noise Analysis as it is more robust than the old skin logic
        // or we could keep the old one, but user asked to implement the Python script
        // logic.
        return analyzeNoise(fileBytes);
    }

    /**
     * 4. Error Level Analysis (ELA)
     */
    public double performELA(byte[] fileBytes) {
        try {
            BufferedImage original = ImageIO.read(new ByteArrayInputStream(fileBytes));
            if (original == null)
                return 0.0;

            // Convert to RGB to ensure consistency
            BufferedImage rgbOriginal = new BufferedImage(original.getWidth(), original.getHeight(),
                    BufferedImage.TYPE_INT_RGB);
            rgbOriginal.getGraphics().drawImage(original, 0, 0, null);

            // 1. Resave at 90% Quality (Python script uses 90)
            java.io.ByteArrayOutputStream compressedStream = new java.io.ByteArrayOutputStream();
            javax.imageio.ImageWriter writer = javax.imageio.ImageIO.getImageWritersByFormatName("jpg").next();
            javax.imageio.ImageWriteParam param = writer.getDefaultWriteParam();
            param.setCompressionMode(javax.imageio.ImageWriteParam.MODE_EXPLICIT);
            param.setCompressionQuality(0.90f);

            writer.setOutput(javax.imageio.ImageIO.createImageOutputStream(compressedStream));
            writer.write(null, new javax.imageio.IIOImage(rgbOriginal, null, null), param);
            writer.dispose();

            BufferedImage compressed = ImageIO.read(new ByteArrayInputStream(compressedStream.toByteArray()));

            // 2. Measure difference
            long maxError = 0;
            double totalGrayDiff = 0;
            int width = rgbOriginal.getWidth();
            int height = rgbOriginal.getHeight();

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int rgb1 = rgbOriginal.getRGB(x, y);
                    int rgb2 = compressed.getRGB(x, y);

                    int rDiff = Math.abs(((rgb1 >> 16) & 0xFF) - ((rgb2 >> 16) & 0xFF));
                    int gDiff = Math.abs(((rgb1 >> 8) & 0xFF) - ((rgb2 >> 8) & 0xFF));
                    int bDiff = Math.abs((rgb1 & 0xFF) - (rgb2 & 0xFF));

                    // Python calculates Scale * Diff (Scale=15).
                    // But we just need statistics.
                    // Gray Diff for stats:
                    int grayD = (rDiff + gDiff + bDiff) / 3;
                    totalGrayDiff += grayD;
                    if (grayD > maxError)
                        maxError = grayD;
                }
            }

            double avgError = totalGrayDiff / (width * height);

            // Python Logic:
            // "Potential Manipulation" if max_error > 50 and avg_error < 5 (Relaxed to 10
            // for sensitivity)

            if (maxError > 50 && avgError < 10.0)
                return 90.0; // High confidence manipulation
            return 10.0; // Consistent

        } catch (Exception e) {
            e.printStackTrace();
            return 0.0;
        }
    }

    // --- Helpers ---

    private double[][] getGrayScaleData(BufferedImage img, int w, int h) {
        BufferedImage scaled = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        scaled.getGraphics().drawImage(img, 0, 0, w, h, null);
        double[][] data = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                data[y][x] = (scaled.getRGB(x, y) & 0xFF);
            }
        }
        return data;
    }

    private BufferedImage limitSize(BufferedImage img, int maxDim) {
        int w = img.getWidth();
        int h = img.getHeight();
        if (w <= maxDim && h <= maxDim)
            return img;

        double ratio = Math.min((double) maxDim / w, (double) maxDim / h);
        int newW = (int) (w * ratio);
        int newH = (int) (h * ratio);

        BufferedImage resized = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB);
        resized.getGraphics().drawImage(img, 0, 0, newW, newH, null);
        return resized;
    }

    private double[][] performMedianFilter(double[][] input, int w, int h) {
        double[][] output = new double[h][w];
        int radius = 1; // 3x3 kernel

        // Median filter over all pixels; borders fall back to original values.
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (x < radius || x >= w - radius || y < radius || y >= h - radius) {
                    output[y][x] = input[y][x];
                    continue;
                }

                double[] window = new double[9];
                int k = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        window[k++] = input[y + dy][x + dx];
                    }
                }
                java.util.Arrays.sort(window);
                output[y][x] = window[4];
            }
        }
        return output;
    }
}
