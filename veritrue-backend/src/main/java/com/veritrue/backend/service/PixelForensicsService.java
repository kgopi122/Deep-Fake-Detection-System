package com.veritrue.backend.service;

import org.springframework.stereotype.Service;

import java.awt.image.BufferedImage;

@Service
public class PixelForensicsService {

    public record Result(
            int width,
            int height,
            double entropy,
            double noiseStd,
            double tileNoiseUniformity,
            double laplacianStd,
            double tileEdgeDispersion,
            double smoothRatio,
            double edgeHighRatio,
            double clippedRatio,
            double blockiness,
            double pixelAiScore,
            String summary
    ) {
    }

    /**
     * Reads ALL pixels and computes robust statistical signals.
     * These are not perfect "watermark" detectors (there is no universal invisible mark),
     * but they provide measurable forensic evidence that often differs between camera images and AIGC.
     */
    public Result analyze(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        long total = (long) width * height;
        int[] hist = new int[256];

        double sum = 0;
        long clipped = 0;
        double boundaryDiff = 0;
        double overallDiff = 0;
        double noiseSumSq = 0;

        // Laplacian (sharpness/texture) stats
        double lapSumSq = 0;

        // Gradient-based stats (portrait-friendly)
        long smoothCount = 0;
        long edgeHighCount = 0;

        // Tile-wise noise uniformity: compute residual energy per tile.
        // AIGC often shows unusually uniform residual/noise across tiles.
        final int tileSize = 64;
        int tilesX = Math.max(1, (int) Math.ceil(width / (double) tileSize));
        int tilesY = Math.max(1, (int) Math.ceil(height / (double) tileSize));
        double[] tileSumSq = new double[tilesX * tilesY];
        int[] tileCount = new int[tilesX * tilesY];

        // Tile-wise edge/texture energy (laplacian)
        double[] tileLapSumSq = new double[tilesX * tilesY];

        // Bulk read all pixels once (fast) and derive luminance for each.
        int[] rgbPixels = image.getRGB(0, 0, width, height, null, 0, width);
        int nPix = width * height;
        double[] lum = new double[nPix];

        for (int i = 0; i < nPix; i++) {
            int rgb = rgbPixels[i];
            int r = (rgb >> 16) & 0xFF;
            int g = (rgb >> 8) & 0xFF;
            int b = rgb & 0xFF;
            double yVal = (0.299 * r + 0.587 * g + 0.114 * b);
            lum[i] = yVal;

            int yi = (int) Math.round(yVal);
            if (yi < 0) yi = 0;
            if (yi > 255) yi = 255;
            hist[yi]++;
            sum += yi;
            if (yi <= 1 || yi >= 254) clipped++;
        }

        // Prefix sum for fast local 3x3 mean over luminance
        double[] prefix = new double[(width + 1) * (height + 1)];
        for (int y = 1; y <= height; y++) {
            double rowSum = 0;
            int base = (y - 1) * width;
            for (int x = 1; x <= width; x++) {
                rowSum += lum[base + (x - 1)];
                prefix[y * (width + 1) + x] = prefix[(y - 1) * (width + 1) + x] + rowSum;
            }
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                double yVal = lum[idx];

                // Blockiness
                if (x > 0) {
                    double diff = Math.abs(yVal - lum[idx - 1]);
                    overallDiff += diff;
                    if ((x % 8) == 0) boundaryDiff += diff;
                }
                if (y > 0) {
                    double diff = Math.abs(yVal - lum[idx - width]);
                    overallDiff += diff;
                    if ((y % 8) == 0) boundaryDiff += diff;
                }

                // Local 3x3 mean residual via prefix sum
                int x0 = Math.max(0, x - 1);
                int y0 = Math.max(0, y - 1);
                int x1 = Math.min(width - 1, x + 1);
                int y1 = Math.min(height - 1, y + 1);

                // Convert to prefix coordinates (+1)
                int px0 = x0;
                int py0 = y0;
                int px1 = x1 + 1;
                int py1 = y1 + 1;

                double areaSum = prefix[py1 * (width + 1) + px1]
                        - prefix[py0 * (width + 1) + px1]
                        - prefix[py1 * (width + 1) + px0]
                        + prefix[py0 * (width + 1) + px0];

                int count = (x1 - x0 + 1) * (y1 - y0 + 1);
                double localMean = areaSum / Math.max(1, count);
                // Normalize residual to 0..1 scale so thresholds are meaningful across inputs
                double residual = (yVal - localMean) / 255.0;
                noiseSumSq += residual * residual;

                int tx = Math.min(tilesX - 1, x / tileSize);
                int ty = Math.min(tilesY - 1, y / tileSize);
                int tIdx = ty * tilesX + tx;
                tileSumSq[tIdx] += residual * residual;
                tileCount[tIdx] += 1;

                // 4-neighbor Laplacian (skip borders)
                if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                    // Central-difference gradient magnitude (cheap Sobel-like)
                    double gx = lum[idx + 1] - lum[idx - 1];
                    double gy = lum[idx + width] - lum[idx - width];
                    double grad = Math.sqrt((gx * gx) + (gy * gy));

                    // Many AIGC portraits: very large smooth regions + unnaturally crisp edges
                    if (grad < 2.2) smoothCount++;
                    if (grad > 18.0) edgeHighCount++;

                    double lap = (-4.0 * yVal)
                            + lum[idx - 1] + lum[idx + 1]
                            + lum[idx - width] + lum[idx + width];
                    // Normalize laplacian to 0..1 scale for stable thresholds
                    double lapN = lap / 255.0;
                    lapSumSq += lapN * lapN;
                    tileLapSumSq[tIdx] += lapN * lapN;
                }
            }
        }

        // Entropy
        double entropy = 0;
        for (int i = 0; i < 256; i++) {
            if (hist[i] == 0) continue;
            double p = (double) hist[i] / total;
            entropy += -p * (Math.log(p) / Math.log(2));
        }

        // Noise std (normalized residual scale: ~0..0.1 typical)
        double noiseStd = Math.sqrt(noiseSumSq / total);

        // Laplacian std (normalized)
        double laplacianStd = Math.sqrt(lapSumSq / Math.max(1.0, (double) total));

        // Smooth/edge ratios computed over interior pixels only
        long interior = Math.max(1L, (long) (width - 2) * (height - 2));
        double smoothRatio = (double) smoothCount / interior;
        double edgeHighRatio = (double) edgeHighCount / interior;

        // Tile noise uniformity
        double tileMean = 0.0;
        int tileN = tilesX * tilesY;
        double[] tileStd = new double[tileN];
        for (int i = 0; i < tileN; i++) {
            double v = tileCount[i] > 0 ? Math.sqrt(tileSumSq[i] / tileCount[i]) : 0.0;
            tileStd[i] = v;
            tileMean += v;
        }
        tileMean /= Math.max(1, tileN);

        double tileVar = 0.0;
        for (int i = 0; i < tileN; i++) {
            double d = tileStd[i] - tileMean;
            tileVar += d * d;
        }
        tileVar /= Math.max(1, tileN);
        double tileStdDev = Math.sqrt(tileVar);

        // Coefficient of variation; lower => more uniform.
        double tileNoiseUniformity = tileMean <= 1e-9 ? 0.0 : (tileStdDev / tileMean);

        // Tile-wise edge dispersion: how uneven texture/sharpness is across the image.
        // AI portraits often have extremely smooth backgrounds + very crisp facial edges => high dispersion.
        double lapTileMean = 0.0;
        double[] tileLapStd = new double[tileN];
        for (int i = 0; i < tileN; i++) {
            double v = tileCount[i] > 0 ? Math.sqrt(tileLapSumSq[i] / tileCount[i]) : 0.0;
            tileLapStd[i] = v;
            lapTileMean += v;
        }
        lapTileMean /= Math.max(1, tileN);
        double lapTileVar = 0.0;
        for (int i = 0; i < tileN; i++) {
            double d = tileLapStd[i] - lapTileMean;
            lapTileVar += d * d;
        }
        lapTileVar /= Math.max(1, tileN);
        double lapTileStdDev = Math.sqrt(lapTileVar);
        double tileEdgeDispersion = lapTileMean <= 1e-9 ? 0.0 : (lapTileStdDev / lapTileMean);

        // Clipped ratio
        double clippedRatio = (double) clipped / total;

        // Blockiness: normalize boundary diff by overall diff
        double blockiness = overallDiff <= 1e-9 ? 0.0 : (boundaryDiff / overallDiff);

        // Heuristic AIGC score from pixel statistics (0..100)
        // NOTE: noiseStd and laplacianStd are normalized to 0..1; thresholds assume that scale.
        double score = 25.0;

        // Very smooth images + low entropy are common for diffusion/upsampled assets
        if (noiseStd < 0.010) score += 30;
        else if (noiseStd < 0.016) score += 18;

        if (entropy < 6.6) score += 18;
        else if (entropy < 7.25) score += 9;

        // Extremely low blockiness with very low noise can indicate synthetic origin.
        if (blockiness < 0.10 && noiseStd < 0.014) score += 10;

        // Tile-wise uniformity: camera images often have spatially varying noise (lighting, sensor pattern, compression).
        // Very uniform residual energy is a common synthetic indicator.
        if (tileNoiseUniformity < 0.18) score += 22;
        else if (tileNoiseUniformity < 0.28) score += 10;

        // Edge dispersion: smooth background + overly crisp foreground is a frequent synthetic signature.
        if (tileEdgeDispersion > 0.95) score += 12;
        else if (tileEdgeDispersion > 0.75) score += 6;

        // Portrait-friendly signal and 2D Illustration detector (e.g., anime/Ghibli)
        if (smoothRatio > 0.90 && edgeHighRatio > 0.015) score += 35; // Strong 2D illustration indicator
        else if (smoothRatio > 0.88) score += 18;
        else if (smoothRatio > 0.84) score += 10;

        // Smooth areas + lots of very strong edges is a common AIGC look
        if (smoothRatio > 0.86 && edgeHighRatio > 0.020) score += 10;

        // Extremely low overall texture can be suspicious (diffusion + aggressive denoise)
        if (laplacianStd < 0.012 && noiseStd < 0.016) score += 8;

        // Excessive clipping can indicate post-processing or synthetic HDR artifacts.
        if (clippedRatio > 0.08) score += 10;

        if (score > 100) score = 100;

        String summary = "Pixel scan: " + width + "x" + height
                + ", entropy=" + String.format("%.2f", entropy)
                + ", noiseStd=" + String.format("%.2f", noiseStd)
                + ", tileUniform=" + String.format("%.3f", tileNoiseUniformity)
                + ", lapStd=" + String.format("%.2f", laplacianStd)
                + ", edgeDisp=" + String.format("%.3f", tileEdgeDispersion)
                + ", smooth=" + String.format("%.2f", smoothRatio)
                + ", edgeHi=" + String.format("%.3f", edgeHighRatio)
                + ", blockiness=" + String.format("%.3f", blockiness)
                + ", clipped=" + String.format("%.2f", clippedRatio * 100) + "%";

        return new Result(width, height, entropy, noiseStd, tileNoiseUniformity, laplacianStd, tileEdgeDispersion, smoothRatio, edgeHighRatio, clippedRatio, blockiness, score, summary);
    }
}
