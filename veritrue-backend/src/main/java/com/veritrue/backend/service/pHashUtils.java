package com.veritrue.backend.service;

import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.io.ByteArrayInputStream;
import java.io.IOException;

@Component
public class pHashUtils {

    private static final int SIZE = 32;
    private static final int SMALLER_SIZE = 8;

    public String computePHash(byte[] imageBytes) {
        try {
            BufferedImage img = ImageIO.read(new ByteArrayInputStream(imageBytes));
            if (img == null)
                return null;

            // 1. Reduce size to 32x32
            img = resize(img, SIZE, SIZE);

            // 2. Reduce color (Grayscale)
            img = grayscale(img);

            // 3. Compute DCT
            double[][] dctVals = applyDCT(img);

            // 4. Reduce DCT (Keep top-left 8x8)
            // 5. Compute average value
            double total = 0;
            for (int x = 0; x < SMALLER_SIZE; x++) {
                for (int y = 0; y < SMALLER_SIZE; y++) {
                    total += dctVals[x][y];
                }
            }
            total -= dctVals[0][0]; // Exclude DC term
            double avg = total / (double) ((SMALLER_SIZE * SMALLER_SIZE) - 1);

            // 6. Construct Hash
            StringBuilder hash = new StringBuilder();
            for (int x = 0; x < SMALLER_SIZE; x++) {
                for (int y = 0; y < SMALLER_SIZE; y++) {
                    if (x != 0 || y != 0) {
                        hash.append(dctVals[x][y] > avg ? "1" : "0");
                    }
                }
            }

            // Convert binary string to hex
            return binaryToHex(hash.toString());

        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private BufferedImage resize(BufferedImage image, int width, int height) {
        BufferedImage resizedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = resizedImage.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(image, 0, 0, width, height, null);
        g.dispose();
        return resizedImage;
    }

    private BufferedImage grayscale(BufferedImage img) {
        ColorConvertOp op = new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null);
        return op.filter(img, null);
    }

    private double[][] applyDCT(BufferedImage img) {
        int N = SIZE;
        double[][] vals = new double[N][N];
        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) {
                vals[x][y] = (img.getRGB(x, y) & 0xFF);
            }
        }
        // Ideally reuse the DCT code from ForensicEngine or make a shared utility
        // For now implementing a simple version here or reusing
        // Since ForensicEngine is a Service and this is Utils, we might want to just
        // duplicate the math
        // or move DCT to a MathUtils static class. For speed, I'll inline a simple DCT
        // II.

        double[][] c = new double[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                c[i][j] = 0;
            }
        }

        // ... (Full DCT implementation is heavy again, maybe we can simplify for pHash
        // by just using average based hash if DCT is too complex for this snippet?)
        // Standard pHash uses DCT.
        // Let's use a simplified Mean Hash (aHash) instead for "Global Intelligence"
        // quick check if DCT is too much code duplication.
        // Actually, let's stick to the plan: "pHash".
        // I'll copy the DCT logic.

        // Use 1D DCT separability for speed? loops are fine for 32x32.

        // (Implementation omitted for brevity in thought, but will write in file)
        return performDCT(vals, N);
    }

    private double[][] performDCT(double[][] f, int N) {
        double[][] F = new double[N][N];
        double[] c = new double[N];
        c[0] = 1.0 / Math.sqrt(2.0);
        for (int i = 1; i < N; i++)
            c[i] = 1.0;

        for (int u = 0; u < N; u++) {
            for (int v = 0; v < N; v++) {
                double sum = 0.0;
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        sum += Math.cos(((2 * i + 1) / (2.0 * N)) * u * Math.PI) *
                                Math.cos(((2 * j + 1) / (2.0 * N)) * v * Math.PI) * f[i][j];
                    }
                }
                F[u][v] = 0.25 * c[u] * c[v] * sum;
            }
        }
        return F;
    }

    private String binaryToHex(String binary) {
        long decimal = Long.parseLong(binary, 2);
        return Long.toHexString(decimal);
    }
}
