package com.aimesh.node;

import android.util.Base64;

import org.json.JSONArray;
import org.json.JSONObject;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
final class TinyModel {
    static final int VOCAB_SIZE = 128;
    static final int DIM = 64;
    static final int HIDDEN = 128;
    static final int LAYERS = 4;
    static final int STAGE_1_LAYERS = 2;

    private final float[][] embedding = new float[VOCAB_SIZE][DIM];
    private final float[][][] w1 = new float[LAYERS][DIM][HIDDEN];
    private final float[][] b1 = new float[LAYERS][HIDDEN];
    private final float[][][] w2 = new float[LAYERS][HIDDEN][DIM];
    private final float[][] b2 = new float[LAYERS][DIM];
    private final float[][] lmHead = new float[DIM][VOCAB_SIZE];
    private int cursor = 0;

    TinyModel() {
        fill(embedding);
        for (int i = 0; i < LAYERS; i++) {
            fill(w1[i]);
            fill(w2[i]);
        }
        fill(lmHead);
    }

    long parameterBytes(int stage) {
        long arrays = stage == 1 ? (long) VOCAB_SIZE * DIM : (long) DIM * VOCAB_SIZE;
        int start = stage == 1 ? 0 : STAGE_1_LAYERS;
        int end = stage == 1 ? STAGE_1_LAYERS : LAYERS;
        for (int i = start; i < end; i++) {
            arrays += (long) DIM * HIDDEN + HIDDEN + (long) HIDDEN * DIM + DIM;
        }
        return arrays * 4L;
    }

    float[][] stage1(int[] tokens) {
        float[][] x = new float[tokens.length][DIM];
        for (int t = 0; t < tokens.length; t++) {
            System.arraycopy(embedding[Math.floorMod(tokens[t], VOCAB_SIZE)], 0, x[t], 0, DIM);
        }
        for (int i = 0; i < STAGE_1_LAYERS; i++) {
            x = block(x, i);
        }
        return x;
    }

    float[] stage2(float[][] hidden) {
        float[][] x = hidden;
        for (int i = STAGE_1_LAYERS; i < LAYERS; i++) {
            x = block(x, i);
        }
        float[] last = layerNorm(x[x.length - 1]);
        float[] logits = new float[VOCAB_SIZE];
        for (int j = 0; j < VOCAB_SIZE; j++) {
            float sum = 0f;
            for (int k = 0; k < DIM; k++) {
                sum += last[k] * lmHead[k][j];
            }
            logits[j] = sum;
        }
        return logits;
    }

    static JSONObject encode2d(float[][] array) throws Exception {
        int rows = array.length;
        int cols = rows == 0 ? 0 : array[0].length;
        ByteBuffer buffer = ByteBuffer.allocate(rows * cols * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float[] row : array) {
            for (float value : row) {
                buffer.putFloat(value);
            }
        }
        return encoded(buffer.array(), new int[]{rows, cols});
    }

    static JSONObject encode1d(float[] array) throws Exception {
        ByteBuffer buffer = ByteBuffer.allocate(array.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : array) {
            buffer.putFloat(value);
        }
        return encoded(buffer.array(), new int[]{array.length});
    }

    static float[][] decode2d(JSONObject object) throws Exception {
        JSONArray shape = object.getJSONArray("shape");
        int rows = shape.getInt(0);
        int cols = shape.getInt(1);
        byte[] bytes = Base64.decode(object.getString("data"), Base64.DEFAULT);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[][] result = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result[r][c] = buffer.getFloat();
            }
        }
        return result;
    }

    private static JSONObject encoded(byte[] data, int[] shape) throws Exception {
        JSONObject object = new JSONObject();
        JSONArray jsonShape = new JSONArray();
        for (int dim : shape) {
            jsonShape.put(dim);
        }
        object.put("dtype", "float32");
        object.put("shape", jsonShape);
        object.put("data", Base64.encodeToString(data, Base64.NO_WRAP));
        return object;
    }

    private float[][] block(float[][] input, int index) {
        float[][] output = new float[input.length][DIM];
        for (int row = 0; row < input.length; row++) {
            float[] norm = layerNorm(input[row]);
            float[] hidden = new float[HIDDEN];
            for (int h = 0; h < HIDDEN; h++) {
                float sum = b1[index][h];
                for (int d = 0; d < DIM; d++) {
                    sum += norm[d] * w1[index][d][h];
                }
                hidden[h] = (float) Math.tanh(sum);
            }
            for (int d = 0; d < DIM; d++) {
                float sum = b2[index][d];
                for (int h = 0; h < HIDDEN; h++) {
                    sum += hidden[h] * w2[index][h][d];
                }
                output[row][d] = input[row][d] + sum;
            }
        }
        return output;
    }

    private static float[] layerNorm(float[] input) {
        float mean = 0f;
        for (float value : input) {
            mean += value;
        }
        mean /= input.length;
        float variance = 0f;
        for (float value : input) {
            float diff = value - mean;
            variance += diff * diff;
        }
        variance /= input.length;
        float scale = (float) (1.0 / Math.sqrt(variance + 1e-5));
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (input[i] - mean) * scale;
        }
        return output;
    }

    private void fill(float[][] array) {
        for (float[] row : array) {
            for (int i = 0; i < row.length; i++) {
                double value = Math.sin(cursor * 12.9898 + 20260927.0) * 43758.5453;
                value = value - Math.floor(value);
                row[i] = (float) (((value * 2.0) - 1.0) * 0.04);
                cursor++;
            }
        }
    }
}
