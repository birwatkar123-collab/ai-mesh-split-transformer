package com.aimesh.node;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

final class MeshHttpServer {
    interface Listener {
        void onLog(String message);
    }

    private final TinyModel model = new TinyModel();
    private final int stage;
    private final int port;
    private final Listener listener;
    private final ExecutorService workers = Executors.newCachedThreadPool();
    private volatile boolean running;
    private ServerSocket serverSocket;
    private Thread acceptThread;
    private long startedAt;

    MeshHttpServer(int stage, int port, Listener listener) {
        this.stage = stage;
        this.port = port;
        this.listener = listener;
    }

    void start() throws Exception {
        serverSocket = new ServerSocket(port);
        running = true;
        startedAt = System.currentTimeMillis();
        acceptThread = new Thread(this::acceptLoop, "ai-mesh-http");
        acceptThread.start();
        log("Stage " + stage + " server started on port " + port);
    }

    void stop() {
        running = false;
        try {
            if (serverSocket != null) {
                serverSocket.close();
            }
        } catch (Exception ignored) {
        }
        log("Server stopped");
    }

    private void acceptLoop() {
        while (running) {
            try {
                Socket socket = serverSocket.accept();
                workers.submit(() -> handle(socket));
            } catch (Exception error) {
                if (running) {
                    log("Accept failed: " + error);
                }
            }
        }
    }

    private void handle(Socket socket) {
        try (Socket closeable = socket) {
            BufferedReader reader = new BufferedReader(new InputStreamReader(closeable.getInputStream(), StandardCharsets.UTF_8));
            String requestLine = reader.readLine();
            if (requestLine == null) {
                return;
            }
            String[] parts = requestLine.split(" ");
            String method = parts.length > 0 ? parts[0] : "";
            String path = parts.length > 1 ? parts[1] : "";
            int contentLength = 0;
            String line;
            while ((line = reader.readLine()) != null && !line.isEmpty()) {
                String lower = line.toLowerCase(Locale.US);
                if (lower.startsWith("content-length:")) {
                    contentLength = Integer.parseInt(line.substring(line.indexOf(':') + 1).trim());
                }
            }
            char[] bodyChars = new char[contentLength];
            int read = 0;
            while (read < contentLength) {
                int n = reader.read(bodyChars, read, contentLength - read);
                if (n < 0) {
                    break;
                }
                read += n;
            }
            route(closeable, method, path, new String(bodyChars, 0, read));
        } catch (Exception error) {
            log("Request failed: " + error);
        }
    }

    private void route(Socket socket, String method, String path, String body) throws Exception {
        if ("GET".equals(method) && "/info".equals(path)) {
            JSONObject response = new JSONObject();
            response.put("ok", true);
            response.put("stage", stage);
            response.put("parameter_bytes", model.parameterBytes(stage));
            response.put("uptime_s", (System.currentTimeMillis() - startedAt) / 1000.0);
            send(socket, 200, response);
            return;
        }
        if ("POST".equals(method) && "/forward".equals(path)) {
            send(socket, 200, forward(new JSONObject(body)));
            return;
        }
        send(socket, 404, new JSONObject().put("ok", false).put("error", "not found"));
    }

    private JSONObject forward(JSONObject request) throws Exception {
        long begin = System.nanoTime();
        JSONObject response = new JSONObject();
        response.put("ok", true);
        response.put("stage", stage);
        if (stage == 1) {
            JSONArray rawTokens = request.getJSONArray("tokens");
            int[] tokens = new int[rawTokens.length()];
            for (int i = 0; i < tokens.length; i++) {
                tokens[i] = rawTokens.getInt(i);
            }
            response.put("hidden", TinyModel.encode2d(model.stage1(tokens)));
        } else {
            response.put("logits", TinyModel.encode1d(model.stage2(TinyModel.decode2d(request.getJSONObject("hidden")))));
        }
        double elapsedMs = (System.nanoTime() - begin) / 1_000_000.0;
        response.put("elapsed_ms", Math.round(elapsedMs * 1000.0) / 1000.0);
        return response;
    }

    private static void send(Socket socket, int code, JSONObject body) throws Exception {
        byte[] bytes = body.toString().getBytes(StandardCharsets.UTF_8);
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8));
        writer.write("HTTP/1.1 " + code + " OK\r\n");
        writer.write("Content-Type: application/json\r\n");
        writer.write("Content-Length: " + bytes.length + "\r\n");
        writer.write("Connection: close\r\n");
        writer.write("\r\n");
        writer.flush();
        socket.getOutputStream().write(bytes);
        socket.getOutputStream().flush();
    }

    private void log(String message) {
        if (listener != null) {
            listener.onLog(message);
        }
    }
}
