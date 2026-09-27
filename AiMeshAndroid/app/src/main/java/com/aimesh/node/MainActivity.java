package com.aimesh.node;

import android.app.Activity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.RadioGroup;
import android.widget.TextView;

import java.net.Inet4Address;
import java.net.NetworkInterface;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends Activity {
    private TextView status;
    private TextView log;
    private EditText port;
    private RadioGroup stageGroup;
    private MeshHttpServer server;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(buildUi());
        refreshStatus("Stopped");
    }

    @Override
    protected void onDestroy() {
        stopServer();
        super.onDestroy();
    }

    private View buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(32, 32, 32, 32);

        TextView title = new TextView(this);
        title.setText("AI Mesh Node");
        title.setTextSize(26);
        title.setGravity(Gravity.START);
        root.addView(title);

        status = new TextView(this);
        status.setTextSize(16);
        root.addView(status);

        stageGroup = new RadioGroup(this);
        stageGroup.setOrientation(RadioGroup.HORIZONTAL);
        android.widget.RadioButton stage1 = new android.widget.RadioButton(this);
        stage1.setText("Stage 1");
        stage1.setId(1);
        android.widget.RadioButton stage2 = new android.widget.RadioButton(this);
        stage2.setText("Stage 2");
        stage2.setId(2);
        stageGroup.addView(stage1);
        stageGroup.addView(stage2);
        stageGroup.check(2);
        root.addView(stageGroup);

        port = new EditText(this);
        port.setHint("Port");
        port.setText("9002");
        port.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        root.addView(port);

        Button start = new Button(this);
        start.setText("Start Node");
        start.setOnClickListener(v -> startServer());
        root.addView(start);

        Button stop = new Button(this);
        stop.setText("Stop Node");
        stop.setOnClickListener(v -> stopServer());
        root.addView(stop);

        log = new TextView(this);
        log.setTextSize(14);
        log.setMovementMethod(new ScrollingMovementMethod());
        root.addView(log, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1f
        ));

        return root;
    }

    private void startServer() {
        stopServer();
        int selectedStage = stageGroup.getCheckedRadioButtonId();
        int selectedPort = Integer.parseInt(port.getText().toString());
        try {
            server = new MeshHttpServer(selectedStage, selectedPort, this::appendLog);
            server.start();
            refreshStatus("Running stage " + selectedStage + " at http://" + localIp() + ":" + selectedPort);
        } catch (Exception error) {
            appendLog("Start failed: " + error);
            refreshStatus("Start failed");
        }
    }

    private void stopServer() {
        if (server != null) {
            server.stop();
            server = null;
        }
        refreshStatus("Stopped");
    }

    private void refreshStatus(String message) {
        if (status != null) {
            status.setText(message + "\nPhone IP: " + localIp());
        }
    }

    private void appendLog(String message) {
        runOnUiThread(() -> {
            String stamp = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
            log.append(stamp + "  " + message + "\n");
        });
    }

    private static String localIp() {
        try {
            for (NetworkInterface networkInterface : Collections.list(NetworkInterface.getNetworkInterfaces())) {
                for (java.net.InetAddress address : Collections.list(networkInterface.getInetAddresses())) {
                    if (!address.isLoopbackAddress() && address instanceof Inet4Address) {
                        return address.getHostAddress();
                    }
                }
            }
        } catch (Exception ignored) {
        }
        return "127.0.0.1";
    }
}
