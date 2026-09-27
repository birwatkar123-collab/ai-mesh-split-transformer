# AI Mesh Android Node

This is a native Android proof-of-concept node for the AI Mesh split-transformer
experiment.

The app can run as:

- Stage 1: token embedding + early transformer blocks
- Stage 2: later transformer blocks + logits

It exposes the same local HTTP endpoints as the Python node:

- `GET /info`
- `POST /forward`

## Open In Android Studio

1. Open Android Studio.
2. Choose **Open**.
3. Select:

   ```text
   D:\New folder (2)\AiMeshAndroid
   ```

4. Let Android Studio sync Gradle.
5. Connect your Android phone with USB debugging enabled.
6. Press **Run**.

## Use Phone As Stage 2

On the phone:

1. Open **AI Mesh Node**.
2. Select **Stage 2**.
3. Keep port `9002`.
4. Tap **Start Node**.
5. Note the phone IP shown in the app.

On Windows, start stage 1:

```powershell
cd "D:\New folder (2)\ai_mesh_split_transformer"
python split_node.py --stage 1 --port 9001
```

Then run:

```powershell
python split_coordinator.py --stage1 http://127.0.0.1:9001 --stage2 http://PHONE_IP:9002 --text "AI mesh" --steps 3 --validate
```

Replace `PHONE_IP` with the IP displayed in the Android app.

## Use Phone As Stage 1

On the phone:

1. Select **Stage 1**.
2. Use port `9001`.
3. Tap **Start Node**.

On Windows, start stage 2:

```powershell
cd "D:\New folder (2)\ai_mesh_split_transformer"
python split_node.py --stage 2 --port 9002
```

Then run:

```powershell
python split_coordinator.py --stage1 http://PHONE_IP:9001 --stage2 http://127.0.0.1:9002 --text "AI mesh" --steps 3 --validate
```

## Notes

- Your phone and laptop must be on the same Wi-Fi network.
- Windows Firewall may ask for permission for Python. Allow private network access.
- This is still a tiny random model. It proves distributed execution, not useful
  chat quality.
