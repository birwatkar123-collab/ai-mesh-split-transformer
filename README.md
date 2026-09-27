# AI Mesh Split Transformer

AI Mesh Split Transformer is a proof-of-concept for splitting one small
transformer-style model across multiple nearby devices.

The current project includes:

- `ai_mesh_split_transformer`: Python coordinator and desktop worker nodes
- `AiMeshAndroid`: Android node app that can run stage 1 or stage 2

This is an experimental architecture prototype. The model is intentionally tiny
and randomly initialized, so it does not produce useful chatbot output yet. The
goal is to prove that one model computation can be partitioned between devices
and still match a monolithic baseline.

## What Works

- Stage 1 computes embeddings and early transformer blocks.
- Stage 2 receives hidden activations and computes later blocks plus logits.
- The Python coordinator validates distributed output against a single-process
  baseline.
- The Android app exposes compatible `/info` and `/forward` endpoints.

Successful validation looks like:

```text
validation max_abs=0.00000000
argmax_match=True
```

## Quick Start

Install NumPy:

```powershell
cd "ai_mesh_split_transformer"
pip install numpy
```

Start stage 1:

```powershell
python split_node.py --stage 1 --port 9001
```

Start stage 2:

```powershell
python split_node.py --stage 2 --port 9002
```

Run the coordinator:

```powershell
python split_coordinator.py --stage1 http://127.0.0.1:9001 --stage2 http://127.0.0.1:9002 --text "AI mesh" --steps 3 --validate
```

## Android Node

Open the Android project in Android Studio:

```text
AiMeshAndroid
```

Run it on a phone, choose Stage 1 or Stage 2, tap **Start Node**, and use the IP
address shown in the app from the Python coordinator.

Example with Android as stage 2:

```powershell
python split_coordinator.py --stage1 http://127.0.0.1:9001 --stage2 http://PHONE_IP:9002 --text "AI mesh" --steps 3 --validate
```

## Roadmap

- Binary tensor transport instead of base64 JSON
- KV cache support
- Real small pretrained model
- Wi-Fi Direct discovery
- Device benchmarking and scheduling
- Larger model sharding research

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
