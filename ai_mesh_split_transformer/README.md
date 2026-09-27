# AI Mesh Split Transformer MVP

This is a tiny proof-of-concept for splitting one transformer-like model across
two devices on a local network.

It is intentionally small and randomly initialized. The goal is to prove the
architecture, not produce useful language yet.

## Install

```bash
pip install numpy
```

## Run Locally

Terminal 1:

```bash
python split_node.py --stage 1 --port 9001
```

Terminal 2:

```bash
python split_node.py --stage 2 --port 9002
```

Terminal 3:

```bash
python split_coordinator.py --stage1 http://127.0.0.1:9001 --stage2 http://127.0.0.1:9002 --text "AI mesh" --steps 3 --validate
```

Success looks like:

```text
validation max_abs=0.00000000
argmax_match=True
```

## Two Devices

Run stage 1 on one computer:

```bash
python split_node.py --stage 1 --port 9001
```

Run stage 2 on another computer or Android Termux phone:

```bash
python split_node.py --stage 2 --port 9002
```

Then point the coordinator at both machines:

```bash
python split_coordinator.py --stage1 http://LAPTOP_IP:9001 --stage2 http://PHONE_IP:9002 --text "AI" --steps 3 --validate
```

Open firewall access for the selected ports if the devices cannot connect.
