#!/usr/bin/env bash
# scripts/net_guard.sh — run the demo with loopback-only networking.
# Ollama listens on localhost; we keep loopback up but cut external routes so the
# process physically cannot reach the internet. The /health badge then verifies
# offline status live on screen.
# Demo-grade equivalent if netns is fiddly: enable airplane mode + rely on the
# /health badge.
set -e
sudo ip netns add mnemo 2>/dev/null || true
sudo ip netns exec mnemo ip link set lo up
echo "Running Mnemo with loopback-only networking."
sudo ip netns exec mnemo env PYTHONPATH=src uvicorn mnemo.server:app --host 127.0.0.1 --port 8000
