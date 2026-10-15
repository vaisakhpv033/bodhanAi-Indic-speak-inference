"""
Deploy Indic-Speak to Active RunPod Pod
=======================================
Synchronizes local indicspeak directory to /workspace/indicspeak on the active
RunPod instance and starts the streaming server in a persistent tmux session.

Usage:
    python deploy_to_runpod.py --host <RUNPOD_IP> --port <RUNPOD_SSH_PORT> --key <SSH_KEY_PATH>
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent.parent

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Deploy Indic-Speak to RunPod")
    parser.add_argument("--host", default=os.getenv("RUNPOD_HOST"), help="RunPod SSH IP/Host")
    parser.add_argument("--port", default=os.getenv("RUNPOD_PORT"), help="RunPod SSH Port")
    parser.add_argument("--key", default=os.getenv("SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519")), help="SSH private key path")
    args = parser.parse_args()

    if not args.host or not args.port:
        print("[INFO] No active RUNPOD_HOST/RUNPOD_PORT provided.")
        print("You can deploy via RunPod Template UI using RUNPOD_TEMPLATE_GUIDE.md,")
        print("or specify --host and --port to deploy directly over SSH.")
        return

    os.environ["RUNPOD_HOST"] = str(args.host)
    os.environ["RUNPOD_PORT"] = str(args.port)

    import runpod_remote

    print(f"[*] Uploading Indic-Speak from {CURRENT_DIR} to RunPod /workspace/indicspeak...")
    runpod_remote.upload_dir_tar(str(CURRENT_DIR), "/workspace")

    startup_command = """
#!/usr/bin/env bash
cd /workspace/indicspeak
chmod +x runpod_start.sh
tmux kill-session -t indicspeak 2>/dev/null || true
tmux new-session -d -s indicspeak "bash /workspace/indicspeak/runpod_start.sh 2>&1 | tee /workspace/indicspeak.log"
echo "[*] Indic-Speak started in tmux session 'indicspeak'!"
echo "[*] Check logs with: tmux attach -t indicspeak or cat /workspace/indicspeak.log"
"""
    print("[*] Launching service on RunPod...")
    runpod_remote.run_script(startup_command)
    print("[*] Deployment script finished. Check pod health at port 8000!")


if __name__ == "__main__":
    main()
