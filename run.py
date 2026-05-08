#!/usr/bin/env python3
"""Entry point for the DeepFace face recognition system.

Usage:
    python run.py                  # Start web demo (default)
    python run.py --port 5001      # Custom port
    python run.py --host 0.0.0.0   # Bind all interfaces
    python run.py --debug          # Debug mode
"""

import argparse
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="DeepFace 人脸识别系统 — Demo UI + REST API"
    )
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址 (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5001, help="端口 (默认 5001)")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    # Ensure we can import from the project root
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from face_system.app import create_app

    app = create_app()
    logger.info("🚀 DeepFace 人脸识别系统启动中...")
    logger.info("   Demo UI:  http://%s:%d", args.host, args.port)
    logger.info("   API 文档:  http://%s:%d/api", args.host, args.port)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
