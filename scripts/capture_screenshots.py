"""Capture Streamlit screenshots with Playwright using the system Edge browser.

Usage:
    python -m scripts.capture_screenshots

The script starts Streamlit, waits for the app to render, captures the landing
page, sends a sample message, captures the conversation and stops the server.
"""

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
URL = "http://localhost:8601"


def wait_for_server(timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(URL, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise RuntimeError("Streamlit did not start in time")


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/streamlit_app.py",
            "--server.headless=true",
            "--server.port=8601",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        wait_for_server()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(URL, wait_until="networkidle", timeout=60000)
            page.wait_for_selector("text=Cognevance AI Chatbot", timeout=60000)
            page.wait_for_timeout(2500)
            page.screenshot(path=str(DOCS_DIR / "streamlit_ui.png"))

            chat_input = page.get_by_placeholder("Ask me anything...")
            chat_input.fill("where is my order?")
            chat_input.press("Enter")
            page.wait_for_selector("text=track_order", timeout=30000)
            page.wait_for_timeout(1500)
            page.screenshot(path=str(DOCS_DIR / "streamlit_chat.png"))

            browser.close()
            print("Saved docs/streamlit_ui.png")
            print("Saved docs/streamlit_chat.png")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
