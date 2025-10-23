from playwright.sync_api import sync_playwright
import sys

URL = "http://127.0.0.1:8501/"
EXPECTED_TEXT = "Prediction Demo"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    try:
        page.goto(URL, timeout=30000)
        # Wait for Streamlit to set prerenderReady true and render the app
        page.wait_for_function("() => window.prerenderReady === true", timeout=30000)
        content = page.content()
        if EXPECTED_TEXT in content:
            print("FOUND")
            browser.close()
            sys.exit(0)
        else:
            print("NOT FOUND")
            print(page.content())
            browser.close()
            sys.exit(2)
    except Exception as e:
        print(f"ERROR: {e}")
        browser.close()
        sys.exit(3)
