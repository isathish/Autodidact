"""
Phase 1, Milestone M1.2 — Motor System
Defines low-level browser interaction actions.
"""

from playwright.sync_api import sync_playwright


class MotorSystem:
    def __init__(self, headless=True):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.page = self.browser.new_page()

    def navigate(self, url: str):
        self.page.goto(url)

    def click(self, x: int, y: int):
        self.page.mouse.click(x, y)

    def type_text(self, text: str):
        self.page.keyboard.type(text)

    def press_enter(self):
        self.page.keyboard.press("Enter")

    def scroll(self, delta_y: int):
        self.page.mouse.wheel(0, delta_y)

    def go_back(self):
        self.page.go_back()

    def close(self):
        self.browser.close()
        self.playwright.stop()


if __name__ == "__main__":
    motor = MotorSystem(headless=True)
    motor.navigate("https://example.com")
    motor.scroll(500)
    motor.close()
