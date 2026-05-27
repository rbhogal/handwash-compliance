"""
LED Output Controller
Drives a physical green/red LED via GPIO on the Modalix/edge hardware.
Falls back to a console stub when GPIO is unavailable (dev machines).
"""

from __future__ import annotations
import logging
import time

logger = logging.getLogger(__name__)


class LEDController:
    """
    Abstracts GPIO-driven LED feedback.

    Green  = compliant wash detected.
    Red    = violation or partial compliance.

    Usage:
        led = LEDController(gpio_pin=17)
        led.green()          # turn green on for 3 s
        led.red()
        led.off()
    """

    DEFAULT_DURATION_SEC = 3.0

    def __init__(self, gpio_pin: int = 17, use_gpio: bool | None = None) -> None:
        self.gpio_pin = gpio_pin
        self._gpio_available = self._detect_gpio() if use_gpio is None else use_gpio
        self._gpio = None

        if self._gpio_available:
            self._setup_gpio()
        else:
            logger.info("GPIO not available — using console LED stub")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def green(self, duration_sec: float = DEFAULT_DURATION_SEC) -> None:
        """Flash green for `duration_sec` seconds."""
        self._set(state="GREEN")
        time.sleep(duration_sec)
        self.off()

    def red(self, duration_sec: float = DEFAULT_DURATION_SEC) -> None:
        """Flash red for `duration_sec` seconds."""
        self._set(state="RED")
        time.sleep(duration_sec)
        self.off()

    def off(self) -> None:
        self._set(state="OFF")

    def signal_verdict(self, compliant: bool) -> None:
        """Convenience: pick colour based on compliance result."""
        if compliant:
            self.green()
        else:
            self.red()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set(self, state: str) -> None:
        if self._gpio_available and self._gpio is not None:
            # TODO: map state → actual GPIO HIGH/LOW logic for your wiring
            # self._gpio.output(self.gpio_pin, self._gpio.HIGH if state == "GREEN" else self._gpio.LOW)
            pass
        else:
            symbols = {"GREEN": "🟢", "RED": "🔴", "OFF": "⚫"}
            print(f"[LED] {symbols.get(state, '?')} {state}")

    @staticmethod
    def _detect_gpio() -> bool:
        try:
            import RPi.GPIO  # noqa: F401
            return True
        except ImportError:
            return False

    def _setup_gpio(self) -> None:
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.OUT)
            GPIO.output(self.gpio_pin, GPIO.LOW)
            self._gpio = GPIO
        except Exception as exc:
            logger.warning(f"GPIO setup failed: {exc} — falling back to stub")
            self._gpio_available = False

    def cleanup(self) -> None:
        if self._gpio is not None:
            self._gpio.cleanup()
