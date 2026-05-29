"""
BioVision AI — Voice Assistant
Local TTS using pyttsx3 only. No external APIs required.
Speaks prediction results with configurable language and speed.
"""

import logging
import threading

logger = logging.getLogger(__name__)

# Try importing pyttsx3 — it may not be available in all envs
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False
    logger.warning("pyttsx3 not installed — voice assistant disabled.")


class VoiceAssistant:
    """
    Wrapper around pyttsx3 that provides thread-safe, non-blocking speech.
    """

    def __init__(self, language="en", rate=145, volume=1.0):
        """
        Args:
            language : 'en' for English, 'hi' for Hindi (if voice installed).
            rate     : Speech rate in words-per-minute (default 145).
            volume   : Volume 0.0 – 1.0 (default 1.0).
        """
        self.enabled   = False
        self.language  = language
        self.rate      = rate
        self.volume    = volume
        self._engine   = None
        self._lock     = threading.Lock()

        if _TTS_AVAILABLE:
            self._init_engine()

    # ──────────────────────────────────────────────────────────────────────
    def _init_engine(self):
        """Initialise the pyttsx3 engine and configure voice properties."""
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   self.rate)
            self._engine.setProperty("volume", self.volume)
            self._set_language(self.language)
            self.enabled = True
            logger.info("Voice assistant initialised.")
        except Exception as e:
            logger.error(f"Voice init error: {e}")
            self.enabled = False

    def _set_language(self, lang_code):
        """
        Attempt to select a voice that matches the requested language code.
        Falls back to the default system voice if none is found.
        """
        if self._engine is None:
            return
        try:
            voices = self._engine.getProperty("voices")
            for voice in voices:
                if lang_code in voice.languages or lang_code in voice.id.lower():
                    self._engine.setProperty("voice", voice.id)
                    logger.info(f"Voice set to: {voice.name}")
                    return
            # Fallback: first available voice
            if voices:
                self._engine.setProperty("voice", voices[0].id)
        except Exception as e:
            logger.warning(f"Could not set language '{lang_code}': {e}")

    # ──────────────────────────────────────────────────────────────────────
    def speak(self, text):
        """
        Speak the given text in a background thread (non-blocking).

        Args:
            text : str — the message to speak.
        """
        if not self.enabled or not _TTS_AVAILABLE:
            logger.info(f"[Voice OFF] Would say: {text}")
            return

        def _run():
            with self._lock:
                try:
                    engine = pyttsx3.init()
                    engine.setProperty("rate",   self.rate)
                    engine.setProperty("volume", self.volume)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    logger.error(f"Speech error: {e}")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def announce_prediction(self, disease, confidence, severity, category=""):
        """
        Build and speak a natural-language prediction announcement.

        Example output:
            "Plant disease detected: Tomato Blight with 92 percent confidence.
             Severity level is HIGH. Please consult a specialist."
        """
        try:
            conf_pct = int(round(confidence * 100))
            cat_str  = f"{category.capitalize()} disease detected: " if category else ""
            message  = (
                f"{cat_str}{disease} with {conf_pct} percent confidence. "
                f"Severity level is {severity}. "
            )
            if severity == "HIGH":
                message += "This is a serious condition. Please seek immediate expert advice."
            elif severity == "MEDIUM":
                message += "Consider consulting a specialist soon."
            else:
                message += "Monitor the situation and apply preventive measures."

            self.speak(message)
        except Exception as e:
            logger.error(f"Announce error: {e}")

    def toggle(self):
        """Toggle voice on/off and return the new state."""
        self.enabled = not self.enabled
        state = "ON" if self.enabled else "OFF"
        logger.info(f"Voice assistant turned {state}.")
        return self.enabled

    def set_rate(self, rate):
        """Change speech rate dynamically."""
        self.rate = rate
        if self._engine:
            self._engine.setProperty("rate", rate)

    def set_language(self, lang_code):
        """Change the TTS language/voice at runtime."""
        self.language = lang_code
        self._set_language(lang_code)


# Module-level singleton for convenience
_assistant = None


def get_assistant():
    """Return the module-level VoiceAssistant singleton (lazy init)."""
    global _assistant
    if _assistant is None:
        _assistant = VoiceAssistant()
    return _assistant
