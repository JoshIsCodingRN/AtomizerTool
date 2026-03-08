from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PySide6 import QtCore

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - depends on local audio device stack
    sd = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover - depends on optional decoder support
    sf = None


class AtomizerAudioEngine(QtCore.QObject):
    error_occurred = QtCore.Signal(str)

    def __init__(self, sample_rate: int = 44100) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self._stream = None
        self._lock = threading.Lock()
        self._burst_total_samples = 0
        self._burst_remaining_samples = 0
        self._burst_intensity = 0.0
        self._burst_tone_hz = 380.0
        self._burst_texture = 0.5
        self._phase = 0.0
        self._pulse_phase = 0.0
        self._noise_state = 0.0
        self._active_buffer = np.zeros(0, dtype=np.float32)
        self._playhead = 0
        self._custom_sample = None
        self._custom_sample_name = "Procedural mist"

    def available(self) -> bool:
        return sd is not None

    def custom_audio_available(self) -> bool:
        return sf is not None

    def current_sample_name(self) -> str:
        return self._custom_sample_name

    def start(self) -> None:
        if not self.available() or self._stream is not None:
            return

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
                blocksize=0,
            )
            self._stream.start()
        except Exception as exc:  # pragma: no cover - depends on audio hardware
            self._stream = None
            self.error_occurred.emit(f"Audio output unavailable: {exc}")

    def stop(self) -> None:
        if self._stream is None:
            return

        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def load_custom_sample(self, file_path: str) -> str:
        if sf is None:
            raise RuntimeError("Custom audio loading requires the soundfile package.")

        samples, sample_rate = sf.read(file_path, dtype="float32", always_2d=False)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)

        max_samples = int(sample_rate * 5.0)
        samples = np.asarray(samples[:max_samples], dtype=np.float32)
        if samples.size == 0:
            raise RuntimeError("The selected file does not contain readable audio samples.")

        if sample_rate != self.sample_rate:
            duration = samples.size / sample_rate
            target_count = max(1, int(duration * self.sample_rate))
            source_x = np.linspace(0.0, 1.0, samples.size, dtype=np.float64)
            target_x = np.linspace(0.0, 1.0, target_count, dtype=np.float64)
            samples = np.interp(target_x, source_x, samples).astype(np.float32)

        peak = float(np.max(np.abs(samples)))
        if peak > 1e-5:
            samples = samples / peak

        with self._lock:
            self._custom_sample = samples
            self._custom_sample_name = Path(file_path).name

        return self._custom_sample_name

    def clear_custom_sample(self) -> None:
        with self._lock:
            self._custom_sample = None
            self._custom_sample_name = "Procedural mist"

    def trigger_burst(
        self,
        *,
        flow_rate_ml_s: float,
        velocity_m_s: float,
        turbulence: float,
        burst_duration_s: float,
    ) -> None:
        total_samples = max(1, int(self.sample_rate * max(burst_duration_s, 0.04)))
        with self._lock:
            self._burst_total_samples = total_samples
            self._burst_remaining_samples = total_samples
            self._burst_intensity = np.clip(flow_rate_ml_s * 1.5 + velocity_m_s * 0.02, 0.0, 1.0)
            self._burst_tone_hz = 180.0 + velocity_m_s * 6.0
            self._burst_texture = np.clip(0.30 + turbulence * 0.18, 0.2, 0.95)
            custom_sample = None if self._custom_sample is None else self._custom_sample.copy()

        if custom_sample is not None:
            buffer = self._render_custom_burst(custom_sample, flow_rate_ml_s, velocity_m_s, turbulence)
        else:
            buffer = self._render_procedural_burst(total_samples)

        with self._lock:
            self._active_buffer = buffer.astype(np.float32, copy=False)
            self._playhead = 0

    def silence(self) -> None:
        with self._lock:
            self._burst_total_samples = 0
            self._burst_remaining_samples = 0
            self._burst_intensity = 0.0
            self._active_buffer = np.zeros(0, dtype=np.float32)
            self._playhead = 0

    def _render_procedural_burst(self, total_samples: int) -> np.ndarray:
        t = np.arange(total_samples, dtype=np.float64) / self.sample_rate
        noise = np.random.normal(0.0, 1.0, total_samples)
        filtered = np.empty(total_samples, dtype=np.float64)
        state = self._noise_state
        cutoff = 0.80 - self._burst_texture * 0.12
        for index, sample in enumerate(noise):
            state = cutoff * state + (1.0 - cutoff) * sample
            filtered[index] = sample - state
        self._noise_state = state

        progress = np.linspace(0.0, 1.0, total_samples, dtype=np.float64)
        envelope = np.clip(progress / 0.08, 0.0, 1.0) * np.power(np.clip(1.0 - progress, 0.0, 1.0), 1.25)
        tone = np.sin(2.0 * np.pi * self._burst_tone_hz * t + self._pulse_phase) * 0.015
        return (filtered * (0.22 + self._burst_texture * 0.10) + tone) * envelope * self._burst_intensity

    def _render_custom_burst(
        self,
        sample: np.ndarray,
        flow_rate_ml_s: float,
        velocity_m_s: float,
        turbulence: float,
    ) -> np.ndarray:
        playback_speed = np.clip(0.82 + velocity_m_s / 32.0 + flow_rate_ml_s * 0.75, 0.75, 1.55)
        output_count = max(1, int(sample.size / playback_speed))
        source_index = np.linspace(0.0, sample.size - 1, output_count, dtype=np.float64)
        stretched = np.interp(source_index, np.arange(sample.size, dtype=np.float64), sample).astype(np.float64)

        progress = np.linspace(0.0, 1.0, output_count, dtype=np.float64)
        envelope = np.clip(progress / 0.05, 0.0, 1.0) * np.power(np.clip(1.0 - progress, 0.0, 1.0), 1.1)
        lowpass = np.empty_like(stretched)
        state = 0.0
        cutoff = 0.92 - np.clip(turbulence * 0.06, 0.02, 0.16)
        for index, value in enumerate(stretched):
            state = cutoff * state + (1.0 - cutoff) * value
            lowpass[index] = state
        airy_component = stretched - lowpass
        modulated = lowpass * 0.55 + airy_component * (0.90 + turbulence * 0.16)
        intensity = np.clip(0.45 + flow_rate_ml_s * 2.1 + velocity_m_s * 0.018, 0.35, 1.0)
        peak = max(np.max(np.abs(modulated)), 1e-5)
        return (modulated / peak) * envelope * intensity

    def _callback(self, outdata, frames, _time, _status) -> None:  # pragma: no cover - real-time callback
        with self._lock:
            buffer = self._active_buffer
            playhead = self._playhead

        if playhead >= buffer.size:
            outdata.fill(0.0)
            return

        sample_count = min(frames, buffer.size - playhead)
        outdata.fill(0.0)
        outdata[:sample_count, 0] = np.clip(buffer[playhead:playhead + sample_count], -0.30, 0.30).astype(np.float32)
        with self._lock:
            self._playhead = playhead + sample_count