from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtWidgets

from .audio import AtomizerAudioEngine
from .simulation import AtomizerParameters, AtomizerSimulation


NOZZLE_TYPES: dict[str, tuple[str, dict[str, float | int | str]]] = {
    "Single Cone": (
        "Classic single-orifice atomizer with modest swirl.",
        {
            "nozzle_pattern": "single",
            "nozzle_orifices": 1,
            "orifice_ring_mm": 0.0,
            "swirl_strength": 0.08,
            "lateral_spread": 1.0,
            "vertical_spread": 0.8,
        },
    ),
    "Twin Jet": (
        "Two offset outlets that produce stronger side-to-side structure.",
        {
            "nozzle_pattern": "twin",
            "nozzle_orifices": 2,
            "orifice_ring_mm": 0.22,
            "swirl_strength": 0.10,
            "lateral_spread": 1.45,
            "vertical_spread": 0.72,
        },
    ),
    "Triad Swirl": (
        "Three-orifice swirl plate that produces corkscrew breakup in 3D.",
        {
            "nozzle_pattern": "triad",
            "nozzle_orifices": 3,
            "orifice_ring_mm": 0.30,
            "swirl_strength": 0.42,
            "lateral_spread": 1.25,
            "vertical_spread": 1.05,
        },
    ),
    "Fan Sheet": (
        "Multi-hole plate that flattens the plume into a wider sheet.",
        {
            "nozzle_pattern": "fan",
            "nozzle_orifices": 4,
            "orifice_ring_mm": 0.26,
            "swirl_strength": 0.05,
            "lateral_spread": 1.9,
            "vertical_spread": 0.45,
        },
    ),
}


ATOMIZER_PRESETS: dict[str, tuple[str, AtomizerParameters]] = {
    "Fine Mist": (
        "Balanced perfume sprayer with a narrow mist and short burst.",
        AtomizerParameters(
            nozzle_radius_mm=0.16,
            flow_rate_ml_s=0.09,
            spray_angle_deg=22.0,
            mean_velocity_m_s=18.0,
            particle_count=14000,
            droplet_mean_um=42.0,
            droplet_spread_um=14.0,
            turbulence=1.10,
            evaporation_rate=0.07,
            burst_duration_s=0.14,
            nozzle_pattern="single",
            nozzle_orifices=1,
            orifice_ring_mm=0.0,
            swirl_strength=0.08,
            lateral_spread=1.0,
            vertical_spread=0.8,
        ),
    ),
    "High Output": (
        "Wider cone with more air and flow for a stronger room-fill spray.",
        AtomizerParameters(
            nozzle_radius_mm=0.24,
            flow_rate_ml_s=0.18,
            spray_angle_deg=34.0,
            mean_velocity_m_s=21.0,
            particle_count=17000,
            droplet_mean_um=60.0,
            droplet_spread_um=22.0,
            turbulence=1.60,
            drag_coefficient=0.44,
            evaporation_rate=0.05,
            burst_duration_s=0.20,
            nozzle_pattern="twin",
            nozzle_orifices=2,
            orifice_ring_mm=0.22,
            swirl_strength=0.12,
            lateral_spread=1.5,
            vertical_spread=0.82,
        ),
    ),
    "Swirl Tube": (
        "Higher swirl and lower flow to mimic a tube-heavy, highly atomized head.",
        AtomizerParameters(
            nozzle_radius_mm=0.17,
            flow_rate_ml_s=0.08,
            spray_angle_deg=29.0,
            mean_velocity_m_s=16.0,
            particle_count=15000,
            droplet_mean_um=36.0,
            droplet_spread_um=11.0,
            turbulence=1.95,
            evaporation_rate=0.09,
            burst_duration_s=0.17,
            nozzle_pattern="triad",
            nozzle_orifices=3,
            orifice_ring_mm=0.30,
            swirl_strength=0.45,
            lateral_spread=1.25,
            vertical_spread=1.08,
        ),
    ),
    "Pressurized Jet": (
        "Fast, tight, high-pressure plume with sharper velocity separation.",
        AtomizerParameters(
            nozzle_radius_mm=0.14,
            flow_rate_ml_s=0.11,
            spray_angle_deg=16.0,
            mean_velocity_m_s=27.0,
            particle_count=12500,
            droplet_mean_um=33.0,
            droplet_spread_um=10.0,
            turbulence=0.85,
            drag_coefficient=0.42,
            evaporation_rate=0.06,
            burst_duration_s=0.12,
            nozzle_pattern="fan",
            nozzle_orifices=4,
            orifice_ring_mm=0.26,
            swirl_strength=0.06,
            lateral_spread=1.85,
            vertical_spread=0.44,
        ),
    ),
}


def _make_slider(
    minimum: int,
    maximum: int,
    value: int,
    on_change: Callable[[int], None],
) -> QtWidgets.QSlider:
    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(value)
    slider.valueChanged.connect(on_change)
    return slider


def _set_tooltip(*widgets: QtWidgets.QWidget, text: str) -> None:
    for widget in widgets:
        widget.setToolTip(text)


def _build_nozzle_item(length: float, radius: tuple[float, float], color: tuple[float, float, float, float]) -> gl.GLMeshItem:
    mesh = gl.MeshData.cylinder(rows=12, cols=32, radius=radius, length=length)
    item = gl.GLMeshItem(meshdata=mesh, smooth=True, shader="shaded", color=color, glOptions="opaque")
    item.translate(0.0, 0.0, -length)
    return item


class AtomizerWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("3D Atomizer Simulation Tool")
        self.resize(1460, 920)

        self.params = AtomizerParameters()
        self.simulation = AtomizerSimulation(self.params)
        self.audio = AtomizerAudioEngine()
        self.audio.error_occurred.connect(self._show_status)
        self.audio.start()
        self.is_running = False
        self.time_scale = 1.0
        self.particle_render_scale = 1.0

        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        self._apply_preset(self.preset_combo.currentText())

    def closeEvent(self, event) -> None:  # noqa: N802
        self.audio.stop()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=2.4, elevation=16, azimuth=-72)
        self.view.setBackgroundColor((11, 13, 20))

        axis = gl.GLAxisItem()
        axis.setSize(0.25, 0.25, 0.8)
        self.view.addItem(axis)

        self.nozzle_body = _build_nozzle_item(0.11, (0.018, 0.018), (0.72, 0.74, 0.78, 1.0))
        self.view.addItem(self.nozzle_body)

        self.nozzle_tip = _build_nozzle_item(0.05, (0.028, 0.009), (0.86, 0.81, 0.68, 1.0))
        self.view.addItem(self.nozzle_tip)

        self.press_stem = _build_nozzle_item(0.032, (0.006, 0.006), (0.60, 0.64, 0.70, 1.0))
        self.view.addItem(self.press_stem)

        self.press_cap = _build_nozzle_item(0.022, (0.026, 0.026), (0.90, 0.92, 0.96, 1.0))
        self.view.addItem(self.press_cap)

        spray_guide = np.array(
            [
                [0.0, 0.0, -0.12],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.22],
            ],
            dtype=float,
        )
        self.nozzle_axis = gl.GLLinePlotItem(
            pos=spray_guide,
            color=(0.98, 0.93, 0.62, 0.65),
            width=2.0,
            antialias=True,
            mode="line_strip",
        )
        self.view.addItem(self.nozzle_axis)

        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=np.ones((1, 4)), size=1.0, pxMode=True)
        self.view.addItem(self.scatter)
        root.addWidget(self.view, stretch=3)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)
        root.addWidget(side, stretch=2)

        controls = QtWidgets.QGroupBox("Atomizer controls")
        form = QtWidgets.QFormLayout(controls)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self._toggle_running)
        self.spray_button = QtWidgets.QPushButton("Spray once")
        self.spray_button.clicked.connect(self._spray_once)
        self.reset_button = QtWidgets.QPushButton("Reset plume")
        self.reset_button.clicked.connect(self._reset_simulation)
        self.audio_toggle = QtWidgets.QCheckBox("Audio preview")
        self.audio_toggle.setChecked(True)
        _set_tooltip(self.start_button, text="Start or pause the current plume without resetting the particle field.")
        _set_tooltip(self.spray_button, text="Instantly reset the plume and fire one fresh atomizer burst using the current settings.")
        _set_tooltip(self.reset_button, text="Clear the current particle field and rebuild the plume at the nozzle without starting motion.")
        _set_tooltip(self.audio_toggle, text="Enable or disable the mist sound for each spray burst.")

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.spray_button)
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.audio_toggle)

        button_widget = QtWidgets.QWidget()
        button_widget.setLayout(button_row)
        form.addRow(button_widget)

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(list(ATOMIZER_PRESETS))
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        form.addRow("Preview preset", self.preset_combo)
        _set_tooltip(self.preset_combo, text="Load a saved atomizer behavior profile with different flow, spread, velocity, and burst timing.")

        self.preset_description = QtWidgets.QLabel()
        self.preset_description.setWordWrap(True)
        form.addRow("Preset notes", self.preset_description)

        self.nozzle_combo = QtWidgets.QComboBox()
        self.nozzle_combo.addItems(list(NOZZLE_TYPES))
        self.nozzle_combo.currentTextChanged.connect(self._apply_nozzle_type)
        form.addRow("Nozzle type", self.nozzle_combo)
        _set_tooltip(self.nozzle_combo, text="Choose the internal nozzle arrangement to change cross-axis structure, swirl, and plume shape.")

        self.nozzle_description = QtWidgets.QLabel()
        self.nozzle_description.setWordWrap(True)
        form.addRow("Nozzle notes", self.nozzle_description)

        self.audio_button = QtWidgets.QPushButton("Load custom mist sound")
        self.audio_button.clicked.connect(self._load_custom_audio)
        self.clear_audio_button = QtWidgets.QPushButton("Clear sound")
        self.clear_audio_button.clicked.connect(self._clear_custom_audio)
        _set_tooltip(self.audio_button, text="Load a short audio file such as an mp3 or wav and use it as the one-shot mist sound for sprays.")
        _set_tooltip(self.clear_audio_button, text="Return to the built-in procedural mist burst sound.")
        audio_row = QtWidgets.QHBoxLayout()
        audio_row.addWidget(self.audio_button)
        audio_row.addWidget(self.clear_audio_button)
        audio_widget = QtWidgets.QWidget()
        audio_widget.setLayout(audio_row)
        form.addRow(audio_widget)

        self.audio_label = QtWidgets.QLabel()
        self.audio_label.setWordWrap(True)
        form.addRow("Mist sound", self.audio_label)
        _set_tooltip(self.audio_label, text="Shows the active sound source used for spray bursts.")

        self.velocity_label = QtWidgets.QLabel()
        self.velocity_slider = _make_slider(5, 40, 17, self._on_velocity_change)
        form.addRow(self.velocity_label, self.velocity_slider)
        _set_tooltip(self.velocity_label, self.velocity_slider, text="Initial droplet exit speed from the nozzle. Higher values produce a faster, more forceful plume.")

        self.angle_label = QtWidgets.QLabel()
        self.angle_slider = _make_slider(8, 55, 26, self._on_angle_change)
        form.addRow(self.angle_label, self.angle_slider)
        _set_tooltip(self.angle_label, self.angle_slider, text="Cone opening angle of the spray. Larger values widen the mist faster.")

        self.particle_label = QtWidgets.QLabel()
        self.particle_slider = _make_slider(3000, 28000, 12000, self._on_particle_change)
        form.addRow(self.particle_label, self.particle_slider)
        _set_tooltip(self.particle_label, self.particle_slider, text="Number of simulated particles. Higher values produce a fuller mist but cost more performance.")

        self.flow_label = QtWidgets.QLabel()
        self.flow_slider = _make_slider(3, 40, 12, self._on_flow_change)
        form.addRow(self.flow_label, self.flow_slider)
        _set_tooltip(self.flow_label, self.flow_slider, text="Liquid flow through the atomizer. Higher flow increases density and sound energy.")

        self.droplet_label = QtWidgets.QLabel()
        self.droplet_slider = _make_slider(20, 100, 55, self._on_droplet_change)
        form.addRow(self.droplet_label, self.droplet_slider)
        _set_tooltip(self.droplet_label, self.droplet_slider, text="Median droplet size in micrometers. Smaller droplets hang longer and feel more mist-like.")

        self.turbulence_label = QtWidgets.QLabel()
        self.turbulence_slider = _make_slider(0, 300, 135, self._on_turbulence_change)
        form.addRow(self.turbulence_label, self.turbulence_slider)
        _set_tooltip(self.turbulence_label, self.turbulence_slider, text="Random airflow disturbance strength. Higher values add breakup, wobble, and swirl to the cloud.")

        self.evaporation_label = QtWidgets.QLabel()
        self.evaporation_slider = _make_slider(0, 150, 60, self._on_evaporation_change)
        form.addRow(self.evaporation_label, self.evaporation_slider)
        _set_tooltip(self.evaporation_label, self.evaporation_slider, text="Rate at which droplets shrink over time. Higher values make the mist disappear faster.")

        self.gravity_label = QtWidgets.QLabel()
        self.gravity_slider = _make_slider(0, 180, 98, self._on_gravity_change)
        form.addRow(self.gravity_label, self.gravity_slider)
        _set_tooltip(self.gravity_label, self.gravity_slider, text="Downward acceleration applied to the particle field. Reduce it for slow floating mist, increase it for stronger falloff.")

        self.yaw_label = QtWidgets.QLabel()
        self.yaw_slider = _make_slider(-75, 75, 0, self._on_yaw_change)
        form.addRow(self.yaw_label, self.yaw_slider)
        _set_tooltip(self.yaw_label, self.yaw_slider, text="Rotate the nozzle left or right around the vertical axis.")

        self.pitch_label = QtWidgets.QLabel()
        self.pitch_slider = _make_slider(-45, 45, 0, self._on_pitch_change)
        form.addRow(self.pitch_label, self.pitch_slider)
        _set_tooltip(self.pitch_label, self.pitch_slider, text="Aim the nozzle upward or downward.")

        self.time_scale_label = QtWidgets.QLabel()
        self.time_scale_slider = _make_slider(10, 300, 100, self._on_time_scale_change)
        form.addRow(self.time_scale_label, self.time_scale_slider)
        _set_tooltip(self.time_scale_label, self.time_scale_slider, text="Change how fast simulated time advances. Slower values now preserve progressive emission instead of overemphasizing the initial cone.")

        self.render_size_label = QtWidgets.QLabel()
        self.render_size_slider = _make_slider(25, 300, 100, self._on_render_size_change)
        form.addRow(self.render_size_label, self.render_size_slider)
        _set_tooltip(self.render_size_label, self.render_size_slider, text="Visual size multiplier for the mist points only. This does not change the underlying physics.")

        side_layout.addWidget(controls)

        metrics = QtWidgets.QGroupBox("Reach, spread, density")
        metrics_layout = QtWidgets.QVBoxLayout(metrics)
        self.metrics_text = QtWidgets.QPlainTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMinimumHeight(150)
        metrics_layout.addWidget(self.metrics_text)
        side_layout.addWidget(metrics)
        _set_tooltip(self.metrics_text, text="Live summary of reach, spread, speed, and overall plume duration.")

        pg.setConfigOptions(antialias=True, background="#11151c", foreground="#e6edf3")
        self.reach_plot = pg.PlotWidget(title="Axial density by forward distance")
        self.reach_plot.showGrid(x=True, y=True, alpha=0.25)
        self.reach_curve = self.reach_plot.plot(pen=pg.mkPen("#78dce8", width=2))
        self.reach_plot.setLabel("left", "Relative density")
        self.reach_plot.setLabel("bottom", "Distance", units="m")
        side_layout.addWidget(self.reach_plot, stretch=1)
        self.reach_plot.setToolTip("Shows relative particle density by forward distance from the nozzle.")

        self.spread_plot = pg.PlotWidget(title="Radial envelope")
        self.spread_plot.showGrid(x=True, y=True, alpha=0.25)
        self.spread_curve = self.spread_plot.plot(pen=pg.mkPen("#ffbf69", width=2), fillLevel=0.0, brush=(255, 191, 105, 60))
        self.spread_plot.setLabel("left", "Radius", units="m")
        self.spread_plot.setLabel("bottom", "Distance", units="m")
        side_layout.addWidget(self.spread_plot, stretch=1)
        self.spread_plot.setToolTip("Shows the approximate widening of the plume as it travels away from the atomizer.")

        self.status_label = QtWidgets.QLabel("Ready")
        side_layout.addWidget(self.status_label)
        _set_tooltip(self.status_label, text="Current app status, including preset loads, spray triggers, and audio load messages.")

    def _sync_controls_from_params(self, *, push_slider_values: bool = False) -> None:
        if push_slider_values:
            self._set_slider_value(self.velocity_slider, int(round(self.params.mean_velocity_m_s)))
            self._set_slider_value(self.angle_slider, int(round(self.params.spray_angle_deg)))
            self._set_slider_value(self.particle_slider, int(round(self.params.particle_count)))
            self._set_slider_value(self.flow_slider, int(round(self.params.flow_rate_ml_s * 100.0)))
            self._set_slider_value(self.droplet_slider, int(round(self.params.droplet_mean_um)))
            self._set_slider_value(self.turbulence_slider, int(round(self.params.turbulence * 100.0)))
            self._set_slider_value(self.evaporation_slider, int(round(self.params.evaporation_rate * 1000.0)))
            self._set_slider_value(self.gravity_slider, int(round(self.params.gravity_m_s2 * 10.0)))
            self._set_slider_value(self.yaw_slider, int(round(self.params.aim_yaw_deg)))
            self._set_slider_value(self.pitch_slider, int(round(self.params.aim_pitch_deg)))

        self.velocity_label.setText(f"Mean velocity: {self.params.mean_velocity_m_s:.1f} m/s")
        self.angle_label.setText(f"Spray angle: {self.params.spray_angle_deg:.0f} deg")
        self.particle_label.setText(f"Particle budget: {self.params.particle_count}")
        self.flow_label.setText(f"Flow rate: {self.params.flow_rate_ml_s:.2f} ml/s")
        self.droplet_label.setText(f"Droplet median: {self.params.droplet_mean_um:.0f} um")
        self.turbulence_label.setText(f"Turbulence: {self.params.turbulence:.2f}")
        self.evaporation_label.setText(f"Evaporation: {self.params.evaporation_rate:.3f}")
        self.gravity_label.setText(f"Gravity: {self.params.gravity_m_s2:.1f} m/s^2")
        self.yaw_label.setText(f"Aim yaw: {self.params.aim_yaw_deg:.0f} deg")
        self.pitch_label.setText(f"Aim pitch: {self.params.aim_pitch_deg:.0f} deg")
        self.time_scale_label.setText(f"Time scale: {self.time_scale:.2f}x")
        self.render_size_label.setText(f"Particle render size: {self.particle_render_scale:.2f}x")
        self.audio_label.setText(self.audio.current_sample_name())

        preset_name = self.preset_combo.currentText()
        if preset_name in ATOMIZER_PRESETS:
            self.preset_description.setText(ATOMIZER_PRESETS[preset_name][0])

        nozzle_name = self.nozzle_combo.currentText()
        if nozzle_name in NOZZLE_TYPES:
            self.nozzle_description.setText(NOZZLE_TYPES[nozzle_name][0])

    def _set_slider_value(self, slider: QtWidgets.QSlider, value: int) -> None:
        slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(False)

    def _apply_preset(self, preset_name: str) -> None:
        if preset_name not in ATOMIZER_PRESETS:
            return

        self.params = replace(ATOMIZER_PRESETS[preset_name][1])
        self.simulation.reset(self.params)
        self.is_running = False
        self.start_button.setText("Start")
        self.audio.silence()
        self.nozzle_combo.blockSignals(True)
        nozzle_name = self._matching_nozzle_type()
        if nozzle_name is not None:
            self.nozzle_combo.setCurrentText(nozzle_name)
        self.nozzle_combo.blockSignals(False)
        self._sync_controls_from_params(push_slider_values=True)
        self._refresh_scene()
        self._refresh_metrics()
        self._update_nozzle_pose()
        self._show_status(f"Loaded {preset_name} preset")

    def _matching_nozzle_type(self) -> str | None:
        for nozzle_name, (_description, settings) in NOZZLE_TYPES.items():
            if (
                self.params.nozzle_pattern == settings["nozzle_pattern"]
                and self.params.nozzle_orifices == settings["nozzle_orifices"]
                and abs(self.params.orifice_ring_mm - float(settings["orifice_ring_mm"])) < 1e-6
            ):
                return nozzle_name
        return None

    def _apply_nozzle_type(self, nozzle_name: str) -> None:
        if nozzle_name not in NOZZLE_TYPES:
            return

        for field_name, value in NOZZLE_TYPES[nozzle_name][1].items():
            setattr(self.params, field_name, value)

        self.simulation.reset(self.params)
        self.audio.silence()
        self.is_running = False
        self.start_button.setText("Start")
        self._sync_controls_from_params(push_slider_values=True)
        self._refresh_scene()
        self._refresh_metrics()
        self._update_nozzle_pose()
        self._show_status(f"Applied {nozzle_name} nozzle")

    def _load_custom_audio(self) -> None:
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select mist sound",
            "",
            "Audio Files (*.mp3 *.wav *.flac *.ogg);;All Files (*)",
        )
        if not path:
            return

        try:
            sample_name = self.audio.load_custom_sample(path)
        except Exception as exc:
            self._show_status(f"Failed to load audio: {exc}")
            return

        self._sync_controls_from_params()
        self._show_status(f"Loaded custom mist sound: {sample_name}")

    def _clear_custom_audio(self) -> None:
        self.audio.clear_custom_sample()
        self._sync_controls_from_params()
        self._show_status("Using procedural mist sound")

    def _reset_simulation(self) -> None:
        self.simulation.reset(self.params)
        self.audio.silence()
        self._refresh_scene()
        self._refresh_metrics()
        self._update_nozzle_pose()
        self._show_status("Plume regenerated from current atomizer settings")

    def _toggle_running(self) -> None:
        self.is_running = not self.is_running
        self.start_button.setText("Pause" if self.is_running else "Start")
        if self.is_running:
            if self.simulation.time_s <= 1e-6 and self.audio_toggle.isChecked():
                self.audio.trigger_burst(
                    flow_rate_ml_s=self.params.flow_rate_ml_s,
                    velocity_m_s=self.params.mean_velocity_m_s,
                    turbulence=self.params.turbulence,
                    burst_duration_s=self.params.burst_duration_s,
                )
        else:
            self.audio.silence()
        self._sync_controls_from_params()
        self._show_status("Simulation running" if self.is_running else "Simulation paused")

    def _spray_once(self) -> None:
        self.simulation.reset(self.params)
        self.is_running = True
        self.start_button.setText("Pause")
        if self.audio_toggle.isChecked():
            self.audio.trigger_burst(
                flow_rate_ml_s=self.params.flow_rate_ml_s,
                velocity_m_s=self.params.mean_velocity_m_s,
                turbulence=self.params.turbulence,
                burst_duration_s=self.params.burst_duration_s,
            )
        self._refresh_scene()
        self._refresh_metrics()
        self._update_nozzle_pose()
        self._sync_controls_from_params()
        self._show_status("Triggered one-shot spray")

    def _tick(self) -> None:
        if not self.is_running:
            return

        self.simulation.step(self.time_scale / 60.0)
        self._refresh_scene()
        self._refresh_metrics()
        self._update_nozzle_pose()

        if not np.any(self.simulation.alive):
            self.is_running = False
            self.start_button.setText("Start")
            self.audio.silence()
            self._sync_controls_from_params()
            self._show_status("All droplets left the domain. Reset the plume to run again")

    def _update_nozzle_pose(self) -> None:
        burst = max(self.params.burst_duration_s, 1e-6)
        return_window = 0.08
        if self.simulation.time_s <= burst:
            press_amount = np.clip(self.simulation.time_s / burst, 0.0, 1.0)
        elif self.simulation.time_s <= burst + return_window:
            press_amount = 1.0 - np.clip((self.simulation.time_s - burst) / return_window, 0.0, 1.0)
        else:
            press_amount = 0.0
        press_amount = 1.0 - np.power(1.0 - press_amount, 2.0)
        press_travel = -0.018 * press_amount

        for item, length, x_offset, y_offset, z_offset in (
            (self.nozzle_body, 0.11, 0.0, 0.0, 0.0),
            (self.nozzle_tip, 0.05, 0.0, press_travel * 0.45, 0.0),
            (self.press_stem, 0.032, 0.0, 0.014, press_travel),
            (self.press_cap, 0.022, 0.0, 0.028, press_travel),
        ):
            item.resetTransform()
            item.translate(x_offset, y_offset, z_offset)
            item.translate(0.0, 0.0, -length)
            item.rotate(self.params.aim_yaw_deg, 0.0, 1.0, 0.0)
            item.rotate(-self.params.aim_pitch_deg, 1.0, 0.0, 0.0)

        forward, _right, up = self.simulation.spray_basis()
        nozzle_origin = up * (press_travel * 0.45)
        axis_points = np.array(
            [
                nozzle_origin - forward * 0.12,
                nozzle_origin,
                nozzle_origin + forward * 0.26,
            ]
        )
        self.nozzle_axis.setData(pos=axis_points, color=(0.98, 0.93, 0.62, 0.65), width=2.0, antialias=True, mode="line_strip")

    def _refresh_scene(self) -> None:
        pos = self.simulation.current_positions()
        if pos.size == 0:
            self.scatter.setData(pos=np.zeros((1, 3)), color=np.zeros((1, 4)), size=1.0, pxMode=True)
            return

        self.scatter.setData(
            pos=pos,
            color=self.simulation.current_colors(),
            size=self.simulation.current_point_sizes() * self.particle_render_scale,
            pxMode=True,
        )

    def _refresh_metrics(self) -> None:
        metrics = self.simulation.metrics()
        self.metrics_text.setPlainText(
            "\n".join(
                [
                    f"Active particles: {metrics.active_fraction * 100.0:5.1f}%",
                    f"P50 reach:       {metrics.p50_reach_m:5.2f} m",
                    f"P90 reach:       {metrics.p90_reach_m:5.2f} m",
                    f"P99 reach:       {metrics.p99_reach_m:5.2f} m",
                    f"Max reach:       {metrics.max_reach_m:5.2f} m",
                    f"Radial spread:   {metrics.radial_spread_m:5.2f} m",
                    f"Cloud width:     {metrics.cloud_width_m:5.2f} m",
                    f"Mean speed:      {metrics.mean_speed_m_s:5.2f} m/s",
                    f"Sim time:        {self.simulation.time_s:5.2f} s",
                ]
            )
        )

        bins = metrics.density_bins_m[:-1]
        self.reach_curve.setData(bins, metrics.density_profile)

        spread_x = np.array([0.0, metrics.p50_reach_m, metrics.p90_reach_m, metrics.max_reach_m])
        spread_y = np.array([
            0.0,
            metrics.radial_spread_m * 0.55,
            metrics.radial_spread_m,
            metrics.cloud_width_m * 0.5,
        ])
        self.spread_curve.setData(spread_x, spread_y)

    def _show_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_velocity_change(self, value: int) -> None:
        self.params.mean_velocity_m_s = float(value)
        self._sync_controls_from_params()

    def _on_angle_change(self, value: int) -> None:
        self.params.spray_angle_deg = float(value)
        self._sync_controls_from_params()

    def _on_particle_change(self, value: int) -> None:
        snapped = int(round(value / 250.0) * 250)
        self.params.particle_count = max(3000, snapped)
        self.particle_slider.blockSignals(True)
        self.particle_slider.setValue(self.params.particle_count)
        self.particle_slider.blockSignals(False)
        self._sync_controls_from_params()
        self._spray_once()

    def _on_flow_change(self, value: int) -> None:
        self.params.flow_rate_ml_s = value / 100.0
        self._sync_controls_from_params()

    def _on_droplet_change(self, value: int) -> None:
        self.params.droplet_mean_um = float(value)
        self.params.droplet_spread_um = max(10.0, value * 0.32)
        self._sync_controls_from_params()

    def _on_turbulence_change(self, value: int) -> None:
        self.params.turbulence = value / 100.0
        self._sync_controls_from_params()

    def _on_evaporation_change(self, value: int) -> None:
        self.params.evaporation_rate = value / 1000.0
        self._sync_controls_from_params()

    def _on_gravity_change(self, value: int) -> None:
        self.params.gravity_m_s2 = value / 10.0
        self._sync_controls_from_params()

    def _on_yaw_change(self, value: int) -> None:
        self.params.aim_yaw_deg = float(value)
        self._sync_controls_from_params()
        self._update_nozzle_pose()

    def _on_pitch_change(self, value: int) -> None:
        self.params.aim_pitch_deg = float(value)
        self._sync_controls_from_params()
        self._update_nozzle_pose()

    def _on_time_scale_change(self, value: int) -> None:
        self.time_scale = value / 100.0
        self._sync_controls_from_params()

    def _on_render_size_change(self, value: int) -> None:
        self.particle_render_scale = value / 100.0
        self._sync_controls_from_params()
        self._refresh_scene()