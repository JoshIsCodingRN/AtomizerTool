from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LIQUID_DENSITY = 860.0


def _normalize(vector: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(vector)
    if magnitude <= 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / magnitude


def _direction_from_angles(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    return _normalize(
        np.array(
            [
                np.sin(yaw) * np.cos(pitch),
                np.sin(pitch),
                np.cos(yaw) * np.cos(pitch),
            ],
            dtype=np.float64,
        )
    )


def _basis_from_direction(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _normalize(direction)
    reference_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(np.dot(forward, reference_up)) > 0.97:
        reference_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = _normalize(np.cross(reference_up, forward))
    up = _normalize(np.cross(forward, right))
    return forward, right, up


@dataclass(slots=True)
class AtomizerParameters:
    nozzle_radius_mm: float = 0.18
    flow_rate_ml_s: float = 0.12
    spray_angle_deg: float = 26.0
    mean_velocity_m_s: float = 17.0
    particle_count: int = 12000
    droplet_mean_um: float = 55.0
    droplet_spread_um: float = 18.0
    turbulence: float = 1.35
    drag_coefficient: float = 0.47
    evaporation_rate: float = 0.06
    air_density: float = 1.225
    gravity_m_s2: float = 9.81
    burst_duration_s: float = 0.16
    aim_yaw_deg: float = 0.0
    aim_pitch_deg: float = 0.0
    nozzle_pattern: str = "single"
    nozzle_orifices: int = 1
    orifice_ring_mm: float = 0.0
    swirl_strength: float = 0.08
    lateral_spread: float = 1.0
    vertical_spread: float = 0.8
    emission_offset_mm: float = 34.0
    actuator_travel_mm: float = 7.0


@dataclass(slots=True)
class SimulationMetrics:
    active_fraction: float
    p50_reach_m: float
    p90_reach_m: float
    p99_reach_m: float
    max_reach_m: float
    radial_spread_m: float
    cloud_width_m: float
    mean_speed_m_s: float
    density_profile: np.ndarray
    density_bins_m: np.ndarray


class AtomizerSimulation:
    def __init__(self, params: AtomizerParameters | None = None, seed: int = 7) -> None:
        self.params = params or AtomizerParameters()
        self._rng = np.random.default_rng(seed)
        self.time_s = 0.0
        self.reset(self.params)

    def reset(self, params: AtomizerParameters | None = None) -> None:
        if params is not None:
            self.params = params

        count = self.params.particle_count
        nozzle_radius_m = self.params.nozzle_radius_mm / 1000.0
        emission_offset_m = self.params.emission_offset_mm / 1000.0
        spread_sigma = max(self.params.droplet_spread_um / max(self.params.droplet_mean_um, 1e-3), 0.08)
        forward, right, up = self.spray_basis()
        burst = max(self.params.burst_duration_s, 1e-3)
        actuator_travel_m = self.params.actuator_travel_mm / 1000.0

        self.release_time_s = self._rng.random(count) * burst
        press_progress = np.clip(self.release_time_s / burst, 0.0, 1.0)
        press_offset = -up[None, :] * (actuator_travel_m * press_progress)[:, None]

        nozzle_r = nozzle_radius_m * np.sqrt(self._rng.random(count))
        nozzle_phi = self._rng.random(count) * np.pi * 2.0
        orifice_count = max(1, int(self.params.nozzle_orifices))
        orifice_index = self._rng.integers(0, orifice_count, count)
        orifice_phase = (2.0 * np.pi * orifice_index / orifice_count) + (0.25 * (orifice_count > 2))
        ring_radius = self.params.orifice_ring_mm / 1000.0
        orifice_offsets = (
            np.cos(orifice_phase)[:, None] * right[None, :] + np.sin(orifice_phase)[:, None] * up[None, :]
        ) * ring_radius

        local_offset = (
            np.cos(nozzle_phi)[:, None] * right[None, :] * self.params.lateral_spread
            + np.sin(nozzle_phi)[:, None] * up[None, :] * self.params.vertical_spread
        ) * nozzle_r[:, None]

        self.positions = (
            forward[None, :] * emission_offset_m
            + orifice_offsets
            + local_offset
            + press_offset
        )

        half_angle = np.deg2rad(self.params.spray_angle_deg * 0.5)
        theta = half_angle * np.sqrt(self._rng.random(count))
        phi = self._rng.random(count) * np.pi * 2.0
        speed = np.clip(
            self._rng.normal(self.params.mean_velocity_m_s, self.params.mean_velocity_m_s * 0.18, count),
            0.5,
            None,
        )

        local_x = np.sin(theta) * np.cos(phi) * self.params.lateral_spread
        local_y = np.sin(theta) * np.sin(phi) * self.params.vertical_spread
        local_z = np.cos(theta)
        swirl = self.params.swirl_strength * (
            -np.sin(phi)[:, None] * right[None, :] + np.cos(phi)[:, None] * up[None, :]
        )
        direction = (
            right[None, :] * local_x[:, None]
            + up[None, :] * local_y[:, None]
            + forward[None, :] * local_z[:, None]
            + swirl * (0.3 + 0.7 * np.sin(theta))[:, None]
        )
        direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
        self.velocities = direction * speed[:, None]

        diameter_um = self._rng.lognormal(np.log(self.params.droplet_mean_um), spread_sigma, count)
        self.radius_m = np.clip(diameter_um * 0.5e-6, 4e-6, 140e-6)
        self.mass_kg = LIQUID_DENSITY * (4.0 / 3.0) * np.pi * np.power(self.radius_m, 3)
        self.released = np.zeros(count, dtype=bool)
        self.alive = np.zeros(count, dtype=bool)
        self.age_s = np.zeros(count, dtype=np.float64)
        self.time_s = 0.0

    def step(self, dt: float) -> None:
        remaining = max(0.0, float(dt))
        max_step = 1.0 / 180.0
        while remaining > 1e-9:
            step_dt = min(remaining, max_step)
            self._step_once(step_dt)
            remaining -= step_dt

    def _step_once(self, dt: float) -> None:
        next_time = self.time_s + dt
        release_now = (~self.released) & (self.release_time_s <= next_time)
        if np.any(release_now):
            self.released[release_now] = True
            self.alive[release_now] = True

        active_idx = np.flatnonzero(self.alive)
        if active_idx.size == 0:
            self.time_s = next_time
            return

        pos = self.positions[active_idx]
        vel = self.velocities[active_idx]
        radius = self.radius_m[active_idx]
        mass = self.mass_kg[active_idx]
        local_dt = np.full(active_idx.size, dt, dtype=np.float64)
        newly_released = release_now[active_idx]
        if np.any(newly_released):
            local_dt[newly_released] = np.maximum(next_time - self.release_time_s[active_idx[newly_released]], 0.0)

        speed = np.linalg.norm(vel, axis=1) + 1e-6
        direction = vel / speed[:, None]
        area = np.pi * np.square(radius)
        drag_mag = 0.5 * self.params.drag_coefficient * self.params.air_density * area * np.square(speed) / mass
        drag = -direction * drag_mag[:, None]

        gravity = np.zeros_like(vel)
        gravity[:, 1] = -self.params.gravity_m_s2

        turbulence = self._rng.normal(0.0, 1.0, size=vel.shape)
        turbulence *= self.params.turbulence * (0.65 + 0.05 * speed)[:, None]

        acceleration = drag + gravity + turbulence
        vel = vel + acceleration * local_dt[:, None]
        pos = pos + vel * local_dt[:, None]

        evaporation = self.params.evaporation_rate * (1.0 + 0.08 * speed) * local_dt * 1e-5
        radius = np.clip(radius - evaporation, 2.5e-6, None)
        mass = LIQUID_DENSITY * (4.0 / 3.0) * np.pi * np.power(radius, 3)

        self.positions[active_idx] = pos
        self.velocities[active_idx] = vel
        self.radius_m[active_idx] = radius
        self.mass_kg[active_idx] = mass
        self.age_s[active_idx] += local_dt
        self.time_s = next_time

        out_of_bounds = (
            (pos[:, 2] > 3.5)
            | (np.abs(pos[:, 0]) > 1.8)
            | (np.abs(pos[:, 1]) > 1.8)
            | (self.age_s[active_idx] > 2.8)
            | (radius <= 2.6e-6)
        )
        self.alive[active_idx[out_of_bounds]] = False

    def current_positions(self) -> np.ndarray:
        return self.positions[self.released & self.alive]

    def spray_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _basis_from_direction(_direction_from_angles(self.params.aim_yaw_deg, self.params.aim_pitch_deg))

    def current_point_sizes(self) -> np.ndarray:
        radius = self.radius_m[self.released & self.alive]
        return np.clip(radius * 38000.0, 0.55, 1.75)

    def current_colors(self) -> np.ndarray:
        active_mask = self.released & self.alive
        positions = self.positions[active_mask]
        if positions.size == 0:
            return np.empty((0, 4), dtype=np.float32)

        distance = np.linalg.norm(positions, axis=1)
        reference_distance = max(np.percentile(distance, 96), 0.25)
        normalized = np.clip(distance / reference_distance, 0.0, 1.0)
        alpha = np.clip(0.16 + normalized * 0.24 + self.radius_m[active_mask] * 2200.0, 0.16, 0.62)
        red = np.interp(normalized, [0.0, 0.22, 0.5, 0.76, 1.0], [0.18, 0.22, 0.36, 0.96, 1.0])
        green = np.interp(normalized, [0.0, 0.22, 0.5, 0.76, 1.0], [0.86, 0.98, 0.86, 0.52, 0.24])
        blue = np.interp(normalized, [0.0, 0.22, 0.5, 0.76, 1.0], [1.0, 0.92, 0.42, 0.16, 0.10])
        colors = np.column_stack(
            (
                red,
                green,
                blue,
                alpha,
            )
        )
        return colors.astype(np.float32)

    def spray_audio_active(self) -> bool:
        return self.time_s <= self.params.burst_duration_s

    def metrics(self) -> SimulationMetrics:
        active_mask = self.released & self.alive
        active_positions = self.positions[active_mask]
        active_velocities = self.velocities[active_mask]

        if active_positions.size == 0:
            bins = np.linspace(0.0, 2.5, 26)
            return SimulationMetrics(
                active_fraction=0.0,
                p50_reach_m=0.0,
                p90_reach_m=0.0,
                p99_reach_m=0.0,
                max_reach_m=0.0,
                radial_spread_m=0.0,
                cloud_width_m=0.0,
                mean_speed_m_s=0.0,
                density_profile=np.zeros(bins.size - 1, dtype=np.float64),
                density_bins_m=bins,
            )

        forward = np.clip(active_positions[:, 2], 0.0, None)
        radial = np.linalg.norm(active_positions[:, :2], axis=1)
        speeds = np.linalg.norm(active_velocities, axis=1)

        bins = np.linspace(0.0, 2.5, 26)
        hist, _ = np.histogram(forward, bins=bins)
        volume_proxy = np.maximum(1e-4, np.diff(bins) * np.square(0.03 + 0.18 * bins[:-1]))
        density = hist / volume_proxy
        density /= max(density.max(), 1.0)

        return SimulationMetrics(
            active_fraction=float(np.mean(active_mask)),
            p50_reach_m=float(np.percentile(forward, 50)),
            p90_reach_m=float(np.percentile(forward, 90)),
            p99_reach_m=float(np.percentile(forward, 99)),
            max_reach_m=float(np.max(forward)),
            radial_spread_m=float(np.percentile(radial, 90)),
            cloud_width_m=float(np.percentile(radial, 99) * 2.0),
            mean_speed_m_s=float(np.mean(speeds)),
            density_profile=density,
            density_bins_m=bins,
        )