# Atomizer Simulation Tool

This project is a Windows desktop tool for exploring a perfume or cologne atomizer in 3D. It simulates droplet motion, visualizes particle spread, estimates airborne density, and synthesizes a spray sound preview based on the current atomizer settings.

## What it models

- A conical atomized spray with stochastic droplet sizes and launch directions.
- Gravity, aerodynamic drag, turbulence, and simple evaporation.
- Forward reach percentiles so you can estimate how far most particles travel.
- Radial spread and axial density so you can reason about concentration in the air.
- Procedural atomizer sound made from filtered noise and pressure-tone components.

This is an engineering exploration tool, not a CFD solver. It is designed to be fast and interactive on a typical Windows desktop.

## Windows setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
atomizer-sim
```

If your system blocks PowerShell script activation, use:

```powershell
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m atomizer_sim.main
```

## Controls

- `Start / Pause`: run or stop the time evolution.
- `Spray once`: instantly reset and fire a fresh one-shot plume using the current settings.
- `Reset plume`: regenerate particles using current settings.
- `Audio preview`: play the synthesized spray while the simulation is running.
- `Load custom mist sound`: load a short `mp3`, `wav`, `flac`, or `ogg` file and modulate it from the current atomizer settings.
- `Clear sound`: switch back to the procedural mist burst.
- `Nozzle type`: choose a nozzle pattern such as `Single Cone`, `Twin Jet`, `Triad Swirl`, or `Fan Sheet`.
- `Mean velocity`: initial droplet speed at the nozzle.
- `Spray angle`: cone angle of the plume.
- `Particle budget`: simulation particle count. Higher values are more stable but heavier.
- `Particle render size`: visual size multiplier for the mist points.
- `Flow rate`: influences density and sound intensity.
- `Droplet size`: median droplet diameter.
- `Turbulence`: random air disturbance strength.
- `Evaporation`: particle shrink rate over time.
- `Gravity`: strength of gravity in the world frame.
- `Aim yaw / Aim pitch`: rotate the nozzle and spray direction.
- `Time scale`: speed up or slow down the simulation clock.

## Custom audio notes

- The custom audio path is intended for short spray effects, ideally under 5 seconds.
- The app attempts to load `mp3` files directly through `soundfile`. If a specific `mp3` fails to decode on your machine, use `wav` as the fallback format.
- Uploaded sounds are played once per spray and modulated by velocity, flow, and turbulence.

## Interpreting results

- `P50 reach` is the forward distance reached by half of the active particles.
- `P90 reach` is the distance reached by most of the active particles.
- `Radial spread` is a 90th percentile radius of the cross-section.
- `Axial density` is a relative occupancy per distance slice.

## Project layout

- `src/atomizer_sim/simulation.py`: particle model and metrics.
- `src/atomizer_sim/audio.py`: real-time procedural spray audio.
- `src/atomizer_sim/ui.py`: PySide6 interface and OpenGL view.
- `src/atomizer_sim/main.py`: entry point.
