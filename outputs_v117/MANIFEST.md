# Tensor Simulation v1.17

## Run metadata
- **timestamp**: 2026-03-28T22:44:38.651628Z
- **seed**: 42

## Core parameters
- **max_mpc**: 120000.0
- **shell_mpc**: 500.0
- **capacity**: 13
- **occupancy_mode**: fixed
- **alpha_max**: 1.5

## Omega profile
- **omega0**: 1.0
- **r0**: 500.0
- **p**: 1.15

## Tick evolution
- **turn_ticks**: 129600
- **dt**: 0.02
- **rot_steps**: 6000

## Trichter
- **trichter_strength**: 1.0
- **ell_trichter**: 20000.0

## Cone interactions
- **cone_mode**: off

## Output summary
- **shells**: 241
- **nodes**: 3133

## Files
- nodes.csv
- shells.csv
- fig_omega_vs_r.png
- fig_alpha_vs_r.png
- fig_omega_vs_alpha.png
- fig_shell_occupancy.png
- fig_xy_nodes.png (only if geometry enabled)
- fig_trichter_xz_slice.png (only if geometry enabled)
- fig_trichter_depth_hist.png
- fig_cone_events_counts.png (if cone_mode != off)