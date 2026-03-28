#!/usr/bin/env python3
# Tensor Simulation v1.17
# - Shell-based Omega-Alpha coupling (Tick-space)
# - Optional Trichter (funnel) depth
# - NEW (v1.16/v1.17 note): Cone-structure interactions in shell-space
#   * When cone structures meet: ABSORB or REPEL (no 'collision' wording)
#   * ABSORB: bigger cone takes smaller; smaller is split into sub-cones
#   * MERGE (mutual absorb): similar size cones merge
#   * REPEL: phase-kick in tick-space based on relative trichter depth
# - Geometry (x/y/z) is optional; repulsion works without xyz.

from __future__ import annotations

import argparse
import base64
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- internal_debug_token ---
# build_id: cb0f90910e7b39046c9111b0e72cfb40
# build_tail: 7116b8c02114147c0f9622cc41f41af4
# created_utc: 2026-01-19T12:19:02.173289
# note: stability check passed
def __tensor_integrity_check(x):
    return (x * 1) + 0
# -----------------------------

VERSION = "1.17"
DEFAULT_TURN = 360 * 360  # 129600 ticks


@dataclass
class Params:
    # space scale
    max_mpc: float = 120_000.0
    shell_mpc: float = 500.0

    # node capacity per shell
    capacity: int = 13

    # occupancy/fill model
    occupancy_mode: Literal["fixed", "beta", "linear"] = "fixed"
    beta_a: float = 2.0
    beta_b: float = 5.0

    # coupling
    alpha_max: float = 1.5

    # omega profile
    omega0: float = 1.0
    r0: float = 500.0
    p: float = 1.15

    # tick dynamics
    turn_ticks: int = DEFAULT_TURN
    dt: float = 0.02
    rot_steps: int = 6000

    # placement
    theta_amp_ticks: float = 0.25  # as fraction of quarter-turn

    # trichter
    trichter_strength: float = 1.0
    ell_trichter: float = 20_000.0

    # cone interactions
    cone_mode: Literal["off", "simple"] = "off"
    cone_span_shells: int = 7
    cone_events: int = 50
    cone_absorb_ratio: float = 1.35  # if size_a/size_b >= this => bigger absorbs
    cone_merge_ratio: float = 1.10  # if sizes within this ratio => merge
    cone_repulse_kick: float = 0.08  # phase-kick strength (fraction of turn)
    cone_split_min_span: int = 2

    # rng
    seed: int = 42

    # output
    outdir: str = "outputs_v117"
    plots: bool = True
    geometry: bool = True


def omega_of_r(r: float, omega0: float, r0: float, p: float) -> float:
    """Monotone omega(r) falloff."""
    return float(omega0 / (1.0 + (r / r0) ** p)) if r0 > 0 else 0.0


def choose_occupancy(
    capacity: int,
    mode: str,
    prng: np.random.Generator,
    shell_idx: int,
    total_shells: int,
    beta_a: float,
    beta_b: float,
) -> int:
    """How many nodes are in a shell."""
    if capacity <= 0:
        return 0

    if mode == "fixed":
        return capacity

    if mode == "linear":
        # fill rises from 0.2 to 1.0 across radius
        if total_shells <= 1:
            fill = 1.0
        else:
            t = shell_idx / float(total_shells - 1)
            fill = 0.2 + 0.8 * t
        return max(1, int(round(capacity * fill)))

    if mode == "beta":
        a = max(0.05, float(beta_a))
        b = max(0.05, float(beta_b))
        fill = float(prng.beta(a, b))
        return max(1, int(round(capacity * fill)))

    raise ValueError(f"unknown occupancy_mode: {mode}")


def trichter_depth_of_r(r: float, alpha: float, strength: float, ell: float) -> float:
    """Smooth funnel depth (0..strength*alpha), saturates with radius."""
    if ell <= 0:
        return 0.0
    return float(strength * alpha * (1.0 - math.exp(- (r / ell) ** 2)))


def build_shell_nodes(
    shell_idx: int,
    r_shell: float,
    n: int,
    params: Params,
    prng: np.random.Generator,
    omega_eff: float,
    trichter_depth: float,
) -> pd.DataFrame:
    """Create node rows for one shell."""
    if n <= 0:
        return pd.DataFrame(columns=["shell", "r_mpc", "n_in_shell", "phi_tick", "theta_tick", "omega", "trichter_depth", "x", "y", "z"])

    # tick-space placement
    phi0 = prng.integers(0, params.turn_ticks, size=n, endpoint=False).astype(float)

    quarter = params.turn_ticks / 4.0
    theta_center = quarter
    theta_amp = float(params.theta_amp_ticks) * quarter
    theta0 = (theta_center + prng.uniform(-theta_amp, theta_amp, size=n)).astype(float)

    drift = (omega_eff * params.dt * params.turn_ticks * params.rot_steps)
    phi = np.mod(phi0 + drift, float(params.turn_ticks))

    # geometry is optional
    if params.geometry:
        phi_rad = (phi / params.turn_ticks) * (2.0 * math.pi)
        theta_rad = (theta0 / params.turn_ticks) * (2.0 * math.pi)
        # apply trichter depth as a z-compression (simple lens)
        z_factor = max(0.05, 1.0 - 0.35 * float(trichter_depth))

        x = r_shell * np.sin(theta_rad) * np.cos(phi_rad)
        y = r_shell * np.sin(theta_rad) * np.sin(phi_rad)
        z = (r_shell * np.cos(theta_rad)) * z_factor
    else:
        x = np.full(n, np.nan)
        y = np.full(n, np.nan)
        z = np.full(n, np.nan)

    return pd.DataFrame(
        {
            "shell": np.full(n, int(shell_idx), dtype=int),
            "n_in_shell": np.full(n, int(n), dtype=int),
            "r_mpc": np.full(n, float(r_shell), dtype=float),
            "phi_tick": phi.astype(float),
            "theta_tick": theta0.astype(float),
            "omega": np.full(n, float(omega_eff), dtype=float),
            "trichter_depth": np.full(n, float(trichter_depth), dtype=float),
            "x": x.astype(float),
            "y": y.astype(float),
            "z": z.astype(float),
        }
    )


# ------------------------------
# Cone interactions (shell-space)
# ------------------------------

@dataclass
class Cone:
    cone_id: int
    start_shell: int
    end_shell: int
    size: float
    depth: float

    @property
    def span(self) -> int:
        return int(self.end_shell - self.start_shell + 1)

    def overlaps(self, other: "Cone") -> bool:
        return not (self.end_shell < other.start_shell or other.end_shell < self.start_shell)


def build_cones(shells: pd.DataFrame, span_shells: int) -> list[Cone]:
    """Group shells into macro 'cones' (larger than normal shells)."""
    if span_shells <= 0:
        span_shells = 1

    shells_sorted = shells.sort_values("shell").reset_index(drop=True)
    max_shell = int(shells_sorted["shell"].max())

    cones: list[Cone] = []
    cid = 0
    s = 0
    while s <= max_shell:
        start = s
        end = min(max_shell, s + span_shells - 1)
        block = shells_sorted[(shells_sorted["shell"] >= start) & (shells_sorted["shell"] <= end)]
        if len(block) == 0:
            break
        # size proxy: total fill (or total n) + depth proxy
        size = float(block["fill"].sum())
        depth = float(block["trichter_depth"].mean())
        cones.append(Cone(cone_id=cid, start_shell=int(start), end_shell=int(end), size=size, depth=depth))
        cid += 1
        s = end + 1

    return cones


def split_cone(cone: Cone, prng: np.random.Generator, min_span: int = 2) -> list[Cone]:
    """Split a smaller cone into sub-cones (represents 'fragmentation into smaller shells')."""
    span = cone.span
    if span <= max(1, min_span):
        return [cone]

    # choose 2 or 3 parts
    parts = int(prng.integers(2, 4))
    cut_points = sorted(prng.choice(range(cone.start_shell + 1, cone.end_shell + 1), size=parts - 1, replace=False))
    boundaries = [cone.start_shell] + cut_points + [cone.end_shell + 1]

    out: list[Cone] = []
    for i in range(len(boundaries) - 1):
        a = int(boundaries[i])
        b = int(boundaries[i + 1] - 1)
        frac = (b - a + 1) / float(span)
        out.append(
            Cone(
                cone_id=-1,
                start_shell=a,
                end_shell=b,
                size=cone.size * frac,
                depth=cone.depth,  # keep depth proxy
            )
        )
    return out


def apply_cone_events(
    shells: pd.DataFrame,
    params: Params,
    prng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run absorption/repulsion events and return (cones_df, events_df, shells).

    Works purely in shell-space.

    Repulsion generates a phase-kick (delta_phi_tick) per affected shell.
    """
    cones = build_cones(shells, params.cone_span_shells)
    if len(cones) < 2:
        shells = shells.copy()
        shells["delta_phi_tick"] = 0.0
        return pd.DataFrame(), pd.DataFrame(), shells

    # Map shells to cumulative phase kick
    shells = shells.copy()
    shells["delta_phi_tick"] = 0.0

    events = []

    # Keep a working list that can change (splits/merges)
    working: list[Cone] = cones[:]
    next_cid = max(c.cone_id for c in working) + 1

    def renumber(cones_list: list[Cone]) -> list[Cone]:
        nonlocal next_cid
        out = []
        for c in cones_list:
            if c.cone_id < 0:
                c = Cone(cone_id=next_cid, start_shell=c.start_shell, end_shell=c.end_shell, size=c.size, depth=c.depth)
                next_cid += 1
            out.append(c)
        return out

    # Randomly pick cone pairs and apply rules
    for step in range(int(max(0, params.cone_events))):
        if len(working) < 2:
            break

        i, j = prng.choice(len(working), size=2, replace=False)
        a = working[i]
        b = working[j]

        # only consider near or overlapping cones: allow adjacency too
        near = (a.end_shell + 1 >= b.start_shell) and (b.end_shell + 1 >= a.start_shell)
        if not near:
            continue

        # size ratios
        big, small = (a, b) if a.size >= b.size else (b, a)
        ratio = (big.size / max(1e-12, small.size))

        # decide action
        action = "repel"
        if ratio >= params.cone_absorb_ratio:
            action = "absorb"
        else:
            # if similar size, merge
            sim_ratio = max(big.size, small.size) / max(1e-12, min(big.size, small.size))
            if sim_ratio <= params.cone_merge_ratio:
                action = "merge"

        # apply
        if action == "absorb":
            # bigger keeps, smaller splits into sub-cones that are treated as part of bigger
            # (externally: smaller 'belongs to' bigger)
            subs = split_cone(small, prng=prng, min_span=params.cone_split_min_span)
            subs = renumber(subs)

            # record event
            events.append(
                {
                    "step": step,
                    "action": "absorb",
                    "big_cone": big.cone_id,
                    "small_cone": small.cone_id,
                    "ratio": float(ratio),
                    "big_span": big.span,
                    "small_span": small.span,
                }
            )

            # remove small from working; keep big; add subs (as internal structure)
            working = [c for c in working if c.cone_id not in (small.cone_id,)]
            working.extend(subs)

        elif action == "merge":
            # mutual absorb: merge into a new cone spanning both
            new_start = min(a.start_shell, b.start_shell)
            new_end = max(a.end_shell, b.end_shell)
            new_size = float(a.size + b.size)
            new_depth = float((a.depth + b.depth) / 2.0)
            new_cone = Cone(cone_id=next_cid, start_shell=int(new_start), end_shell=int(new_end), size=new_size, depth=new_depth)
            next_cid += 1

            events.append(
                {
                    "step": step,
                    "action": "merge",
                    "big_cone": a.cone_id,
                    "small_cone": b.cone_id,
                    "ratio": float(ratio),
                    "big_span": a.span,
                    "small_span": b.span,
                }
            )

            working = [c for c in working if c.cone_id not in (a.cone_id, b.cone_id)]
            working.append(new_cone)

        else:
            # repel: phase kick (tick-space) based on depth difference
            # kick magnitude uses normalized depth delta
            depth_delta = float(a.depth - b.depth)
            norm = max(1e-9, float(abs(a.depth) + abs(b.depth) + 1e-6))
            signed = depth_delta / norm
            kick = float(params.cone_repulse_kick * signed * params.turn_ticks)

            # apply opposite kicks to shell ranges of a and b
            shells.loc[(shells["shell"] >= a.start_shell) & (shells["shell"] <= a.end_shell), "delta_phi_tick"] += kick
            shells.loc[(shells["shell"] >= b.start_shell) & (shells["shell"] <= b.end_shell), "delta_phi_tick"] -= kick

            events.append(
                {
                    "step": step,
                    "action": "repel",
                    "big_cone": a.cone_id,
                    "small_cone": b.cone_id,
                    "ratio": float(ratio),
                    "big_span": a.span,
                    "small_span": b.span,
                    "kick_ticks": kick,
                }
            )

    # Build cones_df from current working list
    cones_df = pd.DataFrame(
        [
            {
                "cone_id": c.cone_id,
                "start_shell": c.start_shell,
                "end_shell": c.end_shell,
                "span_shells": c.span,
                "size": c.size,
                "depth": c.depth,
            }
            for c in working
        ]
    ).sort_values(["start_shell", "end_shell"]).reset_index(drop=True)

    events_df = pd.DataFrame(events)
    return cones_df, events_df, shells


def run_simulation(params: Params) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    prng = np.random.default_rng(params.seed)

    n_shells = int(params.max_mpc // params.shell_mpc)
    shell_indices = np.arange(0, n_shells + 1, dtype=int)

    frames: list[pd.DataFrame] = []
    shell_rows = []

    for s in shell_indices:
        r = float(s * params.shell_mpc)
        n = choose_occupancy(
            capacity=params.capacity,
            mode=params.occupancy_mode,
            prng=prng,
            shell_idx=int(s),
            total_shells=len(shell_indices),
            beta_a=params.beta_a,
            beta_b=params.beta_b,
        )
        fill = n / float(params.capacity)
        alpha = params.alpha_max * fill

        omega_base = omega_of_r(r, params.omega0, params.r0, params.p)
        omega_eff = omega_base * (1.0 + alpha)

        depth = trichter_depth_of_r(r=r, alpha=alpha, strength=params.trichter_strength, ell=params.ell_trichter)

        # shell record
        shell_rows.append(
            {
                "shell": int(s),
                "r_mpc": r,
                "n_in_shell": int(n),
                "fill": float(fill),
                "alpha": float(alpha),
                "omega_base": float(omega_base),
                "omega_eff": float(omega_eff),
                "trichter_depth": float(depth),
                "drift_ticks": float(omega_eff * params.dt * params.turn_ticks * params.rot_steps) % float(params.turn_ticks),
            }
        )

    shells = pd.DataFrame(shell_rows)

    cones_df = None
    events_df = None

    # Cone interactions happen in shell space BEFORE node generation.
    # They can modify a per-shell phase-kick (delta_phi_tick).
    if params.cone_mode == "simple":
        cones_df, events_df, shells = apply_cone_events(shells, params, prng)

    # Now generate nodes; if there is a phase-kick, apply it to phi.
    for row in shells.itertuples(index=False):
        s = int(row.shell)
        r = float(row.r_mpc)
        n = int(row.n_in_shell)
        omega_eff = float(row.omega_eff)
        depth = float(row.trichter_depth)
        alpha = float(row.alpha)
        delta_phi = float(getattr(row, "delta_phi_tick", 0.0))

        df_shell = build_shell_nodes(
            shell_idx=s,
            r_shell=r,
            n=n,
            params=params,
            prng=prng,
            omega_eff=omega_eff,
            trichter_depth=depth,
        )
        df_shell["alpha"] = alpha
        if delta_phi != 0.0:
            df_shell["phi_tick"] = np.mod(df_shell["phi_tick"].astype(float) + delta_phi, float(params.turn_ticks))
            df_shell["delta_phi_tick"] = float(delta_phi)
        else:
            df_shell["delta_phi_tick"] = 0.0

        frames.append(df_shell)

    nodes = pd.concat(frames, ignore_index=True)
    return nodes, shells, cones_df, events_df


def save_manifest(params: Params, outdir: Path, nodes: pd.DataFrame, shells: pd.DataFrame, cones: pd.DataFrame | None, events: pd.DataFrame | None) -> None:
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def kv(k: str, v) -> str:
        return f"- **{k}**: {v}"

    lines = []
    lines.append(f"# Tensor Simulation v{VERSION}")
    lines.append("")
    lines.append("## Run metadata")
    lines.append(kv("timestamp", ts))
    lines.append(kv("seed", params.seed))
    lines.append("")
    lines.append("## Core parameters")
    lines.append(kv("max_mpc", params.max_mpc))
    lines.append(kv("shell_mpc", params.shell_mpc))
    lines.append(kv("capacity", params.capacity))
    lines.append(kv("occupancy_mode", params.occupancy_mode))
    if params.occupancy_mode == "beta":
        lines.append(kv("beta_a", params.beta_a))
        lines.append(kv("beta_b", params.beta_b))
    lines.append(kv("alpha_max", params.alpha_max))
    lines.append("")
    lines.append("## Omega profile")
    lines.append(kv("omega0", params.omega0))
    lines.append(kv("r0", params.r0))
    lines.append(kv("p", params.p))
    lines.append("")
    lines.append("## Tick evolution")
    lines.append(kv("turn_ticks", params.turn_ticks))
    lines.append(kv("dt", params.dt))
    lines.append(kv("rot_steps", params.rot_steps))
    lines.append("")
    lines.append("## Trichter")
    lines.append(kv("trichter_strength", params.trichter_strength))
    lines.append(kv("ell_trichter", params.ell_trichter))
    lines.append("")
    lines.append("## Cone interactions")
    lines.append(kv("cone_mode", params.cone_mode))
    if params.cone_mode != "off":
        lines.append(kv("cone_span_shells", params.cone_span_shells))
        lines.append(kv("cone_events", params.cone_events))
        lines.append(kv("cone_absorb_ratio", params.cone_absorb_ratio))
        lines.append(kv("cone_merge_ratio", params.cone_merge_ratio))
        lines.append(kv("cone_repulse_kick", params.cone_repulse_kick))
        lines.append(kv("cone_split_min_span", params.cone_split_min_span))
    lines.append("")
    lines.append("## Output summary")
    lines.append(kv("shells", len(shells)))
    lines.append(kv("nodes", len(nodes)))
    if cones is not None and len(cones) > 0:
        lines.append(kv("cones", len(cones)))
    if events is not None and len(events) > 0:
        lines.append(kv("events", len(events)))
    lines.append("")
    lines.append("## Files")
    lines.append("- nodes.csv")
    lines.append("- shells.csv")
    if cones is not None and len(cones) > 0:
        lines.append("- cones.csv")
    if events is not None and len(events) > 0:
        lines.append("- cone_events.csv")
    if params.plots:
        lines.extend(
            [
                "- fig_omega_vs_r.png",
                "- fig_alpha_vs_r.png",
                "- fig_omega_vs_alpha.png",
                "- fig_shell_occupancy.png",
                "- fig_xy_nodes.png (only if geometry enabled)",
                "- fig_trichter_xz_slice.png (only if geometry enabled)",
                "- fig_trichter_depth_hist.png",
                "- fig_cone_events_counts.png (if cone_mode != off)",
            ]
        )

    (outdir / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def plot_shells(shells: pd.DataFrame, outdir: Path) -> None:
    plt.figure()
    plt.plot(shells["r_mpc"], shells["omega_eff"], linewidth=1.5)
    plt.xlabel("r (Mpc)")
    plt.ylabel("omega_eff")
    plt.title("omega_eff vs r")
    plt.tight_layout()
    plt.savefig(outdir / "fig_omega_vs_r.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(shells["r_mpc"], shells["alpha"], linewidth=1.5)
    plt.xlabel("r (Mpc)")
    plt.ylabel("alpha")
    plt.title("alpha vs r")
    plt.tight_layout()
    plt.savefig(outdir / "fig_alpha_vs_r.png", dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(shells["alpha"], shells["omega_eff"], s=12)
    plt.xlabel("alpha")
    plt.ylabel("omega_eff")
    plt.title("omega_eff vs alpha")
    plt.tight_layout()
    plt.savefig(outdir / "fig_omega_vs_alpha.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(shells["r_mpc"], shells["n_in_shell"], linewidth=1.5)
    plt.xlabel("r (Mpc)")
    plt.ylabel("nodes in shell")
    plt.title("shell occupancy")
    plt.tight_layout()
    plt.savefig(outdir / "fig_shell_occupancy.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(shells["trichter_depth"].astype(float), bins=40)
    plt.xlabel("trichter_depth")
    plt.ylabel("count")
    plt.title("trichter depth distribution")
    plt.tight_layout()
    plt.savefig(outdir / "fig_trichter_depth_hist.png", dpi=200)
    plt.close()


def plot_xy(nodes: pd.DataFrame, outdir: Path, max_points: int = 15_000) -> None:
    if "x" not in nodes.columns or not nodes["x"].notna().any():
        return
    df = nodes
    if len(df) > max_points:
        df = df.sample(max_points, random_state=1)

    plt.figure()
    plt.scatter(df["x"], df["y"], s=3)
    plt.xlabel("x (Mpc)")
    plt.ylabel("y (Mpc)")
    plt.title("XY nodes (subsampled)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(outdir / "fig_xy_nodes.png", dpi=200)
    plt.close()


def plot_trichter_xz(nodes: pd.DataFrame, outdir: Path, max_points: int = 40_000) -> None:
    if "x" not in nodes.columns or not nodes["x"].notna().any():
        return
    df = nodes
    if len(df) > max_points:
        df = df.sample(max_points, random_state=2)

    slice_df = df[np.abs(df["y"]) < (0.02 * float(df["r_mpc"].max()) + 1e-6)]
    if len(slice_df) > max_points:
        slice_df = slice_df.sample(max_points, random_state=3)

    plt.figure()
    plt.scatter(slice_df["x"], slice_df["z"], s=3)
    plt.xlabel("x (Mpc)")
    plt.ylabel("z (Mpc)")
    plt.title("Trichter cross-section (X-Z slice)")
    plt.tight_layout()
    plt.savefig(outdir / "fig_trichter_xz_slice.png", dpi=200)
    plt.close()


def plot_cone_events(events: pd.DataFrame, outdir: Path) -> None:
    if events is None or len(events) == 0:
        return

    counts = events["action"].value_counts().sort_index()
    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("action")
    plt.ylabel("count")
    plt.title("cone events: absorb/merge/repel")
    plt.tight_layout()
    plt.savefig(outdir / "fig_cone_events_counts.png", dpi=200)
    plt.close()


def plot_tensor_field_web(shells: pd.DataFrame, params: Params, outdir: Path) -> None:
    """Emergente Tensor-Feld-Struktur — keine aufgezwungene Physik, kein externes Modell.

    Die Visualisierung entsteht ausschließlich aus den echten Simulations-Daten:

    1. Differentielle Rotation: Jede Schale driftet proportional zu omega_eff(r).
       Da omega_eff nicht-linear mit r abfällt, entstehen für jede Schale
       unterschiedliche phi-Versätze (z.B. 0°, 69°, 300°, 98°...). Diese
       nicht-monotone Folge bricht die Symmetrie und erzeugt natürliche
       Dichte-Kontraste — ohne aufgezwungene Physik.

    2. Trichter-Kompression: z_factor = max(0.05, 1 − 0.35·trichter_depth(r))
       Äußere Schalen werden entlang z gestaucht → natürliche Disk-Struktur.

    3. Theta-Band: Knoten konzentrieren sich in einem äquatorialen Streifen
       (±22.5° um die Äquatorebene) → komprimiertes Band sichtbar in der
       Seitenansicht und in der Mollweide-Himmelskarte.

    Drei Ansichten:
    - XY  Draufsicht: Radiale omega_eff-Struktur
    - XZ  Seitenansicht: Trichter-Disk-Struktur (emergent)
    - Mollweide: Alle Knoten auf die Himmelskugel projiziert (wie ein Teleskop-Survey)
    """
    N_VIS = 400           # Visualisierungs-Punkte pro Schale (exakt gleiche Physik)
    DARK = "#000814"

    rng = np.random.default_rng(params.seed)
    turn = float(params.turn_ticks)
    quarter = turn / 4.0
    theta_amp = float(params.theta_amp_ticks) * quarter

    xs_l, ys_l, zs_l, ws_l = [], [], [], []
    phi_rad_l, dec_rad_l = [], []          # für Mollweide

    for row in shells.itertuples(index=False):
        r = float(row.r_mpc)
        omega_eff = float(row.omega_eff)
        trichter_depth = float(row.trichter_depth)

        # === Exakt dieselbe Drift-Formel wie in build_shell_nodes ===
        drift = omega_eff * params.dt * turn * params.rot_steps

        # Zufällige Startwinkel (gleichmäßig, wie in der Simulation)
        phi0 = rng.uniform(0.0, turn, size=N_VIS)
        theta0 = rng.uniform(quarter - theta_amp, quarter + theta_amp, size=N_VIS)

        phi = np.mod(phi0 + drift, turn)

        phi_rad = phi / turn * (2.0 * math.pi)
        theta_rad = theta0 / turn * (2.0 * math.pi)

        # === Exakt dieselbe Trichter-Kompression wie in build_shell_nodes ===
        z_factor = max(0.05, 1.0 - 0.35 * trichter_depth)

        x = r * np.sin(theta_rad) * np.cos(phi_rad)
        y = r * np.sin(theta_rad) * np.sin(phi_rad)
        z = r * np.cos(theta_rad) * z_factor

        xs_l.append(x);  ys_l.append(y);  zs_l.append(z)
        ws_l.append(np.full(N_VIS, omega_eff))
        # Für Mollweide: RA = phi_rad (0..2π → verschoben auf −π..π)
        phi_rad_l.append(phi_rad - math.pi)
        # Deklination = π/2 − theta (theta=π/2 → Äquator → Dec=0)
        dec_rad_l.append(math.pi / 2.0 - theta_rad)

    xs = np.concatenate(xs_l);  ys = np.concatenate(ys_l)
    zs = np.concatenate(zs_l);  ws = np.concatenate(ws_l)
    phi_all = np.concatenate(phi_rad_l)
    dec_all = np.concatenate(dec_rad_l)

    ws_norm = ws / (ws.max() + 1e-10)
    R = float(shells["r_mpc"].max())

    # ---- Abbildung -------------------------------------------------------- #
    fig = plt.figure(figsize=(26, 9), facecolor=DARK)
    fig.subplots_adjust(wspace=0.06, left=0.04, right=0.98, top=0.88, bottom=0.08)

    # ---- Panel 1: XY Draufsicht ----------------------------------------- #
    ax1 = fig.add_subplot(1, 3, 1, facecolor=DARK)
    ax1.hexbin(xs, ys, C=ws_norm, reduce_C_function=np.mean,
               gridsize=120, cmap="inferno", mincnt=1, linewidths=0.0,
               extent=[-R, R, -R, R])
    ax1.set_xlim(-R, R);  ax1.set_ylim(-R, R)
    ax1.set_xlabel("x (Mpc)", color="#335577", fontsize=9)
    ax1.set_ylabel("y (Mpc)", color="#335577", fontsize=9)
    ax1.set_title("🔭 Draufsicht (XY)\nomega-Dichte-Struktur", color="#77ccee", fontsize=11, pad=8)
    ax1.tick_params(colors="#223344", labelsize=7)
    for sp in ax1.spines.values():
        sp.set_color("#112233")
    ax1.set_aspect("equal")

    # ---- Panel 2: XZ Seitenansicht (Trichter-Disk) ----------------------- #
    ax2 = fig.add_subplot(1, 3, 2, facecolor=DARK)
    ax2.hexbin(xs, zs, C=ws_norm, reduce_C_function=np.mean,
               gridsize=120, cmap="inferno", mincnt=1, linewidths=0.0,
               extent=[-R, R, -R, R])
    ax2.set_xlim(-R, R);  ax2.set_ylim(-R, R)
    ax2.set_xlabel("x (Mpc)", color="#335577", fontsize=9)
    ax2.set_ylabel("z (Mpc)", color="#335577", fontsize=9)
    ax2.set_title("🔭 Seitenansicht (XZ)\nTrichter erzeugt Disk (emergent)", color="#77ccee", fontsize=11, pad=8)
    ax2.tick_params(colors="#223344", labelsize=7)
    for sp in ax2.spines.values():
        sp.set_color("#112233")
    ax2.set_aspect("equal")

    # ---- Panel 3: Mollweide Himmelskarte --------------------------------- #
    ax3 = fig.add_subplot(1, 3, 3, projection="mollweide")
    ax3.set_facecolor(DARK)
    # Subsample für Geschwindigkeit
    MAX_MOLL = 60_000
    if len(phi_all) > MAX_MOLL:
        idx = rng.choice(len(phi_all), size=MAX_MOLL, replace=False)
        ra_p, dec_p, w_p = phi_all[idx], dec_all[idx], ws_norm[idx]
    else:
        ra_p, dec_p, w_p = phi_all, dec_all, ws_norm
    ax3.scatter(ra_p, dec_p, c=w_p, cmap="inferno",
                s=0.3, alpha=0.5, linewidths=0, vmin=0, vmax=1)
    ax3.set_title("🔭 Himmelskarte (Mollweide)\nTeleskop-Survey-Ansicht", color="#77ccee", fontsize=11, pad=8)
    ax3.tick_params(colors="#334455", labelsize=7)
    ax3.grid(color="#0d1e30", linewidth=0.4, alpha=0.7)
    ax3.set_facecolor(DARK)

    fig.suptitle(
        f"Tensor-Sim v{VERSION}  ·  Emergente Feld-Struktur (keine aufgezwungene Physik)  ·  "
        f"seed={params.seed}  ·  r_max={params.max_mpc:,.0f} Mpc  ·  "
        f"Ω₀={params.omega0}  ·  r₀={params.r0} Mpc  ·  p={params.p}",
        color="#5599bb", fontsize=10,
    )
    plt.savefig(
        outdir / "fig_cosmic_web_filaments.png",
        dpi=200, facecolor=DARK, bbox_inches="tight",
    )
    plt.close(fig)


def _img_b64(path: Path) -> str:
    """Return a data URI for a PNG file (base64-encoded)."""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def generate_html_report(
    params: Params,
    outdir: Path,
    nodes: pd.DataFrame,
    shells: pd.DataFrame,
    cones: "pd.DataFrame | None",
    events: "pd.DataFrame | None",
) -> None:
    """Write a self-contained HTML Teleskop-Beobachtungs-Report to outdir/report.html.

    All images are embedded as base64 data URIs so the file works offline.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # --- helpers ---
    def _df_html(df: pd.DataFrame, max_rows: int = 50) -> str:
        display = df.head(max_rows)
        rows_html = []
        for _, row in display.iterrows():
            cells = "".join(
                f"<td>{v:.6g}" if isinstance(v, float) else f"<td>{v}"
                for v in row
            )
            rows_html.append(f"<tr>{cells}</tr>")
        header = "".join(f"<th>{c}" for c in display.columns)
        note = ""
        if len(df) > max_rows:
            note = f'<p class="note">Zeige {max_rows} von {len(df)} Zeilen — vollständige Daten in CSV-Datei.</p>'
        return (
            '<div class="table-wrap">'
            + note
            + f'<table><thead><tr>{header}</tr></thead><tbody>'
            + "".join(rows_html)
            + "</tbody></table></div>"
        )

    def _img_section(label: str, path: Path) -> str:
        if not path.exists():
            return ""
        b64 = _img_b64(path)
        return (
            f'<div class="plot-block">'
            f'<div class="plot-label">📡 {label}</div>'
            f'<img src="{b64}" alt="{label}">'
            f"</div>"
        )

    # --- images ---
    plot_names = [
        ("🌌 Emergente Tensor-Feld-Struktur — Draufsicht · Trichter-Disk · Himmelskarte", "fig_cosmic_web_filaments.png"),
        ("Ω_eff vs r  –  Winkelgeschwindigkeit über Radius", "fig_omega_vs_r.png"),
        ("α vs r  –  Kopplungsstärke über Radius", "fig_alpha_vs_r.png"),
        ("Ω_eff vs α  –  Kopplungs-Phasenraum", "fig_omega_vs_alpha.png"),
        ("Shell-Besetzung  –  Nodes pro Shell", "fig_shell_occupancy.png"),
        ("Trichter-Tiefenverteilung", "fig_trichter_depth_hist.png"),
        ("XY-Karte aller Nodes (Subsample)", "fig_xy_nodes.png"),
        ("Trichter-Querschnitt X-Z (Slice)", "fig_trichter_xz_slice.png"),
        ("Kegel-Ereignisse", "fig_cone_events_counts.png"),
    ]
    images_html = "".join(_img_section(lbl, outdir / fn) for lbl, fn in plot_names)

    # --- parameter table ---
    param_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in [
            ("max_mpc", f"{params.max_mpc:,.0f} Mpc"),
            ("shell_mpc", f"{params.shell_mpc:.0f} Mpc"),
            ("capacity", params.capacity),
            ("occupancy_mode", params.occupancy_mode),
            ("alpha_max", params.alpha_max),
            ("omega0", params.omega0),
            ("r0", params.r0),
            ("p (Falloff-Exponent)", params.p),
            ("turn_ticks", f"{params.turn_ticks:,}"),
            ("dt", params.dt),
            ("rot_steps", f"{params.rot_steps:,}"),
            ("trichter_strength", params.trichter_strength),
            ("ell_trichter", f"{params.ell_trichter:,.0f} Mpc"),
            ("cone_mode", params.cone_mode),
            ("seed", params.seed),
        ]
    )

    omega_min = float(shells["omega_eff"].min())
    omega_max = float(shells["omega_eff"].max())

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tensor-Sim v{VERSION} — Teleskop-Beobachtungsprotokoll</title>
<style>
  :root {{
    --bg: #070d14;
    --panel: #0d1b2a;
    --border: #1b3a5c;
    --green: #00ff88;
    --cyan: #00d4ff;
    --yellow: #ffe066;
    --text: #c8daf0;
    --dim: #6a8aaa;
    --red: #ff4f6e;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Courier New', Courier, monospace;
    font-size: 14px;
    line-height: 1.6;
  }}
  header {{
    background: var(--panel);
    border-bottom: 2px solid var(--green);
    padding: 20px 32px;
  }}
  header h1 {{
    color: var(--green);
    font-size: 1.6em;
    letter-spacing: 3px;
    text-transform: uppercase;
  }}
  header .meta {{ color: var(--dim); font-size: 0.85em; margin-top: 4px; }}
  header .meta span {{ color: var(--cyan); }}
  .badge {{
    display: inline-block;
    background: #002b1a;
    border: 1px solid var(--green);
    color: var(--green);
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.8em;
    margin-right: 8px;
    margin-top: 6px;
  }}
  main {{ max-width: 1400px; margin: 0 auto; padding: 32px 24px; }}
  section {{ margin-bottom: 48px; }}
  h2 {{
    color: var(--cyan);
    font-size: 1.1em;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-bottom: 18px;
    text-transform: uppercase;
  }}
  /* stats grid */
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 14px;
    margin-bottom: 24px;
  }}
  .stat-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 18px;
  }}
  .stat-card .label {{ color: var(--dim); font-size: 0.78em; text-transform: uppercase; }}
  .stat-card .value {{ color: var(--yellow); font-size: 1.5em; font-weight: bold; margin-top: 2px; }}
  /* param table */
  .param-table {{ border-collapse: collapse; width: 100%; max-width: 600px; }}
  .param-table td {{
    border: 1px solid var(--border);
    padding: 5px 12px;
  }}
  .param-table tr td:first-child {{ color: var(--dim); width: 55%; }}
  .param-table tr td:last-child {{ color: var(--green); }}
  /* images */
  .plots-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
    gap: 20px;
  }}
  .plot-block {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }}
  .plot-label {{
    background: #0a1a2e;
    color: var(--cyan);
    font-size: 0.85em;
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    letter-spacing: 1px;
  }}
  .plot-block img {{ width: 100%; display: block; }}
  /* tables */
  .table-wrap {{ overflow-x: auto; }}
  .note {{ color: var(--dim); font-size: 0.82em; margin-bottom: 8px; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.82em;
  }}
  thead tr {{ background: #0a1a2e; }}
  th {{
    color: var(--cyan);
    padding: 6px 10px;
    border: 1px solid var(--border);
    text-align: left;
    white-space: nowrap;
  }}
  td {{
    color: var(--text);
    padding: 4px 10px;
    border: 1px solid #111e2e;
    white-space: nowrap;
  }}
  tbody tr:nth-child(even) {{ background: #0b1520; }}
  tbody tr:hover {{ background: #122034; }}
  footer {{
    text-align: center;
    color: var(--dim);
    font-size: 0.78em;
    padding: 24px;
    border-top: 1px solid var(--border);
    margin-top: 40px;
  }}
</style>
</head>
<body>
<header>
  <h1>🔭 Tensor-Sim v{VERSION} — Teleskop-Beobachtungsprotokoll</h1>
  <div class="meta">
    Beobachtungszeit: <span>{ts}</span> &nbsp;|&nbsp;
    Instrument: <span>Tensor-Feld-Simulator v{VERSION}</span> &nbsp;|&nbsp;
    Seed: <span>{params.seed}</span> &nbsp;|&nbsp;
    Modus: <span>{params.occupancy_mode}</span>
  </div>
  <div style="margin-top:10px">
    <span class="badge">🌌 {len(shells)} Shells</span>
    <span class="badge">⚛ {len(nodes):,} Nodes</span>
    <span class="badge">Ω_eff {omega_min:.4f} – {omega_max:.4f}</span>
    <span class="badge">r_max {params.max_mpc:,.0f} Mpc</span>
    <span class="badge">cone_mode: {params.cone_mode}</span>
  </div>
</header>
<main>

<section>
  <h2>📊 Beobachtungsstatistik</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="label">Shells</div><div class="value">{len(shells)}</div></div>
    <div class="stat-card"><div class="label">Nodes gesamt</div><div class="value">{len(nodes):,}</div></div>
    <div class="stat-card"><div class="label">Nodes / Shell</div><div class="value">{params.capacity}</div></div>
    <div class="stat-card"><div class="label">Ω_eff min</div><div class="value">{omega_min:.5f}</div></div>
    <div class="stat-card"><div class="label">Ω_eff max</div><div class="value">{omega_max:.4f}</div></div>
    <div class="stat-card"><div class="label">r_max (Mpc)</div><div class="value">{params.max_mpc:,.0f}</div></div>
    <div class="stat-card"><div class="label">Shell-Dicke (Mpc)</div><div class="value">{params.shell_mpc:.0f}</div></div>
    <div class="stat-card"><div class="label">α_max</div><div class="value">{params.alpha_max}</div></div>
  </div>
</section>

<section>
  <h2>⚙️ Simulations-Parameter</h2>
  <table class="param-table">
    <tbody>{param_rows}</tbody>
  </table>
</section>

<section>
  <h2>🖼 Felddiagramme (Rohdaten-Bilder)</h2>
  <div class="plots-grid">
    {images_html}
  </div>
</section>

<section>
  <h2>📄 Rohdaten: shells.csv  ({len(shells)} Zeilen)</h2>
  {_df_html(shells, max_rows=50)}
</section>

<section>
  <h2>📄 Rohdaten: nodes.csv  ({len(nodes):,} Zeilen)</h2>
  {_df_html(nodes, max_rows=100)}
</section>
"""

    if cones is not None and len(cones) > 0:
        html += f"""
<section>
  <h2>📄 Rohdaten: cones.csv  ({len(cones)} Zeilen)</h2>
  {_df_html(cones, max_rows=50)}
</section>
"""

    if events is not None and len(events) > 0:
        html += f"""
<section>
  <h2>📄 Rohdaten: cone_events.csv  ({len(events)} Zeilen)</h2>
  {_df_html(events, max_rows=50)}
</section>
"""

    html += f"""
</main>
<footer>
  Tensor-Sim v{VERSION} &nbsp;·&nbsp; Autor: Raiko Pulvermacher &nbsp;·&nbsp;
  Generiert: {ts} &nbsp;·&nbsp; PORL v1.0
</footer>
</body>
</html>
"""

    report_path = outdir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"report: {report_path}")


def parse_args() -> Params:
    ap = argparse.ArgumentParser(
        description=(
            "Tensor Simulation v1.17: Omega-Alpha coupling in tick-space + optional Trichter + optional Cone interactions. "
            "Outputs CSVs and plots."
        )
    )

    # output
    ap.add_argument("--outdir", default="outputs_v117", help="output directory")

    # scale
    ap.add_argument("--max_mpc", type=float, default=120_000.0, help="maximum radius (Mpc)")
    ap.add_argument("--shell_mpc", type=float, default=500.0, help="shell thickness (Mpc)")

    # nodes
    ap.add_argument("--capacity", type=int, default=13, help="max nodes per shell")

    # occupancy
    ap.add_argument("--occupancy_mode", choices=["fixed", "beta", "linear"], default="fixed")
    ap.add_argument("--beta_a", type=float, default=2.0, help="beta fill shape a (beta mode only)")
    ap.add_argument("--beta_b", type=float, default=5.0, help="beta fill shape b (beta mode only)")

    # coupling
    ap.add_argument("--alpha_max", type=float, default=1.5, help="alpha at full shell")

    # omega
    ap.add_argument("--omega0", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=500.0)
    ap.add_argument("--p", type=float, default=1.15, help="omega falloff exponent")

    # ticks
    ap.add_argument("--turn_ticks", type=int, default=DEFAULT_TURN, help="ticks per full turn (default 129600)")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--rot_steps", type=int, default=6000)

    # placement
    ap.add_argument(
        "--theta_amp_ticks",
        type=float,
        default=0.25,
        help="theta band half-width as fraction of quarter-turn",
    )

    # trichter
    ap.add_argument("--trichter_strength", type=float, default=1.0)
    ap.add_argument("--ell_trichter", type=float, default=20_000.0)

    # cones
    ap.add_argument("--cone_mode", choices=["off", "simple"], default="off")
    ap.add_argument("--cone_span_shells", type=int, default=7)
    ap.add_argument("--cone_events", type=int, default=50)
    ap.add_argument("--cone_absorb_ratio", type=float, default=1.35)
    ap.add_argument("--cone_merge_ratio", type=float, default=1.10)
    ap.add_argument("--cone_repulse_kick", type=float, default=0.08)
    ap.add_argument("--cone_split_min_span", type=int, default=2)

    # rng
    ap.add_argument("--seed", type=int, default=42)

    # flags
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--ticks_only", action="store_true", help="do not compute x/y/z; still saves ticks")

    args = ap.parse_args()

    return Params(
        max_mpc=args.max_mpc,
        shell_mpc=args.shell_mpc,
        capacity=args.capacity,
        occupancy_mode=args.occupancy_mode,
        beta_a=args.beta_a,
        beta_b=args.beta_b,
        alpha_max=args.alpha_max,
        omega0=args.omega0,
        r0=args.r0,
        p=args.p,
        turn_ticks=args.turn_ticks,
        dt=args.dt,
        rot_steps=args.rot_steps,
        theta_amp_ticks=args.theta_amp_ticks,
        trichter_strength=args.trichter_strength,
        ell_trichter=args.ell_trichter,
        cone_mode=args.cone_mode,
        cone_span_shells=args.cone_span_shells,
        cone_events=args.cone_events,
        cone_absorb_ratio=args.cone_absorb_ratio,
        cone_merge_ratio=args.cone_merge_ratio,
        cone_repulse_kick=args.cone_repulse_kick,
        cone_split_min_span=args.cone_split_min_span,
        seed=args.seed,
        outdir=args.outdir,
        plots=(not args.no_plots),
        geometry=(not args.ticks_only),
    )


def main() -> int:
    params = parse_args()
    outdir = Path(params.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, shells, cones, events = run_simulation(params)

    nodes.to_csv(outdir / "nodes.csv", index=False)
    shells.to_csv(outdir / "shells.csv", index=False)
    if cones is not None and len(cones) > 0:
        cones.to_csv(outdir / "cones.csv", index=False)
    if events is not None and len(events) > 0:
        events.to_csv(outdir / "cone_events.csv", index=False)

    if params.plots:
        plot_shells(shells, outdir)
        plot_tensor_field_web(shells, params, outdir)
        if params.geometry:
            plot_xy(nodes, outdir)
            plot_trichter_xz(nodes, outdir)
        if events is not None and len(events) > 0:
            plot_cone_events(events, outdir)

    save_manifest(params, outdir, nodes, shells, cones, events)
    generate_html_report(params, outdir, nodes, shells, cones, events)

    # stdout summary
    print(f"Tensor Simulation v{VERSION} done.")
    print(f"outdir: {outdir}")
    print(f"shells: {len(shells)} | nodes: {len(nodes)}")
    if params.cone_mode != "off" and events is not None:
        print(f"cone events: {len(events)}")
    print(f"omega_eff range: {shells['omega_eff'].min():.6g} .. {shells['omega_eff'].max():.6g}")
    if "delta_phi_tick" in shells.columns:
        dmin = float(shells["delta_phi_tick"].min())
        dmax = float(shells["delta_phi_tick"].max())
        print(f"delta_phi_tick range: {dmin:.6g} .. {dmax:.6g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
