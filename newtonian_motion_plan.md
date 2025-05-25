# Plan: Implement Newtonian Motion and Updated Energy Costs for Probes

This document outlines the plan to modify the Bobiverse simulation to incorporate linear Newtonian physics for probe movement and a revised energy cost model for thrust.

## 1. Goal

Modify probe movement to use linear Newtonian physics principles. Update the energy cost associated with thrusting to be proportional to the magnitude of the applied force.

## 2. Configuration Changes (`config.py`)

The following changes and additions will be made to `config.py`:

*   **Add Probe Mass:**
    *   Introduce `PROBE_MASS = 1.0` (or a suitable initial value).
*   **Modify Thrust Parameter:**
    *   Rename `THRUST_POWER` to `THRUST_FORCE`.
    *   Adjust its values to represent force magnitudes (e.g., `THRUST_FORCE = [0.0, 5.0, 10.0]`). These values will need tuning.
*   **Re-evaluate Maximum Velocity:**
    *   Increase `MAX_VELOCITY` significantly (e.g., `MAX_VELOCITY = 50.0`). This will serve primarily for observation normalization and as a practical upper limit, rather than a hard physical constraint of the old model.
*   **Add Energy Cost Factor for Thrust:**
    *   Introduce `THRUST_ENERGY_COST_FACTOR = 0.05` (or a suitable initial value, representing energy per unit of force per step).

## 3. Environment Logic Changes (`environment.py`)

The following modifications will be made to the `SpaceEnvironment` class:

*   **Probe Initialization (`add_probe` method):**
    *   When a new probe is created, initialize a `mass` attribute for it using `PROBE_MASS` from `config.py`.
*   **Action Processing (`_process_probe_action` method):**
    *   **Force Calculation:** Determine the force vector based on the `thrust_dir` action and the corresponding magnitude from `THRUST_FORCE[thrust_power]`.
    *   **Acceleration Calculation:** Calculate acceleration using `acceleration = force_vector / probe['mass']`.
    *   **Velocity Update:** Update the probe's velocity using `probe['velocity'] += acceleration_vector`.
    *   **Energy Cost:** Calculate energy cost for thrusting: `energy_cost = force_magnitude * THRUST_ENERGY_COST_FACTOR`. Deduct this from `probe['energy']`.
    *   Remove old logic that directly modified velocity based on "power" and the old clamping within this section.
*   **Physics Update (`_update_physics` method):**
    *   **Remove Friction:** Delete the line `probe['velocity'] *= 0.95`.
    *   **Position Update:** The existing line `probe['position'] += probe['velocity']` remains correct for updating position based on the new velocity.
    *   **Optional Velocity Clamping:** After all velocity updates, `probe['velocity']` can be optionally clamped using `np.clip(probe['velocity'], -MAX_VELOCITY, MAX_VELOCITY)` to ensure it stays within the re-evaluated `MAX_VELOCITY` bounds, mainly for stability and normalization.
*   **Observation Normalization (`get_observation` method):**
    *   Ensure that the velocity components of the observation vector (`obs[2:4]`) are consistently normalized using the (potentially new) `MAX_VELOCITY` value.

## 4. Deferred Features

The following related features are considered out of scope for this immediate update and are deferred for potential future enhancements:
*   Rotational physics for probes.
*   More complex energy cost models (e.g., based on change in kinetic energy or specific impulse).
*   Advanced drag models.

## 5. Key Implications
*   **Agent Retraining:** All existing trained `ProbeAgent` models will become invalid due to the fundamental changes in environment physics. Agents will need to be retrained from scratch.
*   **Parameter Tuning:** The new constants (`PROBE_MASS`, `THRUST_FORCE`, `THRUST_ENERGY_COST_FACTOR`, `MAX_VELOCITY`) will likely require iterative tuning to achieve desired simulation behavior and agent performance.

## 6. Mermaid Diagram of Planned Changes

```mermaid
graph TD
    A[Start: User Request for Newtonian Physics & Energy/Rotation] --> B(Phase 1: Modify config.py);
    B --> B1[Define PROBE_MASS];
    B --> B2[Adjust THRUST_POWER to be THRUST_FORCE];
    B --> B3[Increase/Re-evaluate MAX_VELOCITY];
    B --> B4[Define THRUST_ENERGY_COST_FACTOR];

    A --> C(Phase 2: Modify environment.py);
    C --> C1[Update `add_probe` to include mass];
    C --> C2[In `_process_probe_action`:\n- Calculate Force from thrust\n- Calculate Acceleration (F/m)\n- Update Velocity (v += a)\n- Calculate Energy Cost (Force * Factor)];
    C --> C3[In `_update_physics`:\n- Remove old friction (velocity *= 0.95)\n- Position update (p += v) remains\n- Optionally clamp velocity to new MAX_VELOCITY];
    C --> C4[Review velocity normalization in `get_observation`];

    A --> D(Phase 3: Consider Implications);
    D --> D1[Agent retraining will be necessary];
    D --> D2[New constants (mass, force, energy factor) will need tuning];

    A --> F[Decision Point: Rotational Physics];
    F -- Defer for now --> G[Focus on Linear Newtonian Motion & New Energy Model];

    B & C & D & G --> E[Outcome: Probes move based on Linear Newtonian principles with force-based energy cost];