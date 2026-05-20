# my_f1tenth_sim

F1TENTH scaled autonomous vehicle simulation — lane detection and lane-keeping controllers in Gazebo Classic (ROS 2 Humble).

## Quick start

```bash
colcon build --packages-select my_f1tenth_sim
source install/setup.bash
ros2 launch my_f1tenth_sim full_sim.launch.py
```

This opens:
- **Gazebo** with the two-lane track and the car spawned at the saved position
- **LaneAssist peaks** — the lane detection visualisation window
- **Controller Panel** — GUI to select and tune a controller (appears ~12 s after launch)

---

## Lane Detection Algorithm

The detector is a port of the **LaneAssist** algorithm (Tsingiropoulos, BFMC 2023), processing the front camera at 20 Hz through the following pipeline.

### 1. ROI mask
Rows outside 235–395 px are zeroed, removing the sky and the ground too close to the car body.

### 2. Bird's Eye View (BEV) warp
A perspective transform maps the trapezoidal road region to a flat 640×480 top-down view. The source quad is tuned so that the far-field extends 60% wider than the original LaneAssist to handle steep curves.

### 3. CLAHE equalisation
Contrast-Limited Adaptive Histogram Equalisation (clip = 4.0, tile = 4×4) is applied to the grayscale BEV. This lifts dim far-field lane markings into the detectable range without blowing out near-field ones.

### 4. Horizontal slice scan
32 horizontal slices are scanned bottom-to-top across 80% of the BEV height (~12 px apart). Each row is searched for **square-pulse peaks** — bright regions (> 80 ADU) with a sharp rising and falling edge (`sq_min_height_dif = 30`). An adaptive floor (`sq_min_height_curr`) lowers each frame toward the observed peak intensity to stay robust on dim frames.

### 5. Peak clustering
Peaks across slices are grouped into lane tracks using a Hungarian-style nearest-neighbour matcher. A track must span ≥ 2 slices to be valid. The maximum lateral jump between slices is 192 px (30% of BEV width) to track lines through tight curves.

### 6. Left / right lane selection
From all valid tracks the detector selects the rightmost track in the left half and the leftmost track in the right half. On single-line frames it locks the side classification (left/right) until two consecutive dual-lane frames confirm dual detection, preventing noisy frames from flipping the assignment.

### 7. Polynomial fit
Each selected lane track is fitted as **x = f(y)** using least squares — cubic if ≥ 7 points, quadratic if fewer, linear as fallback. A sanity check on the leading coefficient rejects wildly curved fits.

### 8. Certainty and holdout
Each polynomial gets a certainty score (0–100%) combining peak count and frame-to-frame stability. If one lane disappears, its last valid polynomial is held for up to 8 frames (~0.4 s) so heading stays accurate through brief blind spots.

### 9. Crosstrack error
The average x-position of the bottom-3-slice detections from each lane gives `lx` and `rx`. Their midpoint vs. BEV centre gives the lateral offset, converted to metres with `0.8 m / 340 px`. EMA-filtered (α = 0.35). Published on `/perception/crosstrack_error`.

### 10. Heading error
The polynomial slope `dx/dy` is evaluated at a look-ahead row ~1/3 up the scan range (~0.28 m ahead). `heading = −atan(slope)`. EMA-filtered (α = 0.35). Published on `/perception/heading_error`.

---

## Controllers

All three controllers subscribe to `/perception/crosstrack_error`, `/perception/heading_error`, and `/odom`, and publish to `/cmd_vel` at 20 Hz. They are selected and tuned at runtime through the Controller Panel GUI.

---

### Pure Pursuit

> Node: `pure_pursuit_controller`

Despite its name, this node implements the **Stanley lateral law** with a derivative crosstrack term:

```
e_pred  = e + k_d · ė
δ       = heading_gain · θ_e + atan( k · e_pred / (v + k_soft) )
ω       = v · tan(δ) / wheelbase
```

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `target_speed` | 0.50 m/s | 0.1 – 1.5 | Constant forward speed commanded to Gazebo |
| `k` | 1.0 | 0.1 – 5.0 | Crosstrack-to-steer sensitivity. Higher → tighter lane-centring, more oscillation risk |
| `k_soft` | 0.40 | 0.01 – 2.0 | Softening term preventing division by zero and reducing gain at low speed |
| `k_d` | 0.15 | 0.0 – 1.0 | Derivative look-ahead: predicts error at the camera's nearest visible point (~7.5 cm at 0.5 m/s). Reduces overshoot |
| `heading_gain` | 0.0 | 0.0 – 2.0 | Weight on the raw heading error θ_e. Currently 0 — the derivative term already provides anticipation |
| `alpha` | 0.50 | 0.0 – 1.0 | EMA weight on the steering angle. Lower → smoother but laggier response |

---

### Stanley

> Node: `stanley_controller`

Same lateral law as Pure Pursuit. Kept as a separate node to allow independent parameter tuning and direct comparison.

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `target_speed` | 0.50 m/s | 0.1 – 1.5 | Constant forward speed |
| `k` | 1.0 | 0.1 – 5.0 | Crosstrack gain |
| `k_soft` | 0.40 | 0.01 – 2.0 | Low-speed softening |
| `k_d` | 0.15 | 0.0 – 1.0 | Derivative look-ahead |
| `heading_gain` | 0.0 | 0.0 – 2.0 | Heading error weight |
| `alpha` | 0.50 | 0.0 – 1.0 | EMA smoothing |

---

### Linderoth (2008)

> Node: `linderoth_controller`

Implements the nonlinear geometric lateral law from Linderoth's 2008 thesis (eq. 4.53), derived in curvilinear coordinates:

```
numerator   = −cos(e_th) · e_n  −  (k1+k2) · sin(e_th)
denominator =  k1  −  (k1+k2) · cos(e_th)  +  sin(e_th) · e_n
δ           = atan( numerator / denominator )
ω           = v · tan(δ) / wheelbase
```

Near zero error this linearises to `δ ≈ e_n/k2 + (k1/k2 + 1) · θ_e`, so **k2 sets crosstrack sensitivity** and **k1/k2 sets the heading weight**. Longitudinal speed is controlled by a discrete PI:

```
e_v    = v_ref − v_actual
v_cmd += Δt · Kp · ( e_v + (Ki/Kp) · ∫e_v dt )
```

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `k1` | 0.30 | 0.01 – 2.0 | Must be < k2 for stable straight-line behaviour. Controls how quickly heading error is corrected |
| `k2` | 0.60 | 0.01 – 2.0 | Primary crosstrack-to-steer gain. Raise to tighten lane centring; lower if oscillating |
| `v_ref` | 0.35 m/s | 0.1 – 1.5 | Speed setpoint for the PI controller |
| `heading_scale` | 0.40 | 0.0 – 1.5 | Scales θ_e before entering the formula. The detector evaluates heading ~0.28 m ahead, so the heading term fires early and produces large angles in curves. Values < 1 damp this. Raise toward 1.0 for better curve tracking; lower toward 0.3 if turning too aggressively |
| `Kp` | 0.80 | 0.1 – 5.0 | Proportional gain for speed PI. Higher → faster speed convergence, risk of overshoot |
| `Ki` | 0.10 | 0.0 – 2.0 | Integral gain. Eliminates steady-state speed error. Integral is clamped to ±1.0 to prevent windup |
| `alpha` | 0.40 | 0.0 – 1.0 | EMA weight on steering angle output. Lower → smoother steering |

---

## ROS 2 Topics

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | → detector | Raw front camera feed from Gazebo |
| `/perception/crosstrack_error` | `std_msgs/Float32` | detector → controllers | Lateral offset from lane centre (m) |
| `/perception/heading_error` | `std_msgs/Float32` | detector → controllers | Heading difference from road tangent (rad) |
| `/odom` | `nav_msgs/Odometry` | Gazebo → controllers | Vehicle speed and pose |
| `/cmd_vel` | `geometry_msgs/Twist` | controllers → Gazebo | `linear.x` = target speed (m/s), `angular.z` = yaw rate (rad/s) |

## Launch files

| File | Description |
|---|---|
| `full_sim.launch.py` | **Main entry point.** Starts Gazebo, spawns car, lane detector, and controller panel GUI |
| `spawn_car.launch.py` | Starts Gazebo with a configurable world and spawns the car at a given pose |
| `spawn_with_track.launch.py` | Starts Gazebo with the two-lane track and spawns the car at the origin |
