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

Standard geometric pure pursuit. A virtual target point is placed at `lookahead_dist` metres ahead with a lateral offset equal to the crosstrack error. The steering angle is the bicycle-model angle needed to reach that point:

```
alpha = atan2(e, ld)
delta = atan(2 · L · sin(alpha) / ld)
omega = v · tan(delta) / L
```

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `target_speed` | 0.50 m/s | 0.1 – 1.5 | Constant forward speed |
| `lookahead_dist` | 0.30 m | 0.05 – 1.0 | Look-ahead distance. Shorter → reacts faster, more oscillation. Longer → smoother, may cut corners |
| `alpha` | 0.50 | 0.0 – 1.0 | EMA weight on steering angle. Lower → smoother but laggier |

---

### Stanley

> Node: `stanley_controller`

Standard Stanley lateral law. Uses both crosstrack error and heading error:

```
delta = theta_e + atan(k · e / (v + k_soft))
omega = v · tan(delta) / L
```

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `target_speed` | 0.50 m/s | 0.1 – 1.5 | Constant forward speed |
| `k` | 1.0 | 0.1 – 5.0 | Crosstrack gain. Higher → tighter lane-centring, more oscillation risk |
| `k_soft` | 0.40 | 0.01 – 2.0 | Softening term preventing division by zero at low speed |
| `alpha` | 0.50 | 0.0 – 1.0 | EMA weight on steering angle |

---

### Linderoth (2008)

> Node: `linderoth_controller`

Nonlinear geometric lateral law from Linderoth 2008, eq. 4.53, derived in curvilinear coordinates. Runs at constant speed:

```
numerator   = −cos(e_th) · e_n  −  (k1+k2) · sin(e_th)
denominator =  k1  −  (k1+k2) · cos(e_th)  +  sin(e_th) · e_n
delta       = atan(numerator / denominator)
omega       = v · tan(delta) / L
```

Near zero error this linearises to `δ ≈ e_n/k2 + (k1/k2 + 1) · θ_e`, so **k2 is the crosstrack gain** and **k1 controls heading correction**. k1 must be less than k2 for stable straight-line behaviour.

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `target_speed` | 0.35 m/s | 0.1 – 1.5 | Constant forward speed |
| `k1` | 0.30 | 0.01 – 2.0 | Stability gain. Must be < k2 for stable straight-line behaviour |
| `k2` | 0.60 | 0.01 – 2.0 | Crosstrack gain. Raise to tighten lane centring; lower if oscillating |
| `heading_scale` | 0.00 | 0.0 – 1.0 | How much of the look-ahead heading enters the formula. 0 = pure crosstrack `δ=atan(e_n/k2)`, no early turning. 1 = full anticipatory heading (turns into curves before crosstrack error builds). Start at 0 and raise gradually |
| `alpha` | 0.40 | 0.0 – 1.0 | EMA weight on steering angle |

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
