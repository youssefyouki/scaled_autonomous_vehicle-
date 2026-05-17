#!/usr/bin/env python3
"""
Lane Detector — LaneAssist algorithm (ported from github.com/tsiggi/LaneAssist)
=================================================================================
Original: MIT License © 2023 Christos-Alexandros Tsingiropoulos (VROOM/BFMC)

Pipeline
--------
1. ROI mask (rows 235–330 kept)
2. BEV perspective warp
3. CLAHE on grayscale — normalise far-field dimness from 6× BEV stretch
4. Horizontal slice scan with square-pulse peak detection
5. Peak clustering into lane tracks across slices
6. Left / right lane selection
7. 2nd-degree polynomial fit  x = a·y² + b·y + c  per lane
8. Adaptive certainty scoring; trust flags per lane
9. Crosstrack error (m) from polynomial midpoint at bottom row
10. Heading error (rad) from polynomial derivative at bottom row
11. Publish → /perception/crosstrack_error  /perception/heading_error

Tunable params (all in __init__)
---------------------------------
sq_min_height       : grayscale floor for peak detection (auto-adapts each frame)
sq_min_height_dif   : edge sharpness — lower if lane edges are soft after CLAHE
peaks_min/max_width : lane marking width in BEV pixels
bottom_perc         : fraction of BEV height to scan (0.5 = bottom half)
"""

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class LaneDetector(Node):

    def __init__(self):
        super().__init__('lane_detector')

        # ── I/O ───────────────────────────────────────────────────────────────
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()
        self.e_pub  = self.create_publisher(Float32, '/perception/crosstrack_error', 10)
        self.th_pub = self.create_publisher(Float32, '/perception/heading_error', 10)

        # ── BEV calibration ───────────────────────────────────────────────────
        # Bottom source row extended to 390: raw rows 325-390 (beside/in-front
        # of the car) now warp into the bottom of the BEV.  The car body is blue
        # so its pixels are dark in CLAHE grayscale and ignored by peak detection.
        self.src_pts = np.float32([[30, 390], [610, 390], [420, 248], [220, 248]])
        self.dst_pts = np.float32([[0, 480],  [640, 480], [640, 0],   [0, 0]])
        self.M    = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.bev_w = 640
        self.bev_h = 480

        # ── Slice geometry (mirrors LaneAssist choose_455 for our BEV) ────────
        _slices      = 20
        _bot_off     = 3   # scan almost to the BEV bottom to use the new near-field
        _bot_perc    = 0.5
        self.bottom_row_index = self.bev_h - _bot_off                          # 470
        _end          = int((1 - _bot_perc) * self.bev_h)                      # 240
        self.step     = int(-(self.bev_h * _bot_perc / _slices))               # -12
        self.real_slices = int((_end - self.bottom_row_index) // self.step)    # 19
        self.top_row_index = self.bottom_row_index + self.real_slices * self.step  # 242
        self.height_norm   = np.linspace(0, 1, self.real_slices + 1)           # 20 values
        self.slices = _slices
        # Heading evaluated halfway up (~0.25 m ahead) for early curve anticipation.
        self.y_hdg = self.bottom_row_index + (self.real_slices // 2) * self.step  # ≈ 362

        # ── Peak detection ────────────────────────────────────────────────────
        # Tuned for CLAHE-equalized grayscale; lane markings measure V ≈ 80–152.
        # sq_min_height_curr auto-lowers each frame toward observed peak intensity.
        self.peaks_min_width    = 3
        self.peaks_max_width    = 80    # large flat limit: curved/horizontal lines appear
                                        # very wide (width = real_w / sin(angle)); far-field
                                        # BEV stretch also needs headroom
        self.sq_min_height      = 80    # hard floor — reset on sparse frames
        self.sq_min_height_curr = 80    # adaptive threshold used this frame
        self.sq_pix_dif         = 3
        self.sq_min_height_dif  = 30    # edge sharpness (was 60 in original 455 camera)
        self.sq_width_error     = 15

        # ── Clustering ────────────────────────────────────────────────────────
        self.max_allowed_dist = 0.30 * self.bev_w   # 192 px — wide enough to track a
                                                     # line that sweeps laterally on bends
        self.w_width    = 0.3
        self.w_expected = 0.7
        self.min_peaks  = 2     # min peaks for a valid lane track (was 3; 2 handles sparse curve visibility)
        self.opt_perc   = 0.3   # "enough" peak fraction

        # ── Polynomial sanity ─────────────────────────────────────────────────
        # 0.2 (was 0.1) allows tighter-curve polynomials through; the original
        # LaneAssist value was calibrated for a wider-radius track.
        self.extreme_coef_2 = 0.2   # |a| must be < 0.2 for 2nd-degree
        self.extreme_coef_1 = 3.0   # |a| must be < 3.0 for 1st-degree

        # ── Certainty / trust ─────────────────────────────────────────────────
        self.cert_from_peaks  = 0.5
        self.min_single_cert  = 80
        self.min_dual_cert    = 60
        self.allowed_cert_dif = 20.0
        self.prev_left_coef   = None
        self.prev_right_coef  = None

        # ── Output state ──────────────────────────────────────────────────────
        self.smooth_e      = 0.0
        self.smooth_hdg    = 0.0    # EMA-filtered heading; decays slowly when lanes lost
        self.meters_per_px = 0.8 / 340.0
        self._last_lx       = float(self.bev_w // 2 - 150)
        self._last_rx       = float(self.bev_w // 2 + 150)
        self._lane_width_px = float(self._last_rx - self._last_lx)  # ~300 px; updated when both lanes seen
        self._last_centre   = float(self.bev_w / 2.0)               # EMA of lane centre; single-line disambiguation

        self.get_logger().info('Lane Detector (LaneAssist) started.')

    # ══════════════════════ LaneAssist core methods ═══════════════════════════

    def _find_lane_peaks(self, row):
        """Detect square-pulse lane markings in one horizontal grayscale row.

        Flat width limit: a curved line that runs nearly horizontal appears very
        wide in a horizontal scan (apparent_width = real_width / sin(angle)).
        Using a large flat cap lets near-field horizontal sections be detected;
        far-field BEV stretch is also within the same cap.
        """
        upper = self.peaks_max_width + self.sq_width_error   # flat, height-independent

        inside  = False
        peaks   = []
        pix_num = 0
        pd = self.sq_pix_dif

        for i in range(pd, len(row) - pd):
            px      = row[i]
            d_start = int(px) - int(row[i - pd])
            d_end   = int(px) - int(row[i + pd])

            if inside:
                if d_end > self.sq_min_height_dif:         # falling edge → end of pulse
                    inside = False
                    if self.peaks_min_width <= pix_num <= upper:
                        peaks.append(i - (pix_num - pd) // 2)
                    pix_num = 0
                elif px > self.sq_min_height_curr:          # still inside bright region
                    pix_num += 1
                else:
                    inside = False
            else:
                if px > self.sq_min_height_curr and d_start > self.sq_min_height_dif:
                    inside = True
                    pix_num += 1
        return peaks

    def _verify_expected(self, lane, height, x_val):
        """Check if x_val is plausible for this lane track at the given height.

        Extrapolates from previous lane points and returns (ok, distance+penalty).
        """
        dists = []
        xd = lane[-1][0] - lane[0][0]
        yd = lane[-1][1] - lane[0][1]
        xdy = xd / yd if yd != 0 else 0
        dists.append(abs(x_val - ((height - lane[-1][1]) * xdy + lane[-1][0])))

        # Penalise lanes that skipped slices
        punish  = self.max_allowed_dist // 2 if xdy == 0 else 0
        punish += ((height - lane[-1][1]) / self.step - 1) / self.real_slices \
                  * self.max_allowed_dist

        if len(lane) > 3:
            # Additional predictions from last two consecutive point pairs
            for a, b in [(-1, -2), (-2, -3)]:
                xd2 = lane[a][0] - lane[b][0]
                yd2 = lane[a][1] - lane[b][1]
                xdy2 = xd2 / yd2 if yd2 != 0 else 0
                dists.append(
                    abs(x_val - ((height - lane[a][1]) * xdy2 + lane[a][0])))

        d     = min(dists)
        limit = self.max_allowed_dist if xdy == 0 else self.max_allowed_dist // 2
        return (True, d + punish) if d < limit else (False, d + punish)

    def _add_match(self, ld, pd, li, pi, dist):
        ld[li]['point_index'] = pi;  ld[li]['distance'] = dist
        pd[pi]['used']        = True; pd[pi]['lane_index'] = li

    def _find_best_matches(self, ld, pd, pts, lanes, height):
        """Hungarian-style matching: assign each lane its closest valid peak."""
        run_again = False
        for li in range(len(lanes)):
            if ld[li]['point_index'] != -1:
                continue
            at_one = False
            for pi in range(len(pts)):
                x0, y0 = pts[pi], height
                x1, y1 = lanes[li][-1]
                wd = abs((x0 - x1) * self.step / (y0 - y1))

                if wd >= self.max_allowed_dist:
                    if at_one:
                        break   # peaks are sorted; nothing closer further right
                    continue

                ok, we = self._verify_expected(lanes[li], height, pts[pi])
                if not ok:
                    continue
                at_one = True
                td = self.w_width * wd + self.w_expected * we

                pu = pd[pi]['used']
                la = ld[li]['point_index']

                if not pu and la == -1:
                    self._add_match(ld, pd, li, pi, td)

                elif pu and la == -1:
                    pli = pd[pi]['lane_index']
                    if ld[pli]['distance'] > td:
                        run_again = True
                        ld[pli]['point_index'] = -1
                        ld[pli]['distance']    = -1
                        self._add_match(ld, pd, li, pi, td)

                elif not pu and la != -1:
                    ppi = ld[li]['point_index']
                    if ld[li]['distance'] > td:
                        run_again = True
                        pd[ppi]['used']       = False
                        pd[ppi]['lane_index'] = -1
                        self._add_match(ld, pd, li, pi, td)

                else:   # both already assigned
                    if pd[pi]['lane_index'] == li and ld[li]['point_index'] == pi:
                        continue
                    pli = pd[pi]['lane_index']
                    ppi = ld[li]['point_index']
                    if ld[li]['distance'] > td and ld[pli]['distance'] > td:
                        run_again = True
                        pd[ppi]['used']       = False
                        pd[ppi]['lane_index'] = -1
                        ld[pli]['point_index'] = -1
                        ld[pli]['distance']    = -1
                        self._add_match(ld, pd, li, pi, td)

        if run_again:
            all_pts   = all(p['used'] for p in pd)
            all_lanes = all(l['point_index'] != -1 for l in ld)
            if all_pts or all_lanes:
                run_again = False
        return run_again

    def _cluster(self, pts, height, lanes):
        """Assign x-positions (pts) to existing lane tracks or create new ones."""
        if not lanes:
            return [[[x, height]] for x in pts]

        ll  = len(lanes)
        ld  = [{'point_index': -1, 'distance': -1} for _ in range(ll)]
        pd  = [{'used': False, 'lane_index': -1}   for _ in range(len(pts))]

        if self._find_best_matches(ld, pd, pts, lanes, height):
            self._find_best_matches(ld, pd, pts, lanes, height)

        # Append matched points; remove them from pts (index shifts as we pop)
        removed = 0
        for pi, p in enumerate(pd):
            if p['used']:
                lanes[p['lane_index']].append([pts[pi - removed], height])
                pts.pop(pi - removed)
                removed += 1

        # Insert remaining unmatched pts as new lane tracks (preserve sort order)
        ins = 0
        idx = 0
        for x in pts:
            if x < lanes[ins][-1][0]:
                lanes.insert(ins, [[x, height]]); ins += 1; idx = ins
            elif x > lanes[ll + ins - 1][-1][0]:
                lanes.append([[x, height]]); ins += 1
            else:
                for j in range(idx, ll + ins - 1):
                    if lanes[j][-1][0] < x < lanes[j + 1][-1][0]:
                        lanes.insert(j + 1, [[x, height]]); ins += 1; idx = j + 1
                        break
        return lanes

    def _peaks_detection(self, gray):
        """Scan BEV rows bottom→top, detect square-pulse peaks, cluster into lanes."""
        peaks = []
        lanes = []
        for y in range(self.bottom_row_index, self.top_row_index - 1, self.step):
            row = [int(v) for v in gray[y]]
            ps  = self._find_lane_peaks(row)
            peaks.extend([p, y] for p in ps)
            lanes = self._cluster(ps, y, lanes)   # ps modified in-place (matched pts removed)
        return lanes, peaks

    def _choose_lanes(self, lanes):
        """Select most-right left-half lane and most-left right-half lane."""
        left, right = [], []
        lanes = [ln for ln in lanes if len(ln) >= self.min_peaks]
        mid   = self.bev_w / 2.0

        # Single track: assign to whichever side it's closer to based on history.
        # Use the BOTTOMMOST point x (nearest to car) — using all-point mean pulls
        # toward far-field curve data and causes misclassification on bends.
        if len(lanes) == 1:
            lane = lanes[0]
            bot_x = sorted(lane, key=lambda p: p[1], reverse=True)[0][0]
            if abs(bot_x - self._last_lx) <= abs(bot_x - self._last_rx):
                left = lane
            else:
                right = lane
            return left, right

        for lane in lanes:
            n = len(lane)
            if lane[0][0] <= mid:
                if not left or n > self.slices * self.opt_perc or n > len(left):
                    left = lane
            else:
                if not right:
                    right = lane
                    if n > self.slices * self.opt_perc:
                        break
                    continue
                if n > self.slices * self.opt_perc:
                    right = lane
                    break
        return left, right

    def _lstsq_poly(self, pts, degree):
        """Least-squares fit: x = poly(y).  pts = [[y, x], ...]"""
        ys   = [p[0] for p in pts]
        xs   = [p[1] for p in pts]
        coef, _, _, _ = np.linalg.lstsq(np.vander(ys, degree + 1), xs, rcond=None)
        return coef

    def _check_coefs(self, coef):
        if coef is None:
            return None
        limits = {3: self.extreme_coef_2, 2: self.extreme_coef_1}
        return coef if abs(coef[0]) < limits[len(coef)] else None

    def _fit_poly(self, lane):
        """Fit 2nd-degree polynomial to lane points; fall back to 1st-degree if sparse."""
        if not lane:
            return None
        pts = [[p[1], p[0]] for p in lane]     # [[y, x], ...]
        if len(pts) > self.slices * 0.3:        # enough points → quadratic
            return self._check_coefs(self._lstsq_poly(pts, 2))
        coef1 = self._check_coefs(self._lstsq_poly(pts, 1))
        if coef1 is not None:
            return np.array([0.0, coef1[0], coef1[1]])
        return None

    def _certainty(self, new_c, prev_c, peaks):
        if new_c is None or prev_c is None:
            return 0.0
        sim  = max(0.0, 100.0 - float(np.sqrt(np.mean((new_c - prev_c) ** 2))))
        ppct = len(peaks) / self.real_slices * 100.0
        return round(self.cert_from_peaks * ppct + (1.0 - self.cert_from_peaks) * sim, 2)

    def _post_process(self, lc, left, rc, right):
        l_cert = self._certainty(lc, self.prev_left_coef,  left)
        r_cert = self._certainty(rc, self.prev_right_coef, right)
        self.prev_left_coef  = lc
        self.prev_right_coef = rc
        if abs(l_cert - r_cert) > self.allowed_cert_dif:
            trust_l = l_cert >= r_cert
            trust_r = not trust_l
        else:
            trust_l = trust_r = True
        return lc, rc, l_cert, r_cert, trust_l, trust_r

    # ══════════════════════════ ROS callback ══════════════════════════════════

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        H_img, W_img = frame.shape[:2]

        # 1. ROI mask — bottom extended to 395 to include near-field beside the car.
        #    The car body is blue so its pixels stay dark after CLAHE and are
        #    invisible to the peak detector — no body mask needed.
        frame[:235, :] = 0
        frame[395:, :] = 0

        # 2. BEV warp
        warped = cv2.warpPerspective(frame, self.M, (W_img, H_img))

        # 3. CLAHE-equalised grayscale — lift dim far-field pixels into detectable range
        gray    = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        clahe   = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        gray_eq = clahe.apply(gray)

        # 4. LaneAssist peak detection
        lanes, peaks = self._peaks_detection(gray_eq)
        left, right  = self._choose_lanes(lanes)

        # 5. Adaptive min_height — mirror of lanes_detection in detect.py
        if peaks:
            hl = [gray_eq[y][x] for x, y in left]
            hr = [gray_eq[y][x] for x, y in right]
            if hl and hr:
                tmp = min(min(hl), min(hr)) - 20
                if self.sq_min_height <= tmp <= 170:
                    self.sq_min_height_curr = tmp
            if len(left) + len(right) < 10:
                self.sq_min_height_curr = self.sq_min_height

        # 6. Polynomial fitting
        lc = self._fit_poly(left)
        rc = self._fit_poly(right)

        # 7. Certainty / trust
        lc, rc, l_cert, r_cert, trust_l, trust_r = self._post_process(lc, left, rc, right)

        # 8. Crosstrack error — use raw near-field points (bottom 3 slices).
        #    With _bot_off=3, bottom_row_index=477; cutoff_y = 477+3*(-12) = 441.
        #    These slices correspond to raw rows ≈381-390, right beside the car.
        cutoff_y = self.bottom_row_index + 3 * self.step

        def near_x(lane):
            pts = [p for p in lane if p[1] >= cutoff_y]
            if not pts:
                pts = lane  # fallback when no near-field detections
            pts = sorted(pts, key=lambda p: p[1], reverse=True)[:3]
            return float(sum(p[0] for p in pts)) / len(pts)

        lx = near_x(left)  if (trust_l and lc is not None and left)  else None
        rx = near_x(right) if (trust_r and rc is not None and right) else None

        # Keep lane-width estimate fresh (only update when both lines are solid)
        if lx is not None and rx is not None and rx > lx + 50:
            self._lane_width_px = 0.7 * self._lane_width_px + 0.3 * (rx - lx)
        half_w = self._lane_width_px / 2.0

        # Centre estimation:
        # Two-line case: straightforward midpoint.
        # Single-line case: the detected line could be the left or right boundary.
        #   Using lx+half_w / rx-half_w directly trusts _choose_lanes classification,
        #   which can be wrong on bends.  Instead, compute both interpretations and
        #   pick whichever gives a centre closest to the previous frame (_last_centre
        #   continuity).  This is robust even if classification flipped.
        if lx is not None and rx is not None:
            if lx >= rx:
                lx = rx - self._lane_width_px
            self._last_lx = float(lx)
            self._last_rx = float(rx)
            lane_centre = (lx + rx) / 2.0
        elif lx is not None:
            ctr_as_left  = lx + half_w
            ctr_as_right = lx - half_w
            if abs(ctr_as_left - self._last_centre) <= abs(ctr_as_right - self._last_centre):
                self._last_lx = float(lx)
                lane_centre = ctr_as_left
            else:                               # _choose_lanes misclassified: it's the right line
                self._last_rx = float(lx)
                lane_centre = ctr_as_right
        elif rx is not None:
            ctr_as_right = rx - half_w
            ctr_as_left  = rx + half_w
            if abs(ctr_as_right - self._last_centre) <= abs(ctr_as_left - self._last_centre):
                self._last_rx = float(rx)
                lane_centre = ctr_as_right
            else:                               # _choose_lanes misclassified: it's the left line
                self._last_lx = float(rx)
                lane_centre = ctr_as_left
        else:
            lane_centre = self._last_centre     # both lost — hold previous

        self._last_centre = lane_centre
        err_px = self.bev_w / 2.0 - lane_centre

        lanes_active = (trust_l and lc is not None) or (trust_r and rc is not None)
        if lanes_active:
            crosstrack_raw = err_px * self.meters_per_px
            self.smooth_e  = 0.35 * crosstrack_raw + 0.65 * self.smooth_e
        else:
            # Both polynomials lost — decay the existing correction instead of
            # feeding near-zero from stale centred positions (which collapses
            # smooth_e to 0 in ~2 frames via the normal EMA path).
            self.smooth_e *= 0.97

        # 9. Heading error from polynomial derivative at look-ahead row (y_hdg ≈ 0.25 m ahead)
        #    Evaluating mid-scan rather than at the car's position gives curve anticipation:
        #    the derivative already reflects the upcoming bend before crosstrack error builds.
        def poly_slope(coef):
            return 2.0 * coef[0] * self.y_hdg + coef[1]   # dx/dy at look-ahead row

        slopes = []
        if trust_l and lc is not None:
            slopes.append(poly_slope(lc))
        if trust_r and rc is not None:
            slopes.append(poly_slope(rc))

        if slopes:
            # Lanes visible — track heading with a responsive EMA
            hdg_raw = -math.atan(sum(slopes) / len(slopes))
            self.smooth_hdg = 0.35 * hdg_raw + 0.65 * self.smooth_hdg
        else:
            # Lanes lost (curve exit blind spot) — decay slowly so the car
            # continues turning rather than going straight immediately.
            # 0.95/frame → still 77% of last heading after 5 frames (0.25 s).
            self.smooth_hdg *= 0.95

        # 10. Publish
        e_msg  = Float32(); e_msg.data  = float(self.smooth_e)
        th_msg = Float32(); th_msg.data = float(self.smooth_hdg)
        hdg = self.smooth_hdg   # alias for visualisation
        self.e_pub.publish(e_msg)
        self.th_pub.publish(th_msg)

        # 11. Debug visualisation
        self._show(gray_eq, warped, left, right, lc, rc,
                   trust_l, trust_r, l_cert, r_cert,
                   peaks, lane_centre, hdg)

    # ──────────────────────────────────────────────────────────────────────────

    def _show(self, gray_eq, warped, left, right, lc, rc,
              trust_l, trust_r, l_cert, r_cert, peaks, lane_centre, hdg):
        out = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
        H, W = out.shape[:2]

        # Slice scan lines
        for y in range(self.bottom_row_index, self.top_row_index - 1, self.step):
            cv2.line(out, (0, y), (W, y), (50, 50, 50), 1)

        # All detected peaks (magenta)
        for p in peaks:
            cv2.circle(out, (p[0], p[1]), 3, (255, 0, 255), -1)

        # Left lane peaks (orange), right lane peaks (cyan)
        for x, y in left:
            cv2.circle(out, (x, y), 5, (0, 140, 255), -1)
        for x, y in right:
            cv2.circle(out, (x, y), 5, (255, 200, 0), -1)

        # Fitted polynomials — draw only within scan region
        end_y = self.top_row_index

        def draw_coef(coef, col):
            if coef is None:
                return
            a, b, c = coef
            for i in range(self.bottom_row_index, end_y, -3):
                x0 = int(a * i**2 + b * i + c)
                k  = i - 3
                x1 = int(a * k**2 + b * k + c)
                if 0 <= x0 < W and 0 <= x1 < W:
                    cv2.line(out, (x0, i), (x1, k), col, 2)

        draw_coef(lc if trust_l else None, (255, 128, 0))
        draw_coef(rc if trust_r else None, (0, 128, 255))

        # Reference lines
        cv2.line(out, (W // 2, 0),           (W // 2, H),           (0, 255, 255), 1)
        cv2.line(out, (int(lane_centre), 0), (int(lane_centre), H), (255, 255, 0), 2)
        cv2.line(out, (0, self.bottom_row_index), (W, self.bottom_row_index), (0, 200, 100), 1)
        cv2.line(out, (0, self.y_hdg),       (W, self.y_hdg),       (0, 100, 255), 1)  # heading row

        cv2.putText(out,
                    f'CTE={self.smooth_e * 100:+.1f}cm  HDG={math.degrees(hdg):+.1f}deg',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
        cv2.putText(out,
                    f'L={l_cert:.0f}%({trust_l})  R={r_cert:.0f}%({trust_r})'
                    f'  thr={self.sq_min_height_curr}',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow('LaneAssist peaks', out)
        cv2.imshow('CLAHE gray',       gray_eq)
        cv2.imshow('BEV warped',       warped)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
