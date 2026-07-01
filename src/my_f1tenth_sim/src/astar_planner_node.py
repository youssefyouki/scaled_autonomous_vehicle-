#!/usr/bin/env python3
"""
A* Path Planner + Path Tracker
=================================
Plans a collision-free path from the robot's current position to a goal
clicked in RViz, using the A* algorithm on the pre-built occupancy-grid map.

The planned path is tracked at 20 Hz and converted to CTE and heading-error
signals on the same topics as the lane detector / RRT planner, so any
existing controller (Stanley, Pure Pursuit, Linderoth) can follow the path
unchanged.

Subscribes:
  /odom              (nav_msgs/Odometry)          current pose
  /goal_pose         (geometry_msgs/PoseStamped)  goal from RViz "2D Goal Pose"

Publishes:
  /planned_path                (nav_msgs/Path)
  /perception/crosstrack_error (std_msgs/Float32)
  /perception/heading_error    (std_msgs/Float32)
  /astar/markers               (visualization_msgs/MarkerArray)
  /astar/map                   (nav_msgs/OccupancyGrid)
  /astar/active                (std_msgs/Bool)   — mutes lane detector
  /astar/goal_reached          (std_msgs/Bool)   — signals controller to stop

Parameters:
  map_yaml      path to map YAML         (default: .../track_map.yaml)
  goal_tol_m    goal-arrival radius  m   (default 0.20)
  inflate_m     obstacle inflation   m   (default 0.05)
  lookahead_m   path-tracking distance m (default 0.40)
  seed_x / seed_y  world point on drivable surface (for reference only)

RViz setup:
  • Fixed Frame = odom
  • Add → By topic → /planned_path  (Path display)
  • Add → By topic → /astar/markers (MarkerArray)
  • Add → By topic → /astar/map     (Map)
  • Click "2D Goal Pose" to send a goal
"""

import heapq
import math
import os
import threading

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float32
from visualization_msgs.msg import Marker, MarkerArray


# ── Helpers ───────────────────────────────────────────────────────────────────

def _quat_to_yaw(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _wrap(a: float) -> float:
    """Wrap angle to (−π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ── Main node ─────────────────────────────────────────────────────────────────

class AStarPlannerNode(Node):

    # 8-connected moves: (dcol, drow, cost)
    _MOVES = [
        ( 1,  0, 1.0), (-1,  0, 1.0), ( 0,  1, 1.0), ( 0, -1, 1.0),
        ( 1,  1, 1.4142), ( 1, -1, 1.4142), (-1,  1, 1.4142), (-1, -1, 1.4142),
    ]

    def __init__(self):
        super().__init__('astar_planner')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter(
            'map_yaml',
            os.path.expanduser(
                '~/scaled_autonomous_vehicle-/src/my_f1tenth_sim/track_map.yaml'))
        self.declare_parameter('goal_tol_m',  0.20)
        self.declare_parameter('inflate_m',   0.05)
        self.declare_parameter('lookahead_m', 0.40)
        self.declare_parameter('seed_x',  2.2668)
        self.declare_parameter('seed_y', -0.1515)

        map_yaml          = self.get_parameter('map_yaml').value
        self._goal_tol_m  = float(self.get_parameter('goal_tol_m').value)
        self._inflate_m   = float(self.get_parameter('inflate_m').value)
        self._lookahead_m = float(self.get_parameter('lookahead_m').value)

        # ── Load map ──────────────────────────────────────────────────────────
        self._load_map(map_yaml)

        _latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        # ── State ─────────────────────────────────────────────────────────────
        self._pose: tuple | None = None
        self._path: list[tuple[float, float]] = []
        self._path_idx = 0
        self._path_gen = 0
        self._lock = threading.Lock()
        self._stopped = False

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(Odometry,    '/odom',      self._odom_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self._map_pub    = self.create_publisher(OccupancyGrid, '/astar/map',                   _latched)
        self._path_pub   = self.create_publisher(Path,          '/planned_path',                10)
        self._cte_pub    = self.create_publisher(Float32,       '/perception/crosstrack_error', 10)
        self._hdg_pub    = self.create_publisher(Float32,       '/perception/heading_error',    10)
        self._marker_pub = self.create_publisher(MarkerArray,   '/astar/markers',               10)
        self._drive_pub  = self.create_publisher(Twist,         '/cmd_vel',                     10)
        self._active_pub = self.create_publisher(Bool,          '/astar/active',                _latched)
        self._goal_reached_pub = self.create_publisher(Bool,    '/astar/goal_reached',          _latched)

        self._pub_map()
        self._active_pub.publish(Bool(data=False))
        self._goal_reached_pub.publish(Bool(data=False))

        self.create_timer(0.05, self._tick)
        self.create_timer(2.0,  self._pub_map)

        self.get_logger().info(
            'A* Planner ready — open RViz, set Fixed Frame=odom, '
            'click "2D Goal Pose" to plan a path.')

    # ── Map loading ───────────────────────────────────────────────────────────

    def _load_map(self, yaml_path: str):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        self._res      = float(cfg['resolution'])
        self._origin_x = float(cfg['origin'][0])
        self._origin_y = float(cfg['origin'][1])
        negate         = int(cfg.get('negate', 0))
        free_thresh    = float(cfg.get('free_thresh', 0.25))

        pgm = cfg['image']
        if not os.path.isabs(pgm):
            pgm = os.path.join(os.path.dirname(yaml_path), pgm)
        raw = cv2.imread(pgm, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise RuntimeError(f'Cannot read map image: {pgm}')

        self._map_h, self._map_w = raw.shape

        occ = (raw.astype(np.float32) if negate
               else (255 - raw.astype(np.float32))) / 255.0
        free_mask = (occ < free_thresh).astype(np.uint8) * 255

        # ── Texture-based obstacle map ─────────────────────────────────────────
        tex_dir  = os.path.join(os.path.dirname(yaml_path), 'materials', 'textures')
        tex_path = os.path.join(tex_dir, 'Test_Track_no_stop_lines_6x6.png')
        if not os.path.exists(tex_path):
            tex_path = os.path.join(tex_dir, 'Test_Track_6x6.png')
        tex = cv2.imread(tex_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(tex_path) else None
        if tex is not None:
            obs_tex = (tex > 128).astype(np.uint8) * 255
            pre_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            obs_tex = cv2.dilate(obs_tex, pre_k)
            obs_scaled = cv2.resize(obs_tex, (self._map_w, self._map_h),
                                    interpolation=cv2.INTER_AREA)
            _, obs_clean = cv2.threshold(obs_scaled, 30, 255, cv2.THRESH_BINARY)
            free_mask = 255 - obs_clean
            self.get_logger().info(
                f'Texture map loaded: {tex_path}  '
                f'obstacle_px={int((obs_clean > 0).sum())}')
        else:
            self.get_logger().info('Texture not found — using PGM free-space map')

        # Inflate obstacles
        inflate_px = max(1, math.ceil(self._inflate_m / self._res))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * inflate_px + 1, 2 * inflate_px + 1))
        self._free = cv2.erode(free_mask, kernel)

        self.get_logger().info(
            f'Map: {self._map_w}×{self._map_h} px  '
            f'res={self._res} m/px  inflate={inflate_px} px  '
            f'obstacle_px={(self._map_h * self._map_w - int((self._free > 0).sum())):,}  '
            f'world [{self._origin_x:.2f}..{self._origin_x + self._map_w * self._res:.2f}] x '
            f'[{self._origin_y:.2f}..{self._origin_y + self._map_h * self._res:.2f}]')

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _w2p(self, x: float, y: float) -> tuple[int, int]:
        col = int((x - self._origin_x) / self._res)
        row = int(self._map_h - 1 - (y - self._origin_y) / self._res)
        return col, row

    def _p2w(self, col: int, row: int) -> tuple[float, float]:
        x = self._origin_x + col * self._res
        y = self._origin_y + (self._map_h - 1 - row) * self._res
        return x, y

    def _free_px(self, col: int, row: int) -> bool:
        if col < 0 or col >= self._map_w or row < 0 or row >= self._map_h:
            return False
        return bool(self._free[row, col] > 0)

    # ── Line-of-sight (Bresenham) ─────────────────────────────────────────────

    def _los_free(self, c1, r1, c2, r2) -> bool:
        dc, dr = abs(c2 - c1), abs(r2 - r1)
        sc = 1 if c2 > c1 else -1
        sr = 1 if r2 > r1 else -1
        c, r, err = c1, r1, dc - dr
        for _ in range(dc + dr + 2):
            if not self._free_px(c, r):
                return False
            if c == c2 and r == r2:
                return True
            e2 = 2 * err
            if e2 > -dr:
                err -= dr;  c += sc
            if e2 <  dc:
                err += dc;  r += sr
        return True

    # ── A* ────────────────────────────────────────────────────────────────────

    def _astar(self, start: tuple[int, int],
               goal:  tuple[int, int],
               start_yaw: float | None = None) -> list[tuple[int, int]] | None:
        sc, sr = start
        gc, gr = goal

        # ── Virtual wall behind the car ───────────────────────────────────────
        # Prevents stepping directly backward; the planner must go forward and
        # loop around the track to reach a behind-car goal.
        if start_yaw is not None:
            free_work = self._free.copy()
            sw_x, sw_y = self._p2w(sc, sr)
            wall_dist = 0.25
            half_span = 1.50
            wc_x = sw_x - wall_dist * math.cos(start_yaw)
            wc_y = sw_y - wall_dist * math.sin(start_yaw)
            px =  math.sin(start_yaw)
            py = -math.cos(start_yaw)
            p1 = self._w2p(wc_x + half_span * px, wc_y + half_span * py)
            p2 = self._w2p(wc_x - half_span * px, wc_y - half_span * py)
            cv2.line(free_work, p1, p2, color=0, thickness=4)
        else:
            free_work = self._free

        original_free = self._free
        self._free = free_work

        # ── Core A* search ────────────────────────────────────────────────────
        h = lambda c, r: math.hypot(c - gc, r - gr)

        # g-score array and closed-set array (numpy for fast access)
        g_arr    = np.full((self._map_h, self._map_w), np.inf, dtype=np.float32)
        closed   = np.zeros((self._map_h, self._map_w), dtype=bool)
        g_arr[sr, sc] = 0.0

        # came_from stores (pcol, prow) for each expanded cell
        came_from: dict[tuple[int, int], tuple[int, int]] = {}

        # heap entries: (f, g, col, row)
        heap: list[tuple[float, float, int, int]] = []
        heapq.heappush(heap, (h(sc, sr), 0.0, sc, sr))

        found = False
        while heap:
            f, g_cur, c, r = heapq.heappop(heap)

            if closed[r, c]:
                continue
            closed[r, c] = True

            if c == gc and r == gr:
                found = True
                break

            for dc, dr, move_cost in self._MOVES:
                nc, nr = c + dc, r + dr
                if not self._free_px(nc, nr):
                    continue
                if closed[nr, nc]:
                    continue
                new_g = g_cur + move_cost
                if new_g < g_arr[nr, nc]:
                    g_arr[nr, nc] = new_g
                    came_from[(nc, nr)] = (c, r)
                    heapq.heappush(heap, (new_g + h(nc, nr), new_g, nc, nr))

        self._free = original_free

        if not found:
            return None

        # Reconstruct path
        path: list[tuple[int, int]] = []
        node: tuple[int, int] = (gc, gr)
        while node in came_from:
            path.append(node)
            node = came_from[node]
        path.append((sc, sr))
        path.reverse()
        return path

    # ── Path shortcutting ─────────────────────────────────────────────────────

    def _shortcut(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Remove intermediate waypoints that have a clear line of sight."""
        if len(path) < 3:
            return path
        result = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._los_free(*result[-1], *path[j]):
                    break
                j -= 1
            result.append(path[j])
            i = j
        return result

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose
        self._pose = (p.position.x, p.position.y, _quat_to_yaw(p.orientation))

    def _goal_cb(self, msg: PoseStamped):
        if self._pose is None:
            self.get_logger().warn('No odometry yet — cannot plan.')
            return

        gx, gy = msg.pose.position.x, msg.pose.position.y
        sx, sy, syaw = self._pose

        self._stopped = False
        self._goal_reached_pub.publish(Bool(data=False))

        start_px = self._w2p(sx, sy)
        goal_px  = self._w2p(gx, gy)

        if not self._free_px(*start_px):
            self.get_logger().warn(
                f'Start pixel {start_px} is in an obstacle — check map/odom alignment.')

        if not self._free_px(*goal_px):
            self.get_logger().warn(
                f'Goal pixel {goal_px} is in an obstacle — choose a free-space goal.')
            return

        self.get_logger().info(
            f'Goal ({gx:.2f}, {gy:.2f})  start ({sx:.2f}, {sy:.2f})'
            f'  heading={math.degrees(syaw):.1f}° — planning with A*...')

        path_px = self._astar(start_px, goal_px, start_yaw=syaw)
        if path_px is None:
            self.get_logger().error('A* failed — goal may be unreachable.')
            return

        # Shortcut to reduce waypoint count before converting to world coords
        path_px = self._shortcut(path_px)

        path_world = [self._p2w(c, r) for c, r in path_px]
        path_world[-1] = (gx, gy)

        with self._lock:
            self._path      = path_world
            self._path_idx  = 0
            self._path_gen += 1

        self._active_pub.publish(Bool(data=True))
        self.get_logger().info(f'Path ready: {len(path_world)} waypoints.')
        self._pub_path(path_world)
        self._pub_markers(path_world)

    # ── Map publisher ─────────────────────────────────────────────────────────

    def _pub_map(self):
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.info.resolution = self._res
        msg.info.width      = self._map_w
        msg.info.height     = self._map_h
        msg.info.origin.position.x    = self._origin_x
        msg.info.origin.position.y    = self._origin_y
        msg.info.origin.orientation.w = 1.0
        flipped = np.flipud(self._free)
        occ = np.where(flipped > 0, 0, 100).astype(np.int8)
        msg.data = occ.flatten().tolist()
        self._map_pub.publish(msg)

    # ── Path publishers ───────────────────────────────────────────────────────

    def _pub_path(self, path_world: list[tuple[float, float]]):
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        for x, y in path_world:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._path_pub.publish(msg)

    def _pub_markers(self, path_world: list[tuple[float, float]]):
        now = self.get_clock().now().to_msg()
        ma  = MarkerArray()

        line              = Marker()
        line.header.stamp    = now
        line.header.frame_id = 'odom'
        line.ns     = 'astar_path'
        line.id     = 0
        line.type   = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.05
        line.color.r, line.color.g, line.color.b, line.color.a = 0.2, 0.6, 1.0, 1.0
        for x, y in path_world:
            p = Point(); p.x = x; p.y = y; p.z = 0.05
            line.points.append(p)
        ma.markers.append(line)

        for i, (x, y) in enumerate(path_world):
            s = Marker()
            s.header.stamp    = now
            s.header.frame_id = 'odom'
            s.ns     = 'astar_wp'
            s.id     = i + 1
            s.type   = Marker.SPHERE
            s.action = Marker.ADD
            s.pose.position.x = x
            s.pose.position.y = y
            s.pose.position.z = 0.05
            s.pose.orientation.w = 1.0
            s.scale.x = s.scale.y = s.scale.z = 0.10
            s.color.r, s.color.g, s.color.b, s.color.a = 0.2, 0.7, 1.0, 1.0
            ma.markers.append(s)

        self._marker_pub.publish(ma)

    def _pub_car_marker(self):
        if self._pose is None:
            return
        rx, ry, ryaw = self._pose
        now = self.get_clock().now().to_msg()
        ma  = MarkerArray()

        sph = Marker()
        sph.header.stamp    = now
        sph.header.frame_id = 'odom'
        sph.ns     = 'car_pose';  sph.id = 200
        sph.type   = Marker.SPHERE;  sph.action = Marker.ADD
        sph.pose.position.x = rx;  sph.pose.position.y = ry
        sph.pose.position.z = 0.15
        sph.pose.orientation.w = 1.0
        sph.scale.x = sph.scale.y = sph.scale.z = 0.25
        sph.color.r = 1.0; sph.color.g = 0.4; sph.color.b = 0.0; sph.color.a = 1.0
        ma.markers.append(sph)

        arr = Marker()
        arr.header.stamp    = now
        arr.header.frame_id = 'odom'
        arr.ns     = 'car_pose';  arr.id = 201
        arr.type   = Marker.ARROW;  arr.action = Marker.ADD
        arr.pose.position.x = rx;  arr.pose.position.y = ry
        arr.pose.position.z = 0.15
        arr.pose.orientation.z = math.sin(ryaw / 2.0)
        arr.pose.orientation.w = math.cos(ryaw / 2.0)
        arr.scale.x = 0.55
        arr.scale.y = 0.08
        arr.scale.z = 0.08
        arr.color.r = 1.0; arr.color.g = 0.6; arr.color.b = 0.0; arr.color.a = 1.0
        ma.markers.append(arr)

        self._marker_pub.publish(ma)

    # ── Path tracking at 20 Hz ────────────────────────────────────────────────

    def _tick(self):
        self._pub_car_marker()

        with self._lock:
            path = list(self._path)
            idx  = self._path_idx
            gen  = self._path_gen

        if not path or self._pose is None:
            return

        rx, ry, ryaw = self._pose

        if self._stopped:
            self._drive_pub.publish(Twist())
            return

        dist_to_goal = math.hypot(rx - path[-1][0], ry - path[-1][1])
        if dist_to_goal < self._goal_tol_m:
            self.get_logger().info(
                f'Goal reached (dist={dist_to_goal:.2f} m) — stopping.')
            self._stopped = True
            self._goal_reached_pub.publish(Bool(data=True))
            self._drive_pub.publish(Twist())
            return

        # Advance progress index
        while idx < len(path) - 1:
            wp_x, wp_y   = path[idx]
            nxt_x, nxt_y = path[idx + 1]
            seg_x = nxt_x - wp_x;  seg_y = nxt_y - wp_y
            seg_len = math.hypot(seg_x, seg_y)
            if seg_len < 1e-9:
                idx += 1
                continue
            proj = ((rx - wp_x) * seg_x + (ry - wp_y) * seg_y) / seg_len
            if proj > 0:
                idx += 1
            elif math.hypot(rx - wp_x, ry - wp_y) < self._lookahead_m * 0.3:
                idx += 1
            else:
                break

        with self._lock:
            if self._path_gen == gen:
                self._path_idx = idx

        # Lookahead point
        lx, ly = path[min(idx + 1, len(path) - 1)]
        for j in range(idx, len(path)):
            if math.hypot(path[j][0] - rx, path[j][1] - ry) >= self._lookahead_m:
                lx, ly = path[j]
                break

        dx = lx - rx;  dy = ly - ry
        local_x =  dx * math.cos(ryaw) + dy * math.sin(ryaw)
        local_y = -dx * math.sin(ryaw) + dy * math.cos(ryaw)

        if local_x < 0.05:
            for j in range(min(idx + 2, len(path) - 1), len(path)):
                dx2 = path[j][0] - rx;  dy2 = path[j][1] - ry
                lx2 = dx2 * math.cos(ryaw) + dy2 * math.sin(ryaw)
                if lx2 > 0.05:
                    dx, dy = dx2, dy2
                    local_x =  dx * math.cos(ryaw) + dy * math.sin(ryaw)
                    local_y = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
                    break

        cte = local_y

        nxt_idx = min(idx + 1, len(path) - 1)
        seg_dx  = path[nxt_idx][0] - path[idx][0]
        seg_dy  = path[nxt_idx][1] - path[idx][1]
        if abs(seg_dx) > 1e-9 or abs(seg_dy) > 1e-9:
            path_tangent = math.atan2(seg_dy, seg_dx)
        else:
            path_tangent = math.atan2(dy, dx)
        hdg_err = _wrap(path_tangent - ryaw)

        self._cte_pub.publish(Float32(data=float(cte)))
        self._hdg_pub.publish(Float32(data=float(hdg_err)))

        self.get_logger().info(
            f'wp {idx}/{len(path)-1}  '
            f'local=({local_x:+.2f},{local_y:+.2f})m  '
            f'cte={cte:+.3f}m  hdg={math.degrees(hdg_err):+.1f}°',
            throttle_duration_sec=0.5)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._active_pub.publish(Bool(data=False))
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
