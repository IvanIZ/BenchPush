from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math
import random
import yaml
import os
import numpy as np
from benchnpin.common.utils.utils import DotDict

from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union, nearest_points
from shapely.prepared import prep
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
Vec2 = Tuple[float, float]
PolyLike = List[Vec2]



  # --------------------------- Scene builder---------------------------
def _clean_xy_list(v: PolyLike) -> List[Vec2]:
    """Drop NaNs/None and consecutive duplicates."""
    out: List[Vec2] = []
    last: Optional[Vec2] = None
    for xy in v:
        x, y = float(xy[0]), float(xy[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if last is None or (abs(x - last[0]) > 1e-12 or abs(y - last[1]) > 1e-12):
            out.append((x, y))
            last = (x, y)
    return out


class _Scene:
    """
    Shapely-backed scene:
      - walls_union: union of wall lines/polys, later buffered by robot radius
      - walls_infl / walls_prep: robot-inflated walls area (Minkowski sum with a disk)
      - boxes: list of box polygons (movables), with STRtree for broad-phase
    """

    def __init__(self, walls_union, boxes_polys, r_robot: float, bounds_pad: float):
        # Inflate walls (lines or polys) by robot radius
        self.walls_infl = walls_union.buffer(r_robot*1.5, cap_style=1, join_style=1) if walls_union else None
        self.walls_prep = prep(self.walls_infl) if self.walls_infl else None

        # Boxes as area polygons; build spatial index
        self.boxes = [p.buffer(0) for p in boxes_polys]  # buffer(0) fixes most invalidities
        self.box_tree = STRtree(self.boxes) if self.boxes else None

        # Sampling bounds from geometry (+ padding)
        bounds_list: List[Tuple[float, float, float, float]] = []
        if walls_union:
            bounds_list.append(walls_union.bounds)
        for bx in self.boxes:
            bounds_list.append(bx.bounds)
        if bounds_list:
            print("pounds pad:", bounds_pad)
            print("bounds_list:", bounds_list)
            xmin = min(b[0] for b in bounds_list) - bounds_pad
            ymin = min(b[1] for b in bounds_list) - bounds_pad
            xmax = max(b[2] for b in bounds_list) + bounds_pad
            ymax = max(b[3] for b in bounds_list) + bounds_pad
        else:
            xmin, ymin, xmax, ymax = -3.0, -3.0, 3.0, 3.0
        self.bounds = (xmin, xmax, ymin, ymax)

    @classmethod
    def from_any(
        cls,
        wall_items: Optional[List[PolyLike]],
        box_items: Optional[List[PolyLike]],
        r_robot: float,
        bounds_pad: float,
        box_line_buffer: float = 1e-3,
    ):
        # Walls: accept polygons (>=3 pts) or segments (>=2 pts → LineString)
        wall_geoms = []
        for verts in (wall_items or []):
            v = _clean_xy_list(verts)
            if len(v) >= 3:
                try:
                    poly = Polygon(v).buffer(0)
                    if not poly.is_empty and poly.area > 0:
                        wall_geoms.append(poly)
                        continue
                except Exception:
                    pass
            if len(v) >= 2:
                ln = LineString(v)
                if not ln.is_empty and ln.length > 1e-9:
                    wall_geoms.append(ln)
        walls_union = unary_union(wall_geoms) if wall_geoms else None

        # Boxes: prefer polygons; if segments, convert to thin polygons with tiny buffer
        boxes_polys = []
        for verts in (box_items):
            v = _clean_xy_list(verts)
            if len(v) >= 3:
                try:
                    p = Polygon(v).buffer(0)
                    if not p.is_empty and p.area > 0:
                        boxes_polys.append(p)
                        continue
                except Exception:
                    pass
            if len(v) >= 2:
                thin = LineString(v)
                if not thin.is_empty and thin.length > 1e-9:
                    boxes_polys.append(thin.buffer(box_line_buffer, cap_style=2, join_style=2))

        return cls(walls_union, boxes_polys, r_robot, bounds_pad)

    def seg_hit(self, a: Vec2, b: Vec2, boxes_blocking: bool) -> str:
        """
        Returns:
        'static'  -> collides with inflated walls (robot body considered)
        'movable' -> touches a box (when boxes_blocking=True; point-robot)
        'none'    -> free
        """
        line = LineString([a, b])

        if self.walls_prep:
            # intersects/contains cover distance<=radius thanks to buffered walls
            if self.walls_prep.intersects(line) or self.walls_prep.contains(Point(a)) or self.walls_prep.contains(Point(b)):
                return "static"

        if boxes_blocking and self.box_tree:
            # Broad-phase via STRtree then exact intersects/contains
            for poly in self.box_tree.geometries:
                if isinstance(poly, Polygon) or isinstance(poly, LineString):
                    if poly.intersects(line) or poly.contains(Point(a)) or poly.contains(Point(b)):
                        return "movable"

        return "none"

    def snap_to_free(self, p: Vec2) -> Vec2:
        """
        If p lies inside inflated walls, project it to the nearest boundary point.
        (This returns a boundary point; your controller/planner will then take over.)
        """
        if self.walls_infl and self.walls_infl.contains(Point(p)):
            q_on_boundary, _ = nearest_points(Point(p), self.walls_infl.boundary)
            return (q_on_boundary.x, q_on_boundary.y)
        return p


# --------------------------- RRT Planner ---------------------------
#Two-stage RRT planner: first stage with movable obstacles, second stage ignoring them
class RRTPlanner:
    def __init__(self,  config_file=None, plot_path=True) -> None:
        if config_file is None:
            config_file = 'rrt_config.yaml'
        # construct absolute path to the env_config folder
        cfg_file = os.path.join(os.path.dirname(__file__), 'planner_configs', config_file)
        self.cfg = DotDict.load_from_file(cfg_file)
        #plotting flag
        self.plot_path = plot_path
        #set random seed for reproducibility
        random.seed(self.cfg.seed)

    # Public API
    def plan(
        self,
        start: Vec2,
        goal: Vec2,
        movable_obstacles = None,            # boxes: list of polygons or segments
        maze_vertices: Optional[List[PolyLike]] = None,  # walls: list of polygons or segments
        robot_size: Optional[Dict[str, float]] = None,   # {"radius": r} OR {"width": w, "length": l}
        bounds_pad: Optional[float] = None,              # if None: auto from robot size
        snap_to_free: bool = True,
        ignore_boxes_fallback: bool = True,
    ) -> List[Vec2]:
        """
        Returns a densified waypoint path. If both passes fail, returns [start, goal].
        Two-pass behavior:
          1) boxes block (point-robot)
          2) ignore boxes (only walls block)
        """
        # Derive robot effective radius
        r_robot = 0.15  # safe default
        if robot_size:
            if "radius" in robot_size:
                r_robot = float(robot_size["radius"])
            elif "width" in robot_size and "length" in robot_size:
                w = float(robot_size["width"]); l = float(robot_size["length"])
                r_robot = math.hypot(w / 2.0, l / 2.0)

        pad = bounds_pad if bounds_pad is not None else max(0*0.25, 0*1.5 * r_robot)
        print(f"RRT: robot radius {r_robot:.3f}, sampling bounds pad {pad:.3f}")

        # Build scene
        self.scene = _Scene.from_any(
            wall_items=maze_vertices or [],
            box_items=movable_obstacles or [],
            r_robot=r_robot,
            bounds_pad=pad,
        )

        s = self.scene.snap_to_free(start) if snap_to_free else start
        g = self.scene.snap_to_free(goal) if snap_to_free else goal

        # Pass 1: boxes block
        path = self._run_rrt(s, g, self.scene, boxes_blocking=True)
        if path is not None:
            print("RRT: found a collision-free path")
            return path
        print("RRT: no path found that avoids boxes")
        print("Planning a path ignoring boxes...")
        # Pass 2: ignore boxes
        if ignore_boxes_fallback:
            path2 = self._run_rrt(s, g, self.scene, boxes_blocking=False)
            if path2 is not None:
                return path2

        # Fallback
        return [s, g]

    # Internals
    def _run_rrt(self, start: Vec2, goal: Vec2, scene: _Scene, boxes_blocking: bool) -> Optional[List[Vec2]]:
        xmin, xmax, ymin, ymax = scene.bounds

        # Early invalid: start/goal inside inflated walls
        if scene.walls_infl and (scene.walls_infl.contains(Point(start)) or scene.walls_infl.contains(Point(goal))):
            return None
        

        nodes: List[Vec2] = [start]
        parent: List[int] = [-1]

        for _ in range(self.cfg.max_nodes):
            # Sample (goal-biased)
            if random.random() < self.cfg.goal_bias:
                q_rand = goal
            else:
                q_rand = (random.uniform(xmin, xmax), random.uniform(ymin, ymax))

            # Nearest
            i_near = min(range(len(nodes)), key=lambda i: self._dist(nodes[i], q_rand))
            q_new = self._steer(nodes[i_near], q_rand, self.cfg.step)

            # Collision
            hit = scene.seg_hit(nodes[i_near], q_new, boxes_blocking=boxes_blocking)
            if hit != "none":
                continue

            # Accept
            nodes.append(q_new)
            parent.append(i_near)

            # Goal check
            if self._dist(q_new, goal) <= self.cfg.goal_radius:
                nodes.append(goal)
                parent.append(len(nodes) - 2)
                path = self._backtrack(nodes, parent, len(nodes) - 1)
                # if self.plot_path:
                #     self.plot_env(scene, nodes,parent,path)  # For debugging
                return self._densify(path, self.cfg.densify_ds)
        
        
        return None

    @staticmethod
    def _dist(a: Vec2, b: Vec2) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _steer(a: Vec2, b: Vec2, step: float) -> Vec2:
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L <= step or L == 0.0:
            return b
        s = step / L
        return (a[0] + s * dx, a[1] + s * dy)

    @staticmethod
    def _backtrack(nodes: List[Vec2], parent: List[int], idx: int) -> List[Vec2]:
        path: List[Vec2] = []
        while idx != -1:
            path.append(nodes[idx])
            idx = parent[idx]
        return list(reversed(path))

    @staticmethod
    def _densify(path: List[Vec2], ds: float) -> List[Vec2]:
        if not path or len(path) == 1:
            return path
        ds = max(1e-3, ds)
        dense: List[Vec2] = [path[0]]
        for i in range(1, len(path)):
            a, b = dense[-1], path[i]
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, int(L / ds))
            for k in range(1, n + 1):
                t = k / n
                dense.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
        return dense

    def plot_env(self, path, parent=None, nodes=None, ax=None):
        
        """Plot the scene: walls (inflated), boxes.
        path and tree"""

        print("Plotting environment...")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        if self.scene.walls_infl:
            x, y = self.scene.walls_infl.exterior.xy
            ax.fill(x, y, color='lightgray', alpha=0.7, label='Inflated Walls')
            #holes
            for interior in self.scene.walls_infl.interiors:
                x, y = interior.xy
                ax.fill(x, y, color='white')
            
        if self.scene.boxes:
            for i, box in enumerate(self.scene.boxes):
                if isinstance(box, Polygon):
                    x, y = box.exterior.xy
                    ax.fill(x, y, color='orange', alpha=0.7, label='Box' if i == 0 else "")
                elif isinstance(box, LineString):
                    x, y = box.xy
                    ax.plot(x, y, color='orange', linewidth=4, label='Box' if i == 0 else "")
        # Plot RRT tree
        if nodes is not None and parent is not None:
            for i, node in enumerate(nodes):
                if parent[i] != -1:
                    p_node = nodes[parent[i]]
                    ax.plot([node[0], p_node[0]], [node[1], p_node[1]], color='blue', linewidth=0.5, alpha=0.5)
        
        # Plot path if available
        if path is not None and len(path) > 1:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'r', linewidth=2, label='Planned Path')
        # Plot start and goal
        if nodes is not None:
            ax.plot(nodes[0][0], nodes[0][1], 'go', markersize=10, label='Start')
            ax.plot(nodes[-1][0], nodes[-1][1], 'ro', markersize=10, label='Goal')
        ax.set_aspect('equal')
        ax.set_xlim(self.scene.bounds[0], self.scene.bounds[1])
        ax.set_ylim(self.scene.bounds[2], self.scene.bounds[3])
        ax.set_title('RRT Planning Environment')
        ax.legend()
        plt.savefig('maze_namo_rrt_environment.png')
 