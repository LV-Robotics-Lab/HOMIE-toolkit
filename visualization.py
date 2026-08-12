"""
Xperience-10M: Rerun blueprint and visualization helpers.
"""

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import (
    Blueprint,
    Horizontal,
    Vertical,
    Spatial3DView,
    Spatial2DView,
    TimeSeriesView,
)


def build_line3d_skeleton(joints, parent_indices, plus_one=False):
    """Build skeleton lines from joints and parent indices.

    Supports two formats:
    1. Body skeleton: joints[0] is root (not in parent_indices), parent_indices length = J-1
    2. Hand skeleton (MANO): joints and parent_indices have same length, parent_indices[0] = -1 for root

    Args:
        joints: (J, 3) joint positions
        parent_indices: parent index array
            - For body: (J-1,) array where parent_indices[i] corresponds to joints[i+1]
            - For hand: (J,) array where parent_indices[i] directly indexes joints, parent_indices[0] = -1
        plus_one: if True, add 1 to parent indices (body skeleton format)

    Returns:
        skeleton_lines: (N, 2, 3) skeleton line segment array
    """
    line3d_list = []

    if not plus_one:
        num_joints = len(joints)
        for i in range(1, min(len(parent_indices), num_joints)):
            parent_idx = parent_indices[i]
            if parent_idx >= 0 and parent_idx < num_joints:
                line3d_list.append([joints[parent_idx], joints[i]])
    else:
        for i in range(len(parent_indices)):
            line3d_list.append([joints[parent_indices[i] + 1], joints[i + 1]])

    return np.array(line3d_list) if line3d_list else np.array([]).reshape(0, 2, 3)


def depth_to_colormap(depth, depth_min, depth_max, colormap=cv2.COLORMAP_JET):
    """Convert depth to colormap with unified scale. Returns (H, W, 3) RGB."""
    depth_normalized = np.clip((depth - depth_min) / (depth_max - depth_min + 1e-8), 0, 1)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    colormap_image = cv2.applyColorMap(depth_uint8, colormap)
    return cv2.cvtColor(colormap_image, cv2.COLOR_BGR2RGB)


def depth_to_pointcloud(depth, K, rgb_image=None, downsample_factor=2, max_points=50000, near_plane=0.5, far_plane=4.0, confidence=None, confidence_threshold=0.0):
    """Convert depth map to point cloud with RGB colors. Returns (points, colors)."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    depth_ds = depth[::downsample_factor, ::downsample_factor]
    H_ds, W_ds = depth_ds.shape
    conf_ds = confidence[::downsample_factor, ::downsample_factor] if confidence is not None else None

    u = np.arange(W_ds) * downsample_factor + downsample_factor // 2
    v = np.arange(H_ds) * downsample_factor + downsample_factor // 2
    u, v = np.meshgrid(u, v)

    z = depth_ds.flatten()
    valid_mask = (z >= near_plane) & (z <= far_plane)
    if conf_ds is not None and confidence_threshold > 0:
        conf_flat = conf_ds.flatten().astype(np.float32) / 255.0
        valid_mask = valid_mask & (conf_flat >= confidence_threshold)

    if np.sum(valid_mask) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    x = (u.flatten()[valid_mask] - cx) * z[valid_mask] / fx
    y = (v.flatten()[valid_mask] - cy) * z[valid_mask] / fy
    points = np.stack([x, y, z[valid_mask]], axis=-1)

    u_flat = np.clip(u.flatten()[valid_mask].astype(int), 0, W - 1)
    v_flat = np.clip(v.flatten()[valid_mask].astype(int), 0, H - 1)

    if rgb_image is not None:
        if rgb_image.shape[:2] != (H, W):
            rgb_image = cv2.resize(rgb_image, (W, H), interpolation=cv2.INTER_LINEAR)
        colors_rgb = rgb_image[v_flat, u_flat]
    else:
        depth_normalized = (z[valid_mask] - z[valid_mask].min()) / (z[valid_mask].max() - z[valid_mask].min() + 1e-8)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        colors_bgr = cv2.applyColorMap(depth_uint8.reshape(-1, 1), cv2.COLORMAP_JET)
        colors_rgb = cv2.cvtColor(colors_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)

    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors_rgb = colors_rgb[indices]
    return points, colors_rgb


def scale_image(image, scale_factor):
    """Scale image by a factor. Returns (H, W, 3) RGB."""
    if scale_factor == 1.0:
        return image
    H, W = image.shape[:2]
    new_W, new_H = int(W * scale_factor), int(H * scale_factor)
    if new_W <= 0 or new_H <= 0:
        return image
    return cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)


def transform_points_to_world(points, R_c2w, t_c2w):
    """Transform points from camera to world frame. Returns (N, 3)."""
    return (R_c2w @ points.T).T + t_c2w


def create_blueprint(
    show_fisheye=False,
    show_stereo=False,
    show_depth_colormap=False,
    ground_height=-1.8,
    show_imu=False,
    show_caption=False,
    show_3d_view=True,
):
    """Create a Rerun layout similar to the Xperience-10M reference viewer."""

    def _vertical(items, row_shares=None):
        items = [item for item in items if item is not None]
        if not items:
            return None
        kwargs = {}
        if row_shares is not None and len(row_shares) == len(items):
            kwargs["row_shares"] = row_shares
        return Vertical(*items, **kwargs)

    def _horizontal(items, column_shares=None):
        items = [item for item in items if item is not None]
        if not items:
            return None
        kwargs = {}
        if column_shares is not None and len(column_shares) == len(items):
            kwargs["column_shares"] = column_shares
        return Horizontal(*items, **kwargs)

    world_3d_contents = [
        "world/slam_point_cloud",
        "world/stereo/left/**",
        "world/stereo/right/**",
        "world/fisheye/cam0/camera",
        "world/fisheye/cam1/camera",
        "world/fisheye/cam2/camera",
        "world/fisheye/cam3/camera",
        "world/depth/points",
        "world/hand_mocap/**",
        "world/full_body_mocap/**",
        "world/smplh/**",
    ]

    world_view = None
    if show_3d_view:
        world_view = Spatial3DView(
            name="world/3d_view",
            origin="/",
            contents=world_3d_contents,
            line_grid=rrb.archetypes.LineGrid3D(
                visible=True,
                plane=rr.components.Plane3D.XY.with_distance(ground_height),
            ),
            background=[60, 60, 60],
        )

    caption_panel = None
    task_timeline_view = None
    if show_caption:
        caption_panel = _horizontal(
            [
                _vertical(
                    [
                        rrb.TextDocumentView(origin="captions/Main_Task", name="Main Task"),
                        rrb.TextDocumentView(origin="captions/details/objects", name="Objects"),
                    ],
                    row_shares=[1.15, 1.0],
                ),
                _vertical(
                    [
                        rrb.TextDocumentView(origin="captions/Sub_Task", name="Sub Task"),
                        rrb.TextDocumentView(origin="captions/details/interaction", name="Interaction"),
                        rrb.TextDocumentView(origin="captions/Current_Action", name="Current Action"),
                    ],
                    row_shares=[0.8, 1.0, 1.45],
                ),
            ],
            column_shares=[1, 1],
        )
        task_timeline_view = TimeSeriesView(
            origin="timeline",
            name="Task Timeline",
            plot_legend=rrb.PlotLegend(visible=True),
        )

    right_column = _vertical(
        [world_view, caption_panel, task_timeline_view],
        row_shares=[2.5, 1.15, 0.55],
    )

    media_top_columns = []
    if show_fisheye:
        media_top_columns.extend(
            [
                _vertical(
                    [
                        Spatial2DView(name="world/fisheye/cam0", origin="world/fisheye/cam0"),
                        Spatial2DView(name="world/fisheye/cam2", origin="world/fisheye/cam2"),
                    ],
                    row_shares=[1, 1],
                ),
                _vertical(
                    [
                        Spatial2DView(name="world/fisheye/cam1", origin="world/fisheye/cam1"),
                        Spatial2DView(name="world/fisheye/cam3", origin="world/fisheye/cam3"),
                    ],
                    row_shares=[1, 1],
                ),
            ]
        )
    if show_stereo:
        media_top_columns.append(
            _vertical(
                [
                    Spatial2DView(name="world/stereo/vis_cam0", origin="world/stereo/vis_cam0"),
                    Spatial2DView(name="world/stereo/vis_cam1", origin="world/stereo/vis_cam1"),
                ],
                row_shares=[1, 1],
            )
        )
    media_top = _horizontal(media_top_columns, column_shares=[1] * len(media_top_columns))

    media_bottom_items = []
    if show_imu:
        media_bottom_items.append(
            _vertical(
                [
                    TimeSeriesView(name="imu/accel", origin="imu/accel", plot_legend=rrb.PlotLegend(visible=True)),
                    TimeSeriesView(name="imu/gyro", origin="imu/gyro", plot_legend=rrb.PlotLegend(visible=True)),
                ],
                row_shares=[1, 1],
            )
        )
    if show_depth_colormap:
        media_bottom_items.append(Spatial2DView(name="world/depth/vis", origin="world/depth/vis"))

    bottom_column_shares = None
    if len(media_bottom_items) == 2 and show_imu and show_depth_colormap:
        bottom_column_shares = [2, 1]
    elif media_bottom_items:
        bottom_column_shares = [1] * len(media_bottom_items)
    media_bottom = _horizontal(media_bottom_items, column_shares=bottom_column_shares)

    left_column = _vertical([media_top, media_bottom], row_shares=[2.2, 1.0])

    root_view = _horizontal(
        [left_column, right_column],
        column_shares=[1.15, 1.35],
    )
    return Blueprint(root_view, collapse_panels=True)


__all__ = [
    "create_blueprint",
    "depth_to_colormap",
    "depth_to_pointcloud",
    "build_line3d_skeleton",
    "scale_image",
    "transform_points_to_world",
]
