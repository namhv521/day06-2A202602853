"""Topic D, CPU only. Run from repo root: python -m src.obstacles --help.

API reference (implementation written for this lab):
https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
Coordinates: LiDAR x forward, y left, z up; all distances in metres.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
from pathlib import Path
from time import perf_counter

# Single-thread RANSAC makes seeded geometric results easier to reproduce.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import open3d as o3d
import pandas as pd

from starter.datasets import list_frames, load_points

SEED = 42
ROI = (0., 30., -12., 12., -3., 3.)


def cloud(points):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def pipeline(points, voxel_size=.10, distance_threshold=.08, eps=.60, min_points=8):
    """Fit lower-point ground plane, remove its band, then cluster and box.

    Reject steep/unsupported planes; keep all points if fit is invalid.
    nearest_m = minimum XY distance to a clustered point, not box centre.
    No cluster => NaN, never infinity or an assertion of free space.
    """
    if not all(np.isfinite(v) and v > 0 for v in (voxel_size, distance_threshold, eps)):
        raise ValueError('voxel_size, distance_threshold and eps must be finite and positive')
    if min_points < 1:
        raise ValueError('min_points must be positive')
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('points must have shape (N, >=3)')
    t0 = perf_counter()
    xyz = points[:, :3]
    valid = np.isfinite(xyz).all(axis=1)
    lower, upper = np.array(ROI)[[0, 2, 4]], np.array(ROI)[[1, 3, 5]]
    raw = xyz[valid & (xyz >= lower).all(axis=1) & (xyz < upper).all(axis=1)]
    down = np.asarray(cloud(raw).voxel_down_sample(voxel_size).points) if len(raw) else raw.copy()
    t1 = perf_counter()
    ground = np.zeros(len(down), dtype=bool)
    plane = np.full(4, np.nan)
    ground_valid = False
    if len(down) >= 3:
        # Lower 35% reduces the chance of fitting a building wall as ground.
        candidates = down[down[:, 2] <= np.quantile(down[:, 2], .35)]
        if len(candidates) >= 3:
            o3d.utility.random.seed(SEED)
            fitted, inliers = cloud(candidates).segment_plane(
                distance_threshold=distance_threshold, ransac_n=3,
                # Full iteration budget avoids parallel early-stop variability.
                num_iterations=300, probability=1.0)
            fitted = np.asarray(fitted, dtype=float)
            norm = np.linalg.norm(fitted[:3])
            if norm > 0:
                fitted /= norm
                if fitted[2] < 0:
                    fitted *= -1
                ground_valid = bool(fitted[2] >= np.cos(np.deg2rad(20))
                                    and len(inliers) >= .15 * len(candidates))
                if ground_valid:
                    plane = fitted
                    ground = np.abs(down @ plane[:3] + plane[3]) <= distance_threshold
    obstacle = down[~ground]
    t2 = perf_counter()
    labels = np.asarray(cloud(obstacle).cluster_dbscan(eps=eps, min_points=min_points,
                        print_progress=False), dtype=int) if len(obstacle) else np.empty(0, dtype=int)
    boxes = []
    for label in np.unique(labels[labels >= 0]):
        cluster = obstacle[labels == label]
        aabb = cloud(cluster).get_axis_aligned_bounding_box()
        boxes.append(dict(cluster_id=int(label), points=len(cluster),
                          min=np.asarray(aabb.min_bound), max=np.asarray(aabb.max_bound),
                          extent=np.asarray(aabb.get_extent()),
                          nearest_m=float(np.linalg.norm(cluster[:, :2], axis=1).min())))
    t3 = perf_counter()
    return dict(raw=raw, down=down, ground=down[ground], obstacle=obstacle,
                labels=labels, boxes=boxes, plane=plane, ground_valid=ground_valid,
                nearest_m=min((b['nearest_m'] for b in boxes), default=np.nan),
                nearest_return_m=float(np.linalg.norm(obstacle[:, :2], axis=1).min())
                    if len(obstacle) else np.nan,
                voxel_ms=(t1-t0)*1000, ground_ms=(t2-t1)*1000,
                cluster_ms=(t3-t2)*1000, total_ms=(t3-t0)*1000)


def occupancy(points, resolution=.20):
    """Binary endpoint grid; zero = unknown, not ray-traced free space."""
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('resolution must be finite and positive')
    grid = np.zeros((int(np.ceil(24/resolution)), int(np.ceil(30/resolution))), dtype=np.uint8)
    points = np.asarray(points)
    inside = (np.isfinite(points[:, :2]).all(axis=1) & (points[:, 0] >= 0)
              & (points[:, 0] < 30) & (points[:, 1] >= -12) & (points[:, 1] < 12))
    indices = np.floor((points[inside, :2] - [0, -12]) / resolution).astype(int)
    grid[indices[:, 1], indices[:, 0]] = 1
    return grid


def low_scene():
    """Controlled synthetic scene, NOT an annotated KITTI pallet/person.

    Floor z=0; 18 cm low box at 3m; tall box at 8m. Dense sampled surfaces.
    """
    x, y = np.meshgrid(np.arange(.1, 15, .10), np.arange(-8, 8, .10))
    floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    surfaces = [floor]
    for xmin, xmax, ymin, ymax, height in [(3, 3.8, -.3, .3, .18), (8, 9, 1, 2, 1.2)]:
        x, y = np.meshgrid(np.arange(xmin, xmax+.001, .04), np.arange(ymin, ymax+.001, .04))
        surfaces.append(np.column_stack((x.ravel(), y.ravel(), np.full(x.size, height))))
        for z in np.arange(.04, height, .04):
            edge = ((np.isclose(x, xmin)) | (np.isclose(x, xmax))
                    | (np.isclose(y, ymin)) | (np.isclose(y, ymax)))
            surfaces.append(np.column_stack((x[edge], y[edge], np.full(edge.sum(), z))))
    return np.concatenate(surfaces)


def scatter(ax, points, title, colors=None, side=False):
    # ponytail: headless Matplotlib projections; use a 3D viewer for occlusion inspection.
    if len(points):
        ax.scatter(points[:, 0], points[:, 2 if side else 1], s=1,
                   c=points[:, 2] if colors is None else colors,
                   **({'cmap': 'viridis'} if colors is None else {}))
    ax.set(title=f'{title} | n={len(points)}', xlabel='x forward (m)',
           ylabel='z up (m)' if side else 'y left (m)', xlim=(0, 30),
           ylim=(-3, 3) if side else (-12, 12))
    if not side:
        ax.set_aspect('equal')
    ax.grid(alpha=.2)


def draw_boxes(ax, boxes, side=False):
    for box in boxes:
        j = 2 if side else 1
        lo, extent = box['min'], box['extent']
        ax.add_patch(Rectangle((lo[0], lo[j]), extent[0], extent[j], fill=False,
                              edgecolor='red', linewidth=1))
        ax.text(lo[0], lo[j], str(box['cluster_id']), fontsize=7, color='red')


def figures(result, frame, output):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, key, title in zip(axes.ravel()[:4], ('raw', 'down', 'ground', 'obstacle'),
                             ('Input finite + ROI', 'Voxel downsample', 'Removed ground', 'After ground removal')):
        scatter(ax, result[key], title)
    scatter(axes[1, 1], result['obstacle'], f'DBSCAN: {len(result["boxes"])} clusters; gray=noise',
            colors=np.where(result['labels'][:, None] < 0, np.array([[.6, .6, .6, 1]]),
                            plt.get_cmap('tab20')(np.maximum(result['labels'], 0) % 20 / 19)))
    draw_boxes(axes[1, 1], result['boxes'])
    scatter(axes[1, 2], result['obstacle'], 'Cluster AABB: side view', side=True)
    draw_boxes(axes[1, 2], result['boxes'], side=True)
    fig.suptitle(f'{frame} | voxel=0.10m, ground band=0.08m, eps=0.60m')
    fig.tight_layout()
    fig.savefig(output / f'pipeline_{frame}.png', dpi=140)
    plt.close(fig)
    grid = occupancy(result['obstacle'])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(grid, origin='lower', extent=(0, 30, -12, 12), cmap='Greys', vmin=0, vmax=1)
    draw_boxes(ax, result['boxes'])
    ax.plot(0, 0, 'b^', label='LiDAR origin')
    ax.set(title=f'{frame}: BEV endpoints (all non-ground incl. DBSCAN noise)\nBlack=occupied; white=unknown; cell=0.20m',
           xlabel='x forward (m)', ylabel='y left (m)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / f'bev_{frame}.png', dpi=140)
    plt.close(fig)
    return grid


def metrics(result, frame, voxel, threshold, repetition):
    extents = np.array([b['extent'] for b in result['boxes']])
    mean = extents.mean(axis=0) if len(extents) else [np.nan]*3
    return dict(frame=frame, voxel_size=voxel, distance_threshold=threshold, eps=.6,
                min_points=8, repetition=repetition, roi_points=len(result['raw']),
                voxel_points=len(result['down']), ground_points=len(result['ground']),
                obstacle_points=len(result['obstacle']), noise_points=int((result['labels'] < 0).sum()),
                clusters=len(result['boxes']), mean_dx_m=mean[0], mean_dy_m=mean[1], mean_dz_m=mean[2],
                nearest_m=result['nearest_m'], nearest_return_m=result['nearest_return_m'],
                ground_valid=result['ground_valid'],
                **{k: result[k] for k in ('voxel_ms', 'ground_ms', 'cluster_ms', 'total_ms')})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', default='data/kitti_mini')
    ap.add_argument('--frames', nargs='+', default=['000001', '000011', '000049'])
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--repeats', type=int, default=3, help='measured repeats per configuration, after one warm-up')
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error('--repeats must be >=1')
    missing = set(args.frames) - set(list_frames(args.data_root))
    if missing:
        ap.error(f'missing frames: {sorted(missing)}')
    out = Path(args.out_dir)
    figdir = out / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    rows, boxes = [], []
    for frame in args.frames:
        points = load_points(args.data_root, frame)
        for voxel, threshold in itertools.product((.05, .10, .30), (.03, .08, .30)):
            pipeline(points, voxel, threshold)  # warm-up excluded
            for repetition in range(args.repeats):
                result = pipeline(points, voxel, threshold)
                rows.append(metrics(result, frame, voxel, threshold, repetition))
            for box in result['boxes']:
                boxes.append(dict(frame=frame, voxel_size=voxel, distance_threshold=threshold,
                                  cluster_id=box['cluster_id'], points=box['points'], nearest_m=box['nearest_m'],
                                  **{f'{key}_{axis}_m': value for key in ('min', 'max', 'extent')
                                     for axis, value in zip('xyz', box[key])}))
        baseline = pipeline(points)
        np.save(out / f'occupancy_{frame}.npy', figures(baseline, frame, figdir))
        print(f'{frame}: {len(baseline["boxes"])} clusters, nearest={baseline["nearest_m"]:.3f}m', flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / 'obstacle_runs.csv', index=False)
    summary = raw.groupby(['frame', 'voxel_size', 'distance_threshold'], as_index=False).agg(
        clusters=('clusters', 'median'), voxel_points=('voxel_points', 'median'),
        ground_points=('ground_points', 'median'), noise_points=('noise_points', 'median'),
        mean_dx_m=('mean_dx_m', 'median'), mean_dy_m=('mean_dy_m', 'median'), mean_dz_m=('mean_dz_m', 'median'),
        nearest_m=('nearest_m', 'median'), nearest_return_m=('nearest_return_m', 'median'),
        latency_median_ms=('total_ms', 'median'), latency_min_ms=('total_ms', 'min'),
        latency_max_ms=('total_ms', 'max'), latency_p95_ms=('total_ms', lambda s: s.quantile(.95)))
    summary.to_csv(out / 'obstacle_sweep.csv', index=False)
    pd.DataFrame(boxes).to_csv(out / 'obstacle_boxes.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(axes, ('clusters', 'nearest_m', 'latency_median_ms'),
                                ('Cluster count', 'Nearest clustered return (m)', 'Median pipeline latency (ms)')):
        for (frame, voxel), group in summary.groupby(['frame', 'voxel_size']):
            ax.plot(group.distance_threshold, group[metric], 'o-', label=f'{frame}, v={voxel}')
        ax.set(xlabel='Ground distance threshold (m)', title=title)
        ax.grid(alpha=.3)
    axes[-1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(figdir / 'obstacle_sweep.png', dpi=140)
    plt.close(fig)
    synthetic = low_scene()
    failure_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i, threshold in enumerate((.03, .08, .30)):
        result = pipeline(synthetic, .05, threshold)
        low_before = result['down'][(result['down'][:, 0] >= 2.9) & (result['down'][:, 0] <= 3.9)
                                    & (np.abs(result['down'][:, 1]) <= .4) & (result['down'][:, 2] > .01)]
        retained = result['obstacle'][(result['obstacle'][:, 0] >= 2.9) & (result['obstacle'][:, 0] <= 3.9)
                                      & (np.abs(result['obstacle'][:, 1]) <= .4)]
        detected = any(b['min'][0] < 4 and b['max'][0] > 3 for b in result['boxes'])
        failure_rows.append(dict(distance_threshold=threshold, voxel_size=.05, low_height_m=.18,
                                  low_before_points=len(low_before), low_retained_points=len(retained),
                                  low_detected=detected, clusters=len(result['boxes']),
                                  nearest_m=result['nearest_m'], total_ms=result['total_ms']))
        for ax, key in zip(axes[:, i], ('down', 'obstacle')):
            scatter(ax, result[key], f'{key}, threshold={threshold}m', side=True)
            ax.set(xlim=(0, 10), ylim=(-.1, 1.4))
            ax.axvspan(3, 3.8, color='red', alpha=.12)
        draw_boxes(axes[1, i], result['boxes'], side=True)
    fig.suptitle('SYNTHETIC failure: 18 cm low obstacle at 3 m; red band = low object location')
    fig.tight_layout()
    fig.savefig(figdir / 'fail_01_low_obstacle.png', dpi=140)
    plt.close(fig)
    pd.DataFrame(failure_rows).to_csv(out / 'low_obstacle_failure.csv', index=False)
    manifest = dict(seed=SEED, dataset=args.data_root, frames=args.frames, roi_xyz_m=ROI,
                    repeats=args.repeats, warmup_per_config=1, ransac_iterations=300, ransac_probability=1.0,
                    ground_candidate_quantile=.35, max_ground_tilt_deg=20, eps_m=.6, min_points=8,
                    python=platform.python_version(), open3d=o3d.__version__, numpy=np.__version__,
                    omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
                    openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                    platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
                    timing='perf_counter; ROI/filter/voxel + RANSAC + DBSCAN/AABB; excludes I/O, figures and occupancy',
                    occupancy='0.20m endpoint grid: 1=occupied, 0=unknown; no free-space inference',
                    failure_source='procedural synthetic geometry in low_scene; not KITTI ground truth')
    (out / 'obstacle_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
