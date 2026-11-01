import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2


def presentation_2(keypoints1, keypoints2, matches, img1, img2):
    """
    Display matched keypoints from both images on the same plot.
    Shows one-to-one comparison of x and y coordinates.
    """
    if len(matches) == 0:
        print("No matches to display!")
        return
    
    # Extract match coordinates with clear naming
    x1_coords = np.array([keypoints1[m.queryIdx].pt[0] for m in matches])
    y1_coords = np.array([keypoints1[m.queryIdx].pt[1] for m in matches])
    
    x2_coords = np.array([keypoints2[m.trainIdx].pt[0] for m in matches])
    y2_coords = np.array([keypoints2[m.trainIdx].pt[1] for m in matches])
    
    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot both sets of keypoints on the same plot
    ax.scatter(x1_coords, y1_coords, c='red', s=100, marker='o', edgecolors='black', 
               linewidth=1.5, label='Image 1', alpha=0.7)
    ax.scatter(x2_coords, y2_coords, c='blue', s=100, marker='s', edgecolors='black', 
               linewidth=1.5, label='Image 2', alpha=0.7)
    
    # Draw lines connecting matched points
    for i in range(len(matches)):
        ax.plot([x1_coords[i], x2_coords[i]], [y1_coords[i], y2_coords[i]], 
                'g--', alpha=0.3, linewidth=0.8)
    
    ax.set_xlabel('x (pixels)', fontsize=12)
    ax.set_ylabel('y (pixels)', fontsize=12)
    ax.set_title(f'Matched Keypoints Comparison - {len(matches)} matches', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_aspect('equal')  # Same scale on x and y axes
    ax.invert_yaxis()  # Invert y-axis to match image coordinates
    
    plt.tight_layout()
    plt.show()
    
    # Also print the data
    print("\n" + "=" * 80)
    print("              MATCHED KEYPOINTS COORDINATES")
    print("=" * 80)
    print(f"{'Match #':<8} {'Image 1 (x, y)':<25} {'Image 2 (x, y)':<25} {'Disparity':<15}")
    print("-" * 80)
    
    for i, match in enumerate(matches):
        kp1_idx = match.queryIdx
        kp2_idx = match.trainIdx
        
        x1, y1 = keypoints1[kp1_idx].pt
        x2, y2 = keypoints2[kp2_idx].pt
        disparity = x1 - x2
        
        print(f"{i:<8} ({x1:7.2f}, {y1:7.2f})       ({x2:7.2f}, {y2:7.2f})       {disparity:7.2f}")
        
        if i >= 19:  # Show first 20 matches
            remaining = len(matches) - 20
            if remaining > 0:
                print(f"\n... and {remaining} more matches")
            break
    
    print("=" * 80 + "\n")


def _remove_outliers(points, z_thresh=2.5):
    """Remove statistical outliers to clean the point cloud."""
    median = np.median(points, axis=0)
    diffs = np.linalg.norm(points - median, axis=1)
    med_diff = np.median(diffs)
    mask = diffs < z_thresh * (med_diff if med_diff > 0 else 1)
    return points[mask]

def present_3d_points(x, y, z):
    """
    Simple matplotlib 3D scatter plot.
    X = horizontal, Y = vertical, Z (depth) = depth axis.
    """
    x_arr, y_arr, z_arr = np.array(x), np.array(y), np.array(z)

    # Console statistics
    print("\n" + "=" * 55)
    print("         3D POINT CLOUD STATISTICS")
    print("=" * 55)
    print(f"  Total points (raw) : {len(x_arr)}")
    for name, arr in [("X", x_arr), ("Y", y_arr), ("Z (depth)", z_arr)]:
        print(f"\n  {name}:")
        print(f"    min={arr.min():.3f}  max={arr.max():.3f}  "
              f"mean={arr.mean():.3f}  std={arr.std():.3f}")
    print("=" * 55 + "\n")

    # Remove outliers
    points = np.column_stack((x_arr, y_arr, z_arr))
    # points = _remove_outliers(points) # Outlier removal commented out here
    px, py, pz = points[:, 0], points[:, 1], points[:, 2]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot with correct axis mapping:
    # px = horizontal (X), pz = depth (Y axis in plot), py = vertical (Z axis in plot)
    sc = ax.scatter(px, pz, -py, c=pz, cmap='viridis', s=15, alpha=0.7,
                    edgecolors='k', linewidth=0.2)

    ax.set_xlabel('X  (Horizontal, cm)')
    ax.set_ylabel('Z  (depth, cm)')
    ax.set_zlabel('Y  (Vertical, cm)')
    ax.set_title(f'3D Point Cloud  ({len(px)} points)')

    # Equal aspect ratio across all three displayed axes
    max_range = np.array([
        px.max() - px.min(),
        pz.max() - pz.min(),
        py.max() - py.min()
    ]).max() / 2.0

    mid_x = (px.max() + px.min()) * 0.5
    mid_z = (pz.max() + pz.min()) * 0.5
    mid_y = (py.max() + py.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_z - max_range, mid_z + max_range)
    ax.set_zlim(-mid_y - max_range, -mid_y + max_range)

    # Natural front-facing perspective: looking at the object from slightly above
    ax.view_init(elev=20, azim=-60)

    plt.colorbar(sc, ax=ax, shrink=0.6, label='Depth (cm)')
    plt.tight_layout()
    plt.show()

    return points

def show_boxes_from_xyz(x, y, z, box_dims, min_sep_factor=1.5,
                        grid_resolution=50, front_margin_ratio=0.15):
    """
    Parameters
    ----------
    x, y, z            : array-like – 3D point coordinates
    box_dims           : array-like of shape (n_boxes, 3) – each row is (dx, dy, dz)
                         Row 0 → box with lowest Z centroid, row 1 → next, etc.
    min_sep_factor     : float – minimum XZ distance between centroids as a
                         multiple of the current box XZ diameter.
    grid_resolution    : int – grid steps along each axis for dense search.
    front_margin_ratio : float – fraction of dz used as the front-surface
                         margin.  e.g. 0.15 means 15 % of the box depth.
    """
    box_dims = np.array(box_dims, dtype=float)
    n_boxes  = len(box_dims)

    x_arr, y_arr, z_arr = np.array(x), np.array(y), np.array(z)
    points = np.column_stack((x_arr, y_arr, z_arr))

    if len(points) < n_boxes:
        print("Not enough points for clustering.")
        return

    # ------------------------------------------------------------------
    # Step 0: Find the optimal SHARED Y_bottom for all boxes.
    #   In image/camera coords, larger Y = lower in the image.
    #   We want all boxes to share the same bottom surface (max Y edge).
    #   Each box then extends *upward* (toward smaller Y) by its own dy.
    #   Scan for the Y_bottom that captures the most points across
    #   the tallest box height.
    # ------------------------------------------------------------------
    max_dy = max(box_dims[:, 1])            # tallest box height
    y_candidates = np.linspace(y_arr.min(),
                               y_arr.max() + max_dy, 300)
    best_y_bottom, best_y_count = y_arr.max(), 0
    for yb in y_candidates:
        # points must satisfy:  yb - max_dy <= py <= yb
        cnt = np.sum((y_arr <= yb) & (y_arr >= yb - max_dy))
        if cnt > best_y_count:
            best_y_count = cnt
            best_y_bottom = yb
    print(f"\n  Shared Y_bottom (floor): {best_y_bottom:.2f}  "
          f"(captures {best_y_count}/{len(points)} points within "
          f"tallest box dy={max_dy:.1f})")

    # ------------------------------------------------------------------
    # Helper: count points on the FRONT SURFACE of a box
    #   Box goes from  z_front = cz  to  z_back = cz + dz
    #   Front surface margin = [z_front, z_front + dz * front_margin_ratio]
    #   Y range: [y_bottom - dy, y_bottom]   (bottom-aligned)
    #   X range: [cx - dx/2, cx + dx/2]
    # ------------------------------------------------------------------
    def _count_front_surface(pts, cx, cz, dims):
        dx, dy, dz = dims
        margin = dz * front_margin_ratio
        in_x = np.abs(pts[:, 0] - cx) <= dx / 2.0
        in_y = (pts[:, 1] >= best_y_bottom - dy) & (pts[:, 1] <= best_y_bottom)
        in_z = (pts[:, 2] >= cz) & (pts[:, 2] <= cz + margin)
        return np.sum(in_x & in_y & in_z)

    def _count_in_box(pts, cx, cz, dims):
        """Count points fully inside the box volume (used for colouring)."""
        dx, dy, dz = dims
        in_x = np.abs(pts[:, 0] - cx) <= dx / 2.0
        in_y = (pts[:, 1] >= best_y_bottom - dy) & (pts[:, 1] <= best_y_bottom)
        in_z = (pts[:, 2] >= cz) & (pts[:, 2] <= cz + dz)
        return np.sum(in_x & in_y & in_z)

    def _find_best_center(pts, dims, placed, min_dist):
        """Try every point as a potential box front-left-bottom corner.
        cx = X centre,  cz = Z_front (minimum Z of the box).
        Scoring uses the front-surface count.
        Return the best (cx, cz) and the front-surface count."""
        best_c, best_n = None, 0
        for pt in pts:
            cx, cz = pt[0], pt[2]
            # enforce separation (compare XZ centres)
            too_close = any(
                np.linalg.norm([cx - p[0], cz - p[1]]) < min_dist
                for p in placed
            )
            if too_close:
                continue
            n = _count_front_surface(pts, cx, cz, dims)
            if n > best_n:
                best_n = n
                best_c = np.array([cx, cz])
        return best_c, best_n

    # ------------------------------------------------------------------
    # Step 1: Pre-scan – best position per box size (no separation yet)
    # ------------------------------------------------------------------
    candidates = []
    for i in range(n_boxes):
        center, count = _find_best_center(points, box_dims[i], [], 0)
        if center is not None:
            candidates.append((count, i, center))
            print(f"  Pre-scan box {i}: front-surface count={count}  "
                  f"pos=(cx={center[0]:.1f}, z_front={center[1]:.1f})")

    # ------------------------------------------------------------------
    # Step 2: Place boxes densest-first, enforcing separation
    # ------------------------------------------------------------------
    candidates.sort(key=lambda c: c[0], reverse=True)

    remaining        = np.ones(len(points), dtype=bool)
    placed_centroids = []          # list of [cx, cz] for separation check
    raw_boxes        = {}

    for count, i, _ in candidates:
        dims     = box_dims[i]
        diameter = np.linalg.norm(dims[[0, 2]])
        min_dist = min_sep_factor * diameter

        candidate_pts = points[remaining]
        if len(candidate_pts) == 0:
            print(f"Box {i}: no remaining points.")
            raw_boxes[i] = (None, 0)
            continue

        center, n_found = _find_best_center(
            candidate_pts, dims, placed_centroids, min_dist
        )

        if center is None:
            print(f"Box {i}: no valid position found respecting min separation.")
            raw_boxes[i] = (None, 0)
            continue

        cx, cz_front = center
        dx, dy, dz = dims
        # Mark all points inside the full box volume as consumed
        in_x = np.abs(points[:, 0] - cx) <= dx / 2.0
        in_y = (points[:, 1] >= best_y_bottom - dy) & (points[:, 1] <= best_y_bottom)
        in_z = (points[:, 2] >= cz_front) & (points[:, 2] <= cz_front + dz)
        inside_global = in_x & in_y & in_z & remaining
        remaining[inside_global] = False

        placed_centroids.append(center.copy())
        raw_boxes[i] = (center.copy(), int(inside_global.sum()))
        print(f"  Box {i} placed: cx={cx:.1f}, y_bot={best_y_bottom:.1f}, "
              f"z_front={cz_front:.1f}  captured={inside_global.sum()}")

    # ------------------------------------------------------------------
    # Step 3: Sort by ascending Z front → re-pair with box_dims rows
    # ------------------------------------------------------------------
    placed = [(raw_boxes[i][0], raw_boxes[i][1], i)
              for i in range(n_boxes) if i in raw_boxes and raw_boxes[i][0] is not None]
    placed.sort(key=lambda b: b[0][1])   # sort by cz_front (ascending Z)

    boxes = []
    for rank, (center, n_cap, orig_i) in enumerate(placed):
        cx, cz_front = center
        dx, dy, dz = box_dims[rank]
        # pmin / pmax in (X, Y, Z) world coords
        pmin = np.array([cx - dx/2, best_y_bottom - dy, cz_front])
        pmax = np.array([cx + dx/2, best_y_bottom,      cz_front + dz])
        centroid = (pmin + pmax) / 2.0
        boxes.append((pmin, pmax, centroid, n_cap))

    # --- draw wireframe box ---
    def _draw_box(ax, pmin, pmax, color='r'):
        xmin, ymin, zmin = pmin
        xmax, ymax, zmax = pmax
        corners = np.array([
            [xmin, ymin, zmin], [xmax, ymin, zmin],
            [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax],
            [xmax, ymax, zmax], [xmin, ymax, zmax],
        ])
        edges = [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        for i, j in edges:
            ax.plot([corners[i,0], corners[j,0]],
                    [corners[i,2], corners[j,2]],
                    [-corners[i,1], -corners[j,1]],
                    color=color, linewidth=2)

    # --- colour points (full box volume) ---
    point_colors = np.full(len(points), -1, dtype=int)
    for i, (pmin, pmax, centroid, _) in enumerate(boxes):
        in_x = (points[:, 0] >= pmin[0]) & (points[:, 0] <= pmax[0])
        in_y = (points[:, 1] >= pmin[1]) & (points[:, 1] <= pmax[1])
        in_z = (points[:, 2] >= pmin[2]) & (points[:, 2] <= pmax[2])
        point_colors[in_x & in_y & in_z] = i

    px, py, pz = points[:,0], points[:,1], points[:,2]

    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection='3d')

    box_colors = ['red', 'blue', 'orange', 'green', 'magenta']

    noise_mask = point_colors == -1
    if noise_mask.any():
        ax.scatter(px[noise_mask], pz[noise_mask], -py[noise_mask],
                   c='grey', s=10, alpha=0.3, label='Noise')

    for i in range(len(boxes)):
        m = point_colors == i
        if m.any():
            ax.scatter(px[m], pz[m], -py[m],
                       c=box_colors[i % len(box_colors)],
                       s=15, alpha=0.7, edgecolors='k',
                       linewidth=0.2, label=f'Box {i}')

    ax.set_xlabel('X (Horizontal, cm)')
    ax.set_ylabel('Z (Depth, cm)')
    ax.set_zlabel('Y (Vertical, cm)')
    ax.set_title(f'3D Point Cloud — {len(boxes)} Boxes  sep≥{min_sep_factor}×diam')

    max_range = np.array([px.max()-px.min(),
                          pz.max()-pz.min(),
                          py.max()-py.min()]).max() / 2.0
    mid_x = (px.max()+px.min()) * 0.5
    mid_z = (pz.max()+pz.min()) * 0.5
    mid_y = (py.max()+py.min()) * 0.5

    ax.set_xlim(mid_x-max_range, mid_x+max_range)
    ax.set_ylim(mid_z-max_range, mid_z+max_range)
    ax.set_zlim(-mid_y-max_range, -mid_y+max_range)

    for i, (pmin, pmax, centroid, n_cap) in enumerate(boxes):
        _draw_box(ax, pmin, pmax, color=box_colors[i % len(box_colors)])
        dx, dy, dz = box_dims[i]
        print(f"Box {i} (Z-rank {i}): "
              f"y_bottom={best_y_bottom:.1f}, z_front={pmin[2]:.1f}  "
              f"size=({dx:.1f}×{dy:.1f}×{dz:.1f})  captured={n_cap}")

    print(f"\nNoise points: {noise_mask.sum()}")

    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.show()