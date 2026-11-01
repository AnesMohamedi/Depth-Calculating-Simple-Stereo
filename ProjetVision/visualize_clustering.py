import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sift_kp_detection import kp_detection, match_keypoints
from z_dimention import get_xyz_dimension

# Same box array as main.py
PYTHON_BOX_DIMENSIONS = np.array([
    [5.5,  5.5, 5.5],   # Box 0 - lowest  Z
    [5.5, 10.0, 5.5],   # Box 1 - tallest dy  <- used by default
    [10.0, 5.5, 5.5],   # Box 2 - highest Z
])

# Box wireframe edge connectivity (8 corners)
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # front face
    (4, 5), (5, 6), (6, 7), (7, 4),   # back face
    (0, 4), (1, 5), (2, 6), (3, 7),   # connecting edges
]


def _box_corners_plot(cx, y_bottom, cz_front, dx, dy, dz):
    """
    Return the 8 corners of the box in PLOT coordinates.

    World  : X = horizontal,  Y = vertical,  Z = depth
    Plot   : plot_x = X,  plot_y = Z,  plot_z = -Y
             (identical convention to present_3d_points)
    """
    xmin, xmax = cx - dx / 2.0,  cx + dx / 2.0
    ymin, ymax = y_bottom - dy,  y_bottom
    zmin, zmax = cz_front,       cz_front + dz

    corners_world = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin],
        [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax],
        [xmax, ymax, zmax], [xmin, ymax, zmax],
    ])
    # plot_x = world X,  plot_y = world Z,  plot_z = -world Y
    return np.column_stack([
        corners_world[:, 0],
        corners_world[:, 2],
        -corners_world[:, 1],
    ])


def visualize_clustering(img1_path, img2_path, mtx_path,
                         b=12, chosen_box=1,
                         n_frames=150, interval=40,
                         save_gif=False, output_path='sliding_box.gif'):
    """
    3-D animation that replicates exactly what show_boxes_from_xyz does:

      Step 0  Y-scan  : slide best_y_bottom until the slab [yb-max_dy, yb]
                        captures the most points  (vertical floor search).
      Step 1  X-sweep : fix Y at best_y_bottom and Z at best_cz_front,
                        slide the box horizontally (in X) and score each
                        position using the FRONT-SURFACE count
                        (same metric as _find_best_center / _count_front_surface).

    The animation shows Step 1:
      - 3-D scatter of all points (same axis convention as present_3d_points)
      - Box wireframe sliding left -> right in X at fixed Y and Z
      - Points on the front surface highlighted in red
      - Right panel: front-surface count vs X_center
    """
    # ------------------------------------------------------------------
    # 1. Build 3-D point cloud
    # ------------------------------------------------------------------
    mtx  = np.load(mtx_path)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load images. Check paths.")

    kp1, desc1, _ = kp_detection(img1)
    kp2, desc2, _ = kp_detection(img2)
    matches = match_keypoints(desc1, desc2, kp1, kp2, img1,
                              MIN_MATCH_DIST=0.8, MAX_ANGLE=7)

    x_list, y_list, z_list = get_xyz_dimension(b, kp1, kp2, mtx, matches)
    if not x_list:
        raise RuntimeError("No valid 3-D points found. Check images / params.")

    x_arr = np.array(x_list)
    y_arr = np.array(y_list)
    z_arr = np.array(z_list)

    # ------------------------------------------------------------------
    # 2. Box dimensions  (exact values from main.py)
    # ------------------------------------------------------------------
    dx, _dy, dz = PYTHON_BOX_DIMENSIONS[chosen_box]
    max_dy      = float(PYTHON_BOX_DIMENSIONS[:, 1].max())  # = algorithm's max_dy
    dy          = max_dy          # Y-scan always uses tallest dy
    front_margin = dz * 0.15     # same as front_margin_ratio=0.15 in algorithm

    box_label = (f"Box {chosen_box}  "
                 f"({PYTHON_BOX_DIMENSIONS[chosen_box, 0]:.1f} x "
                 f"{dy:.1f} x "
                 f"{PYTHON_BOX_DIMENSIONS[chosen_box, 2]:.1f} cm  "
                 f"dx x dy x dz)")
    print(f"\nUsing {box_label}")

    # ------------------------------------------------------------------
    # 3. Step 0: Y-scan  ->  best_y_bottom  (algorithm Step 0)
    # ------------------------------------------------------------------
    y_scan = np.linspace(y_arr.min(), y_arr.max() + dy, 300)
    best_y_bottom = float(y_scan[
        np.argmax([int(np.sum((y_arr <= yb) & (y_arr >= yb - dy)))
                   for yb in y_scan])
    ])
    print(f"Step 0  best_y_bottom = {best_y_bottom:.2f} cm")

    # ------------------------------------------------------------------
    # 4. Step 1: find best cz_front  (scan Z at best Y)
    #    _find_best_center tries every point as (cx, cz); we need to know
    #    which cz gives the highest count before we animate the X sweep.
    #    We pick best_cz_front = Z of the densest candidate.
    # ------------------------------------------------------------------
    y_mask = (y_arr >= best_y_bottom - dy) & (y_arr <= best_y_bottom)
    cands  = np.column_stack((x_arr, y_arr, z_arr))[y_mask] \
             if y_mask.any() else np.column_stack((x_arr, y_arr, z_arr))

    best_cz, best_cx_pre, best_pre_n = float(z_arr.min()), float(x_arr.mean()), -1
    for pt in cands:
        cx_t, cz_t = float(pt[0]), float(pt[2])
        in_x = np.abs(x_arr - cx_t) <= dx / 2.0
        in_y = y_mask
        in_z = (z_arr >= cz_t) & (z_arr <= cz_t + front_margin)  # front surface
        n = int(np.sum(in_x & in_y & in_z))
        if n > best_pre_n:
            best_pre_n  = n
            best_cz     = cz_t
            best_cx_pre = cx_t
    print(f"Step 1  best cz_front = {best_cz:.2f} cm  "
          f"(pre-scan cx={best_cx_pre:.2f}, front-surface={best_pre_n} pts)")

    # ------------------------------------------------------------------
    # 5. X-sweep: fix Y and Z, slide cx  ->  count front-surface pts
    #    This is exactly what _find_best_center does when iterating cx.
    # ------------------------------------------------------------------
    z_front_mask = (z_arr >= best_cz) & (z_arr <= best_cz + front_margin)
    yz_front_mask = y_mask & z_front_mask

    x_sweep = np.linspace(x_arr.min() - dx, x_arr.max() + dx, 300)
    counts  = np.array([
        int(np.sum((np.abs(x_arr - cx) <= dx / 2.0) & yz_front_mask))
        for cx in x_sweep
    ])

    best_x_idx  = int(np.argmax(counts))
    best_cx     = float(x_sweep[best_x_idx])
    best_cnt    = int(counts[best_x_idx])
    print(f"X-sweep  best cx = {best_cx:.2f} cm  ->  {best_cnt} pts on front surface")

    # ------------------------------------------------------------------
    # 6. Figure: 3-D scatter (left) + count curve (right)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('#f7f7f7')
    fig.suptitle(
        f"Box sliding horizontally (X-axis)  |  Y_bottom fixed = {best_y_bottom:.1f} cm  "
        f"|  Z_front fixed = {best_cz:.1f} cm  |  {box_label}",
        fontsize=11, fontweight='bold'
    )

    ax3d    = fig.add_subplot(121, projection='3d')
    ax_curve = fig.add_subplot(122)

    # ── 3-D scatter (all points, static background) ───────────────────
    ax3d.scatter(x_arr, z_arr, -y_arr,
                 c=z_arr, cmap='Blues_r', s=14, alpha=0.35,
                 edgecolors='none', depthshade=True)

    ax3d.set_xlabel('X  (Horizontal, cm)', fontsize=9, labelpad=5)
    ax3d.set_ylabel('Z  (Depth, cm)',      fontsize=9, labelpad=5)
    ax3d.set_zlabel('Y  (Vertical, cm)',   fontsize=9, labelpad=5)
    ax3d.set_title('3-D Point Cloud', fontsize=11, fontweight='bold')

    # Equal axis ranges (same logic as present_3d_points)
    half = max(x_arr.max() - x_arr.min(),
               z_arr.max() - z_arr.min(),
               y_arr.max() - y_arr.min()) / 2.0
    ax3d.set_xlim((x_arr.max() + x_arr.min()) / 2 - half,
                  (x_arr.max() + x_arr.min()) / 2 + half)
    ax3d.set_ylim((z_arr.max() + z_arr.min()) / 2 - half,
                  (z_arr.max() + z_arr.min()) / 2 + half)
    ax3d.set_zlim(-(y_arr.max() + y_arr.min()) / 2 - half,
                  -(y_arr.max() + y_arr.min()) / 2 + half)
    ax3d.view_init(elev=20, azim=-60)

    # Fixed Y_bottom floor plane (translucent green)
    x_pl = np.array([x_arr.min() - 1, x_arr.max() + 1])
    z_pl = np.array([z_arr.min() - 1, z_arr.max() + 1])
    Xp, Zp = np.meshgrid(x_pl, z_pl)
    ax3d.plot_surface(Xp, Zp,
                      np.full_like(Xp, -best_y_bottom),
                      alpha=0.07, color='green')
    ax3d.text(x_arr.max() + 0.5, z_arr.max() + 0.5, -best_y_bottom,
              f"Y_bot={best_y_bottom:.1f}",
              color='darkgreen', fontsize=8)

    # Pre-create 12 orange wireframe lines (updated each frame)
    box_lines = [
        ax3d.plot([], [], [], color='darkorange', lw=2.2, alpha=0.92)[0]
        for _ in range(12)
    ]

    # Inside / front-surface points (removed & recreated each frame)
    sc_in_ref = [None]

    # Info text (2-D overlay on 3-D axis)
    info_text = ax3d.text2D(
        0.02, 0.97, '',
        transform=ax3d.transAxes,
        fontsize=9, va='top', fontweight='bold', color='darkorange',
        bbox=dict(facecolor='white', alpha=0.87,
                  edgecolor='darkorange', boxstyle='round,pad=0.4')
    )

    # ── Count curve (right panel) ──────────────────────────────────────
    ax_curve.set_facecolor('#edf2f7')
    ax_curve.set_title('Front-Surface Count  vs  X_center',
                        fontsize=12, fontweight='bold')
    ax_curve.set_xlabel('X_center  (cm)', fontsize=11)
    ax_curve.set_ylabel('Points on front surface', fontsize=11)
    ax_curve.grid(True, alpha=0.35, color='white', linewidth=1.4)

    ax_curve.plot(x_sweep, counts, color='steelblue',
                  linewidth=2.2, alpha=0.85, label='Count curve')
    ax_curve.axvline(best_cx, color='#00aa44', linestyle='--', lw=1.8,
                     alpha=0.78,
                     label=f'Best X = {best_cx:.1f} cm  ({best_cnt} pts)')
    ax_curve.legend(fontsize=9, loc='upper right', framealpha=0.88)
    ax_curve.set_xlim(x_sweep[0], x_sweep[-1])

    cur_vline = ax_curve.axvline(x_sweep[0], color='darkorange',
                                  lw=2.5, alpha=0.9)
    cur_dot,  = ax_curve.plot([], [], 'o', color='darkorange', markersize=10)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # 7. Frame index list with hold pause at best X position
    # ------------------------------------------------------------------
    base_frames  = np.linspace(0, len(x_sweep) - 1, n_frames).astype(int)
    hold_insert  = int(np.argmin(np.abs(base_frames - best_x_idx)))
    frame_idxs   = np.concatenate([
        base_frames[:hold_insert],
        np.full(30, base_frames[hold_insert], dtype=int),  # 30-frame hold
        base_frames[hold_insert:]
    ])
    total_frames = len(frame_idxs)

    # ------------------------------------------------------------------
    # 8. Animation update
    # ------------------------------------------------------------------
    def update(frame_num):
        idx = int(frame_idxs[frame_num])
        cx  = float(x_sweep[idx])
        cnt = int(counts[idx])
        best_so_far = int(counts[:idx + 1].max())
        is_best     = (idx == best_x_idx)

        # Update box wireframe
        corners = _box_corners_plot(cx, best_y_bottom, best_cz, dx, dy, dz)
        for k, (i, j) in enumerate(BOX_EDGES):
            box_lines[k].set_data_3d(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]]
            )

        # Update front-surface highlighted points
        if sc_in_ref[0] is not None:
            sc_in_ref[0].remove()
            sc_in_ref[0] = None

        in_front = (np.abs(x_arr - cx) <= dx / 2.0) & yz_front_mask
        if in_front.any():
            sc_in_ref[0] = ax3d.scatter(
                x_arr[in_front], z_arr[in_front], -y_arr[in_front],
                c='crimson', s=65, alpha=0.95,
                edgecolors='darkred', linewidth=0.5, depthshade=False
            )

        # Info box
        clr = '#00aa44' if is_best else 'darkorange'
        info_text.set_text(
            f"X_center    = {cx:.1f} cm\n"
            f"Front pts   = {cnt}\n"
            f"Best so far = {best_so_far}"
            + ("  <-- BEST!" if is_best else "")
        )
        info_text.set_color(clr)
        info_text.get_bbox_patch().set_edgecolor(clr)

        # Count curve tracker
        cur_vline.set_xdata([cx, cx])
        cur_dot.set_data([cx], [cnt])

    ani = animation.FuncAnimation(
        fig, update,
        frames=total_frames,
        interval=interval,
        blit=False,
        repeat=True
    )

    if save_gif:
        ani.save(output_path, writer='pillow', fps=max(1, 1000 // interval))
        print(f"Animation saved to: {output_path}")

    plt.show()
    return ani


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ani = visualize_clustering(
        img1_path='./Images/im15.jpg',
        img2_path='./Images/im16.jpg',
        mtx_path='./camera_params/mtx.npy',
        b=12,
        chosen_box=1,    # Box 1: 5.5 x 10 x 5.5 cm  (tallest dy)
        n_frames=150,
        interval=40,
        save_gif=False,
    )



