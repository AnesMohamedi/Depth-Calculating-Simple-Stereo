import cv2
import numpy as np
import matplotlib.pyplot as plt
from sift_kp_detection import kp_detection, match_keypoints


def visualize_geometry(img1_path, img2_path, mtx_path, b=12, num_matches=5):
    """
    Create a single-figure visualization explaining stereo triangulation:
      - Left / Right images with principal point (Ox, Oy) and selected keypoints.
      - Formulas for disparity, Z, X, Y.
      - A top-down schematic (X-Z plane) showing cameras, baseline, a 3D point,
        and the similar-triangle geometry.
      - A table of real computed values for the selected matches.
    """
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load images. Check paths.")

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    mtx = np.load(mtx_path)
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    Ox = mtx[0, 2]
    Oy = mtx[1, 2]

    # ------------------------------------------------------------------
    # 2. Keypoints & matches (same pipeline as main.py)
    # ------------------------------------------------------------------
    kp1, desc1, _ = kp_detection(img1)
    kp2, desc2, _ = kp_detection(img2)
    matches = match_keypoints(desc1, desc2, kp1, kp2, img1,
                              MIN_MATCH_DIST=0.8, MAX_ANGLE=7)

    # Prefer matches with positive disparity (x1 > x2) so 3D values can be computed
    pos_matches = [m for m in matches
                   if kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0] > 0]
    pool = pos_matches if pos_matches else matches

    # Deduplicate: skip any match whose image-1 keypoint is within 80 px
    # of an already-selected one (avoids clusters of near-identical points)
    MIN_KP_DIST = 80
    selected = []
    for m in pool:
        x1c, y1c = kp1[m.queryIdx].pt
        too_close = any(
            abs(x1c - kp1[s.queryIdx].pt[0]) < MIN_KP_DIST and
            abs(y1c - kp1[s.queryIdx].pt[1]) < MIN_KP_DIST
            for s in selected
        )
        if not too_close:
            selected.append(m)
        if len(selected) == num_matches:
            break
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(selected), 1)))

    # ------------------------------------------------------------------
    # 3. Build figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1],
                          width_ratios=[1, 0.8, 1])

    # ================================================================
    # TOP LEFT: Left image
    # ================================================================
    ax_l = fig.add_subplot(gs[0, 0])
    ax_l.imshow(img1_rgb)
    ax_l.set_aspect('auto')
    ax_l.set_title("Left Image (Camera 1)", fontsize=13, fontweight='bold')
    ax_l.set_xlabel("x (pixels)")
    ax_l.set_ylabel("y (pixels)")

    # Principal axes & optical center
    ax_l.axvline(Ox, color='cyan', linestyle='--', linewidth=1.2, alpha=0.8)
    ax_l.axhline(Oy, color='magenta', linestyle='--', linewidth=1.2, alpha=0.8)
    ax_l.plot(Ox, Oy, marker='+', color='white', markersize=14,
              markeredgewidth=2.5, label=f"Principal Point (Ox={Ox:.1f}, Oy={Oy:.1f})")

    # ================================================================
    # TOP RIGHT: Right image
    # ================================================================
    ax_r = fig.add_subplot(gs[0, 2])
    ax_r.imshow(img2_rgb)
    ax_r.set_aspect('auto')
    ax_r.set_title("Right Image (Camera 2)", fontsize=13, fontweight='bold')
    ax_r.set_xlabel("x (pixels)")
    ax_r.set_ylabel("y (pixels)")

    ax_r.axvline(Ox, color='cyan', linestyle='--', linewidth=1.2, alpha=0.8)
    ax_r.axhline(Oy, color='magenta', linestyle='--', linewidth=1.2, alpha=0.8)
    ax_r.plot(Ox, Oy, marker='+', color='white', markersize=14, markeredgewidth=2.5)

    # ================================================================
    # TOP CENTER: Formulas & explanation
    # ================================================================
    ax_f = fig.add_subplot(gs[0, 1])
    ax_f.axis('off')
    formula_text = (
        "STEREO TRIANGULATION\n"
        "═════════════════════\n\n"
        "Disparity  (d):\n"
        "  d = x₁ – x₂\n\n"
        "Depth  (Z):\n"
        "  Z = b · fx / d\n\n"
        "Horizontal  (X):\n"
        "  X = b · (x₁ – Ox) / d\n\n"
        "Vertical  (Y):\n"
        "  Y = b · (y₁ – Oy) / d\n\n"
        "Where:\n"
        "  b  = baseline (cm)\n"
        "  fx = focal length (pixels)\n"
        "  (Ox, Oy) = principal point"
    )
    ax_f.text(0.5, 0.5, formula_text, transform=ax_f.transAxes,
              fontsize=11, verticalalignment='center', horizontalalignment='center',
              fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', alpha=0.6,
                        edgecolor='black', linewidth=1.5))

    # ================================================================
    # Draw selected matches on both images + collect table rows
    # ================================================================
    table_rows = []
    for i, m in enumerate(selected):
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        d = x1 - x2

        # Draw keypoints
        ax_l.plot(x1, y1, 'o', color=colors[i], markersize=9,
                  markeredgecolor='black', markeredgewidth=1.2)
        ax_r.plot(x2, y2, 's', color=colors[i], markersize=9,
                  markeredgecolor='black', markeredgewidth=1.2)

        # Alternate annotation offsets so labels don't pile up
        offsets = [(40, 20), (-70, 20), (40, -35), (-70, -35), (55, 5)]
        ox, oy = offsets[i % len(offsets)]
        ax_l.annotate(f"P{i+1} ({x1:.0f}, {y1:.0f})",
                      xy=(x1, y1), xytext=(ox, oy),
                      textcoords='offset points',
                      color=colors[i], fontsize=8, fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.0))
        ax_r.annotate(f"P{i+1} ({x2:.0f}, {y2:.0f})",
                      xy=(x2, y2), xytext=(ox, oy),
                      textcoords='offset points',
                      color=colors[i], fontsize=8, fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.0))

        # Compute 3D (same logic as z_dimention.py)
        if d > 0:
            Z = b * fx / d
            X = b * (x1 - Ox) / d
            Y = b * (y1 - Oy) / d
            table_rows.append([
                f"P{i+1}",
                f"{x1:.1f}", f"{y1:.1f}",
                f"{x2:.1f}", f"{y2:.1f}",
                f"{d:.1f}",
                f"{Z:.1f}", f"{X:.1f}", f"{Y:.1f}"
            ])

    ax_l.set_xlim(0, img1.shape[1])
    ax_l.set_ylim(img1.shape[0], 0)
    ax_l.legend(loc='upper right', fontsize=8)

    ax_r.set_xlim(0, img2.shape[1])
    ax_r.set_ylim(img2.shape[0], 0)

    # ================================================================
    # BOTTOM LEFT: Geometric schematic (Top-down X-Z view)
    # ================================================================
    ax_g = fig.add_subplot(gs[1, :2])
    ax_g.set_title("Geometric Interpretation (Top-Down X-Z Plane)",
                   fontsize=13, fontweight='bold')

    # Cameras on the baseline
    ax_g.plot(0, 0, 'ko', markersize=12, label='Camera 1')
    ax_g.plot(b, 0, 'ko', markersize=12, label='Camera 2')
    ax_g.plot([0, b], [0, 0], 'k-', linewidth=3)
    ax_g.text(-0.8, -3.5, "Camera 1\n(Left)", ha='right', fontsize=10)
    ax_g.text(b + 0.8, -3.5, "Camera 2\n(Right)", ha='left', fontsize=10)
    ax_g.text(b / 2, -6.5, f"Baseline  b = {b} cm",
              ha='center', fontsize=11, color='darkblue', fontweight='bold')

    # Illustrative 3D point (use first valid table row if possible)
    if table_rows:
        X_p = float(table_rows[0][7])
        Z_p = float(table_rows[0][6])
        # Clamp for drawing so it always looks reasonable
        X_p = max(min(X_p, b * 2.5), -b * 0.5)
        Z_p = max(min(Z_p, b * 6), 10)
    else:
        X_p, Z_p = b / 2, 40

    ax_g.plot(X_p, Z_p, 'ro', markersize=11)
    ax_g.annotate(f"P  (X≈{X_p:.1f}, Z≈{Z_p:.1f})",
                  xy=(X_p, Z_p), xytext=(X_p + 4, Z_p + 2),
                  fontsize=11, color='red', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    # Rays from cameras to point
    ax_g.plot([0, X_p], [0, Z_p], 'g--', linewidth=1.5, alpha=0.7)
    ax_g.plot([b, X_p], [0, Z_p], 'g--', linewidth=1.5, alpha=0.7)

    # Image planes (vertical lines at each camera position)
    # The image plane is at x=0 (cam1) and x=b (cam2).
    # On each plane the projection of point P is at height = (xi - Ox) / fx * Z
    # We display this as a scaled segment along Y on the image-plane line,
    # with Ox corresponding to y=0 on the schematic.
    ax_g.plot([0, 0], [-8, 18], 'k-', linewidth=1)
    ax_g.plot([b, b], [-8, 18], 'k-', linewidth=1)
    ax_g.text(0, -11.5, "Image plane 1", ha='center', fontsize=9, style='italic')
    ax_g.text(b, -11.5, "Image plane 2", ha='center', fontsize=9, style='italic')

    # x1-Ox and x2-Ox: in the top-down X-Z schematic, these represent the
    # horizontal distance from each camera's principal ray to point P.
    # - Camera 1 is at X=0; its ray to P intersects the image plane at X=X_p.
    #   So x1-Ox (in world units) = X_p - 0 = X_p.
    # - Camera 2 is at X=b; its ray to P intersects the image plane at X=X_p.
    #   So x2-Ox (in world units) = X_p - b.
    # Draw both as horizontal double-headed arrows at a fixed Z height.
    arrow_z = Z_p * 0.40   # Z height where the arrows are drawn

    # Camera 1: arrow from x=0 to x=X_p
    ax_g.annotate("", xy=(X_p, arrow_z), xytext=(0, arrow_z),
                  arrowprops=dict(arrowstyle='<->', color='darkmagenta', lw=2.0))
    ax_g.text(X_p / 2, arrow_z + 2.0, "x₁–Ox",
              ha='center', fontsize=9, color='darkmagenta', fontweight='bold')

    # Camera 2: arrow from x=b to x=X_p  (negative direction since X_p < b)
    ax_g.annotate("", xy=(X_p, arrow_z - 5), xytext=(b, arrow_z - 5),
                  arrowprops=dict(arrowstyle='<->', color='purple', lw=2.0))
    ax_g.text((X_p + b) / 2, arrow_z - 5 + 2.0, "x₂–Ox",
              ha='center', fontsize=9, color='purple', fontweight='bold')

    # Mark the projection hit on each image plane (vertical line) as a dot at Z=0
    ax_g.plot(0, 0, 'm^', markersize=8, zorder=5)  # cam1 at origin
    ax_g.plot(b, 0, 'v', color='purple', markersize=8, zorder=5)  # cam2 at x=b

    # Bracket showing disparity: horizontal double arrow at a fixed height
    # d = x1 - x2 is the pixel disparity; in world X it maps to b = Z*d/fx
    # We show the baseline arrow separately from x1-Ox / x2-Ox.
    y_bracket = arrow_z + 8
    ax_g.annotate("", xy=(0, y_bracket), xytext=(b, y_bracket),
                  arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
    ax_g.text(b / 2, y_bracket + 2.0, "Baseline b",
              ha='center', color='darkorange', fontsize=10, fontweight='bold')

    # Depth arrow
    ax_g.annotate("", xy=(X_p, Z_p), xytext=(X_p, 0),
                  arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax_g.text(X_p + 2.5, Z_p / 2, "Z  (depth)",
              color='indigo', fontsize=11, rotation=90, va='center')

    # Similar-triangles annotation box
    ax_g.text(b * 1.45, Z_p * 0.65,
              "Similar Triangles\n"
              "─────────────────\n"
              "Z / b = fx / (x₁-x₂)\n"
              "  →  Z = b·fx / d\n\n"
              "X / Z = (x₁-Ox) / fx\n"
              "  →  X = b·(x₁-Ox)/d\n\n"
              "Y / Z = (y₁-Oy) / fy\n"
              "  →  Y = b·(y₁-Oy)/d",
              fontsize=10, verticalalignment='center',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                        edgecolor='black', linewidth=1.2))

    ax_g.set_xlim(min(-10, X_p - 5), b * 2.8)
    ax_g.set_ylim(-16, max(Z_p * 1.25, 30, y_bracket + 6))
    ax_g.set_xlabel("X  (horizontal, cm)", fontsize=11)
    ax_g.set_ylabel("Z  (depth, cm)", fontsize=11)
    ax_g.grid(True, alpha=0.3)
    ax_g.legend(loc='upper left', fontsize=9)

    # ================================================================
    # BOTTOM RIGHT: Data table
    # ================================================================
    ax_t = fig.add_subplot(gs[1, 2])
    ax_t.axis('off')
    ax_t.set_title("Computed Values for Selected Points",
                   fontsize=13, fontweight='bold', pad=10)

    columns = ["Pt", "x₁", "y₁", "x₂", "y₂", "d", "Z", "X", "Y"]
    if table_rows:
        tbl = ax_t.table(cellText=table_rows, colLabels=columns,
                         loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.15, 1.8)
        # Header styling
        for j in range(len(columns)):
            tbl[(0, j)].set_facecolor('#4CAF50')
            tbl[(0, j)].set_text_props(weight='bold', color='white')
        # Alternate row colours
        for i in range(1, len(table_rows) + 1):
            for j in range(len(columns)):
                if i % 2 == 0:
                    tbl[(i, j)].set_facecolor('#f0f0f0')
    else:
        ax_t.text(0.5, 0.5, "No valid matches found.",
                  ha='center', va='center', fontsize=12, color='red')

    plt.tight_layout()
    plt.savefig('stereo_geometry_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to: stereo_geometry_visualization.png")


if __name__ == "__main__":
    visualize_geometry(
        img1_path='./Images/im15.jpg',
        img2_path='./Images/im16.jpg',
        mtx_path='./camera_params/mtx.npy',
        b=12,
        num_matches=5
    )
