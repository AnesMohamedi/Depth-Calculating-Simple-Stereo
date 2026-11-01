import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import math

from sift_kp_detection import kp_detection

# ---------------------------------------------------------------------------
# Each filter stage replicated exactly from the algorithm
# ---------------------------------------------------------------------------

def _ratio_test(raw_knn, ratio=0.8):
    """Lowe's ratio test — same threshold as match_keypoints (MIN_MATCH_DIST=0.8)."""
    return [m for m, n in raw_knn if m.distance < ratio * n.distance]


def _angle_filter(matches, kp1, kp2, max_angle=7):
    """Epipolar angle filter — same MAX_ANGLE=7 as match_keypoints."""
    out = []
    for m in matches:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        angle = math.atan2(abs(y2 - y1), abs(x2 - x1)) * 180 / math.pi
        if angle <= max_angle:
            out.append(m)
    return out


def _white_filter(matches, kp1, img1):
    """White-pixel filter — same check as match_keypoints."""
    out = []
    for m in matches:
        x1, y1 = kp1[m.queryIdx].pt
        px = img1[int(y1), int(x1)]
        if px[0] < 240 or px[1] < 240 or px[2] < 240:
            out.append(m)
    return out


def _yshift_filter(matches, kp1, kp2, max_yshift=500):
    """Y-shift filter — same threshold as get_xyz_dimension."""
    return [m for m in matches
            if abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) <= max_yshift]


def _disparity_filter(matches, kp1, kp2):
    """Disparity > 0 guard — same as get_xyz_dimension."""
    return [m for m in matches
            if abs(kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]) > 0]


def _zbounds_filter(matches, kp1, kp2, mtx, b, z_min=30, z_max=100):
    """Z-bounds filter — same range as get_xyz_dimension (30..100 cm)."""
    fx = mtx[0, 0]
    out = []
    for m in matches:
        x1 = kp1[m.queryIdx].pt[0]
        x2 = kp2[m.trainIdx].pt[0]
        d = x1 - x2
        if abs(d) <= 0:
            continue
        z = b * fx / d
        if z_min <= z <= z_max:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def visualize_cleaning(img1_path, img2_path, mtx_path,
                       b=12, max_display=120,
                       hold=25, highlight=18,
                       interval=70,
                       save_gif=False, output_path='cleaning.gif'):
    """
    Animated walk-through of every noise-removal step in the pipeline.

    Performance notes:
      - All match lines are drawn as a single LineCollection (2 calls per frame).
      - RGBA colour arrays for every unique (stage, phase) are precomputed once
        before the animation starts (~19 unique states).
      - Sampling always includes ALL final survivors so the last stage is fully
        visible; remaining slots are filled with representative removed matches.
    """
    C_ALIVE  = '#00dd77'
    C_ORANGE = '#ffaa00'
    C_RED    = '#ff3333'
    C_DEAD   = '#2a2a3a'

    # ------------------------------------------------------------------
    # 1. Load images
    # ------------------------------------------------------------------
    mtx  = np.load(mtx_path)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load images. Check paths.")

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # 2. SIFT keypoints
    # ------------------------------------------------------------------
    kp1, desc1, _ = kp_detection(img1)
    kp2, desc2, _ = kp_detection(img2)

    # ------------------------------------------------------------------
    # 3. Run every filter stage (exact same code as the algorithm)
    # ------------------------------------------------------------------
    bf           = cv2.BFMatcher()
    raw_knn      = bf.knnMatch(desc1, desc2, k=2)
    all_raw      = [m for m, n in raw_knn]
    after_ratio  = _ratio_test(raw_knn)
    after_angle  = _angle_filter(after_ratio,  kp1, kp2)
    after_white  = _white_filter(after_angle,  kp1, img1)
    after_yshift = _yshift_filter(after_white, kp1, kp2)
    after_disp   = _disparity_filter(after_yshift, kp1, kp2)
    after_z      = _zbounds_filter(after_disp, kp1, kp2, mtx, b)

    stages = [
        ("Raw kNN Matches",
         "All SIFT keypoint pairs\nbefore any filtering",
         all_raw),
        ("1. Ratio Test",
         "Lowe's ratio:  d < 0.8 x d_next\nRemoves ambiguous matches",
         after_ratio),
        ("2. Angle Filter",
         "Epipolar constraint:  angle <= 7 deg\nRemoves non-horizontal matches",
         after_angle),
        ("3. White-Pixel Filter",
         "Background removal:\nall channels >= 240 -> rejected",
         after_white),
        ("4. Y-Shift Filter",
         "|y1 - y2| <= 500 px\nRemoves large vertical displacements",
         after_yshift),
        ("5. Disparity Filter",
         "Disparity  d = x1 - x2  > 0\nRemoves zero / negative disparity",
         after_disp),
        ("6. Z-Bounds Filter",
         "30 cm  <=  Z  <=  100 cm\nRemoves out-of-range depth points",
         after_z),
    ]
    counts   = [len(s[2]) for s in stages]
    n_stages = len(stages)

    print("\nCleaning pipeline:")
    for name, _, survivors in stages:
        print(f"  {name:35s}  {len(survivors):5d} matches")

    # ------------------------------------------------------------------
    # 4. Smart sampling:
    #    Always keep ALL final survivors so the last stage is fully visible.
    #    Fill remaining slots evenly from matches removed at each earlier stage.
    # ------------------------------------------------------------------
    rng           = np.random.default_rng(42)
    survivors_set = {id(m) for m in after_z}
    final_surv    = [m for m in all_raw if id(m) in survivors_set]
    non_surv      = [m for m in all_raw if id(m) not in survivors_set]

    remaining = max(0, max_display - len(final_surv))
    if len(non_surv) > remaining:
        idx     = rng.choice(len(non_surv), remaining, replace=False)
        sampled = [non_surv[i] for i in idx]
    else:
        sampled = non_surv

    disp_raw = final_surv + sampled
    n_disp   = len(disp_raw)

    print(f"  Displaying {n_disp} lines "
          f"({len(final_surv)} survivors + {len(sampled)} removed samples)")

    # ------------------------------------------------------------------
    # 5. Determine removal stage for each displayed match
    # ------------------------------------------------------------------
    survivor_id_sets = [{id(m) for m in s[2]} for s in stages]

    removal_stage = np.full(n_disp, -1, dtype=int)
    for k, m in enumerate(disp_raw):
        for si in range(1, n_stages):
            if id(m) not in survivor_id_sets[si]:
                removal_stage[k] = si
                break

    # ------------------------------------------------------------------
    # 6. Precompute RGBA + linewidth arrays for every unique (si, phase)
    #    Only ~19 unique states — runs in milliseconds.
    # ------------------------------------------------------------------
    rgba_alive    = np.array(mcolors.to_rgba(C_ALIVE,  0.55))
    rgba_orange   = np.array(mcolors.to_rgba(C_ORANGE, 0.90))
    rgba_red      = np.array(mcolors.to_rgba(C_RED,    0.80))
    rgba_dead_now = np.array(mcolors.to_rgba(C_DEAD,   0.15))
    rgba_dead_old = np.array(mcolors.to_rgba(C_DEAD,   0.08))

    unique_keys = {(0, 'hold')}
    for si in range(1, n_stages):
        unique_keys.update({(si, 'highlight'), (si, 'remove'), (si, 'hold')})

    frame_cache = {}
    for (si, phase) in unique_keys:
        colors  = np.empty((n_disp, 4), dtype=float)
        lwidths = np.empty(n_disp,      dtype=float)
        rs_arr  = removal_stage

        alive_mask    = (rs_arr == -1) | (rs_arr > si)
        removed_now   = (rs_arr == si)
        removed_before = (rs_arr >= 0) & (rs_arr < si)

        colors[alive_mask]     = rgba_alive
        lwidths[alive_mask]    = 0.75

        colors[removed_before]  = rgba_dead_old
        lwidths[removed_before] = 0.30

        if phase == 'highlight':
            colors[removed_now]  = rgba_orange
            lwidths[removed_now] = 1.60
        elif phase == 'remove':
            colors[removed_now]  = rgba_red
            lwidths[removed_now] = 1.10
        else:  # hold — already gone this stage
            colors[removed_now]  = rgba_dead_now
            lwidths[removed_now] = 0.40

        frame_cache[(si, phase)] = (colors, lwidths)

    print(f"  Precomputed {len(frame_cache)} frame states  ->  animation ready.")

    # ------------------------------------------------------------------
    # 7. Frame schedule
    # ------------------------------------------------------------------
    frame_list = []
    for _ in range(hold):
        frame_list.append((0, 'hold'))
    for si in range(1, n_stages):
        for _ in range(highlight):
            frame_list.append((si, 'highlight'))
        for _ in range(highlight // 2):
            frame_list.append((si, 'remove'))
        for _ in range(hold):
            frame_list.append((si, 'hold'))
    for _ in range(hold):
        frame_list.append((n_stages - 1, 'hold'))

    # ------------------------------------------------------------------
    # 8. Resize images for display
    # ------------------------------------------------------------------
    TARGET_H = 420
    h1, w1   = img1_rgb.shape[:2]
    scale    = TARGET_H / h1
    disp_h   = TARGET_H
    disp_w   = int(w1 * scale)
    img1_d   = cv2.resize(img1_rgb, (disp_w, disp_h))
    img2_d   = cv2.resize(img2_rgb, (disp_w, disp_h))
    combined = np.concatenate([img1_d, img2_d], axis=1)

    def kp_to_disp(kp, side):
        x, y = kp.pt
        return x * scale + side * disp_w, y * scale

    # ------------------------------------------------------------------
    # 9. Build figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(17, 7))
    fig.patch.set_facecolor('#12122a')
    fig.suptitle("SIFT Match Cleaning  —  Noise Removal Pipeline",
                 fontsize=14, fontweight='bold', color='white', y=0.97)

    gs      = fig.add_gridspec(1, 2, width_ratios=[2.3, 1], wspace=0.03)
    ax_img  = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])

    # ── Left panel ───────────────────────────────────────────────────
    ax_img.imshow(combined, aspect='auto')
    ax_img.axis('off')
    ax_img.axvline(disp_w - 0.5, color='yellow', linewidth=1.2, alpha=0.7)
    ax_img.text(disp_w * 0.5, -14, "Left Image  (Camera 1)",
                ha='center', fontsize=10, color='#cccccc')
    ax_img.text(disp_w * 1.5, -14, "Right Image  (Camera 2)",
                ha='center', fontsize=10, color='#cccccc')
    ax_img.set_xlim(0, disp_w * 2)
    ax_img.set_ylim(disp_h + 5, -22)

    # Single LineCollection — all match segments in one artist
    segments = [[[*kp_to_disp(kp1[m.queryIdx], 0)],
                 [*kp_to_disp(kp2[m.trainIdx], 1)]]
                for m in disp_raw]

    init_colors, init_lw = frame_cache[(0, 'hold')]
    lc = LineCollection(segments, colors=init_colors,
                        linewidths=init_lw, zorder=2)
    ax_img.add_collection(lc)

    img_count = ax_img.text(
        0.01, 0.99, '', transform=ax_img.transAxes,
        ha='left', va='top', fontsize=11, fontweight='bold', color='white',
        bbox=dict(facecolor='#00000088', edgecolor='none',
                  boxstyle='round,pad=0.3')
    )

    # ── Right panel ──────────────────────────────────────────────────
    ax_info.set_facecolor('#1a1a35')
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis('off')

    stage_title = ax_info.text(
        0.5, 0.965, '', ha='center', va='top', fontsize=12,
        fontweight='bold', color='white', transform=ax_info.transAxes
    )
    stage_desc = ax_info.text(
        0.5, 0.875, '', ha='center', va='top', fontsize=8.5,
        color='#9999dd', transform=ax_info.transAxes, linespacing=1.45
    )
    count_big = ax_info.text(
        0.5, 0.760, '', ha='center', va='top', fontsize=16,
        fontweight='bold', color=C_ALIVE, transform=ax_info.transAxes
    )
    ax_info.text(
        0.5, 0.700, 'matches remaining', ha='center', va='top',
        fontsize=8, color='#888888', transform=ax_info.transAxes
    )
    removed_label = ax_info.text(
        0.5, 0.655, '', ha='center', va='top', fontsize=8.5,
        color='#ff6666', transform=ax_info.transAxes
    )

    ax_info.add_patch(Rectangle(
        (0.05, 0.585), 0.90, 0.030,
        transform=ax_info.transAxes,
        facecolor='#2a2a4a', edgecolor='#444466', linewidth=1
    ))
    bar_fg = Rectangle(
        (0.05, 0.585), 0.001, 0.030,
        transform=ax_info.transAxes,
        facecolor=C_ALIVE, edgecolor='none'
    )
    ax_info.add_patch(bar_fg)
    ax_info.text(0.5, 0.580, 'matches retained', ha='center', va='top',
                 fontsize=7, color='#666688', transform=ax_info.transAxes)

    row_y           = [0.510 - i * 0.060 for i in range(n_stages)]
    row_name_texts  = []
    row_count_texts = []
    for i, (name, _, _) in enumerate(stages):
        nt = ax_info.text(0.06, row_y[i], name[:26],
                          ha='left', va='top', fontsize=7.5,
                          color='#555577', transform=ax_info.transAxes)
        ct = ax_info.text(0.94, row_y[i], str(counts[i]),
                          ha='right', va='top', fontsize=7.5,
                          color='#555577', transform=ax_info.transAxes)
        row_name_texts.append(nt)
        row_count_texts.append(ct)

    ax_info.axhline(0.072, xmin=0.03, xmax=0.97, color='#333355',
                    linewidth=0.8)
    ax_info.text(0.06, 0.058, 'Green = surviving',
                 ha='left', fontsize=7.5, color=C_ALIVE,
                 transform=ax_info.transAxes)
    ax_info.text(0.06, 0.022, 'Orange/Red = removed',
                 ha='left', fontsize=7.5, color=C_ORANGE,
                 transform=ax_info.transAxes)

    # ------------------------------------------------------------------
    # 10. Animation update — pure cache lookup, no per-frame computation
    # ------------------------------------------------------------------
    _prev_key = [None]

    def update(frame_num):
        si, phase = frame_list[frame_num]
        key = (si, phase)
        if key == _prev_key[0]:
            return                          # hold frame — nothing to do
        _prev_key[0] = key

        # Apply precomputed arrays: 2 calls cover all n_disp lines
        colors, lwidths = frame_cache[key]
        lc.set_colors(colors)
        lc.set_linewidths(lwidths)

        # Text / UI updates
        name, desc, _ = stages[si]
        cnt           = counts[si]

        img_count.set_text(f"{cnt} / {counts[0]}  matches")
        stage_title.set_text(name)
        stage_desc.set_text(desc)
        count_big.set_text(str(cnt))

        removed = counts[0] - cnt
        pct     = removed / counts[0] * 100 if counts[0] > 0 else 0
        removed_label.set_text(f"Removed: {removed}  ({pct:.1f} %)")

        frac = cnt / counts[0] if counts[0] > 0 else 0
        bar_fg.set_width(0.90 * frac)
        bar_fg.set_facecolor(C_ALIVE if frac > 0.5 else
                             C_ORANGE if frac > 0.2 else C_RED)

        for i in range(n_stages):
            col = '#448844' if i < si else '#ffffff' if i == si else '#444466'
            row_name_texts[i].set_color(col)
            row_count_texts[i].set_color(col)
            row_name_texts[i].set_fontsize(8.5 if i == si else 7.5)
            row_count_texts[i].set_fontsize(8.5 if i == si else 7.5)

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(frame_list),
        interval=interval,
        blit=False,
        repeat=True,
        cache_frame_data=False
    )

    if save_gif:
        ani.save(output_path, writer='pillow', fps=max(1, 1000 // interval))
        print(f"Animation saved to: {output_path}")

    plt.show()
    return ani


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ani = visualize_cleaning(
        img1_path='./Images/im15.jpg',
        img2_path='./Images/im16.jpg',
        mtx_path='./camera_params/mtx.npy',
        b=12,
        max_display=120,   # all 104 survivors + 16 removed samples
        hold=25,
        highlight=18,
        interval=70,       # ms between frames (~14 fps)
        save_gif=False,
    )
