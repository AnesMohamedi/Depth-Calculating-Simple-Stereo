def get_xyz_dimension(b, kp1, kp2, mtx, matches):
    x, y, z = [], [], []
    for match in matches:

        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt

        # RELAXED Y-FILTER: Was 50, now 500. 
        # If the Y-shift is > 500 pixels, it's 100% a false SIFT match, not a real physical point.
        if abs(y1 - y2) > 500: 
            print(f"[FILTERED] Y-Shift too large ({abs(y1-y2):.2f} px). False match: ({x1:.2f}, {y1:.2f}) <-> ({x2:.2f}, {y2:.2f})")
            continue

        disparity = x1 - x2
        if abs(disparity) <= 0:   # guard against division by ~zero
            print("[FILTERED] Disparity is 0 (Division by zero risk)")
            continue

        fx = mtx[0, 0]
        fy = mtx[1, 1]
        Ox = mtx[0, 2]
        Oy = mtx[1, 2]

        z_val = b * fx / disparity
        
        # RELAXED Z-FILTER: Was 100, now 1000. 
        # We now allow points up to 1000 units (cm) away!
        if z_val < 30 or z_val > 100:              
            print(f"[FILTERED] Z out of bounds (Z={z_val:.2f}). Point is either behind camera or too far.")
            continue

        # If it survives, print and append!
        print(f"[SUCCESS] Match: ({x1:.2f}, {y1:.2f}) <-> ({x2:.2f}, {y2:.2f}), Disparity: {disparity:.2f}, Z: {z_val:.2f}")
        x.append(b * (x1 - Ox) / disparity)
        y.append(b * (y1 - Oy) / disparity)  
        z.append(z_val)

    return x, y, z