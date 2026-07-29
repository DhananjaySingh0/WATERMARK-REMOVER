import cv2
import numpy as np


def detect_watermark(video_path, sample_frames=5):
    """
    Auto-detect watermark region in a video by analyzing multiple frames.
    Returns list of candidate regions sorted by confidence: [{'x','y','w','h','confidence','label'}]
    Strategy:
      1. Sample several frames spread across the video
      2. Find regions that are CONSISTENT across frames (static overlays = watermarks)
      3. Also detect bright logo-like patches in corners
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frames = _sample_frames(video_path, total, sample_frames)
    if len(frames) < 2:
        return []

    candidates = []

    # ── Method 1: Temporal consistency (static regions across frames) ──────────
    consistency_mask = _temporal_consistency_mask(frames)
    tc_regions = _mask_to_regions(consistency_mask, vid_w, vid_h, min_area=400)
    for r in tc_regions:
        r['label'] = 'Static overlay (likely watermark)'
        r['confidence'] = min(99, int(r['confidence'] * 1.2))
    candidates.extend(tc_regions)

    # ── Method 2: Corner / edge detection (logos usually live in corners) ──────
    corner_regions = _detect_corner_logos(frames[0], vid_w, vid_h)
    candidates.extend(corner_regions)

    # ── Method 3: High-contrast text/logo blobs ────────────────────────────────
    blob_regions = _detect_logo_blobs(frames[0], vid_w, vid_h)
    candidates.extend(blob_regions)

    # Deduplicate overlapping boxes
    candidates = _deduplicate(candidates, iou_threshold=0.3)

    # Sort by confidence descending
    candidates.sort(key=lambda r: r['confidence'], reverse=True)

    # Add small padding to each region
    padded = []
    for r in candidates[:5]:
        pad = 6
        r['x'] = max(0, r['x'] - pad)
        r['y'] = max(0, r['y'] - pad)
        r['w'] = min(vid_w - r['x'], r['w'] + pad * 2)
        r['h'] = min(vid_h - r['y'], r['h'] + pad * 2)
        padded.append(r)

    return padded


def _sample_frames(video_path, total_frames, n):
    cap = cv2.VideoCapture(video_path)
    frames = []
    indices = [int(total_frames * i / (n + 1)) for i in range(1, n + 1)]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def _temporal_consistency_mask(frames):
    """Pixels that barely change across frames = static overlay."""
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    stacked = np.stack(gray, axis=0)
    std_map = np.std(stacked, axis=0)  # low std = static
    # Normalize
    std_norm = cv2.normalize(std_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Invert: static = bright
    static_map = 255 - std_norm
    _, mask = cv2.threshold(static_map, 200, 255, cv2.THRESH_BINARY)
    # Remove very large uniform areas (sky, bg)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _detect_corner_logos(frame, vid_w, vid_h):
    """Check all four corners for non-background content."""
    regions = []
    margin_x = int(vid_w * 0.30)
    margin_y = int(vid_h * 0.30)

    corners = [
        ('top-left',     0,                  0,                  margin_x, margin_y),
        ('top-right',    vid_w - margin_x,   0,                  margin_x, margin_y),
        ('bottom-left',  0,                  vid_h - margin_y,   margin_x, margin_y),
        ('bottom-right', vid_w - margin_x,   vid_h - margin_y,   margin_x, margin_y),
    ]

    for name, cx, cy, cw, ch in corners:
        crop = frame[cy:cy+ch, cx:cx+cw]
        result = _find_logo_in_crop(crop, cx, cy, vid_w, vid_h)
        if result:
            result['label'] = f'Corner logo ({name})'
            regions.append(result)

    return regions


def _find_logo_in_crop(crop, offset_x, offset_y, vid_w, vid_h):
    """Find the tightest bounding box of a logo within a crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    # Dilate to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dilated = cv2.dilate(edges, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find contours with a reasonable size for a watermark
    min_area = (vid_w * vid_h) * 0.001
    max_area = (vid_w * vid_h) * 0.15

    best = None
    best_score = 0
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)
        # Watermarks tend to be wide rectangles
        score = area * (1.5 if 1.5 < aspect < 8 else 1.0)
        if score > best_score:
            best_score = score
            best = (bx, by, bw, bh)

    if best is None:
        return None

    bx, by, bw, bh = best
    return {
        'x': offset_x + bx,
        'y': offset_y + by,
        'w': bw,
        'h': bh,
        'confidence': min(85, int(best_score / (vid_w * vid_h) * 5000))
    }


def _detect_logo_blobs(frame, vid_w, vid_h):
    """Detect semi-transparent or bright logo overlays using alpha-like detection."""
    regions = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Detect near-white / near-gray (common watermark colors)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (vid_w * vid_h) * 0.002
    max_area = (vid_w * vid_h) * 0.12

    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        # Prefer regions near edges/corners
        center_x, center_y = bx + bw // 2, by + bh // 2
        edge_bonus = 1.0
        if center_x < vid_w * 0.25 or center_x > vid_w * 0.75:
            edge_bonus *= 1.5
        if center_y < vid_h * 0.25 or center_y > vid_h * 0.75:
            edge_bonus *= 1.5
        confidence = min(80, int(area / (vid_w * vid_h) * 3000 * edge_bonus))
        if confidence > 20:
            regions.append({
                'x': bx, 'y': by, 'w': bw, 'h': bh,
                'confidence': confidence,
                'label': 'Light/white overlay'
            })

    return regions


def _mask_to_regions(mask, vid_w, vid_h, min_area=400):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        # Skip if it covers most of the frame (background)
        if bw * bh > vid_w * vid_h * 0.4:
            continue
        confidence = min(95, int(area / (vid_w * vid_h) * 4000))
        regions.append({'x': bx, 'y': by, 'w': bw, 'h': bh, 'confidence': confidence})
    return regions


def _iou(a, b):
    ax1, ay1 = a['x'], a['y']
    ax2, ay2 = ax1 + a['w'], ay1 + a['h']
    bx1, by1 = b['x'], b['y']
    bx2, by2 = bx1 + b['w'], by1 + b['h']
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a['w'] * a['h'] + b['w'] * b['h'] - inter
    return inter / max(union, 1)


def _deduplicate(regions, iou_threshold=0.3):
    kept = []
    regions = sorted(regions, key=lambda r: r['confidence'], reverse=True)
    for r in regions:
        if all(_iou(r, k) < iou_threshold for k in kept):
            kept.append(r)
    return kept
