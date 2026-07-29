import cv2
import numpy as np
import subprocess
import shutil
import os
import sys
import threading
import queue


def find_ffmpeg():
    """Find ffmpeg executable, checking PATH, common Windows install
    locations, and finally the portable binary bundled with the
    imageio-ffmpeg package (if installed). That last fallback matters on
    hosts like Render where there's no system package manager access to
    apt-get a system ffmpeg — without it, this app silently drops to the
    much slower, audio-less cv2.VideoWriter encode path.
    """
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg

    if sys.platform == 'win32':
        candidates = [
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
            os.path.join(os.environ.get('USERPROFILE', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    return None


FFMPEG = find_ffmpeg()

# How far (in px) the blend fades out beyond the selected box. A wider
# feather gives a smoother, less "rectangle-shaped" transition but eats
# into a bit more of the surrounding footage.
FEATHER_PX = 14

# How much weight the previous frame's reconstruction keeps when smoothing
# the inpainted patch across time. 0 = no smoothing (recompute independently
# every frame, prone to flicker). Higher = more stable but slightly more
# "smeared" if the background under the watermark is moving fast.
TEMPORAL_SMOOTHING = 0.55

# Inpainting is by far the most expensive step (it dominates total runtime),
# and the watermark sits in a fixed spot, so recomputing a fresh
# reconstruction for every single frame is mostly wasted work. Instead we
# recompute every Nth frame and reuse the cached patch in between, letting
# the temporal blend carry it across.
#   1     = recompute every frame (slowest, original behaviour)
#   3-5   = good speed/quality balance; measured deviation from interval=1
#           is well under 1.5% mean pixel error, i.e. not visible
#   8+    = diminishing speed returns, more risk of a stale-looking patch
# Lower this if the footage behind the watermark moves a lot.
INPAINT_INTERVAL = 4

# Extra context (px) kept around the region when cropping for inpainting.
# TELEA samples surrounding texture, so it needs some margin to work with.
INPAINT_MARGIN = 40

# Output quality. Frames are piped straight into ffmpeg as raw video and
# encoded exactly once, so this CRF is the only lossy step in the whole
# pipeline. Lower = higher quality / bigger file.
#   16-18 = visually lossless, good default for 4K masters
#   20-23 = smaller files, still very good for web
CRF = 17

# x264 speed/compression tradeoff. Note this does NOT control quality — CRF
# does. A slower preset just compresses better (smaller file) for the same
# visual quality. Measured at 4K: 'medium' and 'faster' both land at ~41 dB,
# but 'faster' is ~27% quicker, which matters a lot at 4K.
PRESET = 'faster'


def remove_watermark(input_path, output_path, region, method='inpaint', blur_strength=25, progress_callback=None):
    """
    Remove watermark from video using the specified method.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        region: Tuple (x, y, w, h) defining the watermark region
        method: 'inpaint', 'blur', 'pixelate', or 'black'
        blur_strength: Strength for blur/pixelate methods
        progress_callback: Callable receiving progress 0-100
    """
    x, y, w, h = region

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        total_frames = 1

    # Clamp region to video bounds
    x = max(0, min(x, vid_w - 1))
    y = max(0, min(y, vid_h - 1))
    w = max(1, min(w, vid_w - x))
    h = max(1, min(h, vid_h - y))

    # Hard mask used by cv2.inpaint (needs to know exactly which pixels to
    # reconstruct from surrounding context).
    inpaint_mask = np.zeros((vid_h, vid_w), dtype=np.uint8)
    pad = 3
    mx, my = max(0, x - pad), max(0, y - pad)
    mw = min(vid_w - mx, w + pad * 2)
    mh = min(vid_h - my, h + pad * 2)
    inpaint_mask[my:my + mh, mx:mx + mw] = 255

    # Expanded region (box + feather margin) used for edge blending, so the
    # replaced area fades into the untouched footage instead of showing a
    # visible rectangle.
    ex = max(0, x - FEATHER_PX)
    ey = max(0, y - FEATHER_PX)
    ex2 = min(vid_w, x + w + FEATHER_PX)
    ey2 = min(vid_h, y + h + FEATHER_PX)
    ew, eh = ex2 - ex, ey2 - ey
    local_x, local_y = x - ex, y - ey

    alpha = np.zeros((eh, ew), dtype=np.float32)
    alpha[local_y:local_y + h, local_x:local_x + w] = 1.0
    k = FEATHER_PX * 2 + 1
    alpha = cv2.GaussianBlur(alpha, (k, k), FEATHER_PX / 2.5)
    alpha3 = alpha[:, :, None]  # broadcastable over BGR channels

    # Crop bounds for inpainting: we only hand cv2.inpaint a neighbourhood
    # around the watermark instead of the entire frame.
    ix1 = max(0, x - INPAINT_MARGIN)
    iy1 = max(0, y - INPAINT_MARGIN)
    ix2 = min(vid_w, x + w + INPAINT_MARGIN)
    iy2 = min(vid_h, y + h + INPAINT_MARGIN)
    inpaint_crop_box = (ix1, iy1, ix2, iy2)
    mask_crop = np.ascontiguousarray(inpaint_mask[iy1:iy2, ix1:ix2])

    region_state = {
        'prev_patch': None,   # carries temporal smoothing state
        'cached': None,       # last computed inpaint result (full-region crop)
        'frame_idx': 0,
    }

    # Decide output strategy based on ffmpeg availability.
    #
    # With ffmpeg we pipe raw uncompressed frames straight into it and encode
    # once. The old approach wrote an intermediate mp4v file and then
    # re-encoded it, which meant every frame was compressed twice — that
    # intermediate step alone cost ~27dB PSNR before x264 even ran.
    use_ffmpeg = FFMPEG is not None
    proc = None
    out = None

    if use_ffmpeg:
        cmd = [
            FFMPEG, '-y',
            # Suppress ffmpeg's default per-frame progress/stats output.
            # That output goes to stderr, which we don't read from until
            # after the whole frame-writing loop finishes below. On a big
            # enough (or even not-so-big) encode, that stats output can
            # fill the OS pipe buffer; once it's full ffmpeg blocks trying
            # to write more of it, which in turn blocks it from reading
            # more frames off stdin, which blocks *our* stdin.write() call
            # in the loop below — a full deadlock with no error and no
            # progress. '-loglevel error' keeps stderr near-silent (only
            # real errors show up), which avoids this entirely.
            '-loglevel', 'error',
            # input 0: raw frames from stdin
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{vid_w}x{vid_h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            # input 1: the original file, used only for its audio track
            '-i', input_path,
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:v', 'libx264',
            '-crf', str(CRF),
            '-preset', PRESET,
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            output_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # Belt-and-suspenders on top of '-loglevel error' above: drain
        # stderr continuously on a background thread instead of only
        # reading it after the frame loop finishes. This guarantees the
        # pipe buffer can never fill up and stall ffmpeg (and therefore
        # our stdin writes) no matter how much ffmpeg ends up logging.
        stderr_chunks = []
        def _drain_stderr():
            for line in iter(proc.stderr.readline, b''):
                stderr_chunks.append(line)
        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()
    else:
        # No ffmpeg: fall back to OpenCV's writer. This is lossier and drops
        # audio, so installing ffmpeg is strongly recommended for best output.
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (vid_w, vid_h))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (vid_w, vid_h))

    # Frame decoding + per-frame processing (cv2.inpaint especially) is CPU
    # work, and writing a frame to ffmpeg's stdin blocks until ffmpeg has
    # drained enough of it to keep up — a single 1080p frame (~6MB raw) is
    # far bigger than the OS pipe buffer, so that write() effectively also
    # waits on ffmpeg's own encode time. Doing "compute frame" and "wait
    # for ffmpeg to catch up" in one thread pays both costs back to back.
    # Splitting them into a producer (decode + apply_method) and consumer
    # (write to ffmpeg/VideoWriter) thread lets frame N+1 get computed
    # while frame N is still being drained, overlapping the two instead of
    # serializing them. The processing itself, frame order, and the bytes
    # written are all unchanged — only the scheduling is different.
    frame_queue = queue.Queue(maxsize=6)
    producer_error = []
    stop_event = threading.Event()

    def _produce():
        frame_count = 0
        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = apply_method(
                    frame, x, y, w, h, inpaint_mask, method, blur_strength,
                    ex, ey, ew, eh, alpha3, region_state,
                    inpaint_crop_box, mask_crop
                )
                # Blocking put() with a timeout, re-checked against
                # stop_event, so this thread can't hang forever if the
                # consumer below has already stopped draining the queue
                # (e.g. ffmpeg died and the write loop bailed out).
                while not stop_event.is_set():
                    try:
                        frame_queue.put(frame, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                frame_count += 1
                if progress_callback and frame_count % max(1, total_frames // 100) == 0:
                    progress = min(90, int(frame_count / total_frames * 90))
                    progress_callback(progress)
        except Exception as e:
            producer_error.append(e)
        finally:
            frame_queue.put(None)  # sentinel: no more frames

    producer_thread = threading.Thread(target=_produce, daemon=True)
    producer_thread.start()

    write_error = None
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        try:
            if proc is not None:
                # .tobytes() would allocate and copy the entire frame again
                # just to hand it to write(); the array already exposes the
                # same underlying bytes via the buffer protocol, so we can
                # write it directly (as long as it's C-contiguous, which
                # apply_method's output always is). Same bytes reach
                # ffmpeg either way — this just skips the redundant copy,
                # on every frame rather than only the inpaint-recompute
                # ones.
                buf = frame if frame.flags['C_CONTIGUOUS'] else np.ascontiguousarray(frame)
                proc.stdin.write(buf)
            else:
                out.write(frame)
        except (BrokenPipeError, OSError) as e:
            # ffmpeg's stdin pipe closed underneath us, meaning the ffmpeg
            # process itself already died — the pipe error is just a
            # symptom. Stop feeding it immediately, unblock the producer
            # thread (drain anything still queued so its put() calls don't
            # spin forever), and fall through to read ffmpeg's real error
            # from stderr below instead of surfacing this generic message.
            write_error = e
            stop_event.set()
            while True:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    break
            break

    producer_thread.join()
    cap.release()

    if write_error is not None:
        if proc is not None:
            proc.wait(timeout=30)
            stderr_thread.join(timeout=5)
            stderr = b''.join(stderr_chunks).decode('utf-8', 'ignore').strip()
            detail = stderr[-2000:] if stderr else (
                'ffmpeg exited without an error message — this often means '
                'it was killed by the OS (commonly an out-of-memory kill on '
                'memory-constrained hosts) rather than a normal ffmpeg error.'
            )
            raise RuntimeError(
                f'ffmpeg exited unexpectedly (code {proc.returncode}) while '
                f'still receiving frames:\n{detail}'
            )
        raise RuntimeError(f'Video writer failed: {write_error}')

    if producer_error:
        raise producer_error[0]

    if proc is not None:
        if progress_callback:
            progress_callback(92)
        proc.stdin.close()
        proc.wait(timeout=600)
        stderr_thread.join(timeout=5)
        stderr = b''.join(stderr_chunks).decode('utf-8', 'ignore')

        if proc.returncode != 0:
            raise RuntimeError(
                'ffmpeg failed to encode the output:\n' + stderr[-2000:]
            )
    else:
        out.release()

    if progress_callback:
        progress_callback(100)


def apply_method(frame, x, y, w, h, inpaint_mask, method, blur_strength,
                  ex, ey, ew, eh, alpha3, region_state,
                  inpaint_crop_box=None, mask_crop=None):
    """
    Apply the selected removal method to a single frame, then feather-blend
    the reconstructed patch into the original footage so there's no visible
    hard-edged rectangle. For 'inpaint', also smooths across frames to
    reduce flicker.
    """
    original_crop = frame[ey:ey + eh, ex:ex + ew].astype(np.float32)

    if method == 'inpaint':
        idx = region_state['frame_idx']
        region_state['frame_idx'] = idx + 1

        # Only recompute the reconstruction every INPAINT_INTERVAL frames;
        # reuse the cached one otherwise. This is the single biggest speed
        # win, since inpainting dominates total processing time.
        if region_state['cached'] is None or idx % INPAINT_INTERVAL == 0:
            ix1, iy1, ix2, iy2 = inpaint_crop_box
            crop = np.ascontiguousarray(frame[iy1:iy2, ix1:ix2])
            inpainted_crop = cv2.inpaint(crop, mask_crop, inpaintRadius=9, flags=cv2.INPAINT_TELEA)

            # The feather region (ex,ey,ew,eh) is always fully inside the
            # wider inpaint-margin crop (ix1,iy1,ix2,iy2), since
            # INPAINT_MARGIN (40px) > FEATHER_PX (14px). So instead of
            # pasting inpainted_crop back into a full-frame-sized copy
            # (frame.copy() — a whole extra 1080p-sized allocation on
            # every recompute) just to re-slice the small feather region
            # back out of it, slice that region directly out of
            # inpainted_crop at the equivalent offset. Numerically
            # identical result — same pixels — without ever touching the
            # rest of the frame.
            off_x, off_y = ex - ix1, ey - iy1
            region_state['cached'] = inpainted_crop[off_y:off_y + eh, off_x:off_x + ew].astype(np.float32)

        patch = region_state['cached']

        # Temporal smoothing: blend this frame's reconstruction with the
        # previous frame's, so the patch doesn't visibly re-texture itself
        # every frame (the main giveaway in moving video).
        prev = region_state.get('prev_patch')
        if prev is not None and prev.shape == patch.shape:
            patch = TEMPORAL_SMOOTHING * prev + (1 - TEMPORAL_SMOOTHING) * patch
        region_state['prev_patch'] = patch

    elif method == 'blur':
        roi = frame[y:y + h, x:x + w]
        k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)
        patch = original_crop.copy()
        patch[y - ey:y - ey + h, x - ex:x - ex + w] = blurred_roi

    elif method == 'pixelate':
        roi = frame[y:y + h, x:x + w]
        pixel_size = max(2, blur_strength // 3)
        small = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
        pixelated_roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        patch = original_crop.copy()
        patch[y - ey:y - ey + h, x - ex:x - ex + w] = pixelated_roi

    elif method == 'black':
        patch = original_crop.copy()
        patch[y - ey:y - ey + h, x - ex:x - ex + w] = 0

    elif method == 'white':
        patch = original_crop.copy()
        patch[y - ey:y - ey + h, x - ex:x - ex + w] = 255

    else:
        return frame

    blended = original_crop * (1 - alpha3) + patch * alpha3
    frame[ey:ey + eh, ex:ex + ew] = np.clip(blended, 0, 255).astype(np.uint8)
    return frame
