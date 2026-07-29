// ─── State ────────────────────────────────────────────────────────────────────
let videoFile = null;
let selection = { x: 0, y: 0, w: 0, h: 0 };
let isDrawing = false, startX = 0, startY = 0;
let scaleX = 1, scaleY = 1;
let currentJobId = null, pollInterval = null;
let tmpId = '', tmpExt = '';
let detectedRegions = [];
let selectedDetectedIdx = -1;

// ─── Elements ─────────────────────────────────────────────────────────────────
const dropZone     = document.getElementById('dropZone');
const fileInput    = document.getElementById('fileInput');
const canvas       = document.getElementById('previewCanvas');
const ctx          = canvas.getContext('2d');
const selBox       = document.getElementById('selectionBox');
const previewVideo  = document.getElementById('previewVideo');
const playToggleBtn = document.getElementById('playToggleBtn');
const canvasWrapper  = document.querySelector('.canvas-wrapper');
let isPreviewPlaying = false;
const processBtn   = document.getElementById('processBtn');
const progressFill = document.getElementById('progressFill');
const progressLabel= document.getElementById('progressLabel');
const procStatus   = document.getElementById('procStatus');

// ─── Step Nav ─────────────────────────────────────────────────────────────────
function showStep(id) {
  document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

// ─── Drop / Upload ────────────────────────────────────────────────────────────
['dragenter','dragover'].forEach(e => dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('dragover'); }));
['dragleave','drop'].forEach(e => dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('dragover'); }));
dropZone.addEventListener('drop', ev => { const f = ev.dataTransfer.files[0]; if (f) handleFile(f); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });
dropZone.addEventListener('click', e => { if (e.target.tagName !== 'BUTTON') fileInput.click(); });

const MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024; // must match app.config['MAX_CONTENT_LENGTH']

function handleFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['mp4','avi','mov','mkv','webm','flv'].includes(ext)) { showError('Invalid file type.'); return; }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    showError(`File is too large (${(file.size / (1024*1024)).toFixed(0)}MB). Max size is 500MB.`);
    return;
  }
  videoFile = file;
  showStep('step-analyzing');
  autoDetect(file);
}

// ─── Auto Detect ──────────────────────────────────────────────────────────────
async function autoDetect(file) {
  const steps = ['scan1','scan2','scan3','scan4'];
  let stepIdx = 0;

  const ticker = setInterval(() => {
    if (stepIdx > 0) document.getElementById(steps[stepIdx-1]).classList.replace('active','done');
    if (stepIdx < steps.length) { document.getElementById(steps[stepIdx]).classList.add('active'); stepIdx++; }
  }, 700);

  const form = new FormData();
  form.append('video', file);

  try {
    const res = await fetch('/detect', { method: 'POST', body: form });
    const data = await res.json();
    clearInterval(ticker);
    steps.forEach(s => { const el = document.getElementById(s); el.classList.remove('active'); el.classList.add('done'); });

    if (data.error) { showError(data.error); showStep('step-upload'); return; }

    tmpId  = data.tmp_id  || '';
    tmpExt = data.ext     || '';
    detectedRegions = data.regions || [];

    // Load video preview
    loadVideoPreview(file, () => {
      renderDetectedRegions();
      showStep('step-region');
      // Auto-select the top result
      if (detectedRegions.length > 0) selectDetected(0);
    });

  } catch (err) {
    clearInterval(ticker);
    showError('Detection failed: ' + err.message);
    showStep('step-upload');
  }
}

function loadVideoPreview(file, callback) {
  const url = URL.createObjectURL(file);
  const vid = document.createElement('video');
  vid.src = url; vid.muted = true; vid.preload = 'metadata';
  vid.addEventListener('loadedmetadata', () => {
    canvas.width  = vid.videoWidth;
    canvas.height = vid.videoHeight;
    canvasWrapper.style.aspectRatio = `${vid.videoWidth} / ${vid.videoHeight}`;

    // Wire up the real, playable preview video with the same source.
    previewVideo.src = url;
    previewVideo.currentTime = 0;

    vid.currentTime = 0.001;
  });
  vid.addEventListener('seeked', () => {
    ctx.drawImage(vid, 0, 0);
    computeScale();
    if (callback) callback();
  });
  vid.load();
}

// ─── Render Detected Regions ──────────────────────────────────────────────────
function renderDetectedRegions() {
  const list = document.getElementById('detectedList');
  const count = document.getElementById('detectedCount');
  count.textContent = detectedRegions.length;
  list.innerHTML = '';

  if (detectedRegions.length === 0) {
    list.innerHTML = `<div class="no-detections"><strong>No watermarks detected</strong>Draw manually on the video</div>`;
    return;
  }

  detectedRegions.forEach((r, i) => {
    const item = document.createElement('div');
    item.className = 'detected-item';
    item.dataset.idx = i;
    item.innerHTML = `
      <div class="det-conf">${r.confidence}%</div>
      <div class="det-info">
        <div class="det-label">${r.label || 'Watermark region'}</div>
        <div class="det-coords">${r.x},${r.y} &nbsp;${r.w}×${r.h}</div>
      </div>
      <button class="det-select-btn" onclick="selectDetected(${i})">Use</button>`;
    list.appendChild(item);
  });
}

function selectDetected(idx) {
  const r = detectedRegions[idx];
  if (!r) return;
  selectedDetectedIdx = idx;
  selection = { x: r.x, y: r.y, w: r.w, h: r.h };

  // Highlight item
  document.querySelectorAll('.detected-item').forEach(el => el.classList.remove('selected'));
  const item = document.querySelector(`.detected-item[data-idx="${idx}"]`);
  if (item) item.classList.add('selected');

  // Draw on canvas
  redrawCanvas();
  updateCoords();
  processBtn.disabled = false;
}

function redrawCanvas() {
  // Re-draw base frame (first frame already rendered, just overlay selection)
  updateSelBox();
}

// ─── Canvas Manual Selection ──────────────────────────────────────────────────
canvas.addEventListener('mousedown', e => {
  computeScale();
  const rect = canvas.getBoundingClientRect();
  startX = (e.clientX - rect.left) * scaleX;
  startY = (e.clientY - rect.top)  * scaleY;
  isDrawing = true;
  selBox.classList.remove('hidden');
  selection = { x: startX, y: startY, w: 0, h: 0 };
  selectedDetectedIdx = -1;
  document.querySelectorAll('.detected-item').forEach(el => el.classList.remove('selected'));
});

canvas.addEventListener('mousemove', e => {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * scaleX;
  const cy = (e.clientY - rect.top)  * scaleY;
  selection.x = Math.min(startX, cx);
  selection.y = Math.min(startY, cy);
  selection.w = Math.abs(cx - startX);
  selection.h = Math.abs(cy - startY);
  updateSelBox(); updateCoords();
});

canvas.addEventListener('mouseup', () => {
  isDrawing = false;
  if (selection.w > 5 && selection.h > 5) processBtn.disabled = false;
});

canvas.addEventListener('touchstart', e => { e.preventDefault(); canvas.dispatchEvent(new MouseEvent('mousedown', { clientX: e.touches[0].clientX, clientY: e.touches[0].clientY })); });
canvas.addEventListener('touchmove',  e => { e.preventDefault(); canvas.dispatchEvent(new MouseEvent('mousemove', { clientX: e.touches[0].clientX, clientY: e.touches[0].clientY })); });
canvas.addEventListener('touchend',   e => { e.preventDefault(); canvas.dispatchEvent(new MouseEvent('mouseup')); });

function computeScale() {
  const rect = canvas.getBoundingClientRect();
  scaleX = canvas.width  / rect.width;
  scaleY = canvas.height / rect.height;
}
window.addEventListener('resize', computeScale);

// ─── Video Preview Play/Pause Toggle ─────────────────────────────────────────
function showFrameMode() {
  previewVideo.pause();
  // Capture whatever frame the video was paused/ended on, so the selection
  // canvas reflects what the user last saw instead of jumping back to frame 0.
  if (previewVideo.readyState >= 2) {
    ctx.drawImage(previewVideo, 0, 0, canvas.width, canvas.height);
  }
  previewVideo.classList.remove('active');
  canvas.classList.remove('frame-hidden');
  if (selection.w > 0 && selection.h > 0) selBox.classList.remove('hidden');
  playToggleBtn.textContent = '▶';
  playToggleBtn.setAttribute('aria-label', 'Play preview');
  isPreviewPlaying = false;
}

function showVideoMode() {
  canvas.classList.add('frame-hidden');
  selBox.classList.add('hidden');
  previewVideo.classList.add('active');
  previewVideo.play();
  playToggleBtn.textContent = '⏸';
  playToggleBtn.setAttribute('aria-label', 'Pause preview');
  isPreviewPlaying = true;
}

playToggleBtn.addEventListener('click', () => {
  if (isPreviewPlaying) showFrameMode();
  else showVideoMode();
});

// If the video plays to the end, drop back into selection/frame mode
// automatically instead of leaving a frozen last-frame video on screen.
previewVideo.addEventListener('ended', showFrameMode);

function updateSelBox() {
  if (selection.w === 0 && selection.h === 0) { selBox.classList.add('hidden'); return; }
  selBox.classList.remove('hidden');
  const rect = canvas.getBoundingClientRect();
  const dx = rect.width  / canvas.width;
  const dy = rect.height / canvas.height;
  selBox.style.left   = (selection.x * dx) + 'px';
  selBox.style.top    = (selection.y * dy) + 'px';
  selBox.style.width  = (selection.w * dx) + 'px';
  selBox.style.height = (selection.h * dy) + 'px';
}

function updateCoords() {
  document.getElementById('rX').textContent = Math.round(selection.x);
  document.getElementById('rY').textContent = Math.round(selection.y);
  document.getElementById('rW').textContent = Math.round(selection.w);
  document.getElementById('rH').textContent = Math.round(selection.h);
}

// ─── Method Cards ─────────────────────────────────────────────────────────────
document.querySelectorAll('.method-card').forEach(card => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.method-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    card.querySelector('input').checked = true;
    const m = card.dataset.value;
    document.getElementById('blurControl').classList.toggle('hidden', m === 'inpaint' || m === 'black');
  });
});

document.getElementById('blurStrength').addEventListener('input', function() {
  document.getElementById('blurVal').textContent = this.value;
});

// ─── Process ──────────────────────────────────────────────────────────────────
processBtn.addEventListener('click', async () => {
  if (selection.w <= 0 || selection.h <= 0) return;
  const method = document.querySelector('input[name=method]:checked').value;
  const blurStrength = document.getElementById('blurStrength').value;

  const form = new FormData();
  // Always include the video file itself (the browser already has it in
  // memory). We still pass tmp_id/ext as a hint so the server can reuse
  // its /detect temp file when available and skip re-saving — but we no
  // longer depend on that file still existing. On hosts with ephemeral
  // disk/restarts (e.g. Render), that temp file can disappear between
  // /detect and clicking "Remove Watermark", and relying on it alone
  // caused a hard "No video file provided" failure with no way to recover
  // client-side.
  form.append('video', videoFile);
  form.append('x', Math.round(selection.x));
  form.append('y', Math.round(selection.y));
  form.append('w', Math.round(selection.w));
  form.append('h', Math.round(selection.h));
  form.append('method', method);
  form.append('blur_strength', blurStrength);
  if (tmpId)  form.append('tmp_id', tmpId);
  if (tmpExt) form.append('ext', tmpExt);

  showStep('step-processing');
  setProgress(0);
  procStatus.textContent = 'Starting...';

  try {
    const res = await fetch('/upload', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) { showError(data.error || 'Failed'); showStep('step-region'); return; }
    currentJobId = data.job_id;
    startPolling();
  } catch (err) {
    showError('Network error: ' + err.message);
    showStep('step-region');
  }
});

// ─── Polling ──────────────────────────────────────────────────────────────────
function startPolling() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    try {
      const res  = await fetch(`/status/${currentJobId}`);
      const data = await res.json();
      if (data.status === 'queued') {
        procStatus.textContent = 'Waiting for a free worker...';
      } else if (data.status === 'processing') {
        setProgress(data.progress || 0);
        if      (data.progress < 30) procStatus.textContent = 'Analyzing frames...';
        else if (data.progress < 65) procStatus.textContent = 'Removing watermark...';
        else if (data.progress < 90) procStatus.textContent = 'Encoding video...';
        else                          procStatus.textContent = 'Finalizing...';
      } else if (data.status === 'done') {
        clearInterval(pollInterval);
        setProgress(100);
        setTimeout(() => {
          document.getElementById('downloadBtn').href = `/download/${currentJobId}`;
          const resultVideo = document.getElementById('resultVideo');
          resultVideo.src = `/preview/${currentJobId}`;
          resultVideo.load();
          showStep('step-done');
        }, 600);
      } else if (data.status === 'error') {
        clearInterval(pollInterval);
        showError(data.error || 'Processing failed');
        showStep('step-region');
      }
    } catch (_) {}
  }, 800);
}

function setProgress(v) {
  progressFill.style.width  = v + '%';
  progressLabel.textContent = v + '%';
}

// ─── Reupload / Start Over ────────────────────────────────────────────────────
document.getElementById('reuploadBtn').addEventListener('click', reset);
document.getElementById('startOverBtn').addEventListener('click', reset);

function reset() {
  videoFile = null; selection = {x:0,y:0,w:0,h:0}; tmpId=''; tmpExt='';
  detectedRegions=[]; selectedDetectedIdx=-1;
  processBtn.disabled = true; fileInput.value = '';
  if (pollInterval) clearInterval(pollInterval);
  // Reset video preview back to frame/selection mode for the next upload
  previewVideo.pause();
  previewVideo.removeAttribute('src');
  previewVideo.load();
  previewVideo.classList.remove('active');
  // Also clear the finished-result preview from the Complete step so a
  // previous job's clip isn't briefly visible on the next run.
  const resultVideo = document.getElementById('resultVideo');
  resultVideo.pause();
  resultVideo.removeAttribute('src');
  resultVideo.load();
  canvas.classList.remove('frame-hidden');
  playToggleBtn.textContent = '▶';
  playToggleBtn.setAttribute('aria-label', 'Play preview');
  isPreviewPlaying = false;
  // Reset scan step classes
  ['scan1','scan2','scan3','scan4'].forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('active','done');
  });
  showStep('step-upload');
}

// ─── Error ────────────────────────────────────────────────────────────────────
function showError(msg) {
  document.getElementById('errorMsg').textContent = msg;
  document.getElementById('errorToast').classList.remove('hidden');
  setTimeout(hideError, 6000);
}
function hideError() { document.getElementById('errorToast').classList.add('hidden'); }
window.hideError = hideError;
window.selectDetected = selectDetected;
