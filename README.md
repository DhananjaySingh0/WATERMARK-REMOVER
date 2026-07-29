![image alt](https://github.com/DhananjaySingh0/WATERMARK-REMOVER/blob/86dc9daf3ad568cdd3402d0d0d0a9df4244f84aa/Screenshot.png)
https://watermark-remover-sgvt.onrender.com

# Watermark Remover

A Flask + OpenCV web app to remove watermarks from videos using 4 different methods.

## Features
- **Smart Fill (Inpaint)** — AI-based inpainting using OpenCV TELEA algorithm
- **Blur** — Gaussian blur over the watermark area
- **Pixelate** — Pixelation effect
- **Black Box** — Simple black fill

## Requirements
- Python 3.8+
- ffmpeg installed on system (`sudo apt install ffmpeg` or `brew install ffmpeg`)

## Setup

```bash
# Clone / download the project
cd watermark-remover

# Install Python dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open http://localhost:5000 in your browser.

## Usage
1. **Upload** your video (MP4, AVI, MOV, MKV, WEBM — up to 500MB)
2. **Draw a box** around the watermark on the video preview (or use an auto-detected region)
3. **Choose a method** (Smart Fill recommended for best results)
4. Click **Remove Watermark** and wait for processing
5. **Download** your clean video

## Project Structure
```
watermark-remover/
├── app.py                  # Flask server + routes
├── watermark_remover.py     # Video processing engine
├── watermark_detector.py    # Auto-detection engine
├── requirements.txt
├── templates/
│   └── index.html           # Main UI
├── static/
│   ├── css/style.css
│   └── js/app.js
├── uploads/                 # Temp uploaded videos
└── outputs/                 # Processed videos
```

## Notes
- Processing time depends on video length and resolution
- Smart Fill works best for small, semi-transparent watermarks
- For large opaque logos, Blur or Pixelate may look more natural
- Original audio is preserved in all methods (when ffmpeg is available)
- Temp files in `uploads/` and `outputs/` are cleaned up automatically after each job completes, and a background sweep also removes any leftover `/detect` temp file older than 1 hour (e.g. if a user never followed up with a real job)

## Limits & Safety
- **File validation**: uploads are checked both by extension and by their file signature (magic bytes), so a renamed non-video file is rejected instead of silently failing later
- **Rate limiting**: `/detect` and `/upload` are capped at 20 requests/hour per IP (200/day, 50/hour overall) via Flask-Limiter, to prevent abuse. Uses in-memory storage — fine for a single instance, but point `storage_uri` at Redis in `app.py` if you ever run multiple worker processes
- **Bounded job queue**: watermark removal jobs run on a fixed-size worker pool (up to 4, or fewer on machines with fewer CPU cores) instead of one thread per upload, so a burst of uploads queues up (`status: "queued"`) instead of overloading the CPU all at once#
