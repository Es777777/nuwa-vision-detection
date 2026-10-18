"""Download and annotate the requested reference-video intervals.

The script is intentionally separate from the benchmark: video frames are
reference material, not labelled detection samples. It records every frame's
timestamp and writes RGB, HSV-mask, Canny and candidate-box panels.
"""
from pathlib import Path
import subprocess, cv2, json

ROOT = Path(__file__).resolve().parent
URL = 'https://www.youtube.com/watch?v=mByWoyuU7bo'
VIDEO = ROOT / 'source_media' / 'video' / 'nuwa_reference.mp4'
OUT = ROOT / 'source_media' / 'video' / 'annotated'
INTERVALS = [(74,76),(122,124),(142,145),(191,192),(198,200),(210,212)]

def annotate(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5,35,25), (42,255,255))
    edge = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 45, 130)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    overlay = frame.copy()
    boxes=[]
    for x,y,w,h,a in stats[1:]:
        if 80 <= a <= 50000 and .25 <= w/max(h,1) <= 4:
            cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,0),2); boxes.append([int(x),int(y),int(w),int(h),int(a)])
    panel = cv2.hconcat([overlay, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)])
    return panel, boxes

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not VIDEO.exists():
        raise SystemExit('Video not present. Run: yt-dlp --no-playlist -o source_media/video/nuwa_reference.mp4 ' + URL)
    cap=cv2.VideoCapture(str(VIDEO)); fps=cap.get(cv2.CAP_PROP_FPS) or 30
    records=[]
    for a,b in INTERVALS:
        for sec in range(a,b):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000); ok, frame=cap.read()
            if not ok: continue
            panel,boxes=annotate(frame); fn=f'{sec:03d}s_rgb_mask_canny.png'; cv2.imwrite(str(OUT/fn),panel)
            records.append({'timestamp_s':sec,'interval_s':[a,b],'image':fn,'boxes':boxes})
    (OUT/'manifest.json').write_text(json.dumps({'url':URL,'intervals':INTERVALS,'frames':records},ensure_ascii=False,indent=2))
    print('wrote',len(records),'frames to',OUT)
if __name__=='__main__': main()
