"""Image-only candidate generation and RGB-D geometry checks.

The detector never uses a ground-truth pixel. RGB proposes candidates; depth
and the known field dimensions decide whether a candidate is physically valid.
"""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Candidate:
    u: float
    v: float
    radius_px: float = 0.0
    confidence: float = 0.0
    source: str = 'none'

@dataclass
class ColumnGeometry:
    center_x_m: float
    center_y_m: float
    rim_z_m: float
    radius_m: float = 0.135
    valid: bool = False

@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    # Affine image warp applied after the calibrated camera projection.
    # This is identity for normal streams and supports aligned robustness
    # frames with roll/shear without corrupting metric back-projection.
    image_from_calib: object = None


def _image_to_calib(u, v, cam):
    pixels=np.stack(np.broadcast_arrays(np.asarray(u,dtype=np.float64),
                                        np.asarray(v,dtype=np.float64)),axis=-1)
    if cam.image_from_calib is None: return pixels[...,0],pixels[...,1]
    inv=np.linalg.inv(np.asarray(cam.image_from_calib,dtype=np.float64))
    flat=np.concatenate((pixels.reshape(-1,2),np.ones((pixels.size//2,1))),axis=1)
    mapped=(inv@flat.T).T
    mapped=mapped[:,:2]/mapped[:,2:3]
    mapped=mapped.reshape(pixels.shape)
    return mapped[...,0],mapped[...,1]


def _calib_to_image(u, v, cam):
    pixels=np.stack(np.broadcast_arrays(np.asarray(u,dtype=np.float64),
                                        np.asarray(v,dtype=np.float64)),axis=-1)
    if cam.image_from_calib is None: return pixels[...,0],pixels[...,1]
    mat=np.asarray(cam.image_from_calib,dtype=np.float64)
    flat=np.concatenate((pixels.reshape(-1,2),np.ones((pixels.size//2,1))),axis=1)
    mapped=(mat@flat.T).T
    mapped=mapped[:,:2]/mapped[:,2:3]
    mapped=mapped.reshape(pixels.shape)
    return mapped[...,0],mapped[...,1]

BALL_RADIUS_M = 0.100       # rule document: diameter 200 mm
COLUMN_RADIUS_M = 0.135     # rule document: outside diameter 270 mm
GROOVE_RADIUS_M = 0.090     # rule document: groove diameter 180 mm
GROOVE_DEPTH_M = 0.100      # rule document: groove depth 100 mm

def segment_depth_roi(depth_m, z_min=0.15, z_max=8.0, table_z=None,
                      table_tol=0.12):
    """Create a valid scene ROI from aligned depth.

    When a table plane estimate is supplied, pixels near the plane and above
    it are retained. With no plane estimate this still removes invalid and
    out-of-range depth, which is the safe fallback for a live camera.
    """
    z = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(z) & (z >= z_min) & (z <= z_max)
    if table_z is not None:
        valid &= z <= float(table_z) + float(table_tol)
    return (valid.astype(np.uint8) * 255)

def fit_vertical_cylinder(points, radius_m=COLUMN_RADIUS_M, radius_tol=0.035):
    """Fit a vertical cylinder proxy and return its XY centre and rim height."""
    p = np.asarray(points, dtype=np.float32)
    if p.ndim != 2 or p.shape[0] < 30:
        return None
    xy = p[:, :2]
    # Algebraic circle initialization is much less biased than taking the
    # median of a single visible cylinder arc.
    A=np.column_stack((2*xy[:,0],2*xy[:,1],np.ones(len(xy))))
    b=np.sum(xy*xy,axis=1);sol,*_=np.linalg.lstsq(A,b,rcond=None)
    center=sol[:2]
    for _ in range(6):
        rr = np.linalg.norm(xy - center, axis=1)
        keep = np.abs(rr - radius_m) <= radius_tol
        if keep.sum() < 20: break
        # Refit circle from the current inliers.
        aa=np.column_stack((2*xy[keep,0],2*xy[keep,1],np.ones(keep.sum())))
        ss,*_=np.linalg.lstsq(aa,np.sum(xy[keep]*xy[keep],axis=1),rcond=None)
        center=ss[:2]
    rr = np.linalg.norm(xy - center, axis=1)
    keep = np.abs(rr - radius_m) <= radius_tol
    if keep.sum() < 20: return None
    residual = float(np.sqrt(np.mean((rr[keep] - radius_m) ** 2)))
    rim_z = float(np.percentile(p[keep, 2], 90))
    return ColumnGeometry(float(center[0]), float(center[1]), rim_z,
                          radius_m=float(np.median(rr[keep])),
                          valid=residual <= 0.025)

def depth_mask_points(depth_m, cam, mask):
    """Back-project all valid depth pixels selected by a binary mask."""
    z=np.asarray(depth_m,dtype=np.float32); m=np.asarray(mask)>0
    if m.shape!=z.shape:return np.empty((0,3),np.float32)
    vv,uu=np.nonzero(m & np.isfinite(z) & (z>0.15) & (z<8.0))
    if len(uu)==0:return np.empty((0,3),np.float32)
    zz=z[vv,uu]; uc,vc=_image_to_calib(uu,vv,cam)
    return np.column_stack(((uc-cam.cx)*zz/cam.fx,(vc-cam.cy)*zz/cam.fy,zz)).astype(np.float32)

def depth_points(depth_m, cam, center, radius_px=60):
    """Back-project valid depth pixels around a colour candidate to metres."""
    u0, v0 = center; h, w = depth_m.shape[:2]
    x0=max(0,int(u0-radius_px)); x1=min(w,int(u0+radius_px+1)); y0=max(0,int(v0-radius_px)); y1=min(h,int(v0+radius_px+1))
    z=depth_m[y0:y1,x0:x1]; yy,xx=np.indices(z.shape); uu=xx+x0; vv=yy+y0
    valid=np.isfinite(z)&(z>0.15)&(z<8.0)&(((uu-u0)**2+(vv-v0)**2)<=radius_px**2)
    z=z[valid]; uu=uu[valid]; vv=vv[valid]
    if z.size==0:return np.empty((0,3),np.float32)
    uc,vc=_image_to_calib(uu,vv,cam)
    return np.column_stack(((uc-cam.cx)*z/cam.fx,(vc-cam.cy)*z/cam.fy,z)).astype(np.float32)

def fit_sphere_model(points, radius_m=BALL_RADIUS_M):
    """Fit a sphere and return centre, fixed-radius RMSE, and free radius."""
    if len(points)<20:return None, float('inf'), float('nan')
    p=np.asarray(points,float)
    # The candidate window contains the ball and background. Keep the front
    # depth cluster before fitting; otherwise the least-squares sphere is
    # pulled metres behind the ball and its residual becomes meaningless.
    z=p[:,2]; front=np.percentile(z, 5.0)
    p=p[z <= front + 0.08]
    if len(p)<20: return None, float('inf'), float('nan')
    q=np.sum(p*p,axis=1)
    # 2 p dot c + d = |p|^2, with d = r^2 - |c|^2.
    sol, *_=np.linalg.lstsq(np.column_stack((2*p, np.ones(len(p)))), q, rcond=None)
    c=sol[:3]; err=np.abs(np.linalg.norm(p-c,axis=1)-radius_m)
    fitted_radius=float(np.sqrt(max(0.0,np.dot(c,c)+sol[3])))
    return c.astype(float), float(np.sqrt(np.mean(err*err))), fitted_radius

def fit_sphere(points, radius_m=BALL_RADIUS_M):
    """Fit sphere centre with fixed rule-book radius; return centre and RMSE."""
    center,rmse,_=fit_sphere_model(points,radius_m)
    return center,rmse

def center_ray_ball(depth_m, cam, center, radius_m=BALL_RADIUS_M, patch_radius=2):
    """Estimate ball centre along its projected centre ray from aligned depth."""
    u,v=center; h,w=depth_m.shape[:2]; ui=int(round(u));vi=int(round(v))
    x0=max(0,ui-patch_radius);x1=min(w,ui+patch_radius+1);y0=max(0,vi-patch_radius);y1=min(h,vi+patch_radius+1)
    z=np.asarray(depth_m[y0:y1,x0:x1],dtype=np.float32);z=z[np.isfinite(z)&(z>0.15)&(z<8.0)]
    if z.size<3:return None
    zc=float(np.median(z))+radius_m
    uc,vc=_image_to_calib(float(u),float(v),cam)
    return np.array([(float(uc)-cam.cx)*zc/cam.fx,(float(vc)-cam.cy)*zc/cam.fy,zc],dtype=float)

def stone_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # The range is deliberately broad; the rule-book colour is checked by
    # mean chroma/value after this proposal stage, not by one exact RGB value.
    # Golden stone has substantially higher saturation than the tan concrete
    # and beige field panels. This lower saturation bound removes those panels.
    m = cv2.inRange(hsv, np.array([8, 65, 25], np.uint8),
                    np.array([32, 255, 255], np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

def column_mask(bgr):
    """Narrow brown PVC mask; deliberately excludes the golden ball range."""
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    m=cv2.inRange(hsv,np.array([10,130,15],np.uint8),np.array([24,255,90],np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    # A vertical opening removes the long horizontal brown field stripe that
    # previously merged with the column and shifted its image centre.
    return cv2.morphologyEx(m,cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT,(5,21)))

def propose_ball_candidates(bgr, roi=None, limit=32):
    """Rank circle, compact-colour and partial-ellipse RGB proposals."""
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h) if roi is None else roi
    crop = bgr[y0:y1, x0:x1]
    gray = cv2.medianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 5)
    mask = stone_mask(crop)
    edge = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 45, 130)
    candidates = []
    # A single Hough threshold often sees the partly occluded ball while the
    # other thresholds see only the column rim. Do not discard that proposal
    # here: depth geometry below decides whether it is a real sphere.
    for p2 in (12, 18, 24):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.0, 14,
                                   param1=65, param2=p2, minRadius=7,
                                   maxRadius=max(12, min(w, h)//3))
        if circles is None: continue
        for u, v, r in circles[0]:
            uu, vv = float(u+x0), float(v+y0)
            cx,cy=round(u),round(v); rr=max(2,round(r)); pad=3
            ax0=max(0,cx-rr-pad);ax1=min(crop.shape[1],cx+rr+pad+1)
            ay0=max(0,cy-rr-pad);ay1=min(crop.shape[0],cy+rr+pad+1)
            yy,xx=np.ogrid[ay0:ay1,ax0:ax1]
            local_disk=(xx-cx)**2+(yy-cy)**2<=r*r
            colour=float(np.mean(mask[ay0:ay1,ax0:ax1][local_disk]>0)) if np.any(local_disk) else 0.0
            ring=np.zeros((ay1-ay0,ax1-ax0),np.uint8)
            cv2.circle(ring,(cx-ax0,cy-ay0),rr,1,2)
            support=float(np.mean(edge[ay0:ay1,ax0:ax1][ring>0]>0)) if np.any(ring) else 0.0
            score = 1.4*support + 0.8*colour - 0.001*abs(r-35)
            candidates.append(Candidate(uu, vv, float(r), score, f'hough_p2_{p2}'))
    # Cluster duplicate detections. Single-threshold circles remain proposals;
    # the fixed-radius sphere fit, projected size, colour, and column collision
    # checks are stronger evidence than an arbitrary Hough vote count.
    groups=[]
    for c in candidates:
        g=next((x for x in groups if np.hypot(x.u-c.u,x.v-c.v)<=12),None)
        if g is None: groups.append(Candidate(c.u,c.v,c.radius_px,c.confidence,c.source))
        else:
            g.confidence=max(g.confidence,c.confidence); g.radius_px=(g.radius_px+c.radius_px)/2
            g.source += ','+c.source
    groups=[g for g in groups if g.confidence>=0.04]
    proposals=sorted(groups,key=lambda c:c.confidence,reverse=True)
    # Retain several candidates. A strong circle on the column body must not
    # hide a weaker true ball circle; aligned depth and field geometry choose.
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    choices=[]
    for i in range(1,n):
        area=stats[i,cv2.CC_STAT_AREA]; ww=stats[i,cv2.CC_STAT_WIDTH]; hh=stats[i,cv2.CC_STAT_HEIGHT]
        if area<120 or max(ww,hh)>min(w,h)*.45: continue
        compact=area/(max(1,ww*hh)); aspect=min(ww,hh)/max(ww,hh)
        if compact<0.18 or aspect<0.18: continue
        choices.append((area*compact*aspect,i))
    for score,i in sorted(choices,reverse=True)[:6]:
        x,y,ww,hh=stats[i,:4]; u,v=cents[i]
        proposals.append(Candidate(float(u+x0),float(v+y0),float(max(ww,hh)/2),
                                   min(.45,.20+score/10000),'colour_component_fallback'))
    # At low camera elevations only a narrow arc of the ball may be visible.
    # Recover compact gold contours that are too small for the full-component
    # fallback, while leaving depth geometry as the acceptance gate.
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    partial=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if area < 45 or len(contour) < 5: continue
        (u,v),(major,minor),_=cv2.fitEllipse(contour)
        if minor < 5 or major > min(w,h)*0.35: continue
        aspect=float(minor)/float(major)
        if aspect < 0.28: continue
        perimeter=cv2.arcLength(contour,True)
        circularity=4*np.pi*area/max(perimeter*perimeter,1.0)
        if circularity < 0.22: continue
        partial.append((area*circularity*aspect,Candidate(float(u+x0),float(v+y0),
                        float((major+minor)/4),0.25,'partial_colour_ellipse')))
    proposals.extend(c for _,c in sorted(partial,key=lambda x:x[0],reverse=True)[:6])
    unique=[]
    for c in proposals:
        existing=next((e for e in unique if np.hypot(e.u-c.u,e.v-c.v)<10 and
                       abs(e.radius_px-c.radius_px)<max(12,.5*max(e.radius_px,c.radius_px))),None)
        if existing is None: unique.append(c)
    # Compression can create dozens of strong spurious edges. Keep proposals
    # from each source, and prefer circles repeated across Hough thresholds,
    # so a large number of single-threshold circles cannot evict the ball.
    colour = [c for c in unique if not c.source.startswith('hough_p2_')]
    circles = [c for c in unique if c.source.startswith('hough_p2_')]
    circles.sort(key=lambda c: (len(set(c.source.split(','))), c.confidence),
                 reverse=True)
    reserved = colour[:min(8, limit)]
    selected = circles[:max(0, limit-len(reserved))] + reserved
    return selected


def propose_ball(bgr, roi=None):
    """Return the highest-ranked RGB-only proposal."""
    candidates=propose_ball_candidates(bgr,roi)
    return candidates[0] if candidates else None

def propose_ball_rgbd(bgr, depth_m, cam, roi=None):
    """Select among RGB candidates by fixed-radius depth-sphere consistency."""
    h,w=bgr.shape[:2]; x0,y0,x1,y1=(0,0,w,h) if roi is None else roi
    crop=bgr[y0:y1,x0:x1]; gray=cv2.medianBlur(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),5)
    circles=[]
    for p2 in (18,22,26,30):
        found=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1.0,18,param1=80,param2=p2,
                               minRadius=8,maxRadius=max(12,min(w,h)//3))
        if found is not None:
            circles.extend(Candidate(float(x+x0),float(y+y0),float(r),0.4,f'hough_{p2}') for x,y,r in found[0])
    # Cluster duplicate detections while retaining the number of agreeing thresholds.
    clusters=[]
    for c in circles:
        g=next((g for g in clusters if np.hypot(g.u-c.u,g.v-c.v)<=12),None)
        if g is None: clusters.append(Candidate(c.u,c.v,c.radius_px,1.0,c.source))
        else:
            n=g.source.count('hough_'); g.u=(g.u*n+c.u)/(n+1);g.v=(g.v*n+c.v)/(n+1)
            g.radius_px=(g.radius_px*n+c.radius_px)/(n+1);g.confidence+=1;g.source+=','+c.source
    # Include compact gold components as a recovery source for partial arcs.
    mask=stone_mask(crop); n,lab,stats,cents=cv2.connectedComponentsWithStats(mask,8)
    for i in range(1,n):
        area=stats[i,cv2.CC_STAT_AREA]; bw=stats[i,cv2.CC_STAT_WIDTH];bh=stats[i,cv2.CC_STAT_HEIGHT]
        if area<100 or max(bw,bh)>min(w,h)*.30 or area/max(1,bw*bh)<.22: continue
        u,v=cents[i]; clusters.append(Candidate(float(u+x0),float(v+y0),float(max(bw,bh)/2),.5,'colour_component'))
    valid=[]; color=stone_mask(bgr)
    for c in clusters:
        if c.radius_px<6 or c.radius_px>min(w,h)*.3: continue
        if c.source.startswith('hough_') and len(set(c.source.split(',')))<2: continue
        pts=depth_points(depth_m,cam,(c.u,c.v),max(20,int(c.radius_px*1.8)))
        center,rmse=fit_sphere(pts)
        if center is None or not np.isfinite(rmse) or rmse>.08: continue
        z=float(center[2]); pu=cam.fx*float(center[0])/z+cam.cx;pv=cam.fy*float(center[1])/z+cam.cy
        reproj=float(np.hypot(pu-c.u,pv-c.v))
        expected=cam.fx*BALL_RADIUS_M/z
        radius_error=abs(c.radius_px-expected)/max(expected,1.0)
        if reproj>max(14,c.radius_px*.8) or radius_error>0.45: continue
        yy,xx=np.ogrid[:h,:w];disk=(xx-c.u)**2+(yy-c.v)**2<=c.radius_px**2
        coverage=float(np.mean(color[disk]>0)) if np.any(disk) else 0.0
        if coverage < 0.45: continue
        fallback_penalty=0.020 if c.source=='colour_component' else 0.0
        score=rmse + fallback_penalty + 0.015*radius_error + 0.0005*reproj - 0.002*coverage - 0.0005*min(c.confidence,4)
        valid.append((score,c,center,rmse))
    if valid:
        _,c,center,rmse=min(valid,key=lambda x:x[0]);return c,center,float(rmse)
    return None,None,float('inf')

def propose_column(bgr, roi=None):
    """Return a colour/edge column mask and its image centre proxy."""
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h) if roi is None else roi
    crop = bgr[y0:y1, x0:x1]
    mask = column_mask(crop)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 300: continue
        x, y, ww, hh = stats[i, :4]
        if best is None or area > best[0]:
            best = (area, x+x0, y+y0, ww, hh, cents[i][0]+x0, cents[i][1]+y0)
    out = np.zeros((h, w), np.uint8)
    if best is None: return out, None
    area, x, y, ww, hh, u, v = best
    out[y:y+hh, x:x+ww] = mask[y-y0:y-y0+hh, x-x0:x-x0+ww]
    return out, Candidate(float(u), float(v), 0.5*ww,
                          min(1.0, area/5000.0), 'column_color')

def classify_ball_column(ball_xyz, column: ColumnGeometry, radius_m=BALL_RADIUS_M):
    """Classify a detected ball relative to a detected column rim.

    A ball is on the column only when horizontal axis error, bottom/rim
    contact, and both detections are valid. Missing depth yields UNKNOWN.
    """
    if ball_xyz is None or column is None or not column.valid:
        return 'UNKNOWN', 0.0
    dx = float(ball_xyz[0]) - column.center_x_m
    dy = float(ball_xyz[1]) - column.center_y_m
    axis = float(np.hypot(dx, dy))
    bottom_gap = float(ball_xyz[2]) - radius_m - column.rim_z_m
    # 35 mm axis tolerance and 45 mm contact tolerance account for D435i noise.
    # A Ø200 mm ball seated on a Ø180 mm rim has its lowest point 56.4 mm
    # below the rim plane. Compare against that rule-book contact offset.
    expected_gap = -(radius_m - np.sqrt(max(0.0, radius_m*radius_m-GROOVE_RADIUS_M*GROOVE_RADIUS_M)))
    gap_error = abs(bottom_gap-expected_gap)
    ok = axis <= 0.035 and gap_error <= 0.045 and column.radius_m >= GROOVE_RADIUS_M
    quality = max(0.0, 1.0-axis/0.035) * max(0.0, 1.0-gap_error/0.045)
    return ('ON_COLUMN' if ok else 'OFF_COLUMN'), quality


def _transform_points(points, camera_to_field):
    p=np.asarray(points,dtype=np.float64)
    if p.size == 0: return p.reshape((-1,3))
    return (camera_to_field[:3,:3] @ p.T).T + camera_to_field[:3,3]


def _projected_field_roi(shape, cam, field_to_camera, box):
    """Project a 3-D field box to a conservative image ROI."""
    h,w=shape; corners=[]
    for x in (box[0],box[1]):
        for y in (box[2],box[3]):
            for z in (box[4],box[5]):
                p=field_to_camera @ np.array([x,y,z,1.0])
                if p[2] > 0.05:
                    u=cam.fx*p[0]/p[2]+cam.cx;v=cam.fy*p[1]/p[2]+cam.cy
                    ui,vi=_calib_to_image(u,v,cam)
                    corners.append((float(ui),float(vi)))
    mask=np.zeros((h,w),np.uint8)
    if not corners: return mask
    pts=np.asarray(corners); margin=12
    x0=max(0,int(np.floor(pts[:,0].min()))-margin); x1=min(w,int(np.ceil(pts[:,0].max()))+margin+1)
    y0=max(0,int(np.floor(pts[:,1].min()))-margin); y1=min(h,int(np.ceil(pts[:,1].max()))+margin+1)
    if x1>x0 and y1>y0: mask[y0:y1,x0:x1]=255
    return mask


def _fit_field_column(depth_m, cam, camera_to_field, column_mask_image):
    """Fit the central vertical column from depth within its projected field ROI.

    RGB segmentation limits the proposal when available; the field-space
    cylinder dimensions are the hard gate. The column centre is measured from
    the depth points and is not replaced with the nominal anchor.
    """
    if camera_to_field is None: return None
    h,w=depth_m.shape[:2]
    field_to_camera=np.linalg.inv(camera_to_field)
    anchor=np.array([0.0,0.0,0.0,1.0])
    if (field_to_camera @ anchor)[2] <= 0: return None
    roi=_projected_field_roi((h,w),cam,field_to_camera,
                             (-0.24,0.24,-0.24,0.24,0.78,1.84))
    if not np.any(roi): return None
    # The image mask is a proposal; its depth points must form a tall cylinder
    # with the rule-book radius. This rejects flat floor markings and the
    # floating sphere left in the synthetic "no column" frames.
    support=cv2.bitwise_and(column_mask_image,roi)
    pts=depth_mask_points(depth_m,cam,support)
    fp=_transform_points(pts,camera_to_field)
    if len(fp) < 30: return None
    radial=np.linalg.norm(fp[:,:2],axis=1)
    keep=(fp[:,2]>=0.78)&(fp[:,2]<=1.84)&(radial>=0.085)&(radial<=0.195)
    p=fp[keep]
    if len(p)<30: return None
    z_span=float(np.percentile(p[:,2],95)-np.percentile(p[:,2],5))
    if z_span < 0.22: return None
    fit=fit_vertical_cylinder(p)
    if fit is None or not fit.valid: return None
    centre_error=float(np.hypot(fit.center_x_m,fit.center_y_m))
    if centre_error>0.055 or not (0.105<=fit.radius_m<=0.165): return None
    # The rule-book rim is the reference plane for contact. The fitted points
    # determine the XY axis; the known field CAD height defines the rim plane.
    fit.rim_z_m=1.70
    return fit


def analyze_rgbd(bgr, depth_m, cam, camera_to_field=None, roi=None):
    """Shared detector for offline media and the ROS 2 node.

    RGB generates several ball proposals. Aligned depth ranks them by fixed
    sphere geometry and projected size, then the fitted central column rejects
    candidates that occupy solid column volume.
    """
    h,w=bgr.shape[:2]
    roi=(0,0,w,h) if roi is None else tuple(roi)
    depth=np.asarray(depth_m,dtype=np.float32) if depth_m is not None else None
    if depth is not None and depth.shape[:2] != (h,w):
        raise ValueError('aligned depth and RGB must have identical dimensions')
    depth_roi=segment_depth_roi(depth) if depth is not None else np.zeros((h,w),np.uint8)
    candidates=propose_ball_candidates(bgr,roi)
    ball=candidates[0] if candidates else None
    column_mask_image,col_candidate=propose_column(bgr,roi)
    ball_mask=np.zeros((h,w),np.uint8)
    column=_fit_field_column(depth,cam,camera_to_field,column_mask_image) if depth is not None and cam is not None else None
    # Noise fragments the brown mask; vertical opening can then erase the
    # entire column. Retry a denoised proposal without changing metric gates.
    if column is None and depth is not None and cam is not None and camera_to_field is not None:
        denoised_mask, denoised_candidate=propose_column(cv2.medianBlur(bgr,9),roi)
        denoised_column=_fit_field_column(depth,cam,camera_to_field,denoised_mask)
        if denoised_column is not None:
            column=denoised_column
            column_mask_image=denoised_mask
            col_candidate=denoised_candidate
    result={'ball_candidate':ball is not None,'column_candidate':col_candidate is not None,
            'state':'UNKNOWN','ball_center_3d':None,'column_xy_center_3d':None,
            'coordinate_frame':'field' if camera_to_field is not None else 'camera_color_optical_frame',
            'reason':'no_ball_candidate','sphere_rmse_m':None,'sphere_fit_center_camera_m':None,
            'ball_color_coverage':0.0,'depth_support':0.0,'column_geometry_valid':column is not None}
    if column is not None:
        result['column_xy_center_3d']=[float(column.center_x_m),float(column.center_y_m),float(column.rim_z_m)]
    color=stone_mask(bgr)
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    # The field's orange transfer areas are close to the stone in hue. The
    # high-confidence colour vote starts at H=18 (OpenCV units); the broader
    # mask above remains available for proposal generation and dark pixels.
    saturated_stone=((hsv[:,:,0]>=18)&(hsv[:,:,0]<=35)&
                     (hsv[:,:,1]>=110)&(hsv[:,:,2]>=20))
    if ball is not None:
        proposal_disk=np.zeros((h,w),np.uint8)
        cv2.circle(proposal_disk,(round(ball.u),round(ball.v)),
                   max(3,round(ball.radius_px)),255,-1)
        ball_mask=cv2.bitwise_and(color,proposal_disk)
        proposal_area=proposal_disk>0
        result['ball_color_coverage']=float(np.mean(ball_mask[proposal_area]>0))
    result['ball_mask']=ball_mask
    result['column_mask']=column_mask_image
    result['depth_roi_mask']=depth_roi
    if depth is None or cam is None:
        result['reason']='depth_or_camera_model_unavailable'
        return ball,col_candidate,ball_mask,column_mask_image,depth_roi,result
    if not candidates:
        result['reason']='no_ball_candidate'
        return ball,col_candidate,ball_mask,column_mask_image,depth_roi,result
    accepted=[]
    rejection_counts={}
    for candidate in candidates:
        if candidate.radius_px < 6 or candidate.radius_px > min(h,w)*0.30:
            rejection_counts['projected_radius_out_of_range']=rejection_counts.get('projected_radius_out_of_range',0)+1
            continue
        centre_camera=center_ray_ball(depth,cam,(candidate.u,candidate.v))
        if centre_camera is None:
            rejection_counts['insufficient_depth_support']=rejection_counts.get('insufficient_depth_support',0)+1
            continue
        expected_radius=cam.fx*BALL_RADIUS_M/max(float(centre_camera[2]),1e-6)
        # Hough often fits the merged ball-and-column outline. Use its centre
        # as a proposal, but measure colour, depth, and sphere fit at the
        # rule-book projected ball radius.
        disk=np.zeros((h,w),np.uint8)
        cv2.circle(disk,(round(candidate.u),round(candidate.v)),
                   max(3,round(expected_radius)),255,-1)
        inside=disk>0
        coverage=float(np.mean(color[inside]>0)) if np.any(inside) else 0.0
        saturated_coverage=float(np.mean(saturated_stone[inside])) if np.any(inside) else 0.0
        depth_support=float(np.mean(depth_roi[inside]>0)) if np.any(inside) else 0.0
        if coverage < 0.18 or saturated_coverage < 0.08:
            rejection_counts['insufficient_ball_colour_support']=rejection_counts.get('insufficient_ball_colour_support',0)+1
            continue
        if depth_support < 0.20:
            rejection_counts['insufficient_depth_support']=rejection_counts.get('insufficient_depth_support',0)+1
            continue
        points=depth_points(depth,cam,(candidate.u,candidate.v),
                            max(20,int(expected_radius*1.25)))
        sphere_fit,rmse,fitted_radius=fit_sphere_model(points)
        if sphere_fit is None or centre_camera is None or not np.isfinite(rmse) or rmse>0.08:
            rejection_counts['sphere_fit_rejected']=rejection_counts.get('sphere_fit_rejected',0)+1
            continue
        radius_error=abs(candidate.radius_px-expected_radius)/max(expected_radius,1.0)
        fit_z=float(sphere_fit[2])
        fit_u=cam.fx*float(sphere_fit[0])/fit_z+cam.cx
        fit_v=cam.fy*float(sphere_fit[1])/fit_z+cam.cy
        fit_u,fit_v=_calib_to_image(fit_u,fit_v,cam)
        reprojection=float(np.hypot(float(fit_u)-candidate.u,float(fit_v)-candidate.v))
        if radius_error>1.50 or reprojection>max(80,candidate.radius_px*2.0):
            rejection_counts['projected_sphere_mismatch']=rejection_counts.get('projected_sphere_mismatch',0)+1
            continue
        # When the free-radius fit closely matches the 100 mm rule sphere and
        # has a low residual, it corrects depth-pixel quantization in the
        # centre ray. Otherwise keep the robust centre-ray estimate dominant.
        fit_weight=(0.70 if rmse<=0.008 and
                    abs(fitted_radius-BALL_RADIUS_M)<=0.020 else 0.10)
        refined_camera=(1.0-fit_weight)*centre_camera+fit_weight*sphere_fit
        centre_output=refined_camera
        if camera_to_field is not None:
            centre_output=(camera_to_field[:3,:3]@refined_camera)+camera_to_field[:3,3]
            # The sphere centre must be physically above the field surface.
            if not (BALL_RADIUS_M-0.04 <= float(centre_output[2]) <= 2.2):
                rejection_counts['sphere_height_out_of_range']=rejection_counts.get('sphere_height_out_of_range',0)+1
                continue
            if column is not None:
                axis=float(np.hypot(centre_output[0]-column.center_x_m,
                                    centre_output[1]-column.center_y_m))
                vertical_overlap=(centre_output[2]+BALL_RADIUS_M>0.78 and
                                  centre_output[2]-BALL_RADIUS_M<column.rim_z_m)
                collides=axis < (COLUMN_RADIUS_M+BALL_RADIUS_M-0.01) and vertical_overlap
                expected_z=column.rim_z_m+np.sqrt(max(0.0,BALL_RADIUS_M**2-GROOVE_RADIUS_M**2))
                seated=(axis<=0.055 and abs(float(centre_output[2])-expected_z)<=0.065)
                if collides and not seated:
                    rejection_counts['ball_candidate_intersects_column']=rejection_counts.get('ball_candidate_intersects_column',0)+1
                    continue
        # A planar field patch can fit a fixed-radius sphere over a small
        # window. Check the unconstrained fitted radius as well. Partial views
        # of a seated ball get a slightly wider lower bound because the rim
        # hides its lower hemisphere.
        near_rim=(column is not None and
                  np.hypot(centre_output[0]-column.center_x_m,
                           centre_output[1]-column.center_y_m)<=0.12 and
                  abs(float(centre_output[2])-(column.rim_z_m+
                      np.sqrt(max(0.0,BALL_RADIUS_M**2-GROOVE_RADIUS_M**2))))<=0.14)
        min_fitted_radius=0.075 if near_rim else 0.085
        if not np.isfinite(fitted_radius) or not (min_fitted_radius<=fitted_radius<=0.125):
            rejection_counts['fitted_sphere_radius_mismatch']=rejection_counts.get('fitted_sphere_radius_mismatch',0)+1
            continue
        # Geometry dominates selection. Colour coverage and Hough agreement
        # break ties; an oversized circle on the column body loses on radius.
        votes=candidate.source.count('hough_p2_')
        # Partial spheres can bias the least-squares centre toward the visible
        # cap, so reprojection is only a tie-breaker. The centre-ray depth and
        # projected physical radius carry the localization decision.
        score=(rmse + 0.02*radius_error + 0.0003*reprojection
               - 0.002*coverage - 0.003*saturated_coverage
               - 0.0008*min(votes,4))
        accepted.append((score,candidate,centre_camera,centre_output,sphere_fit,
                         rmse,coverage,depth_support,radius_error,saturated_coverage,
                         expected_radius,fitted_radius,fit_weight))
    if not accepted:
        # Keep the strongest RGB proposal visible in diagnostic images while
        # exposing why every proposal failed the metric gates.
        if rejection_counts:
            result['reason']=max(rejection_counts,key=rejection_counts.get)
            result['candidate_rejections']=rejection_counts
        return ball,col_candidate,ball_mask,column_mask_image,depth_roi,result
    _,ball,centre_camera,centre_output,sphere_fit,rmse,coverage,depth_support,radius_error,saturated_coverage,expected_radius,fitted_radius,fit_weight=min(accepted,key=lambda x:x[0])
    ball.radius_px=float(expected_radius)
    cv2.circle(ball_mask,(round(ball.u),round(ball.v)),max(3,round(ball.radius_px)),255,-1)
    ball_mask=cv2.bitwise_and(color,ball_mask)
    result['ball_color_coverage']=float(coverage)
    result['ball_saturated_color_coverage']=float(saturated_coverage)
    result['depth_support']=float(depth_support)
    result['sphere_rmse_m']=float(rmse)
    result['sphere_fitted_radius_m']=float(fitted_radius)
    result['sphere_fit_blend_weight']=float(fit_weight)
    result['sphere_fit_center_camera_m']=[float(x) for x in sphere_fit]
    result['ball_candidate_source']=ball.source
    result['ball_radius_error_fraction']=float(radius_error)
    if camera_to_field is not None:
        result['ball_center_3d']=[float(x) for x in centre_output]
        result['state']=classify_ball_column(centre_output,column)[0]
        result['reason']='rgbd_geometry_ok' if result['state']!='UNKNOWN' else 'column_geometry_unavailable'
    else:
        result['ball_center_3d']=[float(x) for x in centre_output]
        result['state']='UNKNOWN'
        result['reason']='camera_frame_only_no_field_tf'
    return ball,col_candidate,ball_mask,column_mask_image,depth_roi,result
