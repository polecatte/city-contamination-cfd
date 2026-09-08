#!/usr/bin/env python3
# city3d.py — build an interactive 3D model of the density city from the exported
# city_density.txt (+ meta.txt for the buffer frame). Outputs:
#   model3d.html  — self-contained Three.js scene (orbit/pan/zoom), buildings
#                   coloured by land use, ground + buffer frame + wind arrow.
#   city_model.obj / .mtl — portable mesh (boxes grouped by usage) for any 3D tool.
# Usage: python3 city3d.py [OUT_DIR] [html_out] [obj_out]
import sys, os, json

OUT  = sys.argv[1] if len(sys.argv) > 1 else "forward_city_out"
HTML = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT, "model3d.html")
OBJ  = sys.argv[3] if len(sys.argv) > 3 else os.path.join(OUT, "city_model.obj")

def parse_meta(p):
    d = {}
    if os.path.exists(p):
        for ln in open(p):
            s = ln.strip()
            if s and not s.startswith("#") and len(s.split()) >= 2:
                k, *v = s.split(); d[k] = v
    return d

CELL = 2.0; Sx = Sy = 0.0; blds = []
for ln in open(os.path.join(OUT, "city_density.txt")):
    t = ln.split()
    if not t: continue
    if t[0] == "CELL": CELL = float(t[1])
    elif t[0] == "SIZE": Sx, Sy = float(t[1]), float(t[2])
    elif t[0] == "B":
        # B x0 y0 x1 y1 bx0 by0 bx1 by1 height_cells usage footprint eff_inh
        # x0..x1 = BUILDING footprint (setback applied — matches the voxelized solver
        # geometry); bx0..bx1 = the full lot. Draw the building, not the lot.
        v = list(map(float, t[1:]))
        x0, y0, x1, y1 = v[0], v[1], v[2], v[3]
        hc, usage = v[8], int(v[9])
        x, y = x0*CELL, y0*CELL
        w, d = (x1-x0)*CELL, (y1-y0)*CELL
        h = hc*CELL
        if w > 0 and d > 0 and h > 0:
            blds.append({"x": x, "y": y, "w": w, "d": d, "h": h, "u": usage})

M = parse_meta(os.path.join(OUT, "meta.txt"))
def mv(k, dflt):
    try: return float(M[k][0])
    except Exception: return dflt
bup, bdn, blat = mv("buf_upwind_m", 40), mv("buf_downwind_m", 70), mv("buf_lateral_m", 35)
if Sx == 0: Sx = max((b["x"]+b["w"] for b in blds), default=600)+bdn
if Sy == 0: Sy = max((b["y"]+b["d"] for b in blds), default=600)+blat
wind_deg = mv("wind_deg", 0); U = mv("U_inlet_ms", 4); maxH = mv("maxH_m", max((b["h"] for b in blds), default=60))
city = {"x0": bup, "y0": blat, "x1": Sx-bdn, "y1": Sy-blat}

# ── OBJ (+MTL): one box per building, grouped by usage, plus a ground quad ──────
USAGE = {0: ("park", (0.18, 0.55, 0.34)), 1: ("business", (0.12, 0.23, 0.58)),
         2: ("res_high", (0.90, 0.49, 0.13)), 3: ("res_low", (0.95, 0.76, 0.31))}
def box(f, ox, oy, oz, w, d, h, base):
    # 8 verts (x=east, y=up, z=north): map (x,y_plan,h)->(x, h, y_plan)
    vs = [(ox,oz,oy),(ox+w,oz,oy),(ox+w,oz,oy+d),(ox,oz,oy+d),
          (ox,oz+h,oy),(ox+w,oz+h,oy),(ox+w,oz+h,oy+d),(ox,oz+h,oy+d)]
    for x,yy,z in vs: f.write(f"v {x:.3f} {yy:.3f} {z:.3f}\n")
    q = [(1,2,3,4),(5,8,7,6),(1,5,6,2),(2,6,7,3),(3,7,8,4),(4,8,5,1)]
    for a,b,c,d2 in q: f.write(f"f {base+a} {base+b} {base+c} {base+d2}\n")
    return base+8
mtl = os.path.splitext(OBJ)[0] + ".mtl"
with open(mtl, "w") as f:
    for _,(nm,(r,g,b)) in USAGE.items(): f.write(f"newmtl {nm}\nKd {r:.3f} {g:.3f} {b:.3f}\n")
    f.write("newmtl ground\nKd 0.85 0.87 0.89\n")
with open(OBJ, "w") as f:
    f.write(f"mtllib {os.path.basename(mtl)}\n")
    base = 0
    f.write("g ground\nusemtl ground\n")
    base = box(f, 0, 0, -0.2, Sx, Sy, 0.2, base)
    for uid,(nm,_) in USAGE.items():
        grp = [b for b in blds if b["u"] == uid]
        if not grp: continue
        f.write(f"g {nm}\nusemtl {nm}\n")
        for b in grp: base = box(f, b["x"], b["y"], 0, b["w"], b["d"], b["h"], base)
print("wrote", OBJ, "and", mtl)

# ── self-contained interactive HTML (three.js from cdnjs + tiny custom orbit) ──
payload = {"blds": blds, "Sx": Sx, "Sy": Sy, "city": city,
           "wind_deg": wind_deg, "U": U, "maxH": maxH,
           "colors": {str(k): list(v[1]) for k, v in USAGE.items()}}
HTML_TMPL = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Density city — 3D model</title>
<style>
  html,body{margin:0;height:100%;overflow:hidden;background:#dfe6ee;font-family:system-ui,sans-serif}
  #c{display:block;width:100vw;height:100vh}
  #legend{position:absolute;top:12px;left:12px;background:rgba(255,255,255,.9);
    padding:10px 12px;border-radius:8px;font-size:13px;box-shadow:0 1px 6px rgba(0,0,0,.2)}
  #legend b{display:block;margin-bottom:6px}
  .sw{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:middle}
  #hint{position:absolute;bottom:10px;left:12px;color:#456;font-size:12px}
</style></head><body>
<canvas id="c"></canvas>
<div id="legend"><b>Density city — land use</b>
  <div><span class="sw" style="background:#2e8b57"></span>park</div>
  <div><span class="sw" style="background:#1f3a93"></span>business</div>
  <div><span class="sw" style="background:#e67e22"></span>res-high</div>
  <div><span class="sw" style="background:#f2c14e"></span>res-low</div>
  <div style="margin-top:6px;color:#c0392b">&#8594; wind __WINDDEG__&deg;, U=__U__ m/s</div>
</div>
<div id="hint">drag = orbit · right-drag = pan · wheel = zoom</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const D = __PAYLOAD__;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdfe6ee);
scene.fog = new THREE.Fog(0xdfe6ee, 900, 2600);
const cx = D.Sx/2, cz = D.Sy/2;
const cam = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 1, 8000);
const R = new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});
R.setrpr = R.setPixelRatio(devicePixelRatio); R.setSize(innerWidth,innerHeight);
R.shadowMap.enabled = true;
// lights
scene.add(new THREE.HemisphereLight(0xffffff,0x8899aa,0.75));
const sun = new THREE.DirectionalLight(0xffffff,0.85);
sun.position.set(cx-400, D.maxH*4+300, cz-500); sun.castShadow=true;
sun.shadow.mapSize.set(2048,2048);
const sc=sun.shadow.camera, S=Math.max(D.Sx,D.Sy)*0.7;
sc.left=-S;sc.right=S;sc.top=S;sc.bottom=-S;sc.near=1;sc.far=6000; scene.add(sun);
// ground (buffer) + city plate
const gmat=new THREE.MeshLambertMaterial({color:0xd7dce2});
const g=new THREE.Mesh(new THREE.PlaneGeometry(D.Sx*1.5,D.Sy*1.5),gmat);
g.rotation.x=-Math.PI/2; g.position.set(cx,0,cz); g.receiveShadow=true; scene.add(g);
const cw=D.city.x1-D.city.x0, cd=D.city.y1-D.city.y0;
const plate=new THREE.Mesh(new THREE.PlaneGeometry(cw,cd),
  new THREE.MeshLambertMaterial({color:0xeef1f4}));
plate.rotation.x=-Math.PI/2; plate.position.set((D.city.x0+D.city.x1)/2,0.05,(D.city.y0+D.city.y1)/2);
plate.receiveShadow=true; scene.add(plate);
// city boundary outline
const eg=new THREE.EdgesGeometry(new THREE.BoxGeometry(cw,0.5,cd));
const line=new THREE.LineSegments(eg,new THREE.LineBasicMaterial({color:0x33404d}));
line.position.copy(plate.position); scene.add(line);
// buildings
const box=new THREE.BoxGeometry(1,1,1);
for(const b of D.blds){
  const col=D.colors[String(b.u)]||[0.6,0.6,0.6];
  const m=new THREE.Mesh(box,new THREE.MeshLambertMaterial({color:new THREE.Color(col[0],col[1],col[2])}));
  m.scale.set(b.w,b.h,b.d);
  m.position.set(b.x+b.w/2, b.h/2, b.y+b.d/2);
  m.castShadow=true; m.receiveShadow=true; scene.add(m);
  const e=new THREE.LineSegments(new THREE.EdgesGeometry(box),
     new THREE.LineBasicMaterial({color:0x000000,transparent:true,opacity:0.12}));
  e.scale.copy(m.scale); e.position.copy(m.position); scene.add(e);
}
// wind arrow (red), along +x for 0 deg
const th=D.wind_deg*Math.PI/180, dir=new THREE.Vector3(Math.cos(th),0,Math.sin(th));
const arr=new THREE.ArrowHelper(dir,new THREE.Vector3(D.city.x0-60, D.maxH*0.6, cz),
   130, 0xc0392b, 45, 28); scene.add(arr);
// ── tiny orbit controller (no external dep) ──
let tgt=new THREE.Vector3(cx, D.maxH*0.4, cz);
let rad=Math.max(D.Sx,D.Sy)*1.25, yaw=-0.7, pit=0.62;
function place(){ cam.position.set(
  tgt.x+rad*Math.cos(pit)*Math.cos(yaw), tgt.y+rad*Math.sin(pit), tgt.z+rad*Math.cos(pit)*Math.sin(yaw));
  cam.lookAt(tgt);} place();
let drag=null,px=0,py=0;
addEventListener('mousedown',e=>{drag=e.button;px=e.clientX;py=e.clientY;});
addEventListener('mouseup',()=>drag=null);
addEventListener('contextmenu',e=>e.preventDefault());
addEventListener('mousemove',e=>{ if(drag===null)return;
  const dx=e.clientX-px, dy=e.clientY-py; px=e.clientX; py=e.clientY;
  if(drag===0){ yaw+=dx*0.006; pit=Math.max(0.08,Math.min(1.5,pit+dy*0.006)); }
  else { const s=rad*0.0016; const rt=new THREE.Vector3(-Math.sin(yaw),0,Math.cos(yaw));
    tgt.addScaledVector(rt,-dx*s); tgt.x+=0; tgt.z+=0; tgt.y=Math.max(0,tgt.y+dy*s); }
  place(); });
addEventListener('wheel',e=>{ rad=Math.max(60,Math.min(4000,rad*(1+Math.sign(e.deltaY)*0.08))); place(); },{passive:true});
addEventListener('resize',()=>{ cam.aspect=innerWidth/innerHeight; cam.updateProjectionMatrix(); R.setSize(innerWidth,innerHeight); });
(function loop(){ requestAnimationFrame(loop); R.render(scene,cam); })();
</script></body></html>"""
html = (HTML_TMPL
        .replace("__PAYLOAD__", json.dumps(payload))
        .replace("__WINDDEG__", f"{wind_deg:.0f}")
        .replace("__U__", f"{U:g}"))
open(HTML, "w").write(html)
print("wrote", HTML, "-", len(blds), "buildings")
