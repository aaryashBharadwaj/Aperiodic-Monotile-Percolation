"""Interactive percolation portal for the aperiodic monotile family and its validations.

    streamlit run interface/gui_app.py

Three tabs behind one tiling picker:
  * Visualise  — generate the chosen tiling at a size you control and (optionally) overlay the exact
                 percolation graph, i.e. proof the generator produces the real shape + lattice.
  * Percolate  — launch a run (any patch up to production), sweep L, finite-size-extrapolate the
                 site & bond thresholds, report p_c, check for directional bias, and optionally
                 measure the fractal dimension d_f. Runs auto-save.
  * Analyse saved — reload any run from paper_results/npz/ (saved here or by runner/runner.py)
                 and show the identical analysis.

All numerics come from the same builders/kernels the runner uses (via gui_backend); the Percolate
tab launches runner/runner.py in the background. Converged production numbers come from the same
runner with a larger patch / more trials -- the 'Use paper settings' button, or the console
(see REPRODUCE.md).
"""
import json
import os
import sys
import time
# gui_app is launched as `streamlit run interface/gui_app.py`, which puts interface/ (not the repo
# root) on sys.path. Add the repo root so the package imports (interface.*, generators.*, ...) resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import streamlit.components.v1 as components
import interface.gui_backend as gb
import visualiser.figures as figs
from interface.demo import build_demo, build_scaling_demo, build_nu_demo   # imported as names (a local `demo` dict shadows the module)
import runner.jobs as jobs
import interface.estimate as estimate           # NOT `est` -- a local float `est` shadows it below
from engine.percolation import l_sweep

st.set_page_config(page_title="Aperiodic Percolation Portal", layout="wide")
# Tighten the vertical rhythm: Streamlit's defaults leave a lot of dead space up top -- the main
# container has ~6rem padding, headings carry ~1.25rem of block padding each (the h1 title alone was
# ~90px tall), and the sidebar's collapse-button header is ~60px. Those stack up and push the title,
# the sidebar controls, and the tab content well down the page. Trim them so content sits near the
# top. (Also shrink the metric value font so the longer bands like "a minute or two" don't clip.)
st.markdown(
    "<style>"
    "[data-testid='stMetricValue']{font-size:1.6rem;}"
    # Clear the fixed 60px top header bar so the title isn't cropped under it, but no more than that.
    ".block-container{padding-top:4rem;}"
    "[data-testid='stSidebarHeader']{height:2.25rem;min-height:2.25rem;}"
    # Trim only the heading BOTTOM padding so gaps between headings and content tighten without
    # clipping the glyph tops.
    "h1{padding-top:0.4rem !important;padding-bottom:0.3rem !important;}"
    "h2,h3{padding-top:0.3rem !important;padding-bottom:0.2rem !important;}"
    "</style>",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Rendering…")
def cached_visualise(tiling, size, a, b, graph_type, show_graph):
    return figs.visualise(tiling, size, a, b, graph_type, show_graph)


@st.cache_data(show_spinner="Building demo…")
def cached_demo(lattice):
    return build_demo(lattice=lattice, n=40)


@st.cache_data(show_spinner="Building the scaling grids…")
def cached_scaling():
    return build_scaling_demo()


@st.cache_data(show_spinner="Building the correlation-length grids…")
def cached_nu():
    return build_nu_demo()


@st.cache_data(show_spinner=False)
def _nu_controls():
    """Exact-nu=4/3 reference lattices (square, triangular) for the nu-vs-omega overlay; loaded once."""
    out = []
    for fname, lab in (("square.npz", "square (exact 4/3)"), ("triangular.npz", "triangular (exact 4/3)")):
        try:
            r, _ = gb.load_saved(fname)
            out.append((lab, r))
        except Exception:
            pass
    return out


@st.cache_data(show_spinner=False)
def cached_collapse():
    """Static geometry for the edge-collapse slider: the hat outline as (edge vector, class) pairs so
    the JS can reconstruct Tile(a,b) live, plus the fixed rotation and a viewBox. 'a' edges scale with
    a (unit-length), 'b' edges with b/√3 (√3-length); shrinking a class to 0 walks the hat to an
    endpoint (b→0 comet, a→0 chevron)."""
    import math
    import generators.hat_generator as hg
    SQ3 = math.sqrt(3)
    pts = [(p["x"], p["y"]) for p in hg.hat_outline]
    n = len(pts)
    edges = []
    for i in range(n):                                   # include the CLOSING edge (n-1 -> 0), itself a b-edge
        dx, dy = pts[(i + 1) % n][0] - pts[i][0], pts[(i + 1) % n][1] - pts[i][1]
        L = math.hypot(dx, dy)
        cls = "a" if abs(L - round(L)) < 1e-6 else ("b" if abs(L / SQ3 - round(L / SQ3)) < 1e-6 else "o")
        edges.append({"dx": dx, "dy": dy, "cls": cls})

    def raw(a, b):                                       # first n-1 edges give the n vertices; the last closes
        out = [[0.0, 0.0]]
        for e in edges[:-1]:
            s = a if e["cls"] == "a" else (b / SQ3 if e["cls"] == "b" else 1.0)
            out.append([out[-1][0] + s * e["dx"], out[-1][1] + s * e["dy"]])
        return out
    h = raw(1.0, SQ3)
    ang = math.degrees(math.atan2(h[1][1] - h[0][1], h[1][0] - h[0][0])) % 60.0
    rot = -30.0 if abs(ang - 30.0) < 1e-3 else 0.0
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    big = raw(SQ3, SQ3)                                       # largest reachable tile -> viewBox fits every member
    hr = [[c * x - s * y, -(s * x + c * y)] for x, y in big]  # rotate + flip y for SVG (y down)
    xs = [p[0] for p in hr]; ys = [p[1] for p in hr]; pad = 0.8
    return {"edges": edges, "rot": rot, "sq3": SQ3,
            "view": {"minx": min(xs) - pad, "miny": min(ys) - pad,
                     "w": (max(xs) - min(xs)) + 2 * pad, "h": (max(ys) - min(ys)) + 2 * pad}}


# Self-contained HTML/SVG+JS for the Toy demo. The slider lives INSIDE the component, so dragging it
# recolours the lattice client-side (union-find in JS) with no Streamlit rerun -- it updates smoothly
# while you drag, not only on release. The Streamlit radios (lattice/open/criterion) rebuild it.
_DEMO_HTML = r"""
<style>
 .tw{font-family:sans-serif;color:#333;}
 .trow{display:flex;gap:18px;align-items:center;}
 .tcol{display:flex;flex-direction:column;align-items:center;gap:10px;}
 .tb{padding:5px 10px;border:1px solid #ccc;border-radius:6px;background:#f5f5f7;cursor:pointer;font-size:13px;}
 .vsl{-webkit-appearance:slider-vertical;appearance:slider-vertical;width:26px;height:380px;}
 .st{font-size:14px;margin:8px 2px;} .ok{color:#1a7f37;font-weight:600;} .no{color:#999;}
</style>
<div class="tw">
 <div class="trow">
  <svg id="tsvg" width="440" height="440" style="background:#fff;"></svg>
  <div class="tcol">
   <button class="tb" id="tjmp" title="Jump to the percolation point">Jump</button>
   <input class="vsl" id="tsl" type="range" min="0" value="0" title="Drag to open sites/bonds">
   <button class="tb" id="tres">Reset</button>
  </div>
 </div>
 <div class="st" id="tstat"></div>
</div>
<script>
const D=__DATA__;
(function(){
 const P=["#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd","#17becf","#8c564b","#e377c2","#bcbd22","#393b79","#e6550d","#31a354","#756bb1","#a55194","#843c39","#3182bd"];
 const GOLD="#f4b400", NS="http://www.w3.org/2000/svg";
 const co=D.coords, eg=D.edges, order=D.order, mode=D.mode, N=D.N, kmax=order.length;
 const TOP=new Set(D.top),BOT=new Set(D.bottom),LE=new Set(D.left),RI=new Set(D.right);
 const x0=D.bbox[0],x1=D.bbox[1],y0=D.bbox[2],y1=D.bbox[3], pad=(x1-x0)*0.04;
 const fy=y=>(y0+y1)-y;
 const svg=document.getElementById("tsvg");
 svg.setAttribute("viewBox",(x0-pad)+" "+(y0-pad)+" "+((x1-x0)+2*pad)+" "+((y1-y0)+2*pad));
 svg.setAttribute("preserveAspectRatio","xMidYMid meet");
 const u=(x1-x0)/Math.sqrt(N);
 const R_O=0.30*u,R_C=0.13*u,W_THIN=0.12*u,W_THICK=0.34*u,W_FAINT=0.06*u;
 const fg=document.createElementNS(NS,"g"); svg.appendChild(fg);
 for(const e of eg){const l=document.createElementNS(NS,"line");
  l.setAttribute("x1",co[e[0]][0]);l.setAttribute("y1",fy(co[e[0]][1]));
  l.setAttribute("x2",co[e[1]][0]);l.setAttribute("y2",fy(co[e[1]][1]));
  l.setAttribute("stroke","#e6e6ea");l.setAttribute("stroke-width",W_FAINT);fg.appendChild(l);}
 const eG=document.createElementNS(NS,"g"); svg.appendChild(eG);
 const eEl=eg.map(()=>{const l=document.createElementNS(NS,"line");l.setAttribute("visibility","hidden");eG.appendChild(l);return l;});
 const nG=document.createElementNS(NS,"g"); svg.appendChild(nG);
 const nEl=co.map(c=>{const ci=document.createElementNS(NS,"circle");ci.setAttribute("cx",c[0]);ci.setAttribute("cy",fy(c[1]));nG.appendChild(ci);return ci;});
 let par;
 function find(x){let r=x;while(par[r]!==r)r=par[r];while(par[x]!==r){const n=par[x];par[x]=r;x=n;}return r;}
 function col(m){return P[(m*2654435761)%P.length];}
 function render(k){
  par=new Array(N);for(let i=0;i<N;i++)par[i]=i;
  const open=new Set(); const act=[];
  if(mode==="site"){for(let i=0;i<k;i++)open.add(order[i]);
   for(let e=0;e<eg.length;e++){const a=eg[e][0],b=eg[e][1];if(open.has(a)&&open.has(b)){act.push(e);const ra=find(a),rb=find(b);if(ra!==rb)par[ra]=rb;}}}
  else {for(let i=0;i<k;i++){const e=order[i];act.push(e);const a=eg[e][0],b=eg[e][1];open.add(a);open.add(b);const ra=find(a),rb=find(b);if(ra!==rb)par[ra]=rb;}}
  const rmin={},rh={}; open.forEach(v=>{const r=find(v);if(rmin[r]===undefined||v<rmin[r])rmin[r]=v;let h=rh[r]||(rh[r]={t:0,b:0,l:0,r:0});if(TOP.has(v))h.t=1;if(BOT.has(v))h.b=1;if(LE.has(v))h.l=1;if(RI.has(v))h.r=1;});
  const span=new Set();let tb=false,lr=false;
  for(const r in rh){const h=rh[r];if(h.t&&h.b){span.add(+r);tb=true;}if(h.l&&h.r){span.add(+r);lr=true;}}
  const cOf=v=>{const r=find(v);return span.has(r)?GOLD:col(rmin[r]);};
  // Largest cluster = the incipient infinite cluster (its size ~ L^d_f at criticality). Track it and
  // ring it so you can watch it grow and take over as you drag toward the percolation point.
  const sz={}; open.forEach(v=>{const r=find(v);sz[r]=(sz[r]||0)+1;});
  let bigR=-1,bigN=0; for(const r in sz){if(sz[r]>bigN){bigN=sz[r];bigR=+r;}}
  for(let v=0;v<N;v++){const ci=nEl[v];const big=open.has(v)&&find(v)===bigR;
   if(big){ci.setAttribute("r",R_O*1.18);ci.setAttribute("fill",cOf(v));ci.setAttribute("stroke","#111");ci.setAttribute("stroke-width",W_THIN);}
   else if(mode==="bond"){ci.setAttribute("r",R_C);ci.setAttribute("fill","#9a9aa2");ci.removeAttribute("stroke");}
   else if(open.has(v)){ci.setAttribute("r",R_O);ci.setAttribute("fill",cOf(v));ci.setAttribute("stroke","#fff");ci.setAttribute("stroke-width",W_THIN*0.5);}
   else {ci.setAttribute("r",R_C);ci.setAttribute("fill","#d8d8dd");ci.removeAttribute("stroke");}}
  eEl.forEach(l=>l.setAttribute("visibility","hidden"));
  for(const e of act){const a=eg[e][0],b=eg[e][1],l=eEl[e];l.setAttribute("visibility","visible");
   l.setAttribute("x1",co[a][0]);l.setAttribute("y1",fy(co[a][1]));l.setAttribute("x2",co[b][0]);l.setAttribute("y2",fy(co[b][1]));
   l.setAttribute("stroke",cOf(a));l.setAttribute("stroke-width",mode==="bond"?W_THICK:W_THIN);}
  const unit=mode==="site"?"sites":"bonds";
  const bigFrac=open.size?(100*bigN/open.size).toFixed(0):"0";
  document.getElementById("tstat").innerHTML="<b>"+k+"/"+kmax+"</b> "+unit+" open &nbsp;·&nbsp; p ≈ <b>"+(k/kmax).toFixed(2)+"</b>"
   +"<br>Largest cluster <span style='outline:1px solid #111;padding:0 2px;'>◯</span> : <b>"+bigN+"</b> sites &nbsp;·&nbsp; "+bigFrac+"% of what's open"
   +"<br>Left ↔ right: "+(lr?"<span class=ok>spanning</span>":"<span class=no>not yet</span>")
   +" &nbsp;|&nbsp; Top ↔ bottom: "+(tb?"<span class=ok>spanning</span>":"<span class=no>not yet</span>");
 }
 const sl=document.getElementById("tsl");sl.max=kmax;
 // render synchronously on every 'input' (which fires continuously WHILE dragging) so the lattice
 // updates live under the thumb, not just on release.
 sl.addEventListener("input",()=>render(+sl.value));
 document.getElementById("tjmp").addEventListener("click",()=>{sl.value=D.jump;render(D.jump);});
 document.getElementById("tres").addEventListener("click",()=>{sl.value=0;render(0);});
 render(0);
})();
</script>
"""


def demo_component(demo, mode, jump_k):
    payload = {"coords": demo["coords"], "edges": demo["edges"],
               "order": demo["site_order"] if mode == "site" else demo["bond_order"],
               "mode": mode, "N": demo["N"],
               "top": list(demo["top"]), "bottom": list(demo["bottom"]),
               "left": list(demo["left"]), "right": list(demo["right"]),
               "bbox": demo["bbox"], "jump": int(jump_k)}
    components.html(_DEMO_HTML.replace("__DATA__", json.dumps(payload)), height=540)


# Interactive SCALING toy: three square grids of increasing size, one shared "how open" slider driving
# them all. Client-side union-find per grid finds the largest cluster (gold), and shows its % share --
# at percolation the shares LINE UP across sizes yet shrink as the grid grows (the fractal signature).
_SCALING_HTML = r"""
<style>
 .scw{font-family:sans-serif;color:#333;}
 .scgrids{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-end;justify-content:center;}
 .sccol{display:flex;flex-direction:column;align-items:center;}
 .scsvg{background:#fff;border:1px solid #eee;}
 .sclbl{font-size:11px;margin-top:2px;text-align:center;color:#666;}
 .scctl{display:flex;gap:12px;align-items:center;margin:12px 4px 4px;}
 .scb{padding:5px 12px;border:1px solid #ccc;border-radius:6px;background:#f5f5f7;cursor:pointer;font-size:13px;}
 .scsl{flex:1;} .scsum{font-size:13.5px;margin:4px 2px 0;min-height:3.2em;line-height:1.45;}
 .scplotwrap{display:flex;justify-content:center;margin-top:6px;}
</style>
<div class="scw">
 <div class="scgrids" id="sc_grids"></div>
 <div class="scctl">
  <button class="scb" id="sc_jump" title="Jump to the percolation point p = 0.5927">Jump to percolation</button>
  <input class="scsl" id="sc_sl" type="range" min="0" max="100" value="0" title="Drag to open more of every grid">
  <span id="sc_frac"></span>
 </div>
 <div class="scplotwrap"><svg id="sc_plot" width="560" height="300" style="max-width:100%;height:auto;"></svg></div>
 <div class="scsum" id="sc_sum"></div>
</div>
<script>
const G=__DATA__;
(function(){
 const NS="http://www.w3.org/2000/svg", GOLD="#f4b400", BLUE="#1f5fa8", GREY="#c2c2cc";
 const sizes=G.sizes, pgrid=G.pgrid, P=pgrid.length, F=G.fillings, pc=G.pc;
 const scol=i=>"hsl("+Math.round(222-222*i/Math.max(1,sizes.length-1))+",62%,48%)";   // per-size hue
 const pcIdx=pgrid.reduce((best,p,i)=>Math.abs(p-pc)<Math.abs(pgrid[best]-pc)?i:best,0);
 const el=id=>document.getElementById(id);
 const mk=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};

 // ---- thumbnails: one grid per size (filling 0), drawn live from its open order ----
 const holder=el("sc_grids");
 const grids=G.grids.map((g,gi)=>{
  const box=56, col=document.createElement("div"); col.className="sccol";
  const svg=document.createElementNS(NS,"svg"); svg.setAttribute("width",box); svg.setAttribute("height",box); svg.setAttribute("class","scsvg");
  const b=g.bbox,x0=b[0],x1=b[1],y0=b[2],y1=b[3],pad=(x1-x0)*0.04;
  svg.setAttribute("viewBox",(x0-pad)+" "+(y0-pad)+" "+((x1-x0)+2*pad)+" "+((y1-y0)+2*pad));
  const fy=y=>(y0+y1)-y,u=(x1-x0)/g.n,R=0.5*u;             // touching sites -> clusters look solid
  const nEl=g.coords.map(c=>{const ci=mk("circle",{cx:c[0],cy:fy(c[1]),r:R});svg.appendChild(ci);return ci;});
  const lbl=document.createElement("div"); lbl.className="sclbl"; lbl.textContent="L="+g.n; lbl.style.color=scol(gi);
  col.appendChild(svg); col.appendChild(lbl); holder.appendChild(col);
  return {g,nEl,R};
 });
 function drawThumb(gi,p){
  const G3=grids[gi],g=G3.g,N=g.N,edges=g.edges,order=g.order,k=Math.min(Math.round(p*N),N);
  const par=new Int32Array(N);for(let i=0;i<N;i++)par[i]=i;
  const open=new Uint8Array(N);
  function find(x){let r=x;while(par[r]!==r)r=par[r];while(par[x]!==r){const n=par[x];par[x]=r;x=n;}return r;}
  for(let i=0;i<k;i++)open[order[i]]=1;
  for(let e=0;e<edges.length;e++){const a=edges[e][0],b=edges[e][1];if(open[a]&&open[b]){const ra=find(a),rb=find(b);if(ra!==rb)par[ra]=rb;}}
  const sz=new Int32Array(N);let bigR=-1,bigN=0;
  for(let v=0;v<N;v++){if(open[v]){const r=find(v);sz[r]++;if(sz[r]>bigN){bigN=sz[r];bigR=r;}}}
  for(let v=0;v<N;v++){const ci=G3.nEl[v];
   if(open[v]&&find(v)===bigR){ci.setAttribute("fill",scol(gi));ci.removeAttribute("stroke");}
   else{ci.setAttribute("fill","#a9cbe8");ci.removeAttribute("stroke");}}
 }

 // ---- log-log plot: share (%) of largest cluster vs grid size L ----
 const plot=el("sc_plot"), PW=560, PH=300, mL=46, mR=16, mT=30, mB=34;
 const xlo=Math.log10(sizes[0]*0.88), xhi=Math.log10(sizes[sizes.length-1]*1.12);
 const ylo=Math.log10(2), yhi=Math.log10(100);
 const SX=L=>mL+(Math.log10(L)-xlo)/(xhi-xlo)*(PW-mL-mR);
 const SY=s=>PH-mB-(Math.log10(Math.max(s,2))-ylo)/(yhi-ylo)*(PH-mT-mB);
 // static axes
 plot.appendChild(mk("rect",{x:mL,y:mT,width:PW-mL-mR,height:PH-mT-mB,fill:"#fff",stroke:"#eee"}));
 [2,5,10,20,50,100].forEach(s=>{
  plot.appendChild(mk("line",{x1:mL,y1:SY(s),x2:PW-mR,y2:SY(s),stroke:"#f0f0f2"}));
  const tx=mk("text",{x:mL-6,y:SY(s)+3,"text-anchor":"end","font-size":10,fill:"#999"});tx.textContent=s+"%";plot.appendChild(tx);
 });
 sizes.forEach(L=>{
  const tx=mk("text",{x:SX(L),y:PH-mB+13,"text-anchor":"middle","font-size":10,fill:"#999"});tx.textContent=L;plot.appendChild(tx);
 });
 plot.appendChild(mk("text",{x:(mL+PW-mR)/2,y:PH-4,"text-anchor":"middle","font-size":11,fill:"#666"})).textContent="grid size  L";
 const yl=mk("text",{x:12,y:(mT+PH-mB)/2,"text-anchor":"middle","font-size":11,fill:"#666",transform:"rotate(-90 12 "+((mT+PH-mB)/2)+")"});yl.textContent="largest-cluster share  s_max / N";plot.appendChild(yl);
 // reference line: slope d_f-2 = -0.104 (exact 2D), anchored at the ladder centre
 const xc=(xlo+xhi)/2, refY0=Math.log10(32);
 const refA=refY0+0.104*xc; // log10(share)=refA-0.104*log10(L)
 const refShare=x=>Math.pow(10,refA-0.104*x);
 plot.appendChild(mk("line",{x1:SX(sizes[0]),y1:SY(refShare(Math.log10(sizes[0]))),x2:SX(sizes[sizes.length-1]),y2:SY(refShare(Math.log10(sizes[sizes.length-1]))),stroke:"#bbb","stroke-dasharray":"5 4","stroke-width":1.3}));
 const refLbl=mk("text",{x:SX(sizes[sizes.length-1]),y:SY(refShare(Math.log10(sizes[sizes.length-1])))-5,"text-anchor":"end","font-size":10,fill:"#999"});refLbl.textContent="2D fractal  d_f=1.90";plot.appendChild(refLbl);
 // dynamic layer (dots + fit line + d_f readout)
 const dyn=mk("g",{});plot.appendChild(dyn);

 function linreg(xs,ys){
  const n=xs.length,mx=xs.reduce((a,b)=>a+b,0)/n,my=ys.reduce((a,b)=>a+b,0)/n;
  let sxy=0,sxx=0;for(let i=0;i<n;i++){sxy+=(xs[i]-mx)*(ys[i]-my);sxx+=(xs[i]-mx)*(xs[i]-mx);}
  const b=sxx?sxy/sxx:0;return {b,a:my-b*mx};
 }
 function update(pidx){
  while(dyn.firstChild)dyn.removeChild(dyn.firstChild);
  const p=pgrid[pidx];
  grids.forEach((_,gi)=>drawThumb(gi,p));
  const mLx=[],mLy=[],means=[];
  G.grids.forEach((g,gi)=>{
   let sum=0;
   for(let f=0;f<F;f++){
    const s=g.shares[f][pidx]; sum+=s;
    dyn.appendChild(mk("circle",{cx:SX(g.n),cy:SY(s),r:2.4,fill:f===0?"none":GREY,
      stroke:f===0?scol(gi):"none","stroke-width":f===0?1.4:0,"fill-opacity":0.55}));
   }
   const m=sum/F; means.push(m);
   dyn.appendChild(mk("circle",{cx:SX(g.n),cy:SY(m),r:4.2,fill:scol(gi),stroke:"#fff","stroke-width":1}));
   if(m>0){mLx.push(Math.log10(g.n));mLy.push(Math.log10(m));}
  });
  // fit line through the per-size means; slope of log(share) vs log(L) is d_f-2
  const fit=linreg(mLx,mLy), df=2+fit.b;
  const L0=sizes[0],L1=sizes[sizes.length-1];
  const y0=Math.pow(10,fit.a+fit.b*Math.log10(L0)), y1=Math.pow(10,fit.a+fit.b*Math.log10(L1));
  dyn.appendChild(mk("line",{x1:SX(L0),y1:SY(y0),x2:SX(L1),y2:SY(y1),stroke:BLUE,"stroke-width":2}));
  dyn.appendChild(mk("text",{x:mL+8,y:mT-10,"font-size":13,fill:BLUE,"font-weight":700})).textContent="d_f = "+df.toFixed(2);
  el("sc_frac").innerHTML="p = <b>"+p.toFixed(3)+"</b>";
  // caption by regime
  let msg;
  if(p<pc-0.03){
   msg="Below the percolation point: the largest cluster is a finite blob, so its share falls steeply as the grid grows (the points slope down far faster than the dashed line). Not yet critical.";
  }else if(p<=pc+0.03){
   msg="At the percolation point the ten means sit on a straight line parallel to the 2D reference: the share is nearly the same across a 5&times; range of sizes (scale invariance). Its slope gives d<sub>f</sub> = "+df.toFixed(2)+" (2D exact 91/48 = 1.90). Faint dots are the "+F+" individual fillings behind each mean, so the run-to-run spread is visible and never shrinks, which is exactly why one grid can't settle this and the average can.";
  }else{
   msg="Above the percolation point: the largest cluster is extensive, so its share flattens toward a constant (slope &rarr; 0, d<sub>f</sub> &rarr; 2), filling a fixed fraction of every grid.";
  }
  el("sc_sum").innerHTML=msg;
 }
 const sl=el("sc_sl");
 sl.max=P-1;
 sl.addEventListener("input",()=>update(+sl.value));
 el("sc_jump").addEventListener("click",()=>{sl.value=pcIdx;update(pcIdx);});
 sl.value=0; update(0);
})();
</script>
"""


def scaling_component(demo):
    """Render the interactive scaling toy. `demo` is build_scaling_demo()'s dict (sizes, pgrid, pc,
    fillings, grids); the whole thing is JSON-embedded and driven client-side."""
    components.html(_SCALING_HTML.replace("__DATA__", json.dumps(demo)), height=520)


# ============================================================ CORRELATION-LENGTH (nu) COLLAPSE TOY
_NU_HTML = r"""
<style>
 .nuw{font-family:sans-serif;color:#333;}
 .nustage{font-size:13px;color:#444;margin:16px 2px 6px;line-height:1.5;}
 .nustep{display:inline-block;width:20px;height:20px;line-height:20px;text-align:center;border-radius:50%;background:#2e6db4;color:#fff;font-size:12px;font-weight:700;margin-right:6px;}
 .nutop{display:flex;gap:22px;align-items:flex-start;flex-wrap:wrap;justify-content:center;margin-top:4px;}
 .nurow{display:flex;flex-wrap:wrap;gap:7px;justify-content:center;max-width:352px;}
 .nucolh{font-size:11px;color:#888;margin-bottom:4px;text-align:center;}
 .nuleft,.nuright{display:flex;flex-direction:column;align-items:center;}
 .nucol{display:flex;flex-direction:column;align-items:center;}
 .nulbltop{font-size:11px;font-weight:700;margin-bottom:2px;}
 .nulblbot{font-size:10px;margin-top:2px;min-height:1.1em;text-align:center;line-height:1.2;}
 .nusvg{background:#fff;border:1px solid #eee;}
 .nuctl{display:flex;gap:10px;align-items:center;margin:8px 4px 4px;flex-wrap:wrap;}
 .nub{padding:5px 11px;border:1px solid #ccc;border-radius:6px;background:#f5f5f7;cursor:pointer;font-size:13px;}
 .nusl{flex:1;min-width:150px;} .nusum{font-size:13.5px;margin:8px 2px 0;line-height:1.45;}
 .nuq{display:inline-block;height:9px;border-radius:5px;background:#e6e6ee;width:110px;vertical-align:middle;overflow:hidden;}
 .nuqf{height:100%;background:#2e9e5b;width:0%;}
 .nubig{font-size:14.5px;min-height:2.2em;}
</style>
<div class="nuw">
 <div class="nustage">Drag the occupation slider, and each grid percolates at its own point, freezes green, and drops a dot on the chart at (its size, the p where it crossed). Slide all the way up for all ten dots (slide back to re-watch), then draw the ν = 4/3 curve: the finite-grid points climb toward the true p<sub>c</sub> along p<sub>c</sub> &minus; a&middot;L<sup>&minus;1/ν</sup>.</div>
 <div class="nutop">
  <div class="nuleft">
   <div class="nucolh">grids freeze as they span</div>
   <div class="nurow" id="nu_grids"></div>
  </div>
  <div class="nuright">
   <div class="nucolh">the chart the simulation builds</div>
   <svg class="nusvg" id="nu_plot" width="340" height="260"></svg>
   <div class="nuctl"><button class="nub" id="nu_curve">draw the ν = 4/3 curve</button></div>
  </div>
 </div>
 <div class="nuctl" style="justify-content:center;margin-top:12px;">
  <span>p = <b id="nu_p">0.450</b></span>
  <input class="nusl" id="nu_psl" type="range" min="450" max="620" value="450" step="1" style="max-width:600px;" title="open more sites">
  <button class="nub" id="nu_replay" title="reset">&#8635;</button>
 </div>
 <div class="nusum" id="nu_sum"></div>
</div>
<script>
const G=__DATA__;
(function(){
 const NS="http://www.w3.org/2000/svg";
 const mk=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
 const sizes=G.sizes, pc=G.pc, grids=G.grids, S=sizes.length;
 const col=i=>"hsl("+Math.round(222-222*i/Math.max(1,S-1))+",62%,48%)";   // blue (small) -> red (big)
 const el=id=>document.getElementById(id);
 const GOLD="#f4b400", GREEN="#2e9e5b";

 // ---- (1) ten grids, each drawn with its MEDIAN filling; freeze green when it spans ----
 const holder=el("nu_grids");
 const LG=grids.map((g,gi)=>{
  const box=60, cw=document.createElement("div"); cw.className="nucol";
  const top=document.createElement("div"); top.className="nulbltop"; top.style.color=col(gi); top.textContent="L = "+g.n;
  const svg=document.createElementNS(NS,"svg"); svg.setAttribute("width",box);svg.setAttribute("height",box);svg.setAttribute("class","nusvg");
  const b=g.bbox,x0=b[0],x1=b[1],y0=b[2],y1=b[3],pad=(x1-x0)*0.03;
  svg.setAttribute("viewBox",(x0-pad)+" "+(y0-pad)+" "+((x1-x0)+2*pad)+" "+((y1-y0)+2*pad));
  const fy=y=>(y0+y1)-y,u=(x1-x0)/g.n,R=0.5*u;             // touching sites -> clusters look solid
  const nEl=g.coords.map(c=>{const ci=mk("circle",{cx:c[0],cy:fy(c[1]),r:R});svg.appendChild(ci);return ci;});
  const bot=document.createElement("div"); bot.className="nulblbot";
  cw.appendChild(top);cw.appendChild(svg);cw.appendChild(bot);holder.appendChild(cw);
  // crossK = the exact site count at which THIS drawn filling spans (crossP is rounded, so keying off
  // p directly can mis-colour the spanning shape as "not yet" for a sliver of p just below it).
  return {g,gi,nEl,R,bot,order:g.order,crossP:g.crossP,crossK:Math.round(g.crossP*g.N),spanned:false};
 });
 function drawGrid(G3,k,frozen){
  const g=G3.g,N=g.N,edges=g.edges,order=G3.order;
  const par=new Int32Array(N);for(let i=0;i<N;i++)par[i]=i;const open=new Uint8Array(N);
  function find(x){let r=x;while(par[r]!==r)r=par[r];while(par[x]!==r){const n=par[x];par[x]=r;x=n;}return r;}
  for(let i=0;i<k;i++)open[order[i]]=1;
  for(let e=0;e<edges.length;e++){const a=edges[e][0],b=edges[e][1];if(open[a]&&open[b]){const ra=find(a),rb=find(b);if(ra!==rb)par[ra]=rb;}}
  const sz=new Int32Array(N);let bigR=-1,bigN=0;
  for(let v=0;v<N;v++){if(open[v]){const r=find(v);sz[r]++;if(sz[r]>bigN){bigN=sz[r];bigR=r;}}}
  const cc=frozen?col(G3.gi):GOLD;                          // spanning cluster wears this grid's dot colour
  // every cell is filled: the spanning cluster in its colour, everything else (open or closed) light
  // blue -- no white "holes".
  for(let v=0;v<N;v++){const ci=G3.nEl[v];
   if(open[v]&&find(v)===bigR){ci.setAttribute("fill",cc);ci.removeAttribute("stroke");}
   else{ci.setAttribute("fill","#a9cbe8");ci.removeAttribute("stroke");}}
  G3.bot.innerHTML=frozen?"<b style='color:"+col(G3.gi)+"'>p_c ≈ "+G3.crossP.toFixed(2)+"</b>":"<span style='color:#ccc'>…</span>";
 }

 // ---- (2) the plot the sim builds: each grid's percolation point p* vs L, climbing to p_c ----
 let curP=0.45, showCurve=false;                            // dots shown = grids currently spanning
 const plt=el("nu_plot"),pW=340,pH=260,pmL=44,pmR=14,pmT=12,pmB=34;
 const Lmin=sizes[0],Lmax=sizes[S-1], xlo=Math.log10(Lmin*0.9),xhi=Math.log10(Lmax*1.12);
 const PX=L=>pmL+(Math.log10(L)-xlo)/(xhi-xlo)*(pW-pmL-pmR);
 const ylo=0.53,yhi=0.60, PY=p=>pH-pmB-(Math.min(Math.max(p,ylo),yhi)-ylo)/(yhi-ylo)*(pH-pmT-pmB);
 const pdyn=mk("g",{});plt.appendChild(pdyn);
 function drawPlot(){
  while(pdyn.firstChild)pdyn.removeChild(pdyn.firstChild);
  pdyn.appendChild(mk("rect",{x:pmL,y:pmT,width:pW-pmL-pmR,height:pH-pmT-pmB,fill:"#fff",stroke:"#eee"}));
  [0.54,0.56,0.58,0.60].forEach(p=>{pdyn.appendChild(mk("line",{x1:pmL,y1:PY(p),x2:pW-pmR,y2:PY(p),stroke:"#f4f4f6"}));
   const t=mk("text",{x:pmL-5,y:PY(p)+3,"text-anchor":"end","font-size":9,fill:"#999"});t.textContent=p.toFixed(2);pdyn.appendChild(t);});
  sizes.forEach((L,gi)=>{const t=mk("text",{x:PX(L),y:pH-pmB+12,"text-anchor":"middle","font-size":9,fill:col(gi),"font-weight":700});t.textContent=L;pdyn.appendChild(t);});
  pdyn.appendChild(mk("line",{x1:pmL,y1:PY(pc),x2:pW-pmR,y2:PY(pc),stroke:"#c0392b","stroke-dasharray":"4 3"}));
  pdyn.appendChild(mk("text",{x:pW-pmR-2,y:PY(pc)-4,"text-anchor":"end","font-size":10,fill:"#c0392b"})).textContent="true p_c = 0.593";
  const shown=LG.filter(G3=>G3.spanned);                    // grids currently percolating
  if(showCurve && shown.length===LG.length){                // fit a in p_c - p* = a·L^(-1/nu), nu=4/3
   let sxy=0,sxx=0;shown.forEach(G3=>{const x=Math.pow(G3.g.n,-0.75),y=pc-G3.crossP;sxy+=x*y;sxx+=x*x;});
   const a=sxx?sxy/sxx:0;let dd="";
   for(let s=0;s<=48;s++){const L=Math.pow(10,xlo+(xhi-xlo)*s/48),y=pc-a*Math.pow(L,-0.75);dd+=(s?"L":"M")+PX(L).toFixed(1)+" "+PY(y).toFixed(1)+" ";}
   pdyn.appendChild(mk("path",{d:dd,fill:"none",stroke:"#2e6db4","stroke-width":2}));
   pdyn.appendChild(mk("text",{x:PX(Lmax),y:PY(pc-a*Math.pow(Lmax,-0.75))+14,"text-anchor":"end","font-size":10,fill:"#2e6db4","font-weight":700})).textContent="p_c − a·L^(−1/ν), ν=4/3";
  }
  shown.forEach(G3=>pdyn.appendChild(mk("circle",{cx:PX(G3.g.n),cy:PY(G3.crossP),r:4,fill:col(G3.gi),stroke:"#fff","stroke-width":1})));
  pdyn.appendChild(mk("text",{x:(pmL+pW-pmR)/2,y:pH-3,"text-anchor":"middle","font-size":10,fill:"#666"})).textContent="grid size  L  (log)";
  const yl2=mk("text",{x:12,y:(pmT+pH-pmB)/2,"text-anchor":"middle","font-size":10,fill:"#666",transform:"rotate(-90 12 "+((pmT+pH-pmB)/2)+")"});yl2.textContent="percolation point  p*";pdyn.appendChild(yl2);
 }

 // ---- occupation slider: freeze each grid at its crossing point and drop its dot ----
 function reset(){curP=0.45;showCurve=false;
  el("nu_psl").value=450;el("nu_p").textContent="0.450";
  LG.forEach(G3=>{G3.spanned=false;drawGrid(G3,Math.round(0.45*G3.g.N),false);});drawPlot();
  el("nu_sum").innerHTML="Slide the occupation up, and a dot appears as each grid spans (and vanishes if you slide back); then draw the curve.";}
 function occ(p){curP=p;el("nu_p").textContent=p.toFixed(3);
  // A grid is spanning exactly once the drawn site count reaches crossK. Colour, freeze and the dot all
  // key off that (not off p vs the rounded crossP), so the spanning shape is never shown as "not yet".
  LG.forEach(G3=>{const k=Math.round(p*G3.g.N); G3.spanned=k>=G3.crossK;
   if(G3.spanned)drawGrid(G3,G3.crossK,true); else drawGrid(G3,k,false);});
  drawPlot();}
 el("nu_psl").addEventListener("input",()=>occ(+el("nu_psl").value/1000));
 el("nu_replay").addEventListener("click",reset);
 el("nu_curve").addEventListener("click",()=>{
  if(LG.filter(G3=>G3.spanned).length<sizes.length){el("nu_sum").innerHTML="Slide all the way up first, so every grid has percolated.";return;}
  showCurve=true;drawPlot();
  el("nu_sum").innerHTML="The ν = 4/3 curve threads the dots and flattens onto p<sub>c</sub>; each grid's shortfall p<sub>c</sub> &minus; p* scales as L<sup>&minus;1/ν</sup>. <span style='color:#888'>On grids this small the fit is approximate (their true slope is a hair off 4/3); it tightens as the grids grow.</span>";});
 reset();
})();
</script>
"""


def nu_component(demo):
    """Render the interactive correlation-length (nu) collapse toy. `demo` is build_nu_demo()'s dict
    (sizes, pgrid, pc, per-size spanning-probability R). Client-side; the nu slider rescales the axis."""
    components.html(_NU_HTML.replace("__DATA__", json.dumps(demo)), height=510)


# Edge-collapse slider: reconstruct Tile(a,b) client-side from the hat's (edge, class) list. Two sliders
# shrink the two edge-classes; at a class = 0 the hat lands on a family endpoint (comet / chevron).
_COLLAPSE_HTML = r"""
<style>
 .cw{font-family:sans-serif;color:#333;max-width:440px;margin:0 auto;}
 .csvgwrap{display:flex;justify-content:center;}
 .cbtns{display:flex;gap:6px;justify-content:center;margin:8px 0 2px;}
 .cbtn{padding:4px 12px;border:1px solid #ccc;border-radius:6px;background:#f5f5f7;cursor:pointer;font-size:12px;}
 .crow{display:flex;align-items:center;gap:8px;margin:7px 4px;font-size:13px;}
 .crow input[type=range]{flex:1;}
 .clab{width:168px;}
 .clsum{font-size:13.5px;margin-top:6px;text-align:center;color:#333;min-height:1.3em;}
</style>
<div class="cw">
 <div class="csvgwrap"><svg id="cl_svg" width="380" height="340" style="max-width:100%;height:auto;background:#fff;border:1px solid #eee;"></svg></div>
 <div class="cbtns">
  <button class="cbtn" data-a="58" data-b="100">Hat</button>
  <button class="cbtn" data-a="100" data-b="58">Turtle</button>
  <button class="cbtn" data-a="58" data-b="58">Spectre</button>
  <button class="cbtn" data-a="58" data-b="0">Comet</button>
  <button class="cbtn" data-a="0" data-b="58">Chevron</button>
 </div>
 <div class="crow"><span class="clab">a &nbsp;<span style="color:#2e6f95">unit / blue edges</span></span><input type="range" id="cl_a" min="0" max="100" value="58"></div>
 <div class="crow"><span class="clab">b &nbsp;<span style="color:#d1495b">&#8730;3 / red edges</span></span><input type="range" id="cl_b" min="0" max="100" value="100"></div>
 <div class="clsum" id="cl_sum"></div>
</div>
<script>
(function(){
 const D=__DATA__, NS="http://www.w3.org/2000/svg", svg=document.getElementById("cl_svg");
 svg.setAttribute("viewBox", D.view.minx+" "+D.view.miny+" "+D.view.w+" "+D.view.h);
 const rot=D.rot*Math.PI/180, cr=Math.cos(rot), sr=Math.sin(rot);
 function poly(a,b){let x=0,y=0;const P=[[0,0]];
   for(let i=0;i<D.edges.length-1;i++){const e=D.edges[i];const s=e.cls=="a"?a:(e.cls=="b"?b/D.sq3:1.0);x+=s*e.dx;y+=s*e.dy;P.push([x,y]);}
   return P.map(p=>[cr*p[0]-sr*p[1], -(sr*p[0]+cr*p[1])]);}
 function col(c){return c=="a"?"#2e6f95":(c=="b"?"#d1495b":"#b0b0b0");}
 function near(a,b,ta,tb){return Math.abs(a-ta)<0.06&&Math.abs(b-tb)<0.06;}
 function draw(){
   const a=+document.getElementById("cl_a").value/100*D.sq3;
   const b=+document.getElementById("cl_b").value/100*D.sq3;
   const P=poly(a,b);
   while(svg.firstChild)svg.removeChild(svg.firstChild);
   const fl=document.createElementNS(NS,"polygon");
   fl.setAttribute("points",P.map(p=>p[0].toFixed(4)+","+p[1].toFixed(4)).join(" "));
   fl.setAttribute("fill","#f2f2f4");svg.appendChild(fl);
   for(let i=0;i<P.length;i++){const p=P[i],q=P[(i+1)%P.length];
     const ln=document.createElementNS(NS,"line");
     ln.setAttribute("x1",p[0]);ln.setAttribute("y1",p[1]);ln.setAttribute("x2",q[0]);ln.setAttribute("y2",q[1]);
     ln.setAttribute("stroke",col(i<D.edges.length?D.edges[i].cls:"o"));
     ln.setAttribute("stroke-width","0.14");ln.setAttribute("stroke-linecap","round");ln.setAttribute("stroke-linejoin","round");
     svg.appendChild(ln);}
   let nm="";
   if(near(a,b,1,D.sq3))nm=": the hat, Tile(1,&#8730;3)";
   else if(near(a,b,D.sq3,1))nm=": the turtle, Tile(&#8730;3,1)";
   else if(near(a,b,1,1))nm=": the spectre, Tile(1,1)";
   else if(b<0.04)nm=": the comet endpoint (b = 0)";
   else if(a<0.04)nm=": the chevron endpoint (a = 0)";
   document.getElementById("cl_sum").innerHTML="a = "+a.toFixed(2)+" , b = "+b.toFixed(2)+nm;
 }
 document.getElementById("cl_a").addEventListener("input",draw);
 document.getElementById("cl_b").addEventListener("input",draw);
 document.querySelectorAll(".cbtn").forEach(btn=>btn.addEventListener("click",()=>{
   document.getElementById("cl_a").value=btn.dataset.a;
   document.getElementById("cl_b").value=btn.dataset.b;draw();}));
 draw();
})();
</script>
"""


def collapse_component(data):
    """Render the edge-collapse slider toy. `data` is cached_collapse()'s dict (edges + rotation +
    viewBox). Client-side: the two sliders rebuild Tile(a,b) live."""
    components.html(_COLLAPSE_HTML.replace("__DATA__", json.dumps(data)), height=480)


def _fig_bytes(fig):
    """PNG bytes of a matplotlib figure, for st.download_button."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def show_results(res, label, key_prefix, meta=None):
    """Render a result (freshly run OR loaded from disk): the run parameters (L range/gap/sizes,
    trials, seed, patch), p_c with visible ± error, the direction-bias check that validates the
    estimator, both plots (extrapolation + per-size convergence), and PNG downloads. (The width-line
    nu estimator is deliberately not shown — see the note where the direction-bias check is rendered.)"""
    st.subheader(f"Result — {label}")

    # Parameter line: everything that defined this run, so a saved file is self-describing. Rendered
    # as bold-labelled body text (not a faint caption) so it can't be missed above the results.
    meta = meta or {}
    Ls = sorted(float(x) for x in (res.get("L") or []))
    parts = []
    if Ls:
        diffs = [round(b - a, 6) for a, b in zip(Ls, Ls[1:])]
        gap = f"**Step ΔL:** {diffs[0]:g}" if diffs and len(set(diffs)) == 1 else \
              ("**Step ΔL:** varies" if diffs else "**Step ΔL:** —")
        parts += [f"**L minimum:** {Ls[0]:g}", f"**L maximum:** {Ls[-1]:g}", gap, f"**Sizes:** {len(Ls)}"]
    if meta.get("T"):                        parts.append(f"**Trials per L:** {int(meta['T']):,}")
    if meta.get("patch") not in (None, ""):  parts.append(f"**Patch:** {meta['patch']}")
    if meta.get("seed") not in (None, ""):   parts.append(f"**Seed:** {meta['seed']}")
    if parts:
        st.markdown("  ·  ".join(parts))
    if meta.get("timestamp"):
        st.caption(f"Run at {meta['timestamp']}")
    if not res.get("site"):
        st.warning("Fewer than 3 usable sizes — not enough to extrapolate (raw data is still saveable).")
        return
    s = res["site"]["A"]
    c1, c2 = st.columns(2)
    c1.metric("Site  p_c (∞)", f"{s['pc']:.4f} ± {s['pc_std']:.4f}")
    if res.get("bond"):
        b = res["bond"]["A"]
        c2.metric("Bond  p_c (∞)", f"{b['pc']:.4f} ± {b['pc_std']:.4f}")

    # The width-line WLS critical-exponent estimator is intentionally NOT shown: it carries a
    # ~1-3% method bias (present even on exact-ν lattices), so it's not a defensible interface
    # result. The check below is NOT a universality claim -- it validates the estimator: p_c is
    # measured by crossing a SQUARE window, and left-right vs top-bottom (and the aspect ratio) are
    # arbitrary conventions. p_R = p_D confirms the reported p_c doesn't depend on that choice, which
    # is what justifies the direction-averaged estimator p_A.
    def _bias_block(iso, label):
        lo, hi = iso["d_ci"]
        st.markdown(f"**Direction-bias check — {label} (validates the estimator)**  \n"
                    f"Expected difference if unbiased: p_R − p_D = 0.  \n"
                    f"Estimated difference (L→∞): **{iso['d_inf']:+.5f}**  "
                    f"(95% CI [{lo:+.5f}, {hi:+.5f}])")
        (st.success if iso["isotropic"] else st.warning)(
            f"✓ No directional bias in the {label} threshold — p_R − p_D is consistent with 0 "
            "(95% CI), so the reported p_c is independent of spanning direction."
            if iso["isotropic"]
            else f"Directional bias detected in the {label} threshold — p_R − p_D excludes 0 "
                 "(95% CI); the averaged estimator may be biased for this run.")

    if res.get("isotropy") or res.get("isotropy_bond"):
        st.caption("p_c is read off a square window; left–right vs top–bottom (and the aspect ratio) "
                   "are arbitrary conventions. p_R = p_D confirms the reported p_c doesn't depend on "
                   "that choice — this is an estimator check, not a universality claim.")
        if res.get("isotropy"):      _bias_block(res["isotropy"], "site")
        if res.get("isotropy_bond"): _bias_block(res["isotropy_bond"], "bond")
    else:
        st.caption("Direction-bias check was not recorded for this run.")

    if res.get("exponents"):
        e = res["exponents"]; lo, hi = e["d_f_ci"]; h = e["hyperscaling"]
        st.markdown(r"**Fractal dimension $d_f$ (universality class)**  " + "\n" +
                    r"Incipient spanning cluster: $\langle S_{\max}\rangle \sim L^{d_f}$.  " +
                    rf"$d_f = {e['d_f']:.4f}$  (95% CI $[{lo:.4f}, {hi:.4f}]$; 2D percolation "
                    rf"$= \tfrac{{91}}{{48}} \approx 1.8958$)")
        st.caption(rf"$d_f$ (measured) with $\nu$ (from the crossing width) fix the class; the other "
                   rf"static exponents follow by hyperscaling ($\tau = {h['tau']:.3f}$, "
                   rf"$\gamma/\nu = {h['gamma_nu']:.3f}$, $\beta/\nu = {h['beta_nu']:.4f}$) and are not "
                   rf"measured directly.")
        fdf = figs.df_figure(res)
        if fdf is not None:
            st.pyplot(fdf, width="content")
            st.download_button("Download d_f plot", _fig_bytes(fdf), mime="image/png",
                               file_name=f"{key_prefix}_df.png", key=f"{key_prefix}_dl_df")
        fdc = figs.df_convergence_figure(res)
        if fdc is not None:
            st.caption(r"Convergence check: refit the $d_f$ slope dropping the smallest sizes. With no "
                       r"corrections-to-scaling imposed, the effective exponent drifts to $\tfrac{91}{48}$ "
                       r"on its own as finite-size (small-$L$) points fall away. Solid points are kept on "
                       r"a value-blind error budget (95% CI $\leq 0.02$); the volatile tail (few sizes "
                       r"left) is excluded on the error, not on the value.")
            st.pyplot(fdc, width="content")
            st.download_button("Download d_f convergence plot", _fig_bytes(fdc), mime="image/png",
                               file_name=f"{key_prefix}_df_convergence.png", key=f"{key_prefix}_dl_dfc")

    # correlation-length exponent nu -- shown across the full correction-exponent band (omega is not
    # measurable at these sizes), gated behind a checkbox since it bootstraps three curves.
    if res.get("raw_SI") is not None and sum(1 for L in (res.get("L") or []) if L >= 50) >= 4:
        st.markdown(r"**Correlation-length exponent $\nu$ (universality class)**  " + "\n" +
                    r"$\nu$ sets how fast the finite-size crossing width shrinks, width $\sim L^{-1/\nu}$. "
                    r"It can't be pinned without the correction-to-scaling exponent $\omega$ (not "
                    r"measurable at these sizes), so $\nu$ is shown across the whole plausible band "
                    r"$\omega \in [0.5, 1.5]$: it stays consistent with $\tfrac{4}{3}$ for every $\omega$, "
                    r"and tracks the exact square/triangular lattices analysed identically.")
        if st.checkbox("Compute ν(ω) band  (bootstraps 3 curves, a few seconds)", key=f"{key_prefix}_nucb"):
            fnu = figs.nu_omega_figure(res, controls=_nu_controls(), B=60)
            if fnu is not None:
                st.pyplot(fnu, width="content")
                st.download_button("Download ν plot", _fig_bytes(fnu), mime="image/png",
                                   file_name=f"{key_prefix}_nu.png", key=f"{key_prefix}_dl_nu")

    f1 = figs.fss_figure(res)
    st.pyplot(f1, width="content")
    st.download_button("Download extrapolation plot", _fig_bytes(f1), mime="image/png",
                       file_name=f"{key_prefix}_extrapolation.png", key=f"{key_prefix}_dl_fss")
    f2 = figs.convergence_figure(res)
    st.pyplot(f2, width="content")
    st.download_button("Download convergence plot", _fig_bytes(f2), mime="image/png",
                       file_name=f"{key_prefix}_convergence.png", key=f"{key_prefix}_dl_conv")


def render_job_monitor(s):
    """Progress + controls for the active background job; on completion loads and shows the saved
    result. While the job is live it polls by sleeping briefly then rerunning — but the worker is a
    separate process, so closing the tab or sleeping the machine doesn't stop it."""
    jid = s["job_id"]
    state = s.get("status", "?")
    total = s.get("total") or 0
    frac = (s.get("i", 0) / total) if total else 0.0
    st.progress(min(max(frac, 0.0), 1.0),
                text=f"{state} — {s.get('i', 0)}/{total}  ·  {s.get('last_line', '')}")
    live = state in ("launching", "building", "running", "finalising")
    if live:
        c1, c2 = st.columns([1, 5])
        if c1.button("Stop", key=f"stop_{jid}"):
            jobs.stop_job(jid)
            st.rerun()
        c2.caption("Runs in the background — you can close the tab or let the machine sleep and "
                   "reopen later; it checkpoints after every L and resumes.")
        # Poll for progress, but do the sleep+rerun at the very END of the script (see bottom), NOT
        # here -- this monitor renders inside the Percolate tab, and rerunning here would fire before
        # the later tabs (Analyse saved / Toy demo) get to draw, leaving them blank while a job runs.
        st.session_state["_poll_monitor"] = True
    elif state == "error":
        st.error("Run failed.")
        st.code((s.get("error") or "")[:1200])
        if st.button("Dismiss", key=f"dis_{jid}"):
            st.session_state.pop("active_job", None)
            st.rerun()
    else:   # done | stopped
        if state == "stopped":
            st.warning("Stopped early — the partial result was saved.")
        else:
            st.success("✓ Complete — result saved automatically.")
        rf = s.get("result_file")
        if rf:
            try:
                res, meta = gb.load_saved(rf)
                show_results(res, meta["member"], "job", meta=meta)
                st.caption(f"Saved as paper_results/npz/{rf}")
            except Exception as e:
                st.error(f"Saved but could not load {rf}: {e}")
        else:
            st.info("No result file (fewer than 3 usable sizes).")
        if st.button("Clear this run", key=f"clrdone_{jid}"):
            st.session_state.pop("active_job", None)
            st.rerun()


# ============================================================ sidebar: tiling + graph + seed (shared)
with st.sidebar:
    st.title("Controls")

    st.caption("**Main tilings**")
    tiling = st.selectbox("Tiling", gb.TILINGS, index=0,
                          help="Aperiodic monotiles, the periodic family limits, the Tile(a,b) "
                               "family slider, and the Penrose / triangular validations.")
    st.caption(f"*{gb.CATEGORY[tiling]}*")

    a, b = 1.0, gb.S3
    if tiling == gb.FAMILY:
        member = gb.resolve_member(tiling, a, b)          # (1, √3): the hat, the family's representative
        st.caption("Morph the tile on the Visualise tab by dragging the two edge lengths. Percolation "
                   "depends only on the adjacency, so every generic member runs identically to the hat; "
                   "to percolate an endpoint, pick Comet or Chevron directly.")
    else:
        member = tiling

    graph_opts = gb.GRAPHS_FOR.get(member, gb.GRAPHS)
    graph_type = st.radio("Graph", graph_opts, horizontal=True,
                          help="Direct = tile corners are sites, tile edges are bonds. "
                               "Dual = tiles are sites, adjacent tiles are bonded.")
    if len(graph_opts) == 1:
        st.caption(f"({member} admits only the {graph_opts[0].split()[0].lower()} graph.)")
    # Right next to the Direct/Dual choice, because toggling the graph here + this overlay is exactly
    # what shows the difference between the two graphs on the Visualise tab.
    show_graph = st.checkbox("Overlay graph on the tiling",
                             help="Shows what the direct and dual consider nodes and edges "
                                  "for the given tiling.")

    st.session_state.setdefault("perc_seed", 123456789)
    seed = int(st.number_input("Seed", step=1, key="perc_seed"))
    st.divider()
    if st.button("Recalibrate timer", help="Re-measure the per-trial cost on this machine; "
                                           "the run-time estimates then use it."):
        with st.spinner("Calibrating…"):
            c = estimate.calibrate()
        per = estimate._c_per_node(c, 100_000) * 1e9   # ns per node·trial at a ~100k-node frame
        st.success(f"Recalibrated — estimates now use this machine (~{per:.0f} ns/node·trial "
                   "at a 100k-node frame).")


st.title("Aperiodic Monotile Percolation Portal")
runnable = member is not None
if not runnable:
    st.stop()

tab_vis, tab_run, tab_analyse, tab_toy = st.tabs(
    ["Visualise", "Percolate", "Analyse saved", "Toy demo"])

# ============================================================ TOY DEMO (percolation walk-through)
with tab_toy:
    # ---------------------------------------------------------- What is percolation? (the grid)
    st.subheader("What is percolation?")
    st.markdown(
        "Percolation studies how connected a graph is. We open parts of the lattice and check whether "
        "a connected component forms that spans the entire grid. The point at which it does is sudden "
        "and discrete. On an infinite grid that point is the true critical value; on a finite one, like "
        "the one below, we can estimate it. The left-right and top-bottom crossings let us put a rough "
        "bound on what the infinite lattice would give. The true values for the square are 0.5 (bond) "
        "and 0.5927 (site), so you can see how close we get!")

    s1, s2, _sp = st.columns([1, 1, 3])
    lattice = s1.radio("Lattice", ["Square", "Hat"], horizontal=True, key="demo_lattice").lower()
    mode = s2.radio("Open", ["Site", "Bond"], horizontal=True, key="demo_mode",
                    help="Site opens nodes; bond opens edges.").lower()

    demo = cached_demo(lattice)
    kmax = demo["N"] if mode == "site" else demo["M"]
    cvals = [v for v in demo["cross_" + mode].values() if v]
    jump = min(cvals) if cvals else kmax                      # first step it spans (either direction)

    demo_component(demo, mode, jump)                          # smooth client-side slider on the right
    if len(cvals) == 2:
        k_union, k_inter = min(cvals), max(cvals)
        est = 0.5 * (k_union + k_inter) / kmax
        st.caption(f"left-right / top-bottom crossings bracket it: p ≈ {k_union / kmax:.2f} to "
                   f"{k_inter / kmax:.2f}, estimate ≈ {est:.2f}"
                   + ("  (true: 0.5 bond, 0.5927 site)" if lattice == "square" else ""))

    # universality: same class -> two exponents, nu and d_f
    st.markdown("---")
    st.markdown(
        r"Despite having different critical points, all percolations in the same “universality class” "
        r"act the same as they approach this threshold. This universality class can be uniquely "
        r"determined by two constants, $\nu$ and $d_f$. These need to be $\tfrac{4}{3}$ and "
        r"$\tfrac{91}{48}$ respectively. The following show you what these two represent.")

    # ---------------------------------------------------------- The Critical Scaling Exponent (nu)
    st.subheader("The Critical Scaling Exponent")
    st.markdown(
        r"$\nu$ controls how a finite grid's crossing point closes in on the true threshold: each grid "
        r"spans a little late, and the shortfall $p_c - p^{*}(L)$ shrinks as $L^{-1/\nu}$. Drag the "
        r"slider, and every grid drops a dot where it first spans; the dots climb toward $p_c$ along "
        r"that curve. Draw the $\nu = \tfrac{4}{3}$ line and it threads them.")
    nu_component(cached_nu())

    # ---------------------------------------------------------- The Fractal Dimension (d_f)
    st.subheader("The Fractal Dimension")
    st.markdown(
        r"At the critical point where it percolates, the large percolating cluster doesn't connect "
        r"every single open site; instead it contains a fraction. This fraction decreases as the grid "
        r"gets larger; the rate at which this decreases corresponds to $d_f$. Whilst it isn't exactly "
        r"$d_f$ (it is $d_f - d$, so $d_f - 2$ for 2D percolation) it shows what it does. Drag the "
        r"slider to the percolation point and the ten grids line up on that slope.")
    scaling_component(cached_scaling())

# ============================================================ VISUALISE engine
with tab_vis:
    st.subheader(f"Generate — {member}")
    if tiling == gb.FAMILY:
        # Interactive Tile(a,b) morph: reconstructs the tile live as you shrink an edge-class, so you
        # can walk the whole family (hat / turtle / spectre) and its two collapse endpoints.
        st.markdown(
            r"The hat is one member of a continuous family, Tile$(a,b)$, set by its two edge lengths: "
            r"$a$ for the unit edges (blue) and $b$ for the $\sqrt{3}$ edges (red). The other aperiodic "
            r"monotiles live here too, the spectre is Tile$(1,1)$ and the turtle is Tile$(\sqrt{3},1)$; "
            r"shrink an edge-class to zero and you reach an endpoint, $b \to 0$ the comet and $a \to 0$ "
            r"the chevron. Drag the sliders or hit a preset to morph the tile. The sidebar $a$/$b$ "
            r"sliders pick which concrete member the other tabs render and percolate.")
        collapse_component(cached_collapse())
    else:
        left, right = st.columns([3, 1], gap="large")
        with right:
            rlabel, rlo, rhi, rdflt = gb.RENDER_CTL[member]
            size = st.slider(rlabel, rlo, rhi, rdflt, key="vis_size",
                             help="Render size only, kept small so the figure stays legible. "
                                  "The percolation patch (other tab) can be much larger.")
            st.caption("The visualisation is generated using the same code as the percolation engine. It "
                       "is thus used to verify that the transform used for this algorithm is indeed "
                       "correct. Any mistakes in substitutes show as gaps or discrepancies in the tiling.")
        with left:
            try:
                fig, ntiles, gcounts = cached_visualise(tiling, size, round(a, 3), round(b, 3),
                                                        graph_type, show_graph)
                st.pyplot(fig, width="content")
                cap = f"{ntiles:,} tiles"
                if gcounts:
                    cap += f"  ·  graph: {gcounts[0]:,} nodes, {gcounts[1]:,} edges"
                st.caption(cap)
                st.download_button("Download image", _fig_bytes(fig), mime="image/png",
                                   file_name=f"{member}_tiling.png", key="vis_dl")
            except Exception as e:
                st.warning(f"render unavailable: {e}")

# ============================================================ PERCOLATE engine
with tab_run:
    st.subheader(f"Percolate — {member} · {graph_type.split()[0].lower()} graph")

    with st.expander("What the percolation engine does"):
        st.markdown(
            "We first cut an L×L square window and then run a Monte-Carlo simulation to estimate the "
            "percolation threshold of a given size, employing the Newman-Ziff algorithm. A finite-size "
            "scaling analysis allows us to extrapolate these estimates towards the critical value of "
            "infinity. Further detail is provided in the codebase.")

    kind = "dual" if graph_type.startswith("Dual") else "direct"
    plabel, plo, phi, pdflt, phelp = gb.PATCH_CTL[member]

    def _fmt_eta(s):
        # Finer bands than before: the ETA now includes graph build + per-size frame cost and
        # validates to ~1.1x on the main (hat vertex) case, so each range confidently brackets the
        # estimate's residual error. Still ranges, not exact minutes -- that slop is real.
        if s < 10:    return "a few seconds"
        if s < 30:    return "under half a minute"
        if s < 60:    return "under a minute"
        if s < 120:   return "1–2 minutes"
        if s < 300:   return "2–5 minutes"
        if s < 600:   return "5–10 minutes"
        if s < 1200:  return "10–20 minutes"
        if s < 1800:  return "20–30 minutes"
        if s < 2700:  return "30–45 minutes"
        if s < 3600:  return "45–60 minutes"
        if s < 7200:  return "1–2 hours"
        if s < 14400: return "2–4 hours"
        if s < 28800: return "4–8 hours"
        return "8+ hours"

    def _fmt_count(n):
        if n >= 1_000_000: return f"{n/1e6:.1f}M"
        if n >= 1_000:     return f"{n/1e3:.0f}k"
        return str(int(n))

    patch_key = f"perc_patch_{member}"   # per-member so switching tilings can't leave an out-of-range value
    cset, cnote = st.columns([1, 1], gap="large")
    with cset:
        # One-click "the paper run" for the headline objects -- this is where most of the focus is.
        preset = gb.paper_preset(member, kind)
        if preset:
            def _apply_preset(preset=preset, patch_key=patch_key):
                # Set widget state in an on_click CALLBACK: it runs before any widget is
                # (re-)instantiated this run, so it may set even the sidebar's seed, which is created
                # earlier in the script than this button (inline setting there raises).
                st.session_state[patch_key] = preset["patch"]
                st.session_state["perc_lmin"] = preset["L_min"]
                st.session_state["perc_lmax"] = preset["L_max"]
                st.session_state["perc_gap"] = preset["gap"]
                st.session_state["perc_trials"] = preset["T"]
                st.session_state["perc_seed"] = preset["seed"]
            st.button(f"Use paper settings ({member})", on_click=_apply_preset,
                      help="Load the exact production parameters this object was run with for the paper.")

        st.session_state.setdefault(patch_key, pdflt)
        patch = st.slider(plabel, plo, phi, key=patch_key)
        # Size + ETA come from PRECOMPUTED GEOMETRY -- no graph is built here, so previewing even the
        # r=6 patch is instant. The (heavy) build happens only when you hit Run.
        ge = estimate.geometry(member, kind, patch)
        usable_max = (ge[1] * 0.9) if ge else 200.0
        st.caption(f"Largest useful **L** for this patch ≈ **{usable_max:.0f}** "
                   "(bigger windows fall outside the tiling and are skipped).")

        st.markdown("**Parameters**")
        st.session_state.setdefault("perc_lmin", 10.0)
        st.session_state.setdefault("perc_lmax", float(max(20, round(usable_max))))
        st.session_state.setdefault("perc_gap", 20.0)
        st.session_state.setdefault("perc_trials", 200)
        ra, rb = st.columns(2)
        L_min = ra.number_input("L min", min_value=2.0, step=1.0, key="perc_lmin",
                                help="Smallest square-window side length in the L sweep.")
        L_max = rb.number_input("L max", min_value=3.0, step=1.0, key="perc_lmax",
                                help="Largest square-window side length in the L sweep.")
        rc, rd = st.columns(2)
        gap = rc.number_input("Gap  ΔL", min_value=1.0, step=1.0, key="perc_gap",
                              help="Spacing between window sizes.")
        T = rd.number_input("Trials per L", min_value=1, step=50, key="perc_trials",
                            help="Monte-Carlo Trials per L. More trials means smaller error bars.")

    # L list from the typed min / max / gap -- the SHARED engine definition, so this preview and the
    # runner sweep the identical sizes.
    Ls = l_sweep(L_min, L_max, gap)

    plan = estimate.plan_run(member, kind, patch, Ls, T) if Ls else None
    n_used = plan["n_usable"] if plan else 0
    eta = plan["eta"] if plan else 0.0
    accuracy = estimate.accuracy_estimate(member, kind, patch, Ls, T) if Ls else "—"

    with cnote:
        ref = gb.REFERENCE.get((member, kind))
        if ref:
            st.caption(f"Reference — {ref}")

    # --- readout across the FULL width so nothing truncates ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Graph nodes ~", _fmt_count(plan["nodes_est"]) if plan else "—",
              help=(f"~{plan['nodes_est']:,} nodes, estimated from geometry (not yet built); "
                    "built when you Run." if plan else "Set a valid sweep."))
    m2.metric("Usable L", f"{n_used} / {len(Ls)}")
    m3.metric("Est. run time", _fmt_eta(eta),
              help="Estimated from geometry + this machine's calibration: graph build + the sweep + "
                   "a few seconds of worker startup (fresh process import + numba warm-up; the very "
                   "first run after a cold cache can be longer). A range, not an exact time. Hit "
                   "Recalibrate timer if it drifts.")
    m4.metric("Est. accuracy", accuracy,
              help="Rough band for the digits of p_c this L-range and trial count buy. "
                   "Small patches are also finite-size biased (the value itself is off).")

    # guard-rail warnings. Only flag genuinely long (>20 min) runs loudly.
    if n_used < 3:
        st.error("Fewer than 3 usable L values — raise the patch/block size, lower L min, "
                 "or shrink the gap.")
    elif eta > 1200:
        st.error(f"≈ {_fmt_eta(eta)} — that's a long run. Raise the gap or cut trials, "
                 "or let it run (it's the production-scale sweep).")
    elif eta > 300:
        st.warning(f"≈ {_fmt_eta(eta)} — heads up before you hit Run.")

    # --- background run: a DETACHED worker process owns the sweep, so it survives the tab closing,
    #     the machine sleeping (OS suspend/resume), and even this server dying; it checkpoints after
    #     every L and resumes. We just launch it and poll its status file. The result auto-saves on
    #     completion (an unattended overnight run must persist itself). ---
    st.markdown("---")
    # Auto-reattach: if a run is still going (this tab was closed/reopened, or the server restarted),
    # pick it up so the progress bar comes right back.
    if not st.session_state.get("active_job"):
        _live = [j for j in jobs.list_jobs()
                 if j.get("status") in ("launching", "building", "running", "finalising")]
        if _live:
            st.session_state["active_job"] = _live[0]["job_id"]

    active = st.session_state.get("active_job")
    _s = jobs.read_job_status(active) if active else None
    _busy = bool(_s and _s.get("status") in ("launching", "building", "running", "finalising"))

    want_exp = st.checkbox("Also measure d_f (extra cluster pass, ~+20% time)", value=False,
                           help="Largest-cluster pass: records the incipient-cluster size per trial -> the "
                                "fractal dimension d_f (<s_max> ~ L^d_f), one of the two exponents that "
                                "fix the universality class. Off by default to keep runs fast; the "
                                "paper's universality runs turn it on.")
    if st.button("Run percolation", type="primary", disabled=(n_used < 3) or _busy):
        jid = jobs.launch_job(tiling, member, graph_type, kind, patch, round(a, 3), round(b, 3),
                            L_min, L_max, gap, int(T), int(seed), exponents=want_exp)
        st.session_state["active_job"] = jid
        st.rerun()
    if not _busy:
        st.caption("Launches a background process — it keeps running if you close the tab or the "
                   "machine sleeps. Result auto-saves to paper_results/npz/ when done.")

    # The exact console command for the current settings — the Run button just launches this. Handy
    # for a headless box, a machine you won't keep the browser open on, or scripting a batch.
    with st.expander("Run from a console instead"):
        _gsh = "dual" if kind == "dual" else "direct"
        _parts = [f'--tiling "{tiling}"', f"--graph {_gsh}"]
        if tiling == gb.FAMILY:
            _parts.append(f"--a {round(a, 3):g} --b {round(b, 3):g}")
        _parts += [f"--patch {patch}", f"--lmin {L_min:g}", f"--lmax {L_max:g}", f"--gap {gap:g}",
                   f"--trials {int(T)}", f"--seed {int(seed)}"]
        if want_exp:
            _parts.append("--exponents")
        st.code("python runner/runner.py " + " ".join(_parts), language="bash")
        st.caption("Same computation as the button (identical kernels, seed and results). Run it "
                   "from the project folder. It checkpoints and resumes if interrupted, and saves "
                   "to paper_results/npz/ — so it also shows up under 'Analyse saved' and in the "
                   "background-jobs list above.")

    # Other jobs (finished, or running from another session) — reattach or clean up. Rendered BEFORE
    # the live monitor so it stays visible while the monitor is polling.
    others = [j for j in jobs.list_jobs() if j.get("job_id") != active]
    if others:
        with st.expander(f"Background jobs ({len(others)})"):
            for j in others:
                c0, c1, c2 = st.columns([4, 1, 1])
                c0.write(f"**{j.get('member', '?')} · {j.get('kind', '')}** — "
                         f"{j.get('status', '?')} ({j.get('i', 0)}/{j.get('total', 0)})")
                if c1.button("Monitor", key=f"mon_{j['job_id']}"):
                    st.session_state["active_job"] = j["job_id"]
                    st.rerun()
                if c2.button("Clear", key=f"clr_{j['job_id']}"):
                    jobs.clear_job(j["job_id"])
                    st.rerun()

    if active and _s:
        st.divider()
        render_job_monitor(_s)
    elif active and _s is None:
        st.session_state.pop("active_job", None)   # stale reference; forget it


# ============================================================ ANALYSE SAVED
with tab_analyse:
    st.subheader("Analyse a saved run")
    st.caption("Loads saved runs from the paper_results/npz/ folder and re-derives every result "
               "from the raw per-trial data.")
    saved = gb.list_saved()
    if not saved:
        st.info("No saved runs found in paper_results/npz/. Run something on the Percolate tab, or "
                "run runner/runner.py from a console (see REPRODUCE.md).")
    else:
        pick = st.selectbox("Saved run", saved, key="analyse_pick")
        try:
            res, meta = gb.load_saved(pick)
            show_results(res, meta["member"], "an", meta=meta)
        except Exception as e:
            st.error(f"Could not load {pick}: {e}")


# ============================================================ background-job poll (MUST be last)
# The Percolate tab's job monitor asks to auto-refresh by setting _poll_monitor. We do the actual
# sleep+rerun HERE, after every tab has rendered -- so a running job never pre-empts the later tabs
# (Analyse saved / Toy demo) from drawing. (A GUI Run, or any console `runner.py` writing to the
# shared jobs/ dir, triggers this via auto-reattach.)
if st.session_state.pop("_poll_monitor", False):
    time.sleep(2)
    st.rerun()
