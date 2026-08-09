"""Interactive percolation portal for the aperiodic monotile family and its validations.

    streamlit run interface/gui_app.py

Three tabs behind one tiling picker:
  * Visualise  — generate the chosen tiling at a size you control and (optionally) overlay the exact
                 percolation graph, i.e. proof the generator produces the real shape + lattice.
  * Percolate  — launch a run (any patch up to production), sweep L, finite-size-extrapolate the
                 site & bond thresholds, report p_c, check for directional bias, and optionally
                 measure the fractal dimension d_f. Runs auto-save.
  * Analyse saved — reload any run from results_output/ (saved here or by runner/percolate.py)
                 and show the identical analysis.

All numerics come from the same builders/kernels the runner uses (via gui_backend); the Percolate
tab launches runner/percolate.py in the background. Converged production numbers come from the same
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
  const fy=y=>(y0+y1)-y,u=(x1-x0)/g.n,R=0.40*u;
  const nEl=g.coords.map(c=>{const ci=mk("circle",{cx:c[0],cy:fy(c[1]),r:R});svg.appendChild(ci);return ci;});
  const lbl=document.createElement("div"); lbl.className="sclbl"; lbl.textContent="L="+g.n;
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
   if(open[v]&&find(v)===bigR){ci.setAttribute("fill",GOLD);ci.setAttribute("stroke","#111");ci.setAttribute("stroke-width",G3.R*0.25);}
   else if(open[v]){ci.setAttribute("fill","#a9cbe8");ci.removeAttribute("stroke");}
   else{ci.setAttribute("fill","#e8e8ee");ci.removeAttribute("stroke");}}
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
      stroke:f===0?GOLD:"none","stroke-width":f===0?1.4:0,"fill-opacity":0.55}));
   }
   const m=sum/F; means.push(m);
   dyn.appendChild(mk("circle",{cx:SX(g.n),cy:SY(m),r:4.2,fill:BLUE,stroke:"#fff","stroke-width":1}));
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
   msg="<b>Below</b> the percolation point: the largest cluster is a finite blob, so its share <b>falls steeply</b> as the grid grows (the points slope down far faster than the dashed line). Not yet critical.";
  }else if(p<=pc+0.03){
   msg="<b>At</b> the percolation point the ten means sit on a <b>straight line</b> parallel to the 2D reference — the share is nearly the same across a 5&times; range of sizes (scale invariance). Its slope gives <b>d_f = "+df.toFixed(2)+"</b> (2D exact 91/48 = 1.90). Faint dots are the "+F+" individual fillings behind each mean — the run-to-run spread is real and never shrinks, which is exactly why one grid can't settle this and the average can.";
  }else{
   msg="<b>Above</b> the percolation point: the largest cluster is extensive, so its share <b>flattens</b> toward a constant (slope &rarr; 0, d_f &rarr; 2) — it fills a fixed fraction of every grid.";
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
 .nurow{display:flex;flex-wrap:wrap;gap:14px;justify-content:center;align-items:flex-start;}
 .nucol{display:flex;flex-direction:column;align-items:center;}
 .nulbl{font-size:12px;margin-top:3px;text-align:center;line-height:1.3;font-weight:600;}
 .nusvg{background:#fff;border:1px solid #eee;}
 .nuctl{display:flex;gap:10px;align-items:center;margin:8px 4px 4px;flex-wrap:wrap;}
 .nub{padding:5px 11px;border:1px solid #ccc;border-radius:6px;background:#f5f5f7;cursor:pointer;font-size:13px;}
 .nusl{flex:1;min-width:150px;} .nusum{font-size:13.5px;margin:8px 2px 0;line-height:1.45;}
 .nuq{display:inline-block;height:9px;border-radius:5px;background:#e6e6ee;width:110px;vertical-align:middle;overflow:hidden;}
 .nuqf{height:100%;background:#2e9e5b;width:0%;}
</style>
<div class="nuw">
 <div class="nustage"><span class="nustep">1</span>Drag the occupation up and watch each grid percolate — the largest cluster (gold) grows until it <b>spans across</b> (turns green). Hit <b>new filling</b> a few times: the point where it spans <b>jumps around wildly on the small grid</b>, <b>barely moves on the big one</b>.</div>
 <div class="nurow" id="nu_grids"></div>
 <div class="nuctl">
  <span>occupation p = <b id="nu_p">0.45</b></span>
  <input class="nusl" id="nu_psl" type="range" min="45" max="74" value="45" title="open more sites">
  <button class="nub" id="nu_shuf">&#127922; new filling</button>
 </div>

 <div class="nustage"><span class="nustep">2</span>Repeat over hundreds of fillings: the <b>fraction that have spanned</b> by occupation p is the curve R(p). The blue line marks your p from above; bigger grids switch on more <b>sharply</b>.</div>
 <div class="nucol"><svg class="nusvg" id="nu_raw" width="340" height="210"></svg></div>

 <div class="nustage"><span class="nustep">3</span>That sharpness is set by one exponent ν. Rescale the width by L<tspan>^</tspan>(1/ν) and <b>every size lands on a single master curve</b>. Drag ν to collapse them.</div>
 <div class="nucol"><svg class="nusvg" id="nu_col" width="340" height="220"></svg></div>
 <div class="nuctl">
  <button class="nub" id="nu_43">ν = 4/3</button>
  <button class="nub" id="nu_best">best collapse</button>
  <span>ν = <b id="nu_val">2.00</b></span>
  <input class="nusl" id="nu_sl" type="range" min="100" max="220" value="200" title="Drag to rescale">
  <span>collapse&nbsp;<span class="nuq"><span class="nuqf" id="nu_qf"></span></span></span>
 </div>
 <div class="nusum" id="nu_sum"></div>
</div>
<script>
const G=__DATA__;
(function(){
 const NS="http://www.w3.org/2000/svg";
 const mk=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
 const sizes=G.sizes, pg=G.pgrid, pc=G.pc, grids=G.grids, live=G.live, S=sizes.length;
 const PAL=["#4575b4","#74add1","#f6ad55","#f46d43","#d73027","#7b3294"];
 const col=i=>PAL[i% PAL.length];
 const idxOf=n=>Math.max(0,sizes.indexOf(n));
 const el=id=>document.getElementById(id);
 const GOLD="#f4b400", GREEN="#2e9e5b";

 // ---- (1) live percolating grids: drag occupation, watch the largest cluster span ----
 const holder=el("nu_grids");
 const LG=live.map(g=>{
  const box=132, cw=document.createElement("div"); cw.className="nucol";
  const svg=document.createElementNS(NS,"svg"); svg.setAttribute("width",box);svg.setAttribute("height",box);svg.setAttribute("class","nusvg");
  const b=g.bbox,x0=b[0],x1=b[1],y0=b[2],y1=b[3],pad=(x1-x0)*0.03;
  svg.setAttribute("viewBox",(x0-pad)+" "+(y0-pad)+" "+((x1-x0)+2*pad)+" "+((y1-y0)+2*pad));
  const fy=y=>(y0+y1)-y,u=(x1-x0)/g.n,R=0.42*u;
  const nEl=g.coords.map(c=>{const ci=mk("circle",{cx:c[0],cy:fy(c[1]),r:R});svg.appendChild(ci);return ci;});
  const neigh=Array.from({length:g.N},()=>[]);
  g.edges.forEach(e=>{neigh[e[0]].push(e[1]);neigh[e[1]].push(e[0]);});
  const lbl=document.createElement("div"); lbl.className="nulbl";
  cw.appendChild(svg);cw.appendChild(lbl);holder.appendChild(cw);
  return {g,nEl,R,lbl,neigh,pcross:0,setT:new Set(g.top),setB:new Set(g.bottom),setL:new Set(g.left),setR:new Set(g.right)};
 });
 let ord=0;
 // this filling's crossing point: the occupation at which THIS grid first spans (jitters run-to-run).
 function computeCross(G3){
  const g=G3.g,N=g.N,order=g.orders[ord%g.orders.length],neigh=G3.neigh,T=1,B=2,Lb=4,Rb=8;
  const fm=new Int8Array(N);
  G3.setT.forEach(v=>fm[v]|=T);G3.setB.forEach(v=>fm[v]|=B);G3.setL.forEach(v=>fm[v]|=Lb);G3.setR.forEach(v=>fm[v]|=Rb);
  const par=new Int32Array(N);for(let i=0;i<N;i++)par[i]=i;const sz=new Int32Array(N).fill(1);const open=new Uint8Array(N);
  function find(x){let r=x;while(par[r]!==r)r=par[r];while(par[x]!==r){const n=par[x];par[x]=r;x=n;}return r;}
  for(let step=0;step<N;step++){const v=order[step];open[v]=1;let rv=find(v);
   for(const nb of neigh[v]){if(open[nb]){let ra=rv,rb=find(nb);if(ra!==rb){if(sz[ra]<sz[rb]){const tt=ra;ra=rb;rb=tt;}par[rb]=ra;sz[ra]+=sz[rb];fm[ra]|=fm[rb];rv=ra;}}}
   const m=fm[rv];if((m&(T|B))===(T|B)||(m&(Lb|Rb))===(Lb|Rb))return (step+1)/N;}
  return 1;
 }
 const refreshCross=()=>LG.forEach(G3=>{G3.pcross=computeCross(G3);});
 function drawLive(gi,p){
  const G3=LG[gi],g=G3.g,N=g.N,edges=g.edges,order=g.orders[ord%g.orders.length],k=Math.min(Math.round(p*N),N);
  const par=new Int32Array(N);for(let i=0;i<N;i++)par[i]=i;
  const open=new Uint8Array(N);
  function find(x){let r=x;while(par[r]!==r)r=par[r];while(par[x]!==r){const n=par[x];par[x]=r;x=n;}return r;}
  for(let i=0;i<k;i++)open[order[i]]=1;
  for(let e=0;e<edges.length;e++){const a=edges[e][0],b=edges[e][1];if(open[a]&&open[b]){const ra=find(a),rb=find(b);if(ra!==rb)par[ra]=rb;}}
  const sz=new Int32Array(N);let bigR=-1,bigN=0;
  for(let v=0;v<N;v++){if(open[v]){const r=find(v);sz[r]++;if(sz[r]>bigN){bigN=sz[r];bigR=r;}}}
  let t=false,bt=false,l=false,r=false;
  if(bigR>=0)for(let v=0;v<N;v++){if(open[v]&&find(v)===bigR){if(G3.setT.has(v))t=true;if(G3.setB.has(v))bt=true;if(G3.setL.has(v))l=true;if(G3.setR.has(v))r=true;}}
  const spans=(t&&bt)||(l&&r), cc=spans?GREEN:GOLD;
  for(let v=0;v<N;v++){const ci=G3.nEl[v];
   if(open[v]&&find(v)===bigR){ci.setAttribute("fill",cc);ci.setAttribute("stroke","#111");ci.setAttribute("stroke-width",G3.R*0.2);}
   else if(open[v]){ci.setAttribute("fill","#a9cbe8");ci.removeAttribute("stroke");}
   else{ci.setAttribute("fill","#e8e8ee");ci.removeAttribute("stroke");}}
  G3.lbl.innerHTML="<span style='color:"+col(idxOf(g.n))+"'>L = "+g.n+"</span> "
   +(spans?"<b style='color:#2e9e5b'>SPANS ✓</b>":"<span style='color:#aaa'>not yet</span>")
   +"<br><span style='font-weight:400;color:#777'>this filling spans at p ≈ <b>"+G3.pcross.toFixed(2)+"</b></span>";
 }

 // ---- (2) sigmoid R(p,L) with a movable p-line synced to the occupation slider ----
 const rsvg=el("nu_raw"),rW=340,rH=210,rmL=34,rmR=10,rmT=12,rmB=30;
 const rx0=pg[0],rx1=pg[pg.length-1];
 const RSX=p=>rmL+(Math.min(Math.max(p,rx0),rx1)-rx0)/(rx1-rx0)*(rW-rmL-rmR), RSY=r=>rH-rmB-r*(rH-rmT-rmB);
 rsvg.appendChild(mk("rect",{x:rmL,y:rmT,width:rW-rmL-rmR,height:rH-rmT-rmB,fill:"#fff",stroke:"#eee"}));
 [0,0.5,1].forEach(r=>{rsvg.appendChild(mk("line",{x1:rmL,y1:RSY(r),x2:rW-rmR,y2:RSY(r),stroke:"#f0f0f2"}));
  const t=mk("text",{x:rmL-4,y:RSY(r)+3,"text-anchor":"end","font-size":9,fill:"#999"});t.textContent=r;rsvg.appendChild(t);});
 rsvg.appendChild(mk("line",{x1:RSX(pc),y1:rmT,x2:RSX(pc),y2:rH-rmB,stroke:"#ccc","stroke-dasharray":"2 2"}));
 grids.forEach((g,i)=>{let d="";g.R.forEach((r,k)=>{d+=(k?"L":"M")+RSX(pg[k]).toFixed(1)+" "+RSY(r).toFixed(1)+" ";});
  rsvg.appendChild(mk("path",{d:d,fill:"none",stroke:col(i),"stroke-width":1.7}));});
 rsvg.appendChild(mk("text",{x:(rmL+rW-rmR)/2,y:rH-4,"text-anchor":"middle","font-size":10,fill:"#666"})).textContent="occupation  p    (curve colour = grid size)";
 const pLine=mk("line",{y1:rmT,y2:rH-rmB,stroke:"#2e6db4","stroke-width":1.5});rsvg.appendChild(pLine);
 const setPLine=p=>{const x=RSX(p);pLine.setAttribute("x1",x);pLine.setAttribute("x2",x);};

 // ---- (3) collapse R vs s=(p-p_c(L)) L^{1/nu}, dynamic ----
 const svg=el("nu_col"),W=340,H=220,mL=34,mR=10,mT=12,mB=34;
 const SYr=r=>H-mB-r*(H-mT-mB);
 const dyn=mk("g",{});svg.appendChild(dyn);
 // Each grid is centred on its OWN crossing point (the mean drifts with L) and rescaled by L^(1/nu).
 const cen=grids.map(g=>g.mean);
 const pAt=(R,f)=>{let k=1;while(k<R.length-1&&R[k]<f)k++;const t=(f-R[k-1])/((R[k]-R[k-1])||1);return pg[k-1]+t*(pg[k]-pg[k-1]);};
 const width=grids.map(g=>pAt(g.R,0.8)-pAt(g.R,0.2));     // 20-80 transition width per size (nu-free)
 const sOf=(p,i,nu)=>(p-cen[i])*Math.pow(sizes[i],1.0/nu);
 // Collapse score = spread of the rescaled widths (can't be gamed by shrinking a comparison window):
 // it is minimal exactly when every grid's width scales as L^(-1/nu) with the SAME nu.
 function residual(nu){
  const x=grids.map((g,i)=>width[i]*Math.pow(sizes[i],1.0/nu));
  const m=x.reduce((a,b)=>a+b,0)/S; let v=0; x.forEach(y=>{v+=(y-m)*(y-m);}); return (v/S)/(m*m || 1);
 }
 // one scan: best nu + residual range, so the quality bar auto-calibrates
 let best=2.0,br=1e9,worst=0;
 for(let v=100;v<=220;v++){const nu=v/100,r=residual(nu);if(r<br){br=r;best=nu;}if(r>worst)worst=r;}
 function render(nu){
  while(dyn.firstChild)dyn.removeChild(dyn.firstChild);
  const sArr=grids.map((g,i)=>pg.map(p=>sOf(p,i,nu)));
  const smax=Math.max(...sArr.map(a=>Math.max(Math.abs(a[0]),Math.abs(a[a.length-1]))))*1.05||1;
  const SX=s=>mL+(s+smax)/(2*smax)*(W-mL-mR);
  dyn.appendChild(mk("rect",{x:mL,y:mT,width:W-mL-mR,height:H-mT-mB,fill:"#fff",stroke:"#eee"}));
  [0,0.5,1].forEach(r=>dyn.appendChild(mk("line",{x1:mL,y1:SYr(r),x2:W-mR,y2:SYr(r),stroke:"#f0f0f2"})));
  dyn.appendChild(mk("line",{x1:SX(0),y1:mT,x2:SX(0),y2:H-mB,stroke:"#ddd","stroke-dasharray":"3 3"}));
  grids.forEach((g,i)=>{let d="";g.R.forEach((r,k)=>{d+=(k?"L":"M")+SX(sArr[i][k]).toFixed(1)+" "+SYr(r).toFixed(1)+" ";});
   dyn.appendChild(mk("path",{d:d,fill:"none",stroke:col(i),"stroke-width":1.8,"stroke-opacity":0.85}));});
  dyn.appendChild(mk("text",{x:(mL+W-mR)/2,y:H-4,"text-anchor":"middle","font-size":10,fill:"#666"})).textContent="(p - p_c(L)) · L^(1/ν)";
  el("nu_val").textContent=nu.toFixed(3);
  const res=residual(nu), q=Math.max(0,Math.min(100,100*(worst-res)/((worst-br)||1)));
  el("nu_qf").style.width=q.toFixed(0)+"%";
  let msg;
  if(nu>1.72)msg="High &nu;: barely rescaled — the curves stay <b>separate</b> (bigger grids are steeper). Drag &nu; down.";
  else if(nu<1.18)msg="Low &nu;: <b>over-shot</b> — the rescaling has pushed the curves apart the other way.";
  else msg="The size curves <b>collapse onto one master curve</b> — a single exponent &nu; rescales every grid the same way. That &nu; is the correlation-length exponent.";
  el("nu_sum").innerHTML=msg+" <span style='color:#888'>On these small demo grids the best collapse sits near &nu;&nbsp;&asymp;&nbsp;"+best.toFixed(1)+"; at larger scales it converges to the exact 2D value <b>&nu; = 4/3</b>.</span>";
 }
 // occupation slider drives (1) the live grids and (2) the p-line on the sigmoid
 const psl=el("nu_psl");
 function occ(p){el("nu_p").textContent=p.toFixed(3);LG.forEach((_,gi)=>drawLive(gi,p));setPLine(p);}
 psl.addEventListener("input",()=>occ(+psl.value/100));
 el("nu_shuf").addEventListener("click",()=>{ord++;refreshCross();occ(+psl.value/100);});
 // nu slider drives (3) the collapse
 const sl=el("nu_sl");
 sl.addEventListener("input",()=>render(+sl.value/100));
 el("nu_43").addEventListener("click",()=>{sl.value=133;render(1.333);});
 el("nu_best").addEventListener("click",()=>{sl.value=Math.round(best*100);render(best);});
 refreshCross(); occ(0.45); render(2.0);
})();
</script>
"""


def nu_component(demo):
    """Render the interactive correlation-length (nu) collapse toy. `demo` is build_nu_demo()'s dict
    (sizes, pgrid, pc, per-size spanning-probability R). Client-side; the nu slider rescales the axis."""
    components.html(_NU_HTML.replace("__DATA__", json.dumps(demo)), height=915)


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
        st.markdown("**Fractal dimension d_f (universality class)**  \n"
                    "Incipient spanning cluster: ⟨S_max⟩ ~ L^d_f.  "
                    f"**d_f = {e['d_f']:.4f}**  (95% CI [{lo:.4f}, {hi:.4f}]; 2D percolation = 91/48 ≈ 1.8958)")
        st.caption("d_f (measured) with ν (from the crossing width) fix the class; the other static "
                   f"exponents follow by hyperscaling — τ = {h['tau']:.3f}, γ/ν = {h['gamma_nu']:.3f}, "
                   f"β/ν = {h['beta_nu']:.4f} — and are not measured directly.")
        fdf = figs.df_figure(res)
        if fdf is not None:
            st.pyplot(fdf, width="content")
            st.download_button("Download d_f plot", _fig_bytes(fdf), mime="image/png",
                               file_name=f"{key_prefix}_df.png", key=f"{key_prefix}_dl_df")
        fdc = figs.df_convergence_figure(res)
        if fdc is not None:
            st.caption("Convergence check: refit the d_f slope dropping the smallest sizes. With no "
                       "corrections-to-scaling imposed, the effective exponent drifts to 91/48 on its "
                       "own as finite-size (small-L) points fall away. Solid points are kept on a "
                       "value-blind error budget (95% CI ≤ 0.02); the volatile tail (few sizes left) "
                       "is excluded on the error, not on the value.")
            st.pyplot(fdc, width="content")
            st.download_button("Download d_f convergence plot", _fig_bytes(fdc), mime="image/png",
                               file_name=f"{key_prefix}_df_convergence.png", key=f"{key_prefix}_dl_dfc")

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
                st.caption(f"Saved as results_output/{rf}")
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
        st.markdown("Slide the two edge lengths — the tile morphs. Percolation depends only on the "
                    "adjacency, so every generic aperiodic (a≠b) member shares the **hat's** threshold.")
        a = st.slider("a  (short edges)", 0.0, float(gb.S3), 1.0, 0.05)
        b = st.slider("b  (long edges)", 0.0, float(gb.S3), float(gb.S3), 0.05)
        member = gb.resolve_member(tiling, a, b)
        if member is None:
            st.error("Tile(0,0) is degenerate — pick a>0 or b>0.")
        else:
            st.info(f"**Class:** {gb.family_member(a, b)[1]}\n\n**Runs as:** {member}")
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
    st.subheader("Toy demo")
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

    # --- separate scaling toy: the largest cluster above is ONE grid; run a ladder of sizes and the
    #     fractal dimension falls out of how the largest-cluster share scales. ---
    st.markdown("---")
    st.markdown("**From one blob to the scaling law.** The largest cluster above was a *single* grid. "
                "Here are **ten** square grids of increasing size. Drag the slider (or hit **Jump to "
                "percolation**) and watch each grid drop to a point on the plot — the share of the grid "
                "its largest cluster fills. Below the threshold those points slope down steeply; **at** "
                "the percolation point they line up on a straight line whose slope is the fractal "
                "dimension d_f. The faint dots behind each mean are ten independent fillings, so you can "
                "see the run-to-run spread the average is built from — nothing is hidden.")
    scaling_component(cached_scaling())

    # --- the OTHER exponent: correlation length nu, spelled out grids -> curves -> collapse ---
    st.markdown("---")
    st.markdown("**The sharpness of the transition — the exponent ν.** This one walks all the way from a "
                "grid you can watch percolate, to the spanning-probability curves, to the collapse that "
                "measures ν. Follow the three steps.")
    nu_component(cached_nu())

# ============================================================ VISUALISE engine
with tab_vis:
    st.subheader(f"Generate — {member}")
    left, right = st.columns([3, 1], gap="large")
    with right:
        if tiling == gb.FAMILY:
            st.caption("The family preview shows a single Tile(a,b); pick a concrete member "
                       "(comet / chevron / a≠b) to render a full patch.")
            size = 3
        else:
            rlabel, rlo, rhi, rdflt = gb.RENDER_CTL[member]
            size = st.slider(rlabel, rlo, rhi, rdflt, key="vis_size",
                             help="Render size only — kept small so the figure stays legible. "
                                  "The percolation patch (other tab) can be much larger.")
        st.caption("The visualisation is generated using the same code as the percolation engine. It is "
                   "thus used to verify that the transform used for this algorithm is indeed correct. "
                   "Any mistakes in substitutes show as gaps or discrepancies in the tiling.")
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
                           help="Block-B pass: records the incipient-cluster size per trial -> the "
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
                   "machine sleeps. Result auto-saves to results_output/ when done.")

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
        st.code("python runner/percolate.py " + " ".join(_parts), language="bash")
        st.caption("Same computation as the button (identical kernels, seed and results). Run it "
                   "from the project folder. It checkpoints and resumes if interrupted, and saves "
                   "to results_output/ — so it also shows up under 'Analyse saved' and in the "
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
    st.caption("Loads saved runs from the results_output/ folder and re-derives every result "
               "from the raw per-trial data.")
    saved = gb.list_saved()
    if not saved:
        st.info("No saved runs found in results_output/. Run something on the Percolate tab, or "
                "run runner/percolate.py from a console (see REPRODUCE.md).")
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
# (Analyse saved / Toy demo) from drawing. (A GUI Run, or any console `percolate.py` writing to the
# shared jobs/ dir, triggers this via auto-reattach.)
if st.session_state.pop("_poll_monitor", False):
    time.sleep(2)
    st.rerun()
