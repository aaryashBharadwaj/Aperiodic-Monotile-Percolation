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
import os
import sys
import time
# gui_app is launched as `streamlit run interface/gui_app.py`, which puts interface/ (not the repo
# root) on sys.path. Add the repo root so the package imports (interface.*, generators.*, ...) resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import interface.gui_backend as gb
import visualiser.figures as figs
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
        # (A plain log-log <S_max>-vs-L plot was dropped: on log axes the fit and the 91/48 reference
        #  sit on top of each other, so it showed nothing the number and the convergence plot don't.)
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
        if st.checkbox("Compute ν(ω) band  (bootstraps 3 curves; ~30–40 s, computed once)",
                       key=f"{key_prefix}_nucb"):
            # The bootstrap is expensive (tens of seconds for a large run) and Streamlit reruns the
            # whole script on every interaction / job poll -- recomputing each time froze the app. Cache
            # the finished figure per result in session_state so it's built once, then reused instantly.
            sk = f"{key_prefix}_nufig"
            if sk not in st.session_state:
                with st.spinner("Computing ν(ω) band (bootstrapping)…"):
                    st.session_state[sk] = figs.nu_omega_figure(res, controls=_nu_controls(), B=60)
            fnu = st.session_state[sk]
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
    live = jobs.job_is_live(s)
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
    elif state in jobs.LIVE_STATES:
        # Status says live but the worker process is gone -> it was killed mid-run (crash / machine
        # powered off) and never flipped its status. Don't poll a corpse (that's what kept a dead job
        # showing "running" forever and the Run button disabled); show it as interrupted.
        st.warning("This run's worker is no longer alive — it was interrupted. Its checkpoint is kept, "
                   "so relaunching the same settings resumes from where it stopped.")
        if st.button("Dismiss", key=f"dis_{jid}"):
            st.session_state.pop("active_job", None)
            st.rerun()
    elif state == "error":
        st.error("Run failed.")
        st.code((s.get("error") or "")[:1200])
        if st.button("Dismiss", key=f"dis_{jid}"):
            st.session_state.pop("active_job", None)
            st.rerun()
    else:   # done | stopped
        keep = st.session_state.get(f"keep_{jid}", True)
        verb = "partial result" if state == "stopped" else "result"
        (st.warning if state == "stopped" else st.success)(
            f"✓ Complete — {verb} saved." if keep else
            f"✓ Complete — exploratory run (the {verb} shows below, but was not saved to disk).")
        rf = s.get("result_file")
        if rf:
            # Load ONCE and cache in session: we may delete the file (discard), and Streamlit reruns
            # this block on every poll, so subsequent renders must come from the in-memory copy.
            rkey = f"jobres_{jid}"
            if rkey not in st.session_state:
                try:
                    st.session_state[rkey] = gb.load_saved(rf)
                    if not keep:
                        gb.delete_result(rf)              # exploratory: drop it from disk, keep in memory
                except Exception as e:
                    st.session_state[rkey] = None
                    st.error(f"Run completed and saved, but the file could not be loaded: {e}")
            stored = st.session_state.get(rkey)
            if stored:
                res, meta = stored
                try:
                    show_results(res, meta["member"], "job", meta=meta)
                except Exception as e:
                    # The result is fine; a PLOT failed to render (e.g. the known transient matplotlib
                    # mathtext race under background threads). Say so honestly instead of "could not load".
                    st.error(f"Result computed correctly, but a plot failed to render "
                             f"(likely a transient rendering hiccup — rerun to redraw): {e}")
                st.caption(f"Saved as paper_results/npz/{rf}" if keep
                           else "🗑 Not saved (exploratory run — tick 'Save this run to results' to keep).")
        else:
            st.info("No result file (fewer than 3 usable sizes).")
        if st.button("Clear this run", key=f"clrdone_{jid}"):
            st.session_state.pop(f"jobres_{jid}", None)
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

tab_vis, tab_run, tab_cardy, tab_analyse = st.tabs(
    ["Visualise", "Percolate", "Crossing (Cardy)", "Analyse saved"])

# ============================================================ VISUALISE engine
with tab_vis:
    st.subheader(f"Generate — {member}")
    if tiling == gb.FAMILY:
        # Tile(a,b) shape, drawn live from the sidebar a/b sliders (figs.visualise renders the single
        # morphed tile for the FAMILY choice). Move the sliders to walk the whole family.
        st.markdown(
            r"The hat is one member of a continuous family, Tile$(a,b)$, set by its two edge lengths: "
            r"$a$ for the unit edges (blue) and $b$ for the $\sqrt{3}$ edges (red). The other aperiodic "
            r"monotiles live here too, the spectre is Tile$(1,1)$ and the turtle is Tile$(\sqrt{3},1)$; "
            r"shrink an edge-class to zero and you reach an endpoint, $b \to 0$ the comet and $a \to 0$ "
            r"the chevron. The sidebar $a$/$b$ sliders morph the tile below and pick which concrete "
            r"member the other tabs render and percolate.")
        left, _right = st.columns([3, 1], gap="large")
        with left:
            try:
                fig, _ntiles, _gcounts = cached_visualise(tiling, 1, round(a, 3), round(b, 3),
                                                          graph_type, False)
                st.pyplot(fig, width="content")
                st.caption(f"Tile(a,b) resolves to: **{member}**  ·  the other tabs run this member.")
                st.download_button("Download image", _fig_bytes(fig), mime="image/png",
                                   file_name=f"{member}_tile.png", key="vis_dl")
            except Exception as e:
                st.warning(f"render unavailable: {e}")
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
        meas = estimate.solid_window(member, kind, patch)
        ge = estimate.geometry(member, kind, patch)
        usable_max = meas if meas else ((ge[1] * 0.95) if ge else 200.0)
        how = ("the largest window measured to be fully tiled (0% off-tile) with a coordinated buffer"
               if meas else "0.95× the detected solid square (conservative fallback — this config isn't "
               "calibrated)")
        st.caption(f"Largest useful **L** for this patch ≈ **{usable_max:.0f}** "
                   f"(bigger windows fall outside the tiling and are skipped). This is {how}; "
                   "`builders/verify_solid_windows.py` reproduces the check.")

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
        _live = [j for j in jobs.list_jobs() if jobs.job_is_live(j)]
        if _live:
            st.session_state["active_job"] = _live[0]["job_id"]

    active = st.session_state.get("active_job")
    _s = jobs.read_job_status(active) if active else None
    _busy = bool(jobs.job_is_live(_s))

    want_exp = st.checkbox("Also measure d_f (extra cluster pass, ~+20% time)", value=False,
                           help="Largest-cluster pass: records the incipient-cluster size per trial -> the "
                                "fractal dimension d_f (<s_max> ~ L^d_f), one of the two exponents that "
                                "fix the universality class. Off by default to keep runs fast; the "
                                "paper's universality runs turn it on.")
    want_save = st.checkbox("Save this run to results", value=True,
                            help="On (default): the result is kept in paper_results/npz/. Off: an "
                                 "exploratory run — it still computes and displays, but is discarded "
                                 "from disk when done, so the results folder stays an intentional record.")
    if st.button("Run percolation", type="primary", disabled=(n_used < 3) or _busy):
        jid = jobs.launch_job(tiling, member, graph_type, kind, patch, round(a, 3), round(b, 3),
                            L_min, L_max, gap, int(T), int(seed), exponents=want_exp)
        st.session_state[f"keep_{jid}"] = bool(want_save)   # GUI-side keep/discard decision
        st.session_state["active_job"] = jid
        st.rerun()
    if not _busy:
        st.caption("Launches a background process — it keeps running if you close the tab or the "
                   "machine sleeps. Kept runs save to paper_results/npz/ when done.")

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
                _stat = j.get("status", "?")
                if _stat in jobs.LIVE_STATES and not jobs.job_is_live(j):
                    _stat = "interrupted"     # worker gone; status never flipped
                c0.write(f"**{j.get('member', '?')} · {j.get('kind', '')}** — "
                         f"{_stat} ({j.get('i', 0)}/{j.get('total', 0)})")
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


# ============================================================ CARDY CROSSING TEST
with tab_cardy:
    st.subheader(f"Crossing (Cardy) — {member} · {graph_type.split()[0].lower()} graph")
    st.caption("Conformal-invariance test: at criticality the left–right crossing probability of a "
               "rectangle depends only on its aspect ratio. We anchor at the square's ½-crossing, then "
               "compare the other aspect ratios to Cardy's exact 2D-percolation curve. A separate "
               "experiment from the p_c sweep (rectangles, not one L×L square). (Explainer TBD.)")

    ckind = "dual" if graph_type.startswith("Dual") else "direct"
    cplabel, cplo, cphi, cpdflt, cphelp = gb.PATCH_CTL[member]
    cpk = f"cardy_patch_{member}"
    st.session_state.setdefault(cpk, cpdflt)
    cpatch = st.slider(cplabel, cplo, cphi, key=cpk, help=cphelp)
    cge = estimate.geometry(member, ckind, cpatch)
    cusable = (cge[1] * 0.9) if cge else None
    if cusable:
        st.caption(f"Largest usable window ≈ **{cusable:.0f}** for this patch "
                   "(bigger rectangles fall outside the tiling and are skipped).")

    st.session_state.setdefault("cardy_windows",
                                f"{cusable*0.35:.0f}, {cusable*0.6:.0f}" if cusable else "80, 140")
    cc1, cc2 = st.columns([2, 1])
    windows_txt = cc1.text_input("Window sizes L (comma-separated)", key="cardy_windows",
                                 help="Each window runs a square (the anchor) plus the aspect-ratio "
                                      "sweep at fixed area L². Several windows show the finite-size "
                                      "deviation shrinking.")
    st.session_state.setdefault("cardy_naspect", 15)
    n_aspects = cc2.number_input("Aspect ratios", min_value=5, max_value=40, step=1, key="cardy_naspect")
    cc3, cc4, cc5 = st.columns(3)
    st.session_state.setdefault("cardy_amin", 0.5)
    st.session_state.setdefault("cardy_amax", 2.0)
    st.session_state.setdefault("cardy_T", 4000)
    amin = cc3.number_input("Aspect min", min_value=0.2, max_value=1.0, step=0.1, key="cardy_amin")
    amax = cc4.number_input("Aspect max", min_value=1.0, max_value=5.0, step=0.1, key="cardy_amax")
    cT = cc5.number_input("Trials / rectangle", min_value=200, max_value=20000, step=500, key="cardy_T",
                          help="Crossing probability needs high T (noise floor ~ √(1/4T)); ~4000+ for "
                               "a clean curve. Runs inline — heavy sweeps are better via the CLI "
                               "cardy_runner.")

    if st.button("Run Cardy test", key="cardy_run", type="primary"):
        try:
            windows = [float(x) for x in windows_txt.replace(";", ",").split(",") if x.strip()]
        except ValueError:
            windows = []
        if not windows:
            st.error("Enter one or more numeric window sizes, e.g. 80, 140.")
        else:
            with st.spinner(f"Building {member} · {ckind} and running Cardy "
                            f"({len(windows)} windows × {int(n_aspects)} aspects × {int(cT)} trials)…"):
                _bundle = gb.build_graph(member, graph_type, cpatch)
                st.session_state["cardy_result"] = gb.run_cardy(
                    _bundle, windows, n_aspects=int(n_aspects), aspect_min=float(amin),
                    aspect_max=float(amax), T=int(cT), bt=gb.default_bt(member))
                st.session_state["cardy_result_label"] = f"{member} · {ckind}"

    cres = st.session_state.get("cardy_result")
    if cres is not None and len(cres["windows"]):
        st.caption(f"Result: {st.session_state.get('cardy_result_label', '')}")
        cfig = figs.cardy_figure(cres)
        if cfig is not None:
            st.pyplot(cfig, width="content")
            st.download_button("Download Cardy plot", _fig_bytes(cfig), mime="image/png",
                               file_name=f"{member}_cardy.png", key="cardy_dl")
        floor = cres["noise_floor"]
        for i, L in enumerate(cres["windows"]):
            rms = float(cres["rms"][i])
            flag = "✓ at the noise floor" if rms <= 1.5 * floor else "systematic above floor"
            st.write(f"**L = {L:.0f}**  ·  anchor p\\* = {cres['pstar'][i]:.4f}  ·  "
                     f"RMS(R_h − Cardy) = {rms:.4f}  (noise floor {floor:.4f}) — {flag}")
    elif cres is not None:
        st.warning("No usable windows — all were too large for this patch. Try smaller window sizes.")


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
