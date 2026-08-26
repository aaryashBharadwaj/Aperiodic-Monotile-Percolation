"""Interactive percolation portal for the aperiodic monotile family and its validations.

    streamlit run interface/gui_app.py

Three tabs, each with its own tiling/graph picker (no shared control panel -- the tiling you draw and
the tiling you run are selected independently, on their own pages):
  * Visualise  — pick a tiling + graph and generate it at a size you control, optionally overlaying the
                 exact percolation graph, i.e. proof the generator produces the real shape + lattice.
  * Run        — plan either a p_c sweep OR a Cardy crossing test (patch, sizes, trials): preview the
                 graph size / usable sizes / run-time / accuracy, then emit the exact runner/runner.py
                 or runner/cardy_runner.py command. The run itself happens in a terminal (it
                 checkpoints, resumes and auto-saves); the GUI doesn't launch or monitor the worker.
  * Analyse saved — reload any run from paper_results/npz/ and show its analysis: a p_c sweep gives
                 p_c + direction-bias check + fractal dimension d_f; a Cardy run gives the crossing
                 curve vs Cardy's exact prediction.

All numerics come from the same builders/kernels the runner uses (via gui_backend). Converged
production numbers come from runner/runner.py (or cardy_runner.py) with a larger patch / more trials
-- the 'Use paper settings' button fills the exact production parameters (see REPRODUCE.md).
"""
import os
import sys
# gui_app is launched as `streamlit run interface/gui_app.py`, which puts interface/ (not the repo
# root) on sys.path. Add the repo root so the package imports (interface.*, generators.*, ...) resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import interface.gui_backend as gb
import visualiser.figures as figs
import interface.estimate as estimate           # NOT `est` -- a local float `est` shadows it below
from engine.percolation import l_sweep

st.set_page_config(page_title="Aperiodic Percolation Portal", layout="wide")
# Tighten the vertical rhythm: Streamlit's defaults leave a lot of dead space up top -- the main
# container has ~6rem padding and headings carry ~1.25rem of block padding each (the h1 title alone
# was ~90px tall). Those stack up and push the title and tab content down the page. Trim them so
# content sits near the top. (Also shrink the metric value font so longer values don't clip.)
st.markdown(
    "<style>"
    "[data-testid='stMetricValue']{font-size:1.6rem;}"
    # Clear the fixed 60px top header bar so the title isn't cropped under it, but no more than that.
    ".block-container{padding-top:4rem;}"
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


def show_cardy(result, meta, key_prefix="cardy"):
    """Render a saved Cardy crossing run (loaded via gb.load_cardy): the measured crossing probability
    vs rectangle aspect ratio against Cardy's exact curve, plus a per-window RMS(measured − Cardy)
    read against the √(1/4T) noise floor. This is the conformal-invariance evidence -- at criticality
    the crossing probability depends only on aspect ratio, so the points should sit on the curve and
    tighten onto it as the window grows."""
    import numpy as np
    label = f"{meta.get('member', '?')} · {meta.get('graph', '')}".strip(" ·")
    st.subheader(f"Cardy crossing — {label}")
    bits = []
    if meta.get("patch") not in (None, ""):  bits.append(f"**Patch:** {meta['patch']}")
    if result.get("trials"):                 bits.append(f"**Trials/rectangle:** {int(result['trials']):,}")
    if meta.get("seed") not in (None, ""):   bits.append(f"**Seed:** {meta['seed']}")
    if bits:
        st.markdown("  ·  ".join(bits))
    fig = figs.cardy_figure(result)
    if fig is None:
        st.warning("No usable windows in this file.")
        return
    st.pyplot(fig, width="content")
    st.download_button("Download Cardy plot", _fig_bytes(fig), mime="image/png",
                       file_name=f"{label}_cardy.png".replace(" ", "_"), key=f"{key_prefix}_dl")
    floor = float(result.get("noise_floor", 0.0))
    cardy = np.asarray(result["cardy"], float)
    Rh = np.atleast_2d(np.asarray(result["Rh"], float))
    windows = np.asarray(result["windows"], float)
    pstar = np.asarray(result["pstar"], float)
    st.caption("Each window is anchored at its own square ½-crossing; the other aspect ratios are then "
               "free predictions Cardy's exact curve must reproduce. RMS approaching the noise floor as "
               "L grows is the conformal-invariance fingerprint of the 2D-percolation class.")
    for i, L in enumerate(windows):
        rms = float(np.sqrt(np.nanmean((Rh[i] - cardy) ** 2)))
        flag = "✓ within ~1.5× the noise floor" if floor and rms <= 1.5 * floor \
            else "above the floor (finite-size / low T)"
        ps = pstar[i] if i < len(pstar) else float("nan")
        st.write(f"**L = {L:.0f}**  ·  anchor p\\* = {ps:.4f}  ·  "
                 f"RMS(R_h − Cardy) = {rms:.4f}  (noise floor {floor:.4f}) — {flag}")


def pick_tiling(key_prefix):
    """Per-page tiling picker. There is no shared control panel -- Visualise and Run each own an
    INDEPENDENT selection (distinct `key_prefix`), so the tiling you draw need not be the one you run.
    Renders the tiling dropdown and the Tile(a,b) edge sliders when the family is chosen. Returns
    (tiling, a, b, member); member is None only for the degenerate Tile(0,0). The graph choice is
    page-specific and added by the caller (an overlay selector on Visualise, a --graph flag on Run)."""
    tiling = st.selectbox("Tiling", gb.TILINGS, index=0, key=f"{key_prefix}_tiling",
                          help="Aperiodic monotiles, the periodic family limits, the Tile(a,b) family, "
                               "and the Penrose / triangular validations.")
    st.caption(f"*{gb.CATEGORY[tiling]}*")
    a, b = 1.0, gb.S3
    if tiling == gb.FAMILY:
        a = st.slider("Edge a — unit / blue", 0.0, 2.0, 1.0, 0.05, key=f"{key_prefix}_fam_a")
        b = st.slider("Edge b — √3 / red", 0.0, 2.0, round(float(gb.S3), 2), 0.05, key=f"{key_prefix}_fam_b")
        member = gb.resolve_member(tiling, a, b)
        if member is None:
            st.caption("Tile(0,0) is degenerate — give a or b a non-zero length.")
        else:
            st.caption(f"Tile(a,b) → **{member}**  ·  *{gb.family_member(a, b)[1]}*")
    else:
        member = tiling
    return tiling, a, b, member


st.title("Aperiodic Monotile Percolation Portal")
tab_vis, tab_run, tab_analyse = st.tabs(
    ["Visualise", "Run", "Analyse saved"])

# ============================================================ VISUALISE engine
with tab_vis:
    tiling, a, b, member = pick_tiling("vis")
    graph_opts = gb.GRAPHS_FOR.get(member, gb.GRAPHS) if member else gb.GRAPHS
    # Graph is not a "mode" to visualise -- it's just what (if anything) to overlay on the tile picture.
    if tiling == gb.FAMILY:
        show_graph, graph_type = False, graph_opts[0]       # family draws a single morphing tile, no graph
    else:
        ok = "vis_overlay"
        overlay_opts = ["None"] + list(graph_opts)
        if st.session_state.get(ok) not in overlay_opts:    # a one-graph member would strand a stale pick
            st.session_state.pop(ok, None)
        overlay_sel = st.radio("Overlay graph", overlay_opts, horizontal=True, key=ok,
                               format_func=lambda o: "None" if o == "None" else o.split()[0],
                               help="Draw the exact percolation graph on the tiling: Direct = tile "
                                    "corners (sites) + tile edges (bonds); Dual = tile centroids + "
                                    "adjacency. Same generator the runner percolates.")
        show_graph = overlay_sel != "None"
        graph_type = overlay_sel if show_graph else graph_opts[0]
    st.subheader(f"Generate — {member}" if member else "Generate")
    if tiling == gb.FAMILY:
        # Tile(a,b) shape, drawn live from the a/b sliders above (figs.visualise renders the single
        # morphed tile for the FAMILY choice). Move the sliders to walk the whole family.
        st.markdown(
            r"The hat is one member of a continuous family, Tile$(a,b)$, set by its two edge lengths: "
            r"$a$ for the unit edges (blue) and $b$ for the $\sqrt{3}$ edges (red). The other aperiodic "
            r"monotiles live here too, the spectre is Tile$(1,1)$ and the turtle is Tile$(\sqrt{3},1)$; "
            r"shrink an edge-class to zero and you reach an endpoint, $b \to 0$ the comet and $a \to 0$ "
            r"the chevron. The $a$/$b$ sliders above morph the tile below.")
        left, _right = st.columns([3, 1], gap="large")
        with left:
            try:
                fig, _ntiles, _gcounts = cached_visualise(tiling, 1, round(a, 3), round(b, 3),
                                                          graph_type, False)
                st.pyplot(fig, width="content")
                st.caption(f"Tile(a,b) resolves to: **{member}**.")
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

# ============================================================ RUN engine (p_c sweep OR Cardy crossing)
# One page for both run types. It PLANS the run (geometry -> node count, usable sizes, ETA, accuracy)
# and emits the exact console command; the sweep itself runs in a terminal via runner/runner.py (p_c)
# or runner/cardy_runner.py (Cardy). Both reuse the identical kernels and seed, checkpoint/resume, and
# save to paper_results/npz/, from where "Analyse saved" shows the finished run. The GUI deliberately
# doesn't launch or monitor the worker -- the console owns running, this page owns getting the flags
# right (and, via "Use paper settings", handing you the exact command that reproduces the paper).
with tab_run:
    tiling, a, b, member = pick_tiling("run")
    if member is None:
        st.info("Tile(0,0) is degenerate — give a or b a non-zero length to pick a runnable member.")
    else:
        # Graph is a run PARAMETER here (--graph): direct and dual are different graphs with different
        # thresholds, so you must pick which to percolate. (On Visualise it's only an overlay choice.)
        graph_opts = gb.GRAPHS_FOR.get(member, gb.GRAPHS)
        gk = "run_graph"
        if st.session_state.get(gk) not in graph_opts:       # a one-graph member would strand a stale pick
            st.session_state.pop(gk, None)
        graph_type = st.radio("Graph", graph_opts, horizontal=True, key=gk,
                              format_func=lambda o: o.split()[0],
                              help="Direct = tile corners are sites, tile edges are bonds. Dual = tiles "
                                   "are sites, adjacent tiles are bonded.")
        kind = "dual" if graph_type.startswith("Dual") else "direct"
        _gsh = kind
        st.subheader(f"Run — {member} · {kind} graph")

        run_type = st.radio(
            "Run type", ["Percolation threshold  p_c", "Crossing test  (Cardy)"],
            horizontal=True, key="run_type",
            help="p_c: finite-size sweep of the spanning threshold, extrapolated to L→∞.  "
                 "Cardy: the conformal-invariance crossing test — crossing probability vs rectangle "
                 "aspect ratio, against Cardy's exact 2D-percolation curve.")
        is_cardy = run_type.startswith("Crossing")

        def _fmt_count(n):
            if n >= 1_000_000: return f"{n/1e6:.1f}M"
            if n >= 1_000:     return f"{n/1e3:.0f}k"
            return str(int(n))

        # Every input below is a flag on the emitted command -- one "Parameters" block, in command order.
        # Patch is a FREE number input (the GUI emits a command, no in-browser build), so "Use paper
        # settings" can dial in the real production patch (e.g. 840 cells).
        plabel, _plo, _phi, pdflt, phelp = gb.PATCH_CTL[member]
        patch_key = f"run_patch_{member}"          # per-member so switching tilings keeps a sane value
        st.session_state.setdefault(patch_key, int(pdflt))
        st.session_state.setdefault("run_seed", 123456789)

        st.markdown("**Parameters**  ·  *each becomes a flag on the command below*")
        cset, cnote = st.columns([2, 1], gap="large")
        with cset:
            if not is_cardy:
                preset = gb.paper_preset(member, kind)      # one-click "the paper run" (always fits now)
                if preset:
                    def _apply_preset(preset=preset, patch_key=patch_key):
                        st.session_state[patch_key] = int(preset["patch"])
                        st.session_state["run_lmin"] = float(preset["L_min"])
                        st.session_state["run_lmax"] = float(preset["L_max"])
                        st.session_state["run_gap"] = float(preset["gap"])
                        st.session_state["run_trials"] = int(preset["T"])
                        st.session_state["run_seed"] = int(preset["seed"])
                    st.button(f"Use paper settings ({member})", on_click=_apply_preset,
                              help="Fill the exact production parameters this object was run with for the "
                                   "paper, then copy the command below.")

            pcol, scol = st.columns(2)
            patch = int(pcol.number_input(plabel, min_value=int(_plo), step=1, key=patch_key, help=phelp))
            seed = int(scol.number_input("Seed", step=1, key="run_seed",
                                         help="Monte-Carlo seed; the same seed reproduces the same run."))

            # Usable window comes from PRECOMPUTED GEOMETRY -- no graph is built here.
            meas = estimate.solid_window(member, kind, patch)
            ge = estimate.geometry(member, kind, patch)
            if is_cardy:
                usable_max = meas if meas else ((ge[1] * 0.9) if ge else 200.0)
                st.caption(f"Largest usable **window L** for this patch ≈ **{usable_max:.0f}** "
                           "(bigger rectangles fall outside the tiling and are skipped).")
            else:
                usable_max = meas if meas else ((ge[1] * 0.95) if ge else 200.0)
                how = ("the largest window measured to be fully tiled (0% off-tile) with a coordinated "
                       "buffer" if meas else "0.95× the detected solid square (uncalibrated fallback)")
                st.caption(f"Largest useful **L** for this patch ≈ **{usable_max:.0f}** "
                           f"(bigger windows are skipped). This is {how}; "
                           "`builders/verify_solid_windows.py` reproduces the check.")

            if not is_cardy:
                st.session_state.setdefault("run_lmin", 10.0)
                st.session_state.setdefault("run_lmax", float(max(20, round(usable_max))))
                st.session_state.setdefault("run_gap", 20.0)
                st.session_state.setdefault("run_trials", 200)
                ra, rb = st.columns(2)
                L_min = ra.number_input("L min", min_value=2.0, step=1.0, key="run_lmin",
                                        help="Smallest square-window side length in the L sweep.")
                L_max = rb.number_input("L max", min_value=3.0, step=1.0, key="run_lmax",
                                        help="Largest square-window side length in the L sweep.")
                rc, rd = st.columns(2)
                gap = rc.number_input("Gap  ΔL", min_value=1.0, step=1.0, key="run_gap",
                                      help="Spacing between window sizes.")
                T = rd.number_input("Trials per L", min_value=1, step=50, key="run_trials",
                                    help="Monte-Carlo trials per L. More trials means smaller error bars.")
                want_exp = st.checkbox("Also measure d_f  (--exponents; extra cluster pass, ~+20% time)",
                                       value=False, key="run_exponents",
                                       help="Largest-cluster pass: records the incipient-cluster size per "
                                            "trial -> the fractal dimension d_f (⟨s_max⟩ ~ L^d_f), one of "
                                            "the two exponents that fix the universality class. The paper's "
                                            "universality runs turn it on.")
            else:
                st.session_state.setdefault("cardy_windows",
                                            f"{usable_max*0.4:.0f}, {usable_max*0.7:.0f}" if usable_max
                                            else "150, 250")
                windows_txt = st.text_input("Window sizes L (comma-separated)", key="cardy_windows",
                                            help="Each window runs a square anchor plus the aspect-ratio "
                                                 "sweep at fixed area L². Several windows show the finite-"
                                                 "size deviation shrinking onto Cardy's curve.")
                ca, cb = st.columns(2)
                st.session_state.setdefault("cardy_naspect", 20)
                st.session_state.setdefault("cardy_T", 8000)
                n_aspects = ca.number_input("Aspect ratios", min_value=5, max_value=60, step=1,
                                            key="cardy_naspect")
                cT = cb.number_input("Trials / rectangle", min_value=200, step=500, key="cardy_T",
                                     help="Crossing probability needs high T (noise floor ~ √(1/4T)). The "
                                          "paper's strong tier used 24000.")
                cc, cd = st.columns(2)
                st.session_state.setdefault("cardy_amin", 0.5)
                st.session_state.setdefault("cardy_amax", 2.0)
                amin = cc.number_input("Aspect min", min_value=0.2, max_value=1.0, step=0.1, key="cardy_amin")
                amax = cd.number_input("Aspect max", min_value=1.0, max_value=5.0, step=0.1, key="cardy_amax")

        with cnote:
            ref = gb.REFERENCE.get((member, kind))
            if ref:
                st.caption(f"Reference — {ref}")

        # ---- plan + readout (geometry facts only; NO runtime estimate -- the runner prints a live ETA) ----
        if not is_cardy:
            Ls = l_sweep(L_min, L_max, gap)                    # SHARED engine definition -> identical sizes
            plan = estimate.plan_run(member, kind, patch, Ls) if Ls else None
            n_used = plan["n_usable"] if plan else 0
            # Capacity = the MOST sizes this patch can use at the current L_min/gap: the steps that fit up
            # to the useful ceiling (independent of L_max). Show requested-vs-capacity so the denominator
            # is a stable target -- below it you have room to raise L max, above it the surplus sizes fall
            # outside the tiling and are skipped. (n_used = min(requested, capacity) drives the guard.)
            capacity = len(l_sweep(L_min, plan["usable_max"], gap)) if plan else 0
            m1, m2 = st.columns(2)
            m1.metric("Graph nodes ~", _fmt_count(plan["nodes_est"]) if plan else "—",
                      help=(f"~{plan['nodes_est']:,} nodes, from geometry (not yet built)." if plan
                            else "Set a valid sweep."))
            m2.metric("L sizes", f"{len(Ls)} / {capacity}" if capacity else "—",
                      help="Window sizes you're requesting  /  the most this patch can use (steps up to "
                           "the useful ceiling above). Under the denominator → raise L max for more "
                           "fit-points; over it → the surplus sizes are skipped. Grow the denominator by "
                           "raising the patch (higher ceiling) or shrinking the gap (more steps under it).")
            if n_used < 3:
                st.error("Fewer than 3 usable L values — raise the patch, lower L min, or shrink the gap.")
        else:
            try:
                windows = [float(x) for x in windows_txt.replace(";", ",").split(",") if x.strip()]
            except ValueError:
                windows = []
            cplan = estimate.plan_cardy(member, kind, patch, windows, int(cT)) if windows else None
            n_used = cplan["n_usable"] if cplan else 0
            m1, m2, m3 = st.columns(3)
            m1.metric("Graph nodes ~", _fmt_count(cplan["nodes_est"]) if cplan else "—")
            m2.metric("Usable windows", f"{n_used} / {len(windows)}")
            m3.metric("RMS noise floor", f"{cplan['floor']:.4f}" if cplan else "—",
                      help="√(1/4T): the statistical floor the RMS(measured − Cardy) can reach. Lower T "
                           "means a noisier curve.")
            if n_used < 1:
                st.error("No usable windows — all are larger than this patch allows. Use smaller window "
                         "sizes, or raise the patch.")

        st.caption("Runs scale with the **graph-node count × trials** above: the production settings (large "
                   "patches, high trial counts) can take **hours**, while a small exploratory patch is "
                   "seconds. Copy the command below into a terminal — it prints a live ETA from the first size.")

        # ---- emit the exact command (every parameter above maps to a flag here) ----
        st.markdown("---")
        if not is_cardy:
            parts = [f'--tiling "{tiling}"', f"--graph {_gsh}"]
            if tiling == gb.FAMILY:
                parts.append(f"--a {round(a, 3):g} --b {round(b, 3):g}")
            parts += [f"--patch {patch}", f"--lmin {L_min:g}", f"--lmax {L_max:g}", f"--gap {gap:g}",
                      f"--trials {int(T)}", f"--seed {int(seed)}"]
            if want_exp:
                parts.append("--exponents")
            cmd = "python runner/runner.py " + " ".join(parts)
        else:
            parts = [f'--tiling "{tiling}"', f"--graph {_gsh}"]
            if tiling == gb.FAMILY:
                parts.append(f"--a {round(a, 3):g} --b {round(b, 3):g}")
            win_str = " ".join(f"{w:g}" for w in windows) if windows else "150 250"
            parts += [f"--patch {patch}", f"--windows {win_str}", f"--n-aspects {int(n_aspects)}",
                      f"--aspect-min {amin:g}", f"--aspect-max {amax:g}", f"--trials {int(cT)}",
                      f"--seed {int(seed)}"]
            cmd = "python runner/cardy_runner.py " + " ".join(parts)

        st.markdown("**Run this in a terminal**, from the project folder:")
        st.code(cmd, language="bash")
        st.caption("Identical kernels and seed as the rest of the portal. It checkpoints and resumes if "
                   "interrupted, and saves to paper_results/npz/ — open the finished run under the "
                   "“Analyse saved” tab.")



# ============================================================ ANALYSE SAVED
with tab_analyse:
    st.subheader("Analyse a saved run")
    st.caption("Loads saved runs from paper_results/npz/ and re-derives every result from the raw "
               "per-trial data. Shows both p_c sweeps (runner.py) and Cardy crossing runs "
               "(cardy_runner.py, filenames starting `cardy_`).")
    saved = gb.list_saved()
    if not saved:
        st.info("No saved runs found in paper_results/npz/. Emit a command on the Run tab and run it "
                "in a terminal (see REPRODUCE.md).")
    else:
        pick = st.selectbox("Saved run", saved, key="analyse_pick")
        try:
            if os.path.basename(pick).startswith("cardy_"):
                cres, cmeta = gb.load_cardy(pick)
                show_cardy(cres, cmeta)
            else:
                res, meta = gb.load_saved(pick)
                show_results(res, meta["member"], "an", meta=meta)
        except Exception as e:
            st.error(f"Could not load {pick}: {e}")
