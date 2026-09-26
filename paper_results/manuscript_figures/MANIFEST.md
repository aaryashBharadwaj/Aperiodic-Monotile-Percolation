# Manuscript figures — manifest

17 figures generated from the v2 production run and the generators. Descriptive names (not `Figure_N`)
so they survive any renumbering. Map each to its manuscript slot below.

## Result figures — fill the existing placeholders

| File | Manuscript slot (label / current name) | Content | Source |
|---|---|---|---|
| `hat_direct_convergence.png` | `fig:mean_and_std` (Figure 1) | Hat direct: per-size site & bond I/U estimates vs L; the bracket closes as L grows | hat direct npz |
| `hat_direct_fss.png` | `fig:site` + `fig:bond` (Figures 2 & 3) | Hat direct finite-size scaling — **both site and bond panels in one figure** (I/U/A → common intercept). Either use as one two-panel figure, or crop into the two slots | hat direct npz |
| `hat_dual_convergence.png` | `fig:mean_and_std_tile` (Figure 8) | Hat **dual**: per-size I/U vs L | hat dual npz |
| `hat_dual_fss.png` | `fig:site_tile` (Figure 9) | Hat **dual** finite-size scaling (site + bond panels) | hat dual npz |
| `spectre_direct_convergence.png` | Family / supplement | Spectre **direct** per-size I/U vs L | spectre direct npz |
| `spectre_direct_fss.png` | Family / supplement | Spectre **direct** finite-size scaling | spectre direct npz |
| `spectre_dual_convergence.png` | Family / supplement | Spectre **dual** per-size I/U vs L | spectre dual npz |
| `spectre_dual_fss.png` | Family / supplement | Spectre **dual** finite-size scaling | spectre dual npz |
| `cardy_hat.png` | `fig:cardy` (part) | Crossing prob vs aspect ratio, hat direct, vs Cardy's exact curve | `cardy_Hat_direct_patch6.npz` |
| `cardy_spectre.png` | `fig:cardy` (part) | Same for the spectre — pair with `cardy_hat.png` (the paper wants both on/near one figure) | `cardy_Spectre_direct_patch6.npz` |

## New figures — need new slots (tied to the Layer-2 methods items)

| File | Where it goes | Content | Why |
|---|---|---|---|
| `universality_map.png` | **Universality Results** — new (item A4) | The (ν, d_f) plane: the family clustered at (4/3, 91/48), the two exact class markers, and the **measured 3D cube** at (0.92, 2.43) far away | The falsification test made visual — the pipeline distinguishes classes |
| `hat_direct_df.png` | Universality (supports Table `tab:df`) | Effective d_f vs L_min — drifts onto 91/48 as small-L is dropped, assumption-free | The defensible d_f presentation (convergence, not a single fit) |
| `hat_direct_nu_omega.png` | Universality / Results (item A2) | ν across the ω-band [0.5,1.5], with the exact square & triangular controls overlaid; stays consistent with 4/3 for all ω | The honest ν presentation the code actually produces |
| `hat_direct_pA_zoom_L400.png` | Methods — motivate `p_A` (item: p_A motivation) | Hat direct, L≥400 zoom: I sits high, U low, gap persists — so p_A = ½(I+U) is what cancels the leading correction | Justifies the averaged estimator |
| `tile_ab_to_comet.png` | Model / Introduction — the family thesis (**preferred**) | Hat → comet: 7 steps shrinking the √3 (red, b) edges, Tile(1,√3) → Tile(1,0). Shrinking side bold, unit side faint; one orientation frame throughout | Shows *how* the comet endpoint emerges from the hat |
| `tile_ab_to_chevron.png` | Model / Introduction — the family thesis (**preferred**) | Hat → chevron: 7 steps shrinking the unit (blue, a) edges, Tile(1,√3) → Tile(0,√3). Same treatment | Shows *how* the chevron endpoint emerges (note: this endpoint is the chevron, = Tile(0,1) up to scaling of the √3 edges) |
| `tiling_grid.png` | Model / Introduction (alternative) | The Tile(a,b) grid coloured by percolation-threshold class | The grid version — kept as an alternative; `tile_ab_morph` reads more clearly |
| `spectre_metatiles.png` | Model — spectre substitution | The spectre prototile Tile(1,1) and the mystic Γ (two fused spectres) — the substitution's building blocks | Analogue of the hat H/T/P/F figure (the 9 metatile *labels* are all the spectre except Γ, so only these two shapes matter) |
| `turtle_vs_hat.png` | Model — family invariance | Hat vs turtle patches side by side (independently generated, same graphs) | Visual companion to the isomorphism claim |

## Tiling illustrations — generator validation (Model / Verification)

| File | Content |
|---|---|
| `tiling_hat.png` | Hat tiling patch, gap-free, chirality-coloured (reflected tiles pop) — visual proof the generator is correct |
| `tiling_spectre.png` | Spectre patch, orientation-coloured (rotational classes) |
| `tiling_comet.png` | Comet periodic tiling |
| `tiling_chevron.png` | Chevron periodic tiling |
| `spectre_direct_fss.png` | Spectre direct finite-size scaling (the second headline object; supplementary / family table support) |

## NOT generated here (no result data — hand-drawn illustrations)

`hatunit.png`, `metatile.png` (H/T/P/F), `hattile.png`, `Percolation_Flowchart.png` — these are
schematic illustrations from the tile/metatile structure and the codebase diagram, not data plots.
Keep the existing versions or redraw by hand.

## Notes
- All result figures use the production run (seed 123456789). `fss_figure` shows all three estimators
  (I/U/A) — the author prefers this: I and U meeting at the same intercept is the built-in sanity check.
- The Cardy panels: anchor p* ≈ 0.838 (matches the site threshold), points sit on Cardy's curve within
  ~1.5× the √(1/4T) noise floor.
- The cube point in `universality_map.png` was measured live at ν = 0.92, d_f = 2.43 — decisively 3D,
  incompatible with the 2D family.
