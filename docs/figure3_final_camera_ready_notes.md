# Figure 3 camera-ready notes

## Scope

This version is a plotting-only refinement of Figure 3. It reads the existing V2 A2/C CSV outputs and V3 B3 neighbor-allocation CSV outputs. No checkpoint is loaded, no training is started, and the scientific definitions of Panels (a)–(c) are unchanged.

## Layout and styling

- Final canvas: `7.2 × 5.2 in`, with a compact two-row layout.
- Row 1 contains Panel (a) on the left and the two-sub-axis Panel (b) on the right.
- Row 2 contains Panel (c) across the full width, with Movies and Grocery facets.
- Panel (c) reports the existing retention values as percentages on a shared `80–100%` y-axis; this is a display-unit conversion only.
- Typography uses a paper-friendly serif hierarchy: approximately 8.8 pt panel titles, 8 pt axis labels, and 7–7.5 pt ticks/legends/annotations.
- PDF/SVG text remains editable; PNG/TIFF are exported at 600 dpi.

## Color mapping

- Movies: muted purple (`#7563A6`) with a light fill (`#C9BDE2`).
- Grocery: muted orange (`#D28A3A`) with a light fill (`#F1C58D`).
- Text modality: dark/light blue (`#4C78A8` / `#86A9CF`).
- Visual modality: dark/light green (`#59A14F` / `#91C589`).

Dataset colors are used only for Panels (a) and (b); modality colors are used only for the propagation comparison in Panel (c).

## QA

The final output directory contains render-time alignment JSON files and the plotting manifest. Static source preflight passed 21/21 checks with no warnings. The composite PDF and all three standalone PDFs passed the rendered collision audit with 0 failures and 0 warnings; the composite, Panel (b), and Panel (c) alignment gates passed, and the minimum rendered PDF text size was 5.8 pt. The previous Figure 3 V2/V3/final directories remain untouched.
