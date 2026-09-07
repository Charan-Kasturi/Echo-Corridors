"""
svgplot.py
==========
A tiny, dependency-free SVG line-chart writer (no numpy/matplotlib required).
Good enough for reproducible report figures: axes, ticks, gridlines, a
legend, and an optional shaded +/-1 std-dev band per series.
"""
from __future__ import annotations

PALETTE = ["#2C6E9B", "#C1553B", "#3F9E6B", "#8B5FA8", "#B08B2D", "#5A6B7A"]

# Black-only, dash-differentiated series styling for the print report (no
# color channel available/appropriate on a monochrome printed page -- lines
# are told apart by dash pattern and weight instead, the standard approach
# in black-and-white academic figures).
BW_DASH = ["none", "7,4", "2,3", "9,3,2,3", "1,2"]
BW_WIDTH = [2.6, 2.2, 1.9, 1.9, 1.7]


def _fmt(v):
    return f"{v:.3f}"


def line_chart(path, title, x_label, y_label, series, width=780, height=460,
                y_min=None, y_max=None, bw=False):
    """series: list of dicts with keys:
        label (str), x (list[float]), y (list[float]),
        y_std (optional list[float]), color (optional str)

    bw=True renders the monochrome, dash-differentiated variant used by the
    print report (no baked-in title -- the print layout supplies its own
    "Figure N." caption outside the image, matching how its other figures
    are captioned).
    """
    # Legend is drawn as a stacked list below the title, so its height
    # (and therefore how far down the plot area must start) depends on how
    # many series there are.
    margin_l, margin_r, margin_b = 70, 30, 60
    legend_top = 16 if bw else 40
    legend_row_h = 18
    margin_t = legend_top + legend_row_h * len(series) + (10 if bw else 14)
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    all_x = [v for s in series for v in s["x"]]
    all_y = [v for s in series for v in s["y"]]
    if y_min is None:
        y_min = min(0.0, min(all_y))
    if y_max is None:
        y_max = max(all_y) * 1.08 if max(all_y) > 0 else 1.0
    x_min, x_max = min(all_x), max(all_x)
    if x_max == x_min:
        x_max = x_min + 1

    def sx(v):
        return margin_l + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v):
        return margin_t + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    axis_color = "#000" if bw else "#222"
    label_color = "#000" if bw else "#333"
    grid_color = "#ccc" if bw else "#e3e3e3"
    minor_grid_color = "#e5e5e5" if bw else "#f0f0f0"
    font_family = "'Times New Roman', Times, serif" if bw else "Georgia, 'Times New Roman', serif"

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{font_family}">'
    )
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')

    # gridlines + y ticks
    n_yticks = 5
    for i in range(n_yticks + 1):
        val = y_min + (y_max - y_min) * i / n_yticks
        y = sy(val)
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" y2="{y:.1f}" '
                      f'stroke="{grid_color}" stroke-width="1"/>')
        parts.append(f'<text x="{margin_l - 10}" y="{y+4:.1f}" font-size="12" fill="{label_color}" '
                      f'text-anchor="end">{_fmt(val)}</text>')

    x_range = x_max - x_min
    x_decimals = 0 if x_range >= 5 else (1 if x_range >= 1.5 else 2)
    n_xticks = 5
    for i in range(n_xticks + 1):
        val = x_min + (x_max - x_min) * i / n_xticks
        x = sx(val)
        parts.append(f'<line x1="{x:.1f}" y1="{margin_t}" x2="{x:.1f}" y2="{height - margin_b}" '
                      f'stroke="{minor_grid_color}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - margin_b + 20}" font-size="12" fill="{label_color}" '
                      f'text-anchor="middle">{val:.{x_decimals}f}</text>')

    # axes
    parts.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height - margin_b}" '
                  f'stroke="{axis_color}" stroke-width="1.3"/>')
    parts.append(f'<line x1="{margin_l}" y1="{height - margin_b}" x2="{width - margin_r}" '
                  f'y2="{height - margin_b}" stroke="{axis_color}" stroke-width="1.3"/>')

    # series
    for idx, s in enumerate(series):
        if bw:
            color = "#000000"
            dash = BW_DASH[idx % len(BW_DASH)]
            sw = BW_WIDTH[idx % len(BW_WIDTH)]
        else:
            color = s.get("color", PALETTE[idx % len(PALETTE)])
            dash, sw = "none", 2.4
        xs, ys = s["x"], s["y"]
        if s.get("y_std"):
            top = [y + sd for y, sd in zip(ys, s["y_std"])]
            bot = [y - sd for y, sd in zip(ys, s["y_std"])]
            pts_top = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, top))
            pts_bot = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(reversed(xs), reversed(bot)))
            parts.append(f'<polygon points="{pts_top} {pts_bot}" fill="{color}" opacity="0.10"/>')

        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}"{dash_attr}/>')

    # legend -- one entry per row, stacked below the title, so it never
    # collides with the (centered) title text regardless of label length.
    lx = margin_l + 6
    for idx, s in enumerate(series):
        if bw:
            color = "#000000"
            dash = BW_DASH[idx % len(BW_DASH)]
            sw = BW_WIDTH[idx % len(BW_WIDTH)]
        else:
            color = s.get("color", PALETTE[idx % len(PALETTE)])
            dash, sw = "none", 3
        ly = legend_top + idx * legend_row_h
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        legend_text_color = "#000" if bw else "#222"
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+26}" y2="{ly}" stroke="{color}" '
                      f'stroke-width="{sw}"{dash_attr}/>')
        parts.append(f'<text x="{lx+32}" y="{ly+4}" font-size="12.5" fill="{legend_text_color}">{s["label"]}</text>')

    # titles / axis labels
    if not bw:
        parts.append(f'<text x="{width/2:.1f}" y="24" font-size="16" fill="#111" text-anchor="middle" '
                      f'font-weight="bold">{title}</text>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-14}" font-size="13" fill="{label_color}" '
                  f'text-anchor="middle">{x_label}</text>')
    parts.append(f'<text x="18" y="{height/2:.1f}" font-size="13" fill="{label_color}" text-anchor="middle" '
                  f'transform="rotate(-90 18 {height/2:.1f})">{y_label}</text>')

    parts.append('</svg>')
    svg = "\n".join(parts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return path
