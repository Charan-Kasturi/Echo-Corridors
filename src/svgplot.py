"""
svgplot.py
==========
A tiny, dependency-free SVG line-chart writer (no numpy/matplotlib required).
Good enough for reproducible report figures: axes, ticks, gridlines, a
legend, and an optional shaded +/-1 std-dev band per series.
"""
from __future__ import annotations

PALETTE = ["#2C6E9B", "#C1553B", "#3F9E6B", "#8B5FA8", "#B08B2D", "#5A6B7A"]


def _fmt(v):
    return f"{v:.3f}"


def line_chart(path, title, x_label, y_label, series, width=780, height=460,
                y_min=None, y_max=None):
    """series: list of dicts with keys:
        label (str), x (list[float]), y (list[float]),
        y_std (optional list[float]), color (optional str)
    """
    margin_l, margin_r, margin_t, margin_b = 70, 30, 50, 60
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

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Georgia, \'Times New Roman\', serif">'
    )
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')

    # gridlines + y ticks
    n_yticks = 5
    for i in range(n_yticks + 1):
        val = y_min + (y_max - y_min) * i / n_yticks
        y = sy(val)
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" y2="{y:.1f}" '
                      f'stroke="#e3e3e3" stroke-width="1"/>')
        parts.append(f'<text x="{margin_l - 10}" y="{y+4:.1f}" font-size="12" fill="#333" '
                      f'text-anchor="end">{_fmt(val)}</text>')

    x_range = x_max - x_min
    x_decimals = 0 if x_range >= 5 else (1 if x_range >= 1.5 else 2)
    n_xticks = 5
    for i in range(n_xticks + 1):
        val = x_min + (x_max - x_min) * i / n_xticks
        x = sx(val)
        parts.append(f'<line x1="{x:.1f}" y1="{margin_t}" x2="{x:.1f}" y2="{height - margin_b}" '
                      f'stroke="#f0f0f0" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - margin_b + 20}" font-size="12" fill="#333" '
                      f'text-anchor="middle">{val:.{x_decimals}f}</text>')

    # axes
    parts.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height - margin_b}" '
                  f'stroke="#222" stroke-width="1.3"/>')
    parts.append(f'<line x1="{margin_l}" y1="{height - margin_b}" x2="{width - margin_r}" '
                  f'y2="{height - margin_b}" stroke="#222" stroke-width="1.3"/>')

    # series
    for idx, s in enumerate(series):
        color = s.get("color", PALETTE[idx % len(PALETTE)])
        xs, ys = s["x"], s["y"]
        if s.get("y_std"):
            top = [y + sd for y, sd in zip(ys, s["y_std"])]
            bot = [y - sd for y, sd in zip(ys, s["y_std"])]
            pts_top = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, top))
            pts_bot = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(reversed(xs), reversed(bot)))
            parts.append(f'<polygon points="{pts_top} {pts_bot}" fill="{color}" opacity="0.14"/>')

        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"/>')

    # legend
    lx, ly = margin_l + 10, margin_t - 30
    lx0 = lx
    for idx, s in enumerate(series):
        color = s.get("color", PALETTE[idx % len(PALETTE)])
        parts.append(f'<line x1="{lx0}" y1="{ly}" x2="{lx0+22}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx0+28}" y="{ly+4}" font-size="12.5" fill="#222">{s["label"]}</text>')
        lx0 += 26 + 9 * len(s["label"]) + 22

    # titles / axis labels
    parts.append(f'<text x="{width/2:.1f}" y="24" font-size="16" fill="#111" text-anchor="middle" '
                  f'font-weight="bold">{title}</text>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-14}" font-size="13" fill="#333" '
                  f'text-anchor="middle">{x_label}</text>')
    parts.append(f'<text x="18" y="{height/2:.1f}" font-size="13" fill="#333" text-anchor="middle" '
                  f'transform="rotate(-90 18 {height/2:.1f})">{y_label}</text>')

    parts.append('</svg>')
    svg = "\n".join(parts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return path
