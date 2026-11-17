"""
Forecasts GTA VI launch-day Twitch viewership hour by hour using historical
release data. The program filters comparable games for rolling-midnight or
midnight-local launches, extracts viewer progression curves, and fits an
ordinary least-squares log-ratio regression model scaled by the target's hour-1
print. Residual variance determines quantile spreads (q05, q25, q50, q75, q95),
providing deterministic probabilistic forecasts adhering strictly to constraints.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import math
import os
from datetime import datetime
from statistics import pstdev

def round_q(val):
    val = max(0.01, min(1e9, val))
    rounded = math.floor(val * 100 + 0.5) / 100
    if abs(rounded - round(rounded)) < 1e-9:
        return int(round(rounded))
    return rounded

# Strict rolling-midnight analogues first; the caller falls back to all comparables, then to persistence.
def filter_analogues(comparables, strict=True):
    valid = []
    for comp in comparables:
        name = comp.get('name', '')
        if 'Grand Theft Auto VI' in name:
            continue
        unlock = comp.get('unlock', '')
        if strict:
            if 'Tier: early-access-contaminated' in unlock or 'Tier: weak' in unlock:
                continue
            if 'Simultaneous' in unlock or 'single global unlock' in unlock or 'Single simultaneous' in unlock:
                continue
            lower_unlock = unlock.lower()
            if 'midnight local' not in lower_unlock and 'rolling' not in lower_unlock:
                continue

        day_start_str = comp.get('day_start')
        if not day_start_str:
            continue
        try:
            day_start = datetime.fromisoformat(day_start_str.replace('Z', '+00:00'))
        except Exception:
            continue

        curve = {}
        for item in comp.get('hourly', []):
            f_str = item.get('from')
            v = item.get('viewers')
            if not f_str or v is None or v <= 0:
                continue
            try:
                dt = datetime.fromisoformat(f_str.replace('Z', '+00:00'))
                h_idx = int((dt - day_start).total_seconds() // 3600)
                if 0 <= h_idx <= 23:
                    curve[h_idx] = float(v)
            except Exception:
                continue

        if 0 in curve and len(curve) >= 20:
            valid.append(curve)
    return valid

def main():
    input_dir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    output_dir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)
    if not input_path.exists():
        return

    for in_file in sorted(input_path.glob('*.in')):
        try:
            with open(in_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        known = data.get('known', [])
        p1 = 1
        for row in known:
            if row.get('hour') == 1:
                p1 = max(1, row.get('viewers', 1))
                break

        comparables = data.get('comparables', [])
        analogues = filter_analogues(comparables, strict=True)
        if len(analogues) < 5:
            analogues = filter_analogues(comparables, strict=False)

        z_list = [-1.645, -0.674, 0.0, 0.674, 1.645]
        hours_result = []

        for h in range(2, 25):
            if len(analogues) < 3:
                q_vals = [round_q(p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in z_list]
            else:
                points = []
                for curve in analogues:
                    if (h - 1) in curve:
                        x = math.log(curve[0])
                        y = math.log(curve[h - 1] / curve[0])
                        points.append((x, y))

                n = len(points)
                if n == 0:
                    pred = 0.0
                    sigma = 0.5
                else:
                    mean_x = sum(pt[0] for pt in points) / n
                    mean_y = sum(pt[1] for pt in points) / n
                    cov = sum((pt[0] - mean_x) * (pt[1] - mean_y) for pt in points)
                    var = sum((pt[0] - mean_x) ** 2 for pt in points)
                    b = (cov / var) if var > 0 else 0.0
                    b = b * n / (n + 3)
                    a = mean_y - b * mean_x
                    pred = a + b * math.log(p1)
                    residuals = [pt[1] - (a + b * pt[0]) for pt in points]
                    sigma = pstdev(residuals) if n > 1 else 0.4
                    sigma = max(0.30, min(1.20, sigma))

                q_vals = [round_q(p1 * math.exp(pred + z * sigma)) for z in z_list]

            for i in range(1, len(q_vals)):
                if q_vals[i] < q_vals[i - 1]:
                    q_vals[i] = q_vals[i - 1]

            hours_result.append({"hour": h, "q": q_vals})

        out_data = {"hours": hours_result}
        out_file = Path(output_dir) / f"{in_file.stem}.out"
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(out_data, separators=(',', ':')) + '\n')

if __name__ == '__main__':
    main()
