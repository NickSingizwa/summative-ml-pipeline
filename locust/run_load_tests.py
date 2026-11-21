"""Runs the same Locust test against 1, 2 and 4 API containers and saves the results.

    docker compose up -d --build
    python locust/run_load_tests.py --containers 1 2 4 --users 100 --duration 90s

Results go to locust/results/ (summary table, chart, raw locust reports).
To test a deployed URL instead: --host <url> --no-docker --label "HF Space"
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
RAW = RESULTS / "raw"


def sh(*cmd: str) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def wait_for_replicas(host: str, expected: int, timeout: float = 240) -> set[str]:
    seen: set[str] = set()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{host}/health", timeout=5) as resp:
                data = json.load(resp)
                seen.add(data.get("instance") or data["host"])
        except Exception:
            time.sleep(1)
        if len(seen) >= expected:
            return seen
        time.sleep(0.2)
    raise SystemExit(f"only {len(seen)} of {expected} replicas answered: {seen}")


def run_locust(host: str, tag: str, users: int, spawn_rate: int, duration: str) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    prefix = RAW / tag
    sh(sys.executable, "-m", "locust", "-f", str(HERE / "locustfile.py"), "--host", host, "--headless",
       "-u", str(users), "-r", str(spawn_rate), "-t", duration, "--csv", str(prefix),
       "--html", str(prefix) + ".html", "--only-summary")
    with open(f"{prefix}_stats.csv") as fh:
        rows = {r["Name"]: r for r in csv.DictReader(fh)}
    r = rows["POST /api/predict"]
    total = rows["Aggregated"]
    return {
        "requests": int(r["Request Count"]),
        "failures": int(r["Failure Count"]),
        "failure_rate_pct": round(100 * int(r["Failure Count"]) / max(1, int(r["Request Count"])), 2),
        "rps": round(float(r["Requests/s"]), 2),
        "total_rps": round(float(total["Requests/s"]), 2),
        "avg_ms": round(float(r["Average Response Time"]), 1),
        "min_ms": round(float(r["Min Response Time"]), 1),
        "median_ms": float(r["50%"]),
        "p90_ms": float(r["90%"]),
        "p95_ms": float(r["95%"]),
        "p99_ms": float(r["99%"]),
        "max_ms": round(float(r["Max Response Time"]), 1),
    }


def plot(results: list[dict], out_dir: Path = RESULTS, title: str = "Locust") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink2, muted, grid, accent = "#52514e", "#898781", "#e1e0d9", "#2a78d6"
    labels = [r["label"] for r in results]
    x = range(len(results))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=grid, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors=muted, length=0)

    ax = axes[0]
    bars = ax.bar(x, [r["rps"] for r in results], width=0.45, color=accent)
    ax.bar_label(bars, labels=[f"{r['rps']:.1f}" for r in results], padding=3, color=ink2, fontsize=10)
    ax.set_xticks(list(x), labels)
    ax.set_title("Throughput (successful predictions / s)", loc="left", fontsize=12, color="#0b0b0b")

    ax = axes[1]
    w = 0.36
    ax.bar([i - w / 2 for i in x], [r["median_ms"] for r in results], width=w, color=muted, label="Median")
    b2 = ax.bar([i + w / 2 for i in x], [r["p95_ms"] for r in results], width=w, color=accent, label="95th percentile")
    ax.bar_label(b2, labels=[f"{r['p95_ms']:,.0f}" for r in results], padding=3, color=ink2, fontsize=9)
    ax.set_xticks(list(x), labels)
    ax.set_title("Response time of /api/predict (ms)", loc="left", fontsize=12, color="#0b0b0b")
    ax.legend(frameon=False, labelcolor=ink2, loc="upper right")
    fig.suptitle(f"{title}: {results[0]['users']} concurrent users, {results[0]['duration']} per run",
                 x=0.01, ha="left", color=ink2, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "load_test_results.png", dpi=150)
    print(f"[ok] chart -> {out_dir / 'load_test_results.png'}")


def write_summary(results: list[dict], out_dir: Path = RESULTS) -> None:
    keys = ["label", "containers", "users", "duration", "requests", "failures", "failure_rate_pct", "rps",
            "avg_ms", "median_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms"]
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    lines = ["| Setup | Requests | Failures | Throughput (req/s) | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Max (ms) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['label']} | {r['requests']:,} | {r['failures']} ({r['failure_rate_pct']}%) | {r['rps']} | "
                     f"{r['avg_ms']:,.0f} | {r['median_ms']:,.0f} | {r['p95_ms']:,.0f} | {r['p99_ms']:,.0f} | {r['max_ms']:,.0f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--containers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--users", type=int, default=100)
    p.add_argument("--spawn-rate", type=int, default=10)
    p.add_argument("--duration", default="90s")
    p.add_argument("--host", default="http://localhost:8080")
    p.add_argument("--no-docker", action="store_true", help="test --host as-is, without scaling containers")
    p.add_argument("--label", default=None, help="label for a --no-docker run")
    p.add_argument("--append", action="store_true", help="add to an existing summary instead of replacing it")
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    if args.append and (RESULTS / "summary.json").exists():
        results = json.loads((RESULTS / "summary.json").read_text())

    if args.no_docker:
        label = args.label or args.host
        tag = "".join(c if c.isalnum() else "_" for c in label).strip("_").lower()
        stats = run_locust(args.host, tag, args.users, args.spawn_rate, args.duration)
        results.append({"label": label, "containers": None, "users": args.users, "duration": args.duration, **stats})
    else:
        for n in args.containers:
            sh("docker", "compose", "up", "-d", "--scale", f"api={n}", "--no-recreate")
            sh("docker", "compose", "restart", "lb")  # nginx only resolves the replicas on startup
            hosts = wait_for_replicas(args.host, n)
            print(f"[ok] {n} replica(s) answering: {sorted(hosts)}")
            time.sleep(3)
            stats = run_locust(args.host, f"containers_{n}", args.users, args.spawn_rate, args.duration)
            results.append({"label": f"{n} container{'s' if n > 1 else ''}", "containers": n,
                            "users": args.users, "duration": args.duration, **stats})
            print(json.dumps(results[-1], indent=2))

    (RESULTS / "summary.json").write_text(json.dumps(results, indent=2))
    write_summary(results)
    plot(results)


if __name__ == "__main__":
    main()
