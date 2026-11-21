"""Same Locust test as run_load_tests.py, but against the app deployed on Azure with 1, 2 and 4 replicas.

Locust itself runs in an Azure Container Instance in the same region, so my home upload
speed (~350 KB/s, about one 10 s clip per second) doesn't limit the test.

    az login
    python locust/run_azure_load_tests.py --replicas 1 2 4

Scales the app back to 1 replica at the end. Results go to locust/results/azure/.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_load_tests import plot, wait_for_replicas, write_summary  # noqa: E402

OUT = HERE / "results" / "azure"


def az(*args: str) -> str:
    return subprocess.run(["az", *args, "--only-show-errors"], check=True, capture_output=True, text=True).stdout.strip()


def az_json(*args: str):
    return json.loads(az(*args, "-o", "json"))


def stats_from_csv(text: str) -> dict:
    rows = {r["Name"]: r for r in csv.DictReader(io.StringIO(text))}
    r = rows["POST /api/predict"]
    n = int(r["Request Count"])
    return {
        "requests": n, "failures": int(r["Failure Count"]),
        "failure_rate_pct": round(100 * int(r["Failure Count"]) / max(1, n), 2),
        "rps": round(float(r["Requests/s"]), 2), "avg_ms": round(float(r["Average Response Time"]), 1),
        "median_ms": float(r["50%"]), "p90_ms": float(r["90%"]), "p95_ms": float(r["95%"]),
        "p99_ms": float(r["99%"]), "max_ms": round(float(r["Max Response Time"]), 1),
    }


def wait_for_revision(rg: str, app: str, n: int, timeout: float = 900) -> None:
    # changing the replica count makes a new revision, wait until it's the only one taking traffic
    deadline = time.time() + timeout
    while time.time() < deadline:
        revs = az_json("containerapp", "revision", "list", "-n", app, "-g", rg, "--query",
                       "[?properties.active].{replicas: properties.replicas, state: properties.runningState}")
        if len(revs) == 1 and revs[0]["replicas"] == n and str(revs[0]["state"]).startswith("Running"):
            time.sleep(15)
            return
        time.sleep(10)
    raise SystemExit(f"revision with {n} replicas never became ready")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rg", default="pumpsense-rg")
    p.add_argument("--app", default="pumpsense")
    p.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--users", type=int, default=100)
    p.add_argument("--duration", default="90s")
    p.add_argument("--append", action="store_true", help="add to the existing results instead of replacing them")
    args = p.parse_args()

    app = az_json("containerapp", "show", "-n", args.app, "-g", args.rg)
    url = "https://" + app["properties"]["configuration"]["ingress"]["fqdn"]
    image = app["properties"]["template"]["containers"][0]["image"]
    server = app["properties"]["configuration"]["registries"][0]["server"]
    acr = server.split(".")[0]
    user = az("acr", "credential", "show", "-n", acr, "--query", "username", "-o", "tsv")
    password = az("acr", "credential", "show", "-n", acr, "--query", "passwords[0].value", "-o", "tsv")
    az("provider", "register", "--namespace", "Microsoft.ContainerInstance", "--wait")
    locustfile = (HERE / "locustfile.py").read_text()
    print("app:", url)

    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads((OUT / "summary.json").read_text()) if args.append and (OUT / "summary.json").exists() else []
    try:
        for n in args.replicas:
            # --no-wait because the cli sometimes hangs here even after the replicas are up, we check ourselves below
            az("containerapp", "update", "-n", args.app, "-g", args.rg, "--min-replicas", str(n), "--max-replicas", str(n),
               "--no-wait", "-o", "none")
            wait_for_revision(args.rg, args.app, n)
            hosts = wait_for_replicas(url, n, timeout=600)
            print(f"[ok] {n} replica(s): {sorted(hosts)}")

            # the app image already has the demo clips in /app/data/samples, it just needs locust installed
            cmd = ("sh -c 'pip install --user --quiet locust==2.47.0 && "
                   f"python -m locust -f /app/locust/locustfile.py --host {url} --headless -u {args.users} -r 10 "
                   f"-t {args.duration} --csv /tmp/out --only-summary > /tmp/locust.log 2>&1; "
                   "echo ===CSV===; cat /tmp/out_stats.csv'")
            name = f"{args.app}-locust-{n}"
            az("container", "create", "-g", args.rg, "-n", name, "--image", image, "--os-type", "Linux",
               "--registry-login-server", server, "--registry-username", user, "--registry-password", password,
               "--cpu", "2", "--memory", "3", "--restart-policy", "Never",
               "--secrets", f"locustfile.py={locustfile}", "--secrets-mount-path", "/app/locust",
               "--command-line", cmd, "-o", "none")
            while az("container", "show", "-g", args.rg, "-n", name, "--query",
                     "containers[0].instanceView.currentState.state", "-o", "tsv") != "Terminated":
                time.sleep(10)
            logs = az("container", "logs", "-g", args.rg, "-n", name)
            az("container", "delete", "-g", args.rg, "-n", name, "--yes", "-o", "none")
            if "===CSV===" not in logs:
                raise SystemExit(f"locust run failed:\n{logs[-2000:]}")
            stats = stats_from_csv(logs.split("===CSV===", 1)[1].strip())
            label = f"{n} replica{'s' if n > 1 else ''}"
            results.append({"label": label, "containers": n, "users": args.users, "duration": args.duration, **stats})
            print(json.dumps(results[-1], indent=2))
    finally:
        az("containerapp", "update", "-n", args.app, "-g", args.rg, "--min-replicas", "1", "--max-replicas", "1",
           "--no-wait", "-o", "none")
        print("[ok] scaled back to 1 replica")

    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    write_summary(results, OUT)
    plot(results, OUT, title="Locust on Azure Container Apps")


if __name__ == "__main__":
    main()
