#!/usr/bin/env python3
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request


def post_json(url: str, body: bytes, timeout: float) -> tuple[int, float, bytes]:
    t0 = time.monotonic()
    req = request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            dt = time.monotonic() - t0
            return resp.getcode(), dt, data
    except Exception:
        dt = time.monotonic() - t0
        return 0, dt, b""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--path", default="/generate")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--body")
    grp.add_argument(
        "--bodies",
        help="Comma-separated list of JSON payload files. Round-robin is used.",
    )
    ap.add_argument("--total", type=int, default=80)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}{args.path}"
    if args.bodies:
        body_paths = [p.strip() for p in args.bodies.split(",") if p.strip()]
        if not body_paths:
            print("--bodies provided but empty after parsing", file=sys.stderr)
            return 2
        bodies = [open(p, "rb").read() for p in body_paths]
    else:
        bodies = [open(args.body, "rb").read()]
    t_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(post_json, url, bodies[(i - 1) % len(bodies)], args.timeout): i
            for i in range(1, args.total + 1)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            code, dt, data = fut.result()
            open(f"{args.outdir}/meta_{i}.txt", "w").write(
                f"code:{code} time:{dt:.6f}\n"
            )
            if data:
                open(f"{args.outdir}/out_{i}.json", "wb").write(data)
            results.append((i, code, dt))
    t_end = time.time()
    summary = {
        "total": args.total,
        "concurrency": args.concurrency,
        "window_seconds": t_end - t_start,
        "ok": sum(1 for _, c, _ in results if c == 200),
        "latencies": [dt for _, _, dt in results],
    }
    open(f"{args.outdir}/http_summary.json", "w").write(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    sys.exit(main())
