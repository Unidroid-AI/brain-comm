# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``brain-comm`` console script: serve, check, and smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from brain_comm.check import CheckError, check_submission
from brain_comm.client import BrainError


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="brain-comm")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="serve one replica per GPU")
    serve.add_argument("--entry", default=None)
    serve.add_argument("--replicas", type=int, default=1)
    serve.add_argument("--socket-dir", default=None)
    serve.add_argument("--submission", default=None)
    serve.add_argument("--max-gb", type=float, default=None)

    check = sub.add_parser("check", help="validate a submission and run the image hook")
    check.add_argument("--submission", default="/submission")
    check.add_argument("--max-gb", type=float, default=None)

    probe = sub.add_parser("smoke", help="check a live replica")
    probe.add_argument("--socket", required=True)
    probe.add_argument("--timeout", type=float, default=30.0)

    args = parser.parse_args(argv)
    if args.command == "serve":
        forwarded = []
        if args.entry:
            forwarded += ["--entry", args.entry]
        forwarded += ["--replicas", str(args.replicas)]
        if args.socket_dir:
            forwarded += ["--socket-dir", args.socket_dir]
        if args.submission:
            forwarded += ["--submission", args.submission]
        if args.max_gb is not None:
            forwarded += ["--max-gb", str(args.max_gb)]
        from brain_comm.serve import main as serve_main

        serve_main(forwarded)
        return
    if args.command == "check":
        try:
            check_submission(Path(args.submission), max_gb=args.max_gb)
        except CheckError as exc:
            json.dump({"ok": False, "errors": exc.errors}, sys.stdout)
            sys.stdout.write("\n")
            raise SystemExit(1) from exc
        json.dump({"ok": True}, sys.stdout)
        sys.stdout.write("\n")
        return
    from brain_comm.smoke import SmokeError, smoke

    try:
        spec = smoke(args.socket, timeout_s=args.timeout)
    except (SmokeError, BrainError) as exc:
        json.dump({"ok": False, "error": str(exc)}, sys.stdout)
        sys.stdout.write("\n")
        raise SystemExit(1) from exc
    json.dump({"ok": True, "horizon": spec["horizon"], "action_dim": spec["action_dim"]}, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
