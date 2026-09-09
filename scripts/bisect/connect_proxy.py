#!/usr/bin/env python3
"""Minimal logging HTTP proxy (CONNECT tunnels + plain HTTP forwarding), stdlib only.

No TLS interception: for every tunnel it records the destination host, the bytes that flowed in
each direction and the duration, one JSON object per line. Point installers at it with
HTTPS_PROXY / HTTP_PROXY / ALL_PROXY=http://127.0.0.1:<port> and NO_PROXY=127.0.0.1,localhost.

    python connect_proxy.py serve --port 0 --log proxy.jsonl --port-file proxy.port
    python connect_proxy.py serve --refuse ...                  # 403 every request, log the attempt
    python connect_proxy.py serve --deny-hosts pypi.org,files.pythonhosted.org,github.com ...
    python connect_proxy.py summary proxy.jsonl [--since-ts T]

`--refuse` is how "offline" is measured rather than asserted: the child still has a proxy to talk
to, every attempt is answered 403 and recorded, so a run that claims to do no work has to prove it
made no connections. `--deny-hosts` is the same trick aimed at the package hosts only, which is what
the prefetch job needs: the swap must complete from the warm cache while PyPI and GitHub are dead.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import socket
import sys
import threading
import time
from collections import defaultdict
from urllib.parse import urlsplit

BUF = 1 << 16


def _now() -> float:
    return time.time()


class Proxy:
    def __init__(self, port: int, log_path: str, port_file: str | None,
                 refuse: bool = False, deny_hosts: tuple[str, ...] = ()):
        self.log_path = log_path
        self.refuse = refuse
        self.deny_hosts = tuple(h.strip().lower() for h in deny_hosts if h.strip())
        self.lock = threading.Lock()
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", port))
        self.srv.listen(256)
        self.port = self.srv.getsockname()[1]
        if port_file:
            with open(port_file, "w") as fh:
                fh.write(str(self.port))
        mode = "refuse-all" if refuse else (f"deny={','.join(self.deny_hosts)}" if self.deny_hosts else "allow-all")
        print(f"[proxy] listening on 127.0.0.1:{self.port} log={log_path} mode={mode}", flush=True)

    def denied(self, host: str | None) -> bool:
        if self.refuse:
            return True
        h = (host or "").lower()
        return any(h == d or h.endswith("." + d) for d in self.deny_hosts)

    def log(self, rec: dict) -> None:
        line = json.dumps(rec, separators=(",", ":"))
        with self.lock:
            with open(self.log_path, "a") as fh:
                fh.write(line + "\n")

    def serve(self) -> None:
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    @staticmethod
    def _read_head(conn: socket.socket) -> bytes:
        data = b""
        conn.settimeout(30)
        while b"\r\n\r\n" not in data:
            chunk = conn.recv(BUF)
            if not chunk:
                break
            data += chunk
            if len(data) > 1 << 20:
                break
        return data

    def handle(self, conn: socket.socket) -> None:
        t0 = _now()
        host = port = None
        method = "?"
        down = up = 0
        status = "ok"
        upstream = None
        try:
            head = self._read_head(conn)
            if not head:
                return
            line = head.split(b"\r\n", 1)[0].decode("latin-1")
            parts = line.split()
            if len(parts) < 2:
                return
            method, target = parts[0], parts[1]
            probe = target.rpartition(":")[0] if method == "CONNECT" else (urlsplit(target).hostname or "")
            if self.denied(probe):
                host, status = probe, "refused"
                port = 443 if method == "CONNECT" else 80
                conn.sendall(b"HTTP/1.1 403 Forbidden\r\nProxy-Agent: bisect-refuse\r\n"
                             b"Content-Length: 0\r\nConnection: close\r\n\r\n")
                return
            if method == "CONNECT":
                host, _, p = target.rpartition(":")
                port = int(p or 443)
                upstream = socket.create_connection((host, port), timeout=60)
                conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                initial = b""
            else:
                u = urlsplit(target)
                host = u.hostname or ""
                port = u.port or 80
                upstream = socket.create_connection((host, port), timeout=60)
                initial = head
            conn.settimeout(None)
            upstream.settimeout(None)
            if initial:
                upstream.sendall(initial)
                up += len(initial)
            socks = [conn, upstream]
            while True:
                r, _, x = select.select(socks, [], socks, 600)
                if x or not r:
                    status = "timeout" if not r else "error"
                    break
                done = False
                for s in r:
                    try:
                        data = s.recv(BUF)
                    except OSError:
                        data = b""
                    if not data:
                        done = True
                        break
                    if s is conn:
                        upstream.sendall(data)
                        up += len(data)
                    else:
                        conn.sendall(data)
                        down += len(data)
                if done:
                    break
        except Exception as exc:  # noqa: BLE001
            status = f"error:{type(exc).__name__}"
        finally:
            for s in (conn, upstream):
                try:
                    if s:
                        s.close()
                except OSError:
                    pass
            self.log({
                "ts": round(t0, 3), "host": host, "port": port, "method": method,
                "bytes_down": down, "bytes_up": up, "seconds": round(_now() - t0, 3), "status": status,
            })


def summary(path: str, since_ts: float | None, until_ts: float | None) -> dict:
    by_host: dict[str, dict] = defaultdict(lambda: {"bytes_down": 0, "bytes_up": 0, "connections": 0,
                                                    "refused": 0, "seconds": 0.0})
    total = 0
    connections = 0
    refused = 0
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if since_ts is not None and rec["ts"] < since_ts:
                    continue
                if until_ts is not None and rec["ts"] > until_ts:
                    continue
                h = by_host[rec.get("host") or "?"]
                h["bytes_down"] += rec["bytes_down"]
                h["bytes_up"] += rec["bytes_up"]
                h["connections"] += 1
                h["seconds"] += rec["seconds"]
                connections += 1
                if rec.get("status") == "refused":
                    h["refused"] += 1
                    refused += 1
                total += rec["bytes_down"]
    ordered = dict(sorted(by_host.items(), key=lambda kv: -kv[1]["bytes_down"]))
    return {"total_bytes_down": total, "connections": connections, "refused": refused, "by_host": ordered}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=0)
    s.add_argument("--log", required=True)
    s.add_argument("--port-file")
    s.add_argument("--refuse", action="store_true", help="answer 403 to every request and log it")
    s.add_argument("--deny-hosts", default="", help="comma list of hosts to 403 (suffix match); others pass")
    m = sub.add_parser("summary")
    m.add_argument("log")
    m.add_argument("--since-ts", type=float)
    m.add_argument("--until-ts", type=float)
    a = ap.parse_args()
    if a.cmd == "serve":
        Proxy(a.port, a.log, a.port_file, refuse=a.refuse,
              deny_hosts=tuple(a.deny_hosts.split(","))).serve()
    else:
        json.dump(summary(a.log, a.since_ts, a.until_ts), sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
