"""Network workaround for huggingface.co CloudFront edge failures.

On this node the CloudFront edges that `huggingface.co` resolves to often fail
to answer SYN. At times all four A records were dead. Other HF CloudFront
distributions such as `cdn-lfs.hf.co` stay healthy, and CloudFront edges route
by SNI, so sending a huggingface.co request to one of those edges returns 200
(the certificate is also selected by SNI, so validation passes).

Behavior:
  1. Probe the target host's A records over TCP and keep only the live ones.
  2. If all are dead, borrow a live edge from another HF CloudFront distribution.
  3. Cache the result in process and on disk with a TTL.

No IP is hardcoded, so this keeps working when edges are replaced.
Disable with NET_FIX=0
"""

import json
import os
import socket
import time

if os.environ.get("NET_FIX", "1") != "0":
    _PROBE_TIMEOUT = float(os.environ.get("NET_PROBE_TIMEOUT", "2.5"))
    _CONNECT_TIMEOUT = float(os.environ.get("NET_CONNECT_TIMEOUT", "10"))
    _CACHE_TTL = float(os.environ.get("NET_CACHE_TTL", "900"))
    _CACHE_PATH = os.environ.get("NET_CACHE_PATH", "/NHNHOME/hf-cache/.alive_edges.json")

    # hosts that need fixing (HF API and web)
    _TARGET_HOSTS = {"huggingface.co", "www.huggingface.co"}
    # HF hosts on the same CloudFront we can borrow an edge from
    _DONOR_HOSTS = (
        "cdn-lfs.hf.co",
        "cdn-lfs-us-1.hf.co",
        "cdn-lfs-eu-1.hf.co",
        "cas-bridge.xethub.hf.co",
        "transfer.xethub.hf.co",
    )

    _orig_getaddrinfo = socket.getaddrinfo
    _orig_create_connection = socket.create_connection

    _mem: dict[str, list[str]] = {}

    def _load_cache() -> dict:
        try:
            with open(_CACHE_PATH) as f:
                d = json.load(f)
            if time.time() - d.get("ts", 0) < _CACHE_TTL:
                return d.get("hosts", {})
        except Exception:
            pass
        return {}

    def _save_cache(hosts: dict) -> None:
        try:
            tmp = f"{_CACHE_PATH}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump({"ts": time.time(), "hosts": hosts}, f)
            os.replace(tmp, _CACHE_PATH)
        except Exception:
            pass

    _disk = _load_cache()

    def _probe(ip: str, port: int, family: int) -> bool:
        s = socket.socket(family, socket.SOCK_STREAM)
        s.settimeout(_PROBE_TIMEOUT)
        try:
            s.connect((ip, port))
            return True
        except Exception:
            return False
        finally:
            s.close()

    def _ipv4_of(host: str, port: int) -> list[str]:
        try:
            res = _orig_getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        except Exception:
            return []
        seen, out = set(), []
        for _f, _t, _p, _c, sa in res:
            if sa[0] not in seen:
                seen.add(sa[0])
                out.append(sa[0])
        return out

    def _find_alive(host: str, port: int) -> list[str]:
        alive = [ip for ip in _ipv4_of(host, port) if _probe(ip, port, socket.AF_INET)]
        if alive:
            return alive
        # all own edges are dead, borrow one from another HF distribution
        for donor in _DONOR_HOSTS:
            borrowed = [ip for ip in _ipv4_of(donor, port) if _probe(ip, port, socket.AF_INET)]
            if borrowed:
                return borrowed
        return []

    def _getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if (
            not isinstance(host, str)
            or host not in _TARGET_HOSTS
            or port not in (80, 443)
            or type not in (0, socket.SOCK_STREAM)
        ):
            return _orig_getaddrinfo(host, port, family, type, proto, flags)

        key = f"{host}:{port}"
        alive = _mem.get(key)
        if alive is None:
            alive = _disk.get(key)
            if alive is not None and not _probe(alive[0], port, socket.AF_INET):
                alive = None  # stale cache
        if alive is None:
            alive = _find_alive(host, port)
            if alive:
                _disk[key] = alive
                _save_cache(_disk)
        _mem[key] = alive

        if not alive:
            return _orig_getaddrinfo(host, port, family, type, proto, flags)
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))
            for ip in alive
        ]

    def _create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, *a, **kw):
        if timeout is socket._GLOBAL_DEFAULT_TIMEOUT or timeout is None:
            timeout = _CONNECT_TIMEOUT
        return _orig_create_connection(address, timeout, *a, **kw)

    socket.getaddrinfo = _getaddrinfo
    socket.create_connection = _create_connection
