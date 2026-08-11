"""AISL-GPU4 네트워크 우회 (huggingface.co CloudFront 엣지 장애).

이 노드에서 `huggingface.co`가 해석되는 CloudFront 엣지(icn57, 서울)는
SYN에 응답하지 않는 경우가 잦다. 관측상 4개 A 레코드가 전부 죽는 구간도
있었다. 반면 `cdn-lfs.hf.co` 등 다른 HF CloudFront 배포의 엣지는 정상이며,
CloudFront 엣지는 SNI로 배포를 라우팅하므로 huggingface.co 요청을 그
엣지로 보내도 200이 돌아온다 (인증서도 SNI 기준이라 검증 통과).

동작:
  1. 대상 호스트의 A 레코드를 짧은 TCP 프로브로 걸러 살아있는 것만 쓴다.
  2. 전부 죽었으면, 살아있는 다른 HF CloudFront 엣지로 대체한다.
  3. 결과는 프로세스 + 디스크에 캐시한다 (TTL).

IP를 하드코딩하지 않으므로 엣지가 교체돼도 유효하다.
비활성화: NET_FIX=0
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

    # 고쳐야 하는 호스트 (HF API/웹)
    _TARGET_HOSTS = {"huggingface.co", "www.huggingface.co"}
    # 엣지를 빌려올 수 있는, 같은 CloudFront 위의 HF 호스트들
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
        # 자기 엣지가 전부 죽음 -> 다른 HF CloudFront 배포의 엣지를 빌린다
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
                alive = None  # 캐시가 상했다
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
