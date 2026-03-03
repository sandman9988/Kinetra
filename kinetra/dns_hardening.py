"""
Broker/API-agnostic DNS hardening helpers.

This module centralizes endpoint hostname validation and DNS resolution policy
so connectors can enforce consistent controls before opening sockets.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set

logger = logging.getLogger(__name__)


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _env_value(name: str) -> Optional[str]:
    if name in os.environ:
        return os.getenv(name)
    project_root = Path(__file__).resolve().parent.parent
    merged: dict[str, str] = {}
    for p in (project_root / ".env", project_root / ".env.openapi"):
        merged.update(_parse_env_file(p))
    return merged.get(name)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_value(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> List[str]:
    raw = _env_value(name) or ""
    return [part.strip() for part in raw.split(",") if part.strip()]


def _valid_resolver_ip(raw: str) -> bool:
    try:
        ipaddress.ip_address(raw)
        return True
    except ValueError:
        return False


def _default_public_resolvers() -> List[str]:
    # Cloudflare, Google, Quad9 (+ secondary anycast).
    return ["1.1.1.1", "1.0.0.1", "8.8.8.8", "8.8.4.4", "9.9.9.9", "149.112.112.112"]


@dataclass(frozen=True)
class DNSHardeningPolicy:
    """Policy controls for DNS resolution and endpoint validation."""

    allowed_hosts: Set[str] = field(default_factory=set)
    allowed_suffixes: Set[str] = field(default_factory=set)
    min_unique_ips: int = 1
    block_private_ips: bool = True
    fail_closed: bool = True
    resolvers: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "DNSHardeningPolicy":
        """Build policy from environment variables."""
        resolvers = [r for r in _env_list("KINETRA_DNS_RESOLVERS") if _valid_resolver_ip(r)]
        if not resolvers and _env_bool("KINETRA_DNS_USE_PUBLIC_RESOLVERS", False):
            resolvers = _default_public_resolvers()
        return cls(
            allowed_hosts={h.lower() for h in _env_list("KINETRA_DNS_ALLOW_HOSTS")},
            allowed_suffixes={
                s.lower().lstrip(".") for s in _env_list("KINETRA_DNS_ALLOW_SUFFIXES")
            },
            min_unique_ips=max(int(_env_value("KINETRA_DNS_MIN_UNIQUE_IPS") or "1"), 1),
            block_private_ips=_env_bool("KINETRA_DNS_BLOCK_PRIVATE", True),
            fail_closed=_env_bool("KINETRA_DNS_FAIL_CLOSED", True),
            resolvers=resolvers,
        )


def _host_allowed(host: str, policy: DNSHardeningPolicy) -> bool:
    host_l = host.lower().strip(".")
    if not policy.allowed_hosts and not policy.allowed_suffixes:
        return True
    if host_l in policy.allowed_hosts:
        return True
    return any(
        host_l == suffix or host_l.endswith(f".{suffix}") for suffix in policy.allowed_suffixes
    )


def _ip_allowed(ip: str, policy: DNSHardeningPolicy) -> bool:
    try:
        parsed = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if not policy.block_private_ips:
        return True
    if parsed.is_private or parsed.is_loopback or parsed.is_link_local:
        return False
    if parsed.is_multicast or parsed.is_reserved or parsed.is_unspecified:
        return False
    return True


def resolve_and_validate_host(host: str, *, service_name: str = "endpoint") -> List[str]:
    """
    Resolve and validate a hostname according to DNS hardening policy.

    Returns a list of validated IPs (unique, sorted). Raises ValueError when
    policy is violated in fail-closed mode.
    """
    policy = DNSHardeningPolicy.from_env()
    target = (host or "").strip()
    if not target:
        raise ValueError(f"{service_name}: empty host")

    if not _host_allowed(target, policy):
        msg = f"{service_name}: host {target!r} blocked by allowlist policy"
        if policy.fail_closed:
            raise ValueError(msg)
        logger.warning(msg)
        return []

    resolved_ips: Set[str] = set()

    # Optional resolver override pool (e.g., Cloudflare/Google/Quad9).
    if policy.resolvers:
        try:
            import dns.resolver  # type: ignore[import-untyped]

            timeout_s = float(_env_value("KINETRA_DNS_QUERY_TIMEOUT_S") or "2.0")
            lifetime_s = float(_env_value("KINETRA_DNS_QUERY_LIFETIME_S") or "4.0")
            # Query each provider explicitly so one failing resolver
            # does not block the aggregate result.
            for provider in policy.resolvers:
                resolver = dns.resolver.Resolver(configure=False)
                resolver.nameservers = [provider]
                resolver.timeout = timeout_s
                resolver.lifetime = lifetime_s
                provider_hits = 0
                for rrtype in ("A", "AAAA"):
                    try:
                        answers = resolver.resolve(target, rrtype, raise_on_no_answer=False)
                        if answers:
                            for rr in answers:
                                resolved_ips.add(str(rr))
                                provider_hits += 1
                    except Exception:
                        continue
                if provider_hits > 0:
                    logger.info(
                        "%s: DNS provider %s returned %d records for %s",
                        service_name,
                        provider,
                        provider_hits,
                        target,
                    )
                else:
                    logger.warning(
                        "%s: DNS provider %s returned no usable records for %s",
                        service_name,
                        provider,
                        target,
                    )
        except ImportError:
            logger.warning(
                "%s: KINETRA_DNS_RESOLVERS is set but dnspython is not installed; "
                "falling back to system resolver",
                service_name,
            )

    # Fallback: OS/system resolver.
    if not resolved_ips:
        try:
            infos = socket.getaddrinfo(target, None, proto=socket.IPPROTO_TCP)
            resolved_ips = {info[4][0] for info in infos if info and info[4]}
        except socket.gaierror as exc:
            msg = f"{service_name}: DNS resolution failed for {target!r}: {exc}"
            if policy.fail_closed:
                raise ValueError(msg) from exc
            logger.warning(msg)
            return []

    filtered = [ip for ip in sorted(resolved_ips) if _ip_allowed(ip, policy)]

    if len(filtered) < policy.min_unique_ips:
        msg = (
            f"{service_name}: host {target!r} resolved to {len(filtered)} validated IPs "
            f"(required >= {policy.min_unique_ips})"
        )
        if policy.fail_closed:
            raise ValueError(msg)
        logger.warning(msg)
        return filtered

    logger.info("%s: DNS validation ok for %s -> %s", service_name, target, ", ".join(filtered))
    return filtered


def expand_endpoint_candidates(
    hosts: Iterable[str],
    *,
    include_resolved_ips: bool = False,
    service_name: str = "endpoint",
) -> List[str]:
    """
    Build ordered endpoint candidates from hostnames and optional resolved IPs.
    """
    ordered: List[str] = []
    seen: Set[str] = set()
    for raw in hosts:
        host = (raw or "").strip()
        if not host:
            continue
        key = host.lower()
        if key not in seen:
            ordered.append(host)
            seen.add(key)
        try:
            ips = resolve_and_validate_host(host, service_name=service_name)
        except ValueError:
            continue
        if not include_resolved_ips:
            continue
        for ip in ips:
            if ip not in seen:
                ordered.append(ip)
                seen.add(ip)
    return ordered


def _tcp_probe(host: str, port: int, timeout_s: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _tcp_probe_latency_ms(host: str, port: int, timeout_s: float) -> Optional[float]:
    start = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return (time.perf_counter() - start) * 1000.0
    except OSError:
        return None


def rank_reachable_endpoints(
    candidates: Iterable[str],
    *,
    port: int,
    timeout_s: float = 2.0,
    service_name: str = "endpoint",
) -> List[tuple[str, float]]:
    """Return reachable endpoints ranked by ascending TCP connect latency."""
    ranked: List[tuple[str, float]] = []
    for candidate in candidates:
        latency_ms = _tcp_probe_latency_ms(candidate, port=port, timeout_s=timeout_s)
        if latency_ms is None:
            logger.warning("%s: endpoint probe failed %s:%d", service_name, candidate, port)
            continue
        logger.info(
            "%s: endpoint reachable %s:%d latency=%.1fms",
            service_name,
            candidate,
            port,
            latency_ms,
        )
        ranked.append((candidate, latency_ms))
    ranked.sort(key=lambda x: x[1])
    return ranked


def select_reachable_endpoint(
    candidates: Iterable[str],
    *,
    port: int,
    timeout_s: float = 2.0,
    service_name: str = "endpoint",
) -> Optional[str]:
    """
    Return first TCP-reachable endpoint from *candidates*, else ``None``.
    """
    ranked = rank_reachable_endpoints(
        candidates,
        port=port,
        timeout_s=timeout_s,
        service_name=service_name,
    )
    if not ranked:
        return None
    selected, latency_ms = ranked[0]
    logger.info(
        "%s: selected lowest-latency endpoint %s:%d (%.1fms)",
        service_name,
        selected,
        port,
        latency_ms,
    )
    return selected
