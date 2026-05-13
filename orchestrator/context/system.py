"""System probe — live machine state via read-only subprocess calls."""

from __future__ import annotations

import datetime
import logging

from orchestrator.context.base import ContextBlock, estimate_tokens
from orchestrator.core.security import safe_run

log = logging.getLogger(__name__)

# Keyword → subsystem mapping
_SUBSYSTEM_KEYWORDS: dict[str, set[str]] = {
    "memory": {"ram", "memória", "memory", "swap", "livre", "free", "disponível", "available"},
    "gpu": {"gpu", "vram", "nvidia", "cuda", "gráfica", "graphics"},
    "disk": {"disco", "disk", "storage", "armazenamento", "espaço", "space"},
    "cpu": {"cpu", "processador", "processor", "carga", "load", "uptime", "cores"},
    "processes": {"processos", "processes", "correr", "running", "consumir", "consuming"},
    "system": {"sistema", "system", "kernel", "driver", "máquina", "machine", "hardware"},
    "network": {"rede", "network", "ip", "interface"},
    "temperature": {"temperatura", "temperature", "temp"},
}


def _detect_subsystems(query: str) -> set[str]:
    q_lower = query.lower()
    words = {w.strip(".,!?:;\"'()[]{}") for w in q_lower.split()}
    found: set[str] = set()
    for subsystem, keywords in _SUBSYSTEM_KEYWORDS.items():
        if words & keywords:
            found.add(subsystem)
    if not found:
        found = {"memory", "cpu", "disk"}
    return found


def _collect_memory() -> str:
    out = safe_run(["free", "-h"])
    return f"## Memory\n```\n{out}\n```" if out else ""


def _collect_gpu() -> str:
    out = safe_run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader"])
    return f"## GPU\n```\n{out}\n```" if out else ""


def _collect_disk() -> str:
    out = safe_run(["df", "-h", "--output=source,size,used,avail,pcent", "/", "/home"])
    return f"## Disk\n```\n{out}\n```" if out else ""


def _collect_cpu() -> str:
    parts: list[str] = []
    uptime = safe_run(["uptime"])
    if uptime:
        parts.append(f"Uptime/Load: {uptime}")
    nproc = safe_run(["nproc"])
    if nproc:
        parts.append(f"Cores: {nproc}")
    return "## CPU\n```\n" + "\n".join(parts) + "\n```" if parts else ""


def _collect_processes() -> str:
    out = safe_run(["ps", "aux", "--sort=-%cpu"])
    if out:
        lines = out.split("\n")[:8]
        return "## Top processes\n```\n" + "\n".join(lines) + "\n```"
    return ""


def _collect_system() -> str:
    uname = safe_run(["uname", "-a"])
    return f"## System\n```\n{uname}\n```" if uname else ""


def _collect_network() -> str:
    out = safe_run(["ip", "-br", "addr"])
    if out:
        lines = out.split("\n")[:6]
        return "## Network\n```\n" + "\n".join(lines) + "\n```"
    return ""


def _collect_temperature() -> str:
    gpu_temp = safe_run(["nvidia-smi", "--query-gpu=name,temperature.gpu", "--format=csv,noheader"])
    parts: list[str] = []
    if gpu_temp:
        parts.append(f"GPU: {gpu_temp}°C")
    sensors = safe_run(["sensors"])
    if sensors:
        temp_lines = [ln for ln in sensors.split("\n") if "°C" in ln][:5]
        parts.extend(temp_lines)
    return "## Temperature\n```\n" + "\n".join(parts) + "\n```" if parts else ""


_COLLECTORS: dict[str, callable] = {
    "memory": _collect_memory,
    "gpu": _collect_gpu,
    "disk": _collect_disk,
    "cpu": _collect_cpu,
    "processes": _collect_processes,
    "system": _collect_system,
    "network": _collect_network,
    "temperature": _collect_temperature,
}


class SystemProbeProvider:
    """Live system state via read-only subprocess calls."""

    @property
    def name(self) -> str:
        return "system"

    def get_context(self, query: str, *, budget_tokens: int = 2000) -> ContextBlock | None:
        subsystems = _detect_subsystems(query)
        sections: list[str] = []
        for name in ("system", "cpu", "memory", "gpu", "disk", "processes", "network", "temperature"):
            if name not in subsystems:
                continue
            collector = _COLLECTORS.get(name)
            if collector:
                section = collector()
                if section:
                    sections.append(section)

        if not sections:
            return None

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = f"[SYSTEM STATE — {timestamp}]\n\n" + "\n\n".join(sections) + "\n\n[/SYSTEM STATE]"
        return ContextBlock(
            source="system",
            content=content,
            token_estimate=estimate_tokens(content),
        )

    def health(self) -> bool:
        return True
