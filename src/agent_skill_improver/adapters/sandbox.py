"""The OS mechanism that removes the egress leg, and the probe that proves it.

Two implementations, chosen by host, plus a deliberate third state of *nothing
available*:

* **macOS** -- ``sandbox-exec`` with a generated profile. The profile allows the
  ordinary operations an interpreter needs, then denies ``network*`` and
  ``file-write*`` outright, then re-allows writes under one freshly created
  private directory. Order matters: in this profile language the last matching
  rule wins, so the narrow allow has to come after the broad deny.
* **Linux** -- ``bubblewrap`` with the network namespace unshared, ``/`` bound
  read-only, and the same private directory bound writable.
* **neither** -- :func:`detect` returns ``None`` and the caller refuses. There is
  no degraded mode on purpose. A mechanism that silently becomes advisory when
  the host is unusual is worse than no mechanism, because the receipt still says
  the run was confined.

The part that makes this evidence rather than configuration is :func:`verify`.
It runs a real child process twice over the same three probes -- connect a
socket, write outside the private directory, write inside it -- once unsandboxed
and once sandboxed, and requires the outcomes to differ in the specific way
claimed. Both halves are load-bearing:

* the sandboxed half must FAIL on the socket and the outside write. Without it,
  a profile with a typo reports success.
* the unsandboxed half must SUCCEED on all three. Without it, a probe that
  cannot connect for an unrelated reason (no interpreter, a bad path, a
  mis-serialised spec) reads exactly like a working sandbox. This is the same
  fail-open shape as a leak scanner that never fires.

The socket probe targets a listener this process opens on loopback, so
verification needs no network and no external host, and a machine with no
connectivity cannot make the sandboxed half pass for the wrong reason.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..domain.isolation import (
    MODE_BUBBLEWRAP,
    MODE_NONE,
    MODE_SANDBOX_EXEC,
)

#: Probe source, executed with ``-c`` so no file has to be readable inside the
#: sandbox and nothing is left on disk. It never raises: every outcome is
#: reported as text, because an exception would be indistinguishable from a
#: sandbox that killed the process for an unrelated reason.
PROBE_SOURCE = (
    "import json,socket,sys\n"
    "spec=json.loads(sys.argv[1])\n"
    "out={}\n"
    "try:\n"
    "    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)\n"
    "    s.settimeout(3.0)\n"
    "    s.connect((spec['host'],spec['port']))\n"
    "    s.close()\n"
    "    out['socket']='allowed'\n"
    "except BaseException as exc:\n"
    "    out['socket']='blocked:%s' % type(exc).__name__\n"
    "for key in ('outside_path','inside_path'):\n"
    "    try:\n"
    "        fh=open(spec[key],'w')\n"
    "        fh.write('probe')\n"
    "        fh.close()\n"
    "        out[key]='allowed'\n"
    "    except BaseException as exc:\n"
    "        out[key]='blocked:%s' % type(exc).__name__\n"
    "sys.stdout.write(json.dumps(out,sort_keys=True))\n"
)

PROBE_TIMEOUT_SECONDS = 60


def _sbpl_string(value: str) -> str:
    """Quote a path for the macOS sandbox profile language."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass(frozen=True)
class Mechanism:
    """A no-egress mechanism that exists on this host."""

    mode: str
    executable: str

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "executable": self.executable}


def detect(
    *,
    platform_name: str = sys.platform,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = lambda p: Path(p).exists(),
) -> Mechanism | None:
    """Find a supported mechanism, or return ``None`` so the caller can refuse.

    Injected lookups rather than module-level calls, so a test can exercise the
    unavailable branch on a host where the mechanism does exist. An
    "unavailable" path that is unreachable in the suite is an untested refusal.
    """
    if platform_name == "darwin":
        for path in ("/usr/bin/sandbox-exec", "/usr/local/bin/sandbox-exec"):
            if exists(path):
                return Mechanism(mode=MODE_SANDBOX_EXEC, executable=path)
        found = which("sandbox-exec")
        if found:
            return Mechanism(mode=MODE_SANDBOX_EXEC, executable=found)
        return None
    if platform_name.startswith("linux"):
        for name in ("bwrap", "bubblewrap"):
            found = which(name)
            if found:
                return Mechanism(mode=MODE_BUBBLEWRAP, executable=found)
        return None
    return None


class NoSandbox:
    """Runs a command unchanged. Legal ONLY for a synthetic fixture source.

    Exists as a real object rather than as ``None`` so the isolation mode is
    always a value that can be recorded in a command id and a receipt. A missing
    field reads as "not applicable"; ``mode = "none"`` reads as what it is.
    """

    mode = MODE_NONE

    def wrap(self, argv: Sequence[str], *, workdir: Path) -> list[str]:
        return list(argv)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "executable": ""}


class OsSandbox:
    """Wraps a command so it cannot reach the network or write outside ``workdir``."""

    def __init__(self, mechanism: Mechanism) -> None:
        self.mechanism = mechanism

    @property
    def mode(self) -> str:
        return self.mechanism.mode

    def profile_text(self, workdir: Path) -> str:
        """The generated macOS profile. Public so a test can read what shipped."""
        target = _sbpl_string(str(Path(workdir).resolve()))
        return (
            "(version 1)\n"
            ";; Allow the ordinary operations an interpreter needs, then remove the\n"
            ";; two authorities that make private data exfiltratable. Later rules win.\n"
            "(allow default)\n"
            "(deny network*)\n"
            "(deny file-write*)\n"
            f"(allow file-write* (subpath {target}))\n"
            '(allow file-write-data (literal "/dev/null") (literal "/dev/zero")'
            ' (literal "/dev/random") (literal "/dev/urandom")'
            ' (literal "/dev/dtracehelper"))\n'
            '(allow file-ioctl (literal "/dev/dtracehelper"))\n'
        )

    def wrap(self, argv: Sequence[str], *, workdir: Path) -> list[str]:
        """Return ``argv`` wrapped so it has no egress and one writable path.

        The working directory is deliberately **inherited** rather than changed
        to ``workdir``. Configured argv routinely contains repository-relative
        paths, and moving the child would break them while looking like a
        sandbox failure. Nothing is lost: the child still cannot write to the
        directory it starts in, because the only writable path is ``workdir``.
        """
        resolved = Path(workdir).resolve()
        if self.mode == MODE_SANDBOX_EXEC:
            return [
                self.mechanism.executable,
                "-p",
                self.profile_text(resolved),
                *argv,
            ]
        if self.mode == MODE_BUBBLEWRAP:
            return [
                self.mechanism.executable,
                "--unshare-net",
                "--unshare-ipc",
                "--unshare-uts",
                "--unshare-pid",
                "--die-with-parent",
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/run",
                "--bind",
                str(resolved),
                str(resolved),
                "--",
                *argv,
            ]
        raise AssertionError(f"unsupported isolation mode {self.mode!r}")

    def to_dict(self) -> dict[str, Any]:
        return self.mechanism.to_dict()


@dataclass(frozen=True)
class ProbeOutcome:
    ran: bool
    socket: str
    outside_write: str
    inside_write: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "socket": self.socket,
            "write_outside_private_dir": self.outside_write,
            "write_inside_private_dir": self.inside_write,
            "detail": self.detail,
        }


def _run_probe(argv: Sequence[str], *, cwd: Path, env: dict[str, str]) -> ProbeOutcome:
    try:
        proc = subprocess.run(  # noqa: S603 - argv is constructed, never shell
            list(argv),
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
            cwd=str(cwd),
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ProbeOutcome(False, "", "", "", detail=f"{type(exc).__name__}: {exc}")
    text = (proc.stdout or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return ProbeOutcome(
            False,
            "",
            "",
            "",
            detail=f"exit {proc.returncode}, unparseable probe output: {text[:200]!r} "
            f"stderr: {(proc.stderr or '').strip()[:200]!r}",
        )
    return ProbeOutcome(
        ran=True,
        socket=str(parsed.get("socket", "")),
        outside_write=str(parsed.get("outside_path", "")),
        inside_write=str(parsed.get("inside_path", "")),
        detail=f"exit {proc.returncode}",
    )


def verify(
    sandbox: OsSandbox,
    *,
    python: str = sys.executable,
    workdir_root: Path | None = None,
) -> dict[str, Any]:
    """Prove the sandbox blocks a socket and an escaping write. Two-sided.

    Returns a report; never raises for a failed verification, because the caller
    turns an unverified mechanism into a refusal and a raised exception here
    would be indistinguishable from a broken host.
    """
    root = Path(workdir_root) if workdir_root is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    inside = Path(tempfile.mkdtemp(prefix="asi-verify-in-", dir=str(root) if root else None))
    outside = Path(tempfile.mkdtemp(prefix="asi-verify-out-", dir=str(root) if root else None))
    inside = inside.resolve()
    outside = outside.resolve()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    reasons: list[str] = []
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        host, port = listener.getsockname()[0], listener.getsockname()[1]
        spec = json.dumps(
            {
                "host": host,
                "port": port,
                "outside_path": str(outside / "escaped.txt"),
                "inside_path": str(inside / "allowed.txt"),
            },
            sort_keys=True,
        )
        probe = [python, "-c", PROBE_SOURCE, spec]
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(inside)}

        control = _run_probe(probe, cwd=inside, env=env)
        confined = _run_probe(sandbox.wrap(probe, workdir=inside), cwd=inside, env=env)

        # The control half. Without it a probe that simply never worked would
        # look exactly like a sandbox that blocked everything.
        if not control.ran:
            reasons.append(f"the unsandboxed control probe did not run: {control.detail}")
        else:
            if control.socket != "allowed":
                reasons.append(
                    "the unsandboxed control could not open a loopback socket "
                    f"({control.socket}), so the socket check proves nothing"
                )
            if control.outside_write != "allowed":
                reasons.append(
                    "the unsandboxed control could not write outside the private directory "
                    f"({control.outside_write}), so the escape check proves nothing"
                )
            if control.inside_write != "allowed":
                reasons.append(
                    f"the unsandboxed control could not write inside it either "
                    f"({control.inside_write}); the probe itself is broken"
                )

        # The confined half.
        if not confined.ran:
            reasons.append(f"the sandboxed probe did not run: {confined.detail}")
        else:
            if not confined.socket.startswith("blocked"):
                reasons.append(f"the sandbox did NOT block an outbound socket ({confined.socket})")
            if not confined.outside_write.startswith("blocked"):
                reasons.append(
                    "the sandbox did NOT block a write outside the private directory "
                    f"({confined.outside_write})"
                )
            if confined.inside_write != "allowed":
                reasons.append(
                    "the sandbox blocked the write INSIDE the private directory "
                    f"({confined.inside_write}); the model would have no scratch space"
                )
        return {
            "mode": sandbox.mode,
            "mechanism": sandbox.to_dict(),
            "ok": not reasons,
            "control": control.to_dict(),
            "sandboxed": confined.to_dict(),
            "reasons": reasons,
            "checks": [
                "an outbound socket to a loopback listener is refused",
                "a write outside the fresh private temp directory is refused",
                "a write inside that directory succeeds",
                "all three behave the opposite way with no sandbox (the control)",
            ],
        }
    except OSError as exc:  # pragma: no cover - host networking is unusual
        return {
            "mode": sandbox.mode,
            "mechanism": sandbox.to_dict(),
            "ok": False,
            "control": {},
            "sandboxed": {},
            "reasons": [f"could not open a loopback listener for the probe: {exc}"],
            "checks": [],
        }
    finally:
        try:
            listener.close()
        finally:
            shutil.rmtree(inside, ignore_errors=True)
            shutil.rmtree(outside, ignore_errors=True)
