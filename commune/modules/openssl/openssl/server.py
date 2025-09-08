#!/usr/bin/env python3
"""
openssl.py — one-stop helper for SSL/TLS tasks using the OpenSSL CLI (with graceful fallbacks)

Features
- Generate RSA or ECDSA private keys
- Create self‑signed certificates (with SAN support)
- Create CSRs (certificate signing requests)
- Verify cert ↔ key match
- Generate Diffie–Hellman parameters (dhparam)
- Build hardened SSL contexts for servers & clients
- Launch a minimal HTTPS static server
- Export to PKCS#12 bundles (.p12)

Requirements
- Prefers the system `openssl` binary if available (most Linux/macOS distros).
- If `openssl` is missing, certain features still work via Python stdlib (contexts & serving),
  but key/cert generation requires OpenSSL or the `cryptography` package (not bundled).

Usage (CLI)
  python openssl.py gen-key --algo rsa --bits 2048 --out key.pem
  python openssl.py gen-key --algo ecdsa --curve prime256v1 --out key.pem
  python openssl.py self-signed --key key.pem --out cert.pem \
      --cn example.com --san DNS:example.com,DNS:www.example.com,IP:127.0.0.1 --days 365
  python openssl.py csr --key key.pem --out csr.pem --cn example.com --org "My Org"
  python openssl.py verify --key key.pem --cert cert.pem
  python openssl.py dhparam --bits 2048 --out dhparam.pem
  python openssl.py p12 --cert cert.pem --key key.pem --out bundle.p12 --name "My Cert" --password secret
  python openssl.py serve --cert cert.pem --key key.pem --dir ./public --host 0.0.0.0 --port 4443

Programmatic
  from openssl import OpenSSLManager
  m = OpenSSLManager()
  m.generate_private_key('rsa', bits=2048, out_path='key.pem')
  m.generate_self_signed_cert('key.pem', 'cert.pem', cn='example.com', san=['DNS:example.com'])
  m.start_https_server('cert.pem','key.pem', directory='public', host='0.0.0.0', port=4443)
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# ------------------------- Utility & Errors -------------------------

class OpenSSLError(RuntimeError):
    pass


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _run(cmd: Sequence[str], *, input_bytes: bytes | None = None) -> Tuple[str, str]:
    """Run a command and return (stdout, stderr). Raises OpenSSLError on nonzero exit."""
    try:
        proc = subprocess.run(
            list(cmd),
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as e:
        raise OpenSSLError(f"Command not found: {cmd[0]}") from e

    if proc.returncode != 0:
        raise OpenSSLError(
            f"Command failed ({proc.returncode}): {' '.join(shlex.quote(c) for c in cmd)}\n" \
            f"stderr: {proc.stderr.decode(errors='ignore')}"
        )
    return proc.stdout.decode(), proc.stderr.decode()


# ------------------------- Main Manager -------------------------

@dataclass
class Subject:
    C: str | None = None
    ST: str | None = None
    L: str | None = None
    O: str | None = None
    OU: str | None = None
    CN: str | None = None
    emailAddress: str | None = None

    def to_x509_subj(self) -> str:
        parts = []
        for k in ["C", "ST", "L", "O", "OU", "CN", "emailAddress"]:
            v = getattr(self, k)
            if v:
                # Escape slashes to avoid breaking the subject
                v = str(v).replace("/", r"\/")
                parts.append(f"/{k}={v}")
        return "".join(parts) if parts else "/CN=localhost"


class OpenSSLManager:
    def __init__(self, openssl_bin: Optional[str] = None):
        self.openssl = openssl_bin or _which("openssl")
        if not self.openssl:
            # We keep going for server contexts/serving, but key/cert ops will error
            print(
                "[warn] 'openssl' CLI not found. Key/cert generation and CSR features will not work.",
                file=sys.stderr,
            )

    # --------------------- Key Generation ---------------------
    def generate_private_key(
        self,
        algo: str = "rsa",
        *,
        bits: int = 2048,
        curve: str = "prime256v1",
        out_path: str | os.PathLike = "key.pem",
        passphrase: Optional[str] = None,
    ) -> Path:
        """Generate a private key (RSA or ECDSA) using OpenSSL CLI."""
        if not self.openssl:
            raise OpenSSLError("OpenSSL binary not found. Cannot generate keys.")

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if algo.lower() == "rsa":
            cmd = [self.openssl, "genpkey", "-algorithm", "RSA", "-pkeyopt", f"rsa_keygen_bits:{bits}"]
        elif algo.lower() in {"ecdsa", "ec"}:
            cmd = [self.openssl, "ecparam", "-name", curve, "-genkey"]
        else:
            raise ValueError("Unsupported algo. Use 'rsa' or 'ecdsa'.")

        if passphrase and algo.lower() == "rsa":
            cmd += ["-aes-256-cbc", "-pass", f"pass:{passphrase}"]

        stdout, _ = _run(cmd)
        out.write_text(stdout)
        os.chmod(out, 0o600)
        return out

    # --------------------- CSR Generation ---------------------
    def generate_csr(
        self,
        key_path: str | os.PathLike,
        *,
        out_path: str | os.PathLike = "csr.pem",
        subject: Optional[Subject] = None,
        cn: Optional[str] = None,
        san: Sequence[str] | None = None,
        passphrase: Optional[str] = None,
    ) -> Path:
        """Create a CSR from an existing private key. SAN can be ["DNS:example.com", "IP:127.0.0.1"]."""
        if not self.openssl:
            raise OpenSSLError("OpenSSL binary not found. Cannot generate CSR.")

        subj = subject or Subject(CN=cn or "localhost")
        key_path = Path(key_path)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Use a temp config for SAN if provided
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "req.cnf"
            exts = "" if not san else textwrap.dedent(
                f"""
                [ req ]
                distinguished_name = dn
                req_extensions = v3_req

                [ dn ]

                [ v3_req ]
                subjectAltName = {','.join(san)}
                """
            )
            if san:
                config_path.write_text(exts)

            cmd = [
                self.openssl, "req", "-new",
                "-key", str(key_path),
                "-subj", subj.to_x509_subj(),
                "-out", str(out),
            ]
            if san:
                cmd += ["-config", str(config_path), "-reqexts", "v3_req"]
            if passphrase:
                cmd += ["-passin", f"pass:{passphrase}"]

            _run(cmd)
        return out

    # --------------------- Self-signed Certificate ---------------------
    def generate_self_signed_cert(
        self,
        key_path: str | os.PathLike,
        cert_out_path: str | os.PathLike = "cert.pem",
        *,
        subject: Optional[Subject] = None,
        cn: Optional[str] = None,
        san: Sequence[str] | None = None,
        days: int = 365,
        passphrase: Optional[str] = None,
    ) -> Path:
        """Create a self-signed x509 certificate."""
        if not self.openssl:
            raise OpenSSLError("OpenSSL binary not found. Cannot generate certificate.")

        subj = subject or Subject(CN=cn or "localhost")
        key_path = Path(key_path)
        cert_out = Path(cert_out_path)
        cert_out.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "x509.cnf"
            exts = "" if not san else textwrap.dedent(
                f"""
                [ req ]
                distinguished_name = dn
                x509_extensions = v3_req

                [ dn ]

                [ v3_req ]
                subjectAltName = {','.join(san)}
                keyUsage = critical, digitalSignature, keyEncipherment
                extendedKeyUsage = serverAuth, clientAuth
                """
            )
            if san:
                config_path.write_text(exts)

            cmd = [
                self.openssl, "req", "-x509", "-new",
                "-key", str(key_path),
                "-sha256", "-days", str(days),
                "-subj", subj.to_x509_subj(),
                "-out", str(cert_out),
            ]
            if san:
                cmd += ["-config", str(config_path), "-extensions", "v3_req"]
            if passphrase:
                cmd += ["-passin", f"pass:{passphrase}"]

            _run(cmd)
        return cert_out

    # --------------------- Verify cert ↔ key ---------------------
    def verify_cert_matches_key(
        self,
        key_path: str | os.PathLike,
        cert_path: str | os.PathLike,
        *,
        passphrase: Optional[str] = None,
    ) -> bool:
        """Check whether certificate and private key share the same public key."""
        if not self.openssl:
            raise OpenSSLError("OpenSSL binary not found. Cannot verify match.")

        key_path = Path(key_path)
        cert_path = Path(cert_path)

        key_cmd = [self.openssl, "pkey", "-in", str(key_path), "-pubout"]
        if passphrase:
            key_cmd += ["-passin", f"pass:{passphrase}"]
        key_pub, _ = _run(key_cmd)

        cert_pub, _ = _run([self.openssl, "x509", "-in", str(cert_path), "-pubkey", "-noout"])
        return key_pub.strip() == cert_pub.strip()

    # --------------------- DH params ---------------------
    def generate_dhparam(self, *, bits: int = 2048, out_path: str | os.PathLike = "dhparam.pem") -> Path:
        if not self.openssl:
            raise OpenSSLError("OpenSSL binary not found. Cannot generate dhparam.")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        _run([self.openssl, "dhparam", "-out", str(out), str(bits)])
        return out

    # --------------------- PKCS#12 ---------------------
    def export_pkcs12(
        self,
        cert_path: str | os.PathLike,
        key_path: str | os.PathLike,
        *,
        out_path: str | os.PathLike = "bundle.p12",
        name: str = "certificate",
        password: Optional[str] = None,
        passphrase: Optional[str] = None,
    ) -> Path:
        if not self.openssl:
            raise OpenSSLError("OpenSSL binary not found. Cannot export PKCS#12.")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.openssl, "pkcs12", "-export",
            "-in", str(cert_path),
            "-inkey", str(key_path),
            "-name", name,
            "-out", str(out),
        ]
        if password:
            cmd += ["-passout", f"pass:{password}"]
        if passphrase:
            cmd += ["-passin", f"pass:{passphrase}"]
        _run(cmd)
        return out

    # --------------------- SSL Contexts ---------------------
    def create_server_context(
        self,
        cert_path: str | os.PathLike,
        key_path: str | os.PathLike,
        *,
        require_client_cert: bool = False,
        ca_path: Optional[str] = None,
        alpn_protocols: Optional[List[str]] = None,
    ) -> ssl.SSLContext:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        # Hardened defaults
        if hasattr(ctx, 'minimum_version'):
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3
        # Prefer server ciphers
        if hasattr(ctx, 'set_ciphers'):
            try:
                ctx.set_ciphers('ECDHE+AESGCM:!aNULL:!MD5:!RC4')
            except ssl.SSLError:
                pass
        ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
        if require_client_cert:
            ctx.verify_mode = ssl.CERT_REQUIRED
            if ca_path:
                ctx.load_verify_locations(cafile=str(ca_path))
        if alpn_protocols and hasattr(ctx, 'set_alpn_protocols'):
            try:
                ctx.set_alpn_protocols(alpn_protocols)
            except NotImplementedError:
                pass
        return ctx

    def create_client_context(
        self,
        *,
        ca_path: Optional[str] = None,
        insecure: bool = False,
        alpn_protocols: Optional[List[str]] = None,
    ) -> ssl.SSLContext:
        ctx = ssl.create_default_context()
        if insecure:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        elif ca_path:
            ctx.load_verify_locations(cafile=str(ca_path))
        if alpn_protocols and hasattr(ctx, 'set_alpn_protocols'):
            try:
                ctx.set_alpn_protocols(alpn_protocols)
            except NotImplementedError:
                pass
        return ctx

    # --------------------- Minimal HTTPS Server ---------------------
    def start_https_server(
        self,
        cert_path: str | os.PathLike,
        key_path: str | os.PathLike,
        *,
        directory: str | os.PathLike = ".",
        host: str = "0.0.0.0",
        port: int = 4443,
        require_client_cert: bool = False,
        ca_path: Optional[str] = None,
    ) -> None:
        from http.server import SimpleHTTPRequestHandler
        from socketserver import ThreadingMixIn
        from functools import partial

        class ThreadingHTTPServer(ThreadingMixIn, socketserver.TCPServer):
            daemon_threads = True
            allow_reuse_address = True

        import socketserver  # local import to avoid confusion with typing

        handler = partial(SimpleHTTPRequestHandler, directory=str(Path(directory).resolve()))
        httpd = socketserver.TCPServer((host, port), handler)
        ctx = self.create_server_context(cert_path, key_path,
                                         require_client_cert=require_client_cert,
                                         ca_path=ca_path,
                                         alpn_protocols=["h2", "http/1.1"])
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        print(f"Serving HTTPS on https://{host}:{port} from {Path(directory).resolve()}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            httpd.server_close()


# ------------------------- CLI -------------------------

def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSL/TLS helper using OpenSSL")
    sub = p.add_subparsers(dest="cmd", required=True)

    # gen-key
    s = sub.add_parser("gen-key", help="Generate a private key")
    s.add_argument("--algo", choices=["rsa", "ecdsa"], default="rsa")
    s.add_argument("--bits", type=int, default=2048, help="RSA bits")
    s.add_argument("--curve", default="prime256v1", help="ECDSA curve name")
    s.add_argument("--out", required=True, help="Output key path")
    s.add_argument("--passphrase", help="Encrypt RSA key with passphrase")

    # self-signed
    s = sub.add_parser("self-signed", help="Create a self-signed certificate")
    s.add_argument("--key", required=True)
    s.add_argument("--out", required=True, help="Output cert path")
    s.add_argument("--days", type=int, default=365)
    s.add_argument("--cn", help="Common Name (CN)")
    s.add_argument("--san", help="SubjectAltName list, e.g. DNS:ex.com,IP:127.0.0.1")
    s.add_argument("--C")
    s.add_argument("--ST")
    s.add_argument("--L")
    s.add_argument("--O")
    s.add_argument("--OU")
    s.add_argument("--emailAddress")
    s.add_argument("--passphrase")

    # csr
    s = sub.add_parser("csr", help="Create a CSR from a key")
    s.add_argument("--key", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--cn")
    s.add_argument("--san", help="SubjectAltName list, e.g. DNS:ex.com,IP:127.0.0.1")
    s.add_argument("--C")
    s.add_argument("--ST")
    s.add_argument("--L")
    s.add_argument("--O")
    s.add_argument("--OU")
    s.add_argument("--emailAddress")
    s.add_argument("--passphrase")

    # verify
    s = sub.add_parser("verify", help="Verify a cert matches a key")
    s.add_argument("--key", required=True)
    s.add_argument("--cert", required=True)
    s.add_argument("--passphrase")

    # dhparam
    s = sub.add_parser("dhparam", help="Generate Diffie-Hellman parameters")
    s.add_argument("--bits", type=int, default=2048)
    s.add_argument("--out", required=True)

    # pkcs12
    s = sub.add_parser("p12", help="Export to PKCS#12 bundle")
    s.add_argument("--cert", required=True)
    s.add_argument("--key", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--name", default="certificate")
    s.add_argument("--password")
    s.add_argument("--passphrase")

    # serve
    s = sub.add_parser("serve", help="Run a simple HTTPS static file server")
    s.add_argument("--cert", required=True)
    s.add_argument("--key", required=True)
    s.add_argument("--dir", default=".")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=4443)
    s.add_argument("--require-client-cert", action="store_true")
    s.add_argument("--ca")

    return p.parse_args(argv)


def _subject_from_args(ns: argparse.Namespace) -> Subject:
    return Subject(C=ns.C, ST=ns.ST, L=ns.L, O=ns.O, OU=ns.OU, CN=getattr(ns, 'cn', None), emailAddress=ns.emailAddress)


def main(argv: Sequence[str] | None = None) -> int:
    ns = _parse_args(argv or sys.argv[1:])
    m = OpenSSLManager()

    if ns.cmd == "gen-key":
        m.generate_private_key(ns.algo, bits=ns.bits, curve=ns.curve, out_path=ns.out, passphrase=ns.passphrase)
        print(f"Wrote {ns.out}")
        return 0

    if ns.cmd == "self-signed":
        san = [s.strip() for s in ns.san.split(',')] if ns.san else None
        subj = _subject_from_args(ns)
        m.generate_self_signed_cert(ns.key, ns.out, subject=subj, cn=ns.cn, san=san, days=ns.days, passphrase=ns.passphrase)
        print(f"Wrote {ns.out}")
        return 0

    if ns.cmd == "csr":
        san = [s.strip() for s in ns.san.split(',')] if ns.san else None
        subj = _subject_from_args(ns)
        m.generate_csr(ns.key, out_path=ns.out, subject=subj, cn=ns.cn, san=san, passphrase=ns.passphrase)
        print(f"Wrote {ns.out}")
        return 0

    if ns.cmd == "verify":
        ok = m.verify_cert_matches_key(ns.key, ns.cert, passphrase=ns.passphrase)
        print("Match" if ok else "NOT a match")
        return 0 if ok else 2

    if ns.cmd == "dhparam":
        m.generate_dhparam(bits=ns.bits, out_path=ns.out)
        print(f"Wrote {ns.out}")
        return 0

    if ns.cmd == "p12":
        m.export_pkcs12(ns.cert, ns.key, out_path=ns.out, name=ns.name, password=ns.password, passphrase=ns.passphrase)
        print(f"Wrote {ns.out}")
        return 0

    if ns.cmd == "serve":
        m.start_https_server(ns.cert, ns.key, directory=ns.dir, host=ns.host, port=ns.port,
                             require_client_cert=ns.require_client_cert, ca_path=ns.ca)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
