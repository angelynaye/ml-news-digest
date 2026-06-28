"""SMTP delivery. Supports macOS Keychain for the password."""
from __future__ import annotations
import logging
import smtplib
import subprocess
from email.message import EmailMessage

log = logging.getLogger(__name__)


def _keychain_password(service: str) -> str | None:
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
        log.warning("keychain miss for service=%s: %s", service, out.stderr.strip())
    except Exception as e:
        log.warning("keychain lookup failed: %s", e)
    return None


def send(smtp_cfg: dict, recipients: list[str], subject: str, html: str, text: str) -> None:
    pw = None
    if smtp_cfg.get("keychain_service"):
        pw = _keychain_password(smtp_cfg["keychain_service"])
    if not pw:
        pw = smtp_cfg.get("password")
    if not pw:
        raise RuntimeError(
            "No SMTP password available. Either set smtp.password in config or run:\n"
            f"  security add-generic-password -a \"$USER\" -s {smtp_cfg.get('keychain_service', 'news-digest-smtp')} -w 'APP_PASSWORD'"
        )

    host = smtp_cfg["host"]; port = int(smtp_cfg.get("port", 587))
    user = smtp_cfg["username"]
    from_name = smtp_cfg.get("from_name", "Daily Digest")
    from_addr = smtp_cfg.get("from_address", user)

    with smtplib.SMTP(host, port, timeout=30) as s:
        s.ehlo(); s.starttls(); s.ehlo()
        s.login(user, pw)
        for rcpt in recipients:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"]    = f"{from_name} <{from_addr}>"
            msg["To"]      = rcpt
            msg.set_content(text)
            msg.add_alternative(html, subtype="html")
            s.send_message(msg)
            log.info("sent to %s", rcpt)
