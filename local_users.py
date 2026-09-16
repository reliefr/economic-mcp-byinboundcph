"""
E-mail + password login hosted by the MCP server itself ("local users").

For teams without Google Workspace or Microsoft 365 that still want to share the server
as a claude.ai / Claude Desktop connector (which requires an OAuth login). Users are
plain environment variables:

    MCP_USER_ANNA=anna@firma.dk:pbkdf2_sha256$600000$<salt>$<hash>

`python scripts/new_user.py anna anna@firma.dk` creates the line and shows the password
once. Deleting the variable removes the user.

What this module owns and how it stays safe:
- Passwords are stored as PBKDF2-HMAC-SHA256 hashes (600 000 rounds, random salt) and
  compared in constant time. Unknown e-mails go through the same hashing so timing does
  not reveal which addresses exist.
- Five failed attempts per e-mail or per client IP lock login for 15 minutes.
- Each login page belongs to a single, unguessable, short-lived transaction created by the
  OAuth authorize request, so the form cannot be replayed or driven from another site.
- Authorization codes live 5 minutes and are single-use; PKCE (S256) is enforced by the
  MCP SDK. Access tokens live 1 hour in memory. Refresh tokens live 30 days, are rotated on
  use and stored only as SHA-256 hashes on disk (FASTMCP_HOME), so logins survive deploys.
- The login page sends no cookies, sets a strict Content-Security-Policy and is never cached.
- Anyone can register an OAuth client (that is how MCP clients work), so the login page
  always shows where the user will be sent after login, and the operator can restrict
  redirect destinations with MCP_ALLOWED_CLIENT_REDIRECT_URIS. Unknown destination = do not log in.

The protocol parts (client registration, /authorize, /token, metadata, PKCE checks) come
from FastMCP and the MCP SDK; this file only decides who the user is.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import logging
import os
import re
import secrets
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlencode

from mcp.server.auth.provider import (
    AuthorizationCode,
    RegistrationError,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.routing import Route

from fastmcp.server.auth.auth import AccessToken, ClientRegistrationOptions, OAuthProvider, RevocationOptions
from fastmcp.server.auth.redirect_validation import validate_redirect_uri

logger = logging.getLogger("economic-mcp.auth.users")

USER_PREFIX = "MCP_USER_"
HASH_SCHEME = "pbkdf2_sha256"
PBKDF2_ROUNDS = 600_000
MIN_PASSWORD_LENGTH = 12
SCOPE = "economic:access"

AUTH_CODE_TTL = 5 * 60
ACCESS_TOKEN_TTL = 60 * 60
REFRESH_TOKEN_TTL = 30 * 24 * 60 * 60
LOGIN_TXN_TTL = 10 * 60
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class UserConfigError(ValueError):
    """An MCP_USER_* variable is malformed."""


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def hash_password(password: str, rounds: int = PBKDF2_ROUNDS) -> str:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"{HASH_SCHEME}${rounds}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, rounds, salt, digest = stored.split("$")
        if scheme != HASH_SCHEME:
            return False
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), _unb64(salt), int(rounds))
        return hmac.compare_digest(candidate, _unb64(digest))
    except (ValueError, TypeError):
        return False


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


_DUMMY_HASH = hash_password("dummy-password-for-constant-time", rounds=PBKDF2_ROUNDS)


def _looks_like_hash(value: str) -> bool:
    parts = value.split("$")
    return len(parts) == 4 and parts[0] == HASH_SCHEME and parts[1].isdigit()


# ---------------------------------------------------------------------------
# Users from environment
# ---------------------------------------------------------------------------


def parse_users(env: Mapping[str, str]) -> dict[str, str]:
    """MCP_USER_<NAME>=email:hash  ->  {email: hash}. Raises UserConfigError on mistakes."""
    users: dict[str, str] = {}
    for variable in sorted(env):
        if not variable.startswith(USER_PREFIX):
            continue
        name = variable[len(USER_PREFIX):].lower()
        if not _NAME_PATTERN.match(name):
            raise UserConfigError(f"{variable}: the name after {USER_PREFIX} may only contain letters, digits, '-' and '_'")
        value = (env.get(variable) or "").strip()
        if not value:
            continue
        email, sep, stored = value.partition(":")
        email = email.strip().lower()
        if not sep or not _EMAIL_PATTERN.match(email) or not _looks_like_hash(stored.strip()):
            raise UserConfigError(
                f"{variable} must look like email:{HASH_SCHEME}$rounds$salt$hash "
                "(create it with: python scripts/new_user.py <name> <email>)"
            )
        if email in users:
            raise UserConfigError(f"{variable}: {email} is defined twice")
        users[email] = stored.strip()
    return users


# ---------------------------------------------------------------------------
# Brute-force protection
# ---------------------------------------------------------------------------


@dataclass
class _Attempts:
    failures: list[float] = field(default_factory=list)

    def prune(self, now: float) -> None:
        self.failures = [t for t in self.failures if now - t < LOCKOUT_SECONDS]

    def locked(self, now: float) -> bool:
        self.prune(now)
        return len(self.failures) >= MAX_FAILED_ATTEMPTS


class LoginThrottle:
    def __init__(self) -> None:
        self._by_key: dict[str, _Attempts] = {}

    def is_locked(self, *keys: str) -> bool:
        now = time.time()
        return any(self._by_key.get(k, _Attempts()).locked(now) for k in keys if k)

    def record_failure(self, *keys: str) -> None:
        now = time.time()
        for key in keys:
            if key:
                self._by_key.setdefault(key, _Attempts()).failures.append(now)

    def reset(self, *keys: str) -> None:
        for key in keys:
            self._by_key.pop(key, None)


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------


@dataclass
class _LoginTransaction:
    client: OAuthClientInformationFull
    params: AuthorizationParams
    created_at: float
    failures: int = 0


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class LocalUsersProvider(OAuthProvider):
    """OAuth authorization server whose users are the MCP_USER_* variables."""

    def __init__(
        self,
        users: Mapping[str, str],
        *,
        base_url: str,
        state_dir: Path,
        server_name: str = "e-conomic MCP",
        access_token_ttl: int = ACCESS_TOKEN_TTL,
        refresh_token_ttl: int = REFRESH_TOKEN_TTL,
        allowed_client_redirect_uris: Optional[list[str]] = None,
    ) -> None:
        if not users:
            raise UserConfigError("At least one MCP_USER_<NAME> is required for e-mail login")
        super().__init__(
            base_url=base_url,
            client_registration_options=ClientRegistrationOptions(
                enabled=True, valid_scopes=[SCOPE], default_scopes=[SCOPE]
            ),
            revocation_options=RevocationOptions(enabled=True),
            required_scopes=[SCOPE],
        )
        self._users = {email.lower(): stored for email, stored in users.items()}
        self._server_name = server_name
        self._access_token_ttl = int(access_token_ttl)
        self._refresh_token_ttl = int(refresh_token_ttl)
        self._allowed_redirects = list(allowed_client_redirect_uris) if allowed_client_redirect_uris is not None else None
        self._state_dir = Path(state_dir)
        self._state_file = self._state_dir / "local-users-state.json"
        self.throttle = LoginThrottle()

        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._transactions: dict[str, _LoginTransaction] = {}
        self._auth_codes: dict[str, tuple[AuthorizationCode, str]] = {}  # code -> (code, email)
        self._access_tokens: dict[str, AccessToken] = {}  # sha256(token) -> token
        self._refresh_tokens: dict[str, dict[str, Any]] = {}  # sha256(token) -> record
        self._load_state()

    # ----- persistence (clients + hashed refresh tokens) -----

    def _load_state(self) -> None:
        try:
            raw = json.loads(self._state_file.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s (%s); starting with empty login state", self._state_file, exc)
            return
        for client_id, data in (raw.get("clients") or {}).items():
            try:
                self._clients[client_id] = OAuthClientInformationFull.model_validate(data)
            except Exception:  # noqa: BLE001 - a corrupt entry must not block startup
                logger.warning("Skipping unreadable client registration %s", client_id)
        now = time.time()
        for token_hash, record in (raw.get("refresh_tokens") or {}).items():
            if record.get("expires_at", 0) > now and record.get("email", "").lower() in self._users:
                self._refresh_tokens[token_hash] = record

    def _save_state(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "clients": {cid: c.model_dump(mode="json") for cid, c in self._clients.items()},
            "refresh_tokens": self._refresh_tokens,
        }
        fd, tmp_path = tempfile.mkstemp(dir=self._state_dir, prefix=".state-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self._state_file)
        except OSError:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    # ----- clients -----

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if client_info.client_id is None:
            raise ValueError("client_id is required")
        for uri in client_info.redirect_uris or []:
            if not validate_redirect_uri(uri, self._allowed_redirects):
                logger.warning("Rejected client registration with redirect URI %s", uri)
                raise RegistrationError(error="invalid_redirect_uri", error_description=f"redirect_uri {uri} is not allowed on this server")
        self._clients[client_info.client_id] = client_info
        self._save_state()

    # ----- authorize -> login page -----

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        if client.client_id is None or client.client_id not in self._clients:
            raise AuthorizeError(error="unauthorized_client", error_description="Unknown client")
        if not validate_redirect_uri(params.redirect_uri, self._allowed_redirects):
            raise AuthorizeError(error="invalid_request", error_description="redirect_uri is not allowed on this server")
        self._prune_transactions()
        txn = secrets.token_urlsafe(32)
        self._transactions[txn] = _LoginTransaction(client=client, params=params, created_at=time.time())
        return f"{str(self.base_url).rstrip('/')}/login?{urlencode({'txn': txn})}"

    def _prune_transactions(self) -> None:
        now = time.time()
        for txn in [t for t, tx in self._transactions.items() if now - tx.created_at > LOGIN_TXN_TTL]:
            self._transactions.pop(txn, None)

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def login_page(self, request: Request) -> Response:
        txn = request.query_params.get("txn", "")
        self._prune_transactions()
        if txn not in self._transactions:
            return self._html(_render_page(self._server_name, None, error="expired", txn=None), status=400)
        transaction = self._transactions[txn]
        return self._html(_render_page(self._server_name, transaction.client.client_name, txn=txn, destination=_destination(transaction)))

    async def login_submit(self, request: Request) -> Response:
        form = await request.form()
        txn = str(form.get("txn", ""))
        email = str(form.get("email", "")).strip().lower()
        password = str(form.get("password", ""))
        self._prune_transactions()
        transaction = self._transactions.get(txn)
        if transaction is None:
            return self._html(_render_page(self._server_name, None, error="expired", txn=None), status=400)

        ip = self._client_ip(request)
        client_name = transaction.client.client_name
        if self.throttle.is_locked(f"email:{email}", f"ip:{ip}"):
            logger.warning("Login locked out (too many failures) email=%s ip=%s", email, ip)
            return self._html(_render_page(self._server_name, client_name, error="locked", txn=txn, destination=_destination(transaction)), status=429)

        stored = self._users.get(email)
        # Verify against a dummy hash when the e-mail is unknown so both paths cost the same.
        ok = verify_password(password, stored if stored else _DUMMY_HASH) and stored is not None
        if not ok:
            self.throttle.record_failure(f"email:{email}", f"ip:{ip}")
            transaction.failures += 1
            logger.warning("Login failed email=%s ip=%s", email, ip)
            if transaction.failures >= MAX_FAILED_ATTEMPTS:
                self._transactions.pop(txn, None)
                return self._html(_render_page(self._server_name, None, error="expired", txn=None), status=400)
            return self._html(_render_page(self._server_name, client_name, error="invalid", txn=txn, destination=_destination(transaction)), status=401)

        self.throttle.reset(f"email:{email}")
        self._transactions.pop(txn, None)
        redirect = self._issue_code(transaction, email)
        logger.info("Login accepted for %s (client %s)", email, client_name or transaction.client.client_id)
        return RedirectResponse(redirect, status_code=302, headers=_NO_STORE)

    def _issue_code(self, transaction: _LoginTransaction, email: str) -> str:
        params = transaction.params
        code = secrets.token_urlsafe(32)
        scopes = [s for s in (params.scopes or []) if s == SCOPE] or [SCOPE]
        self._auth_codes[code] = (
            AuthorizationCode(
                code=code,
                client_id=transaction.client.client_id or "",
                redirect_uri=params.redirect_uri,
                redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
                scopes=scopes,
                expires_at=time.time() + AUTH_CODE_TTL,
                code_challenge=params.code_challenge,
                resource=params.resource,
            ),
            email,
        )
        return construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state)

    def _html(self, body: str, status: int = 200) -> HTMLResponse:
        return HTMLResponse(body, status_code=status, headers={**_NO_STORE, **_SECURITY_HEADERS})

    # ----- codes and tokens -----

    async def load_authorization_code(self, client: OAuthClientInformationFull, authorization_code: str) -> AuthorizationCode | None:
        entry = self._auth_codes.get(authorization_code)
        if entry is None:
            return None
        code, _email = entry
        if code.client_id != client.client_id or code.expires_at < time.time():
            self._auth_codes.pop(authorization_code, None)
            return None
        return code

    async def exchange_authorization_code(self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode) -> OAuthToken:
        entry = self._auth_codes.pop(authorization_code.code, None)
        if entry is None:
            raise TokenError("invalid_grant", "Authorization code not found or already used")
        _code, email = entry
        return self._issue_tokens(client, email, authorization_code.scopes)

    def _issue_tokens(self, client: OAuthClientInformationFull, email: str, scopes: list[str]) -> OAuthToken:
        if client.client_id is None:
            raise TokenError("invalid_client", "Client ID is required")
        scopes = sorted(set(scopes) | {SCOPE})
        access = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(32)
        now = int(time.time())
        self._access_tokens[_token_hash(access)] = AccessToken(
            token=access,
            client_id=client.client_id,
            scopes=scopes,
            expires_at=now + self._access_token_ttl,
            claims={"email": email, "sub": email},
        )
        self._refresh_tokens[_token_hash(refresh)] = {
            "client_id": client.client_id,
            "email": email,
            "scopes": scopes,
            "expires_at": now + self._refresh_token_ttl,
        }
        self._save_state()
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=self._access_token_ttl,
            refresh_token=refresh,
            scope=" ".join(scopes),
        )

    async def load_access_token(self, token: str) -> AccessToken | None:  # type: ignore[override]
        key = _token_hash(token)
        access = self._access_tokens.get(key)
        if access is None:
            return None
        if access.expires_at is not None and access.expires_at < time.time():
            self._access_tokens.pop(key, None)
            return None
        if access.claims.get("email") not in self._users:  # user removed while token alive
            self._access_tokens.pop(key, None)
            return None
        return access

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        key = _token_hash(refresh_token)
        record = self._refresh_tokens.get(key)
        if record is None or record["client_id"] != client.client_id:
            return None
        if record["expires_at"] < time.time() or record["email"] not in self._users:
            self._refresh_tokens.pop(key, None)
            self._save_state()
            return None
        return RefreshToken(token=refresh_token, client_id=record["client_id"], scopes=record["scopes"], expires_at=record["expires_at"])

    async def exchange_refresh_token(self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]) -> OAuthToken:
        key = _token_hash(refresh_token.token)
        record = self._refresh_tokens.pop(key, None)
        if record is None:
            raise TokenError("invalid_grant", "Refresh token not found or already used")
        if not set(scopes).issubset(set(record["scopes"])):
            raise TokenError("invalid_scope", "Requested scopes exceed those originally granted")
        return self._issue_tokens(client, record["email"], scopes or record["scopes"])

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        if isinstance(token, AccessToken):
            self._access_tokens.pop(_token_hash(token.token), None)
        else:
            self._refresh_tokens.pop(_token_hash(token.token), None)
            self._save_state()

    # ----- routes -----

    def get_routes(self, mcp_path: str | None = None) -> list[Route]:
        routes = super().get_routes(mcp_path)
        routes.append(Route("/login", endpoint=self.login_page, methods=["GET"]))
        routes.append(Route("/login", endpoint=self.login_submit, methods=["POST"]))
        return routes


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------

_NO_STORE = {"Cache-Control": "no-store", "Pragma": "no-cache"}
_SECURITY_HEADERS = {
           "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self' https://claude.ai https://*.claude.ai http://localhost:* http://127.0.0.1:*; frame-ancestors 'none'; base-uri 'none'",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

_ERRORS = {
    "invalid": "Forkert e-mail eller kodeord. / Wrong e-mail or password.",
    "locked": "For mange forsøg. Prøv igen om 15 minutter. / Too many attempts. Try again in 15 minutes.",
    "expired": "Login-siden er udløbet. Start forbindelsen igen fra din app. / This login page has expired. Start the connection again from your app.",
}


def _destination(transaction: _LoginTransaction) -> str:
    uri = transaction.params.redirect_uri
    host = uri.host or ""
    if uri.port and uri.port not in (80, 443):
        host = f"{host}:{uri.port}"
    return f"{uri.scheme}://{host}"


def _render_page(
    server_name: str,
    client_name: Optional[str],
    *,
    txn: Optional[str],
    error: Optional[str] = None,
    destination: Optional[str] = None,
) -> str:
    title = html.escape(server_name)
    client = html.escape((client_name or "En app")[:60])
    message = f'<p class="error">{html.escape(_ERRORS[error])}</p>' if error else ""
    where = (
        f'<p class="dest">Efter login sendes du tilbage til <b>{html.escape(destination)}</b>. '
        "Genkender du ikke adressen, så log ikke ind. / After login you are sent back to this address; "
        "if you do not recognise it, do not sign in.</p>"
        if destination
        else ""
    )
    form = (
        f"""<form method="post" action="/login" autocomplete="on">
      <input type="hidden" name="txn" value="{html.escape(txn)}">
      <label for="email">E-mail</label>
      <input id="email" name="email" type="email" autocomplete="username" required autofocus>
      <label for="password">Kodeord / Password</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required>
      <button type="submit">Log ind / Sign in</button>
    </form>"""
        if txn
        else ""
    )
    return f"""<!doctype html>
<html lang="da"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} – log ind</title>
<style>
  body {{ font-family: -apple-system, system-ui, Segoe UI, sans-serif; background: #f5f6f8; margin: 0; display: flex; min-height: 100vh; align-items: center; justify-content: center; }}
  main {{ background: #fff; border-radius: 12px; padding: 32px; width: 360px; box-shadow: 0 8px 30px rgba(0,0,0,.08); }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }} p {{ color: #555; margin: 0 0 20px; font-size: 14px; }}
  label {{ display: block; font-size: 13px; margin: 12px 0 4px; }}
  input {{ width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #ccd; border-radius: 8px; font-size: 15px; }}
  button {{ margin-top: 18px; width: 100%; padding: 11px; border: 0; border-radius: 8px; background: #1a56db; color: #fff; font-size: 15px; cursor: pointer; }}
  .error {{ color: #b42318; background: #fef3f2; padding: 10px; border-radius: 8px; }}
  .dest {{ background: #f0f4ff; padding: 10px; border-radius: 8px; font-size: 13px; color: #1e3a8a; }}
</style></head>
<body><main>
  <h1>{title}</h1>
  <p>"{client}" beder om adgang til regnskabet (navnet er oplyst af appen selv). Log ind med den e-mail og det kodeord, du har fået af jeres administrator.</p>
  {where}
  {message}
  {form}
</main></body></html>"""
