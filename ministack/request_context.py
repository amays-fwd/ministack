"""Per-request context that cannot ride the shared handler signature.

AWS routes v1 REST resources on RAW path segments and decodes each segment
after splitting; ASGI hands services a pre-decoded ``scope["path"]``, which
collapses an encoded slash (%2F) inside a path parameter into a real slash
BEFORE segmentation — `/accounts/acct_x%2F` matched `{id}=acct_x` plus a
filtered empty segment and answered 200 where AWS surfaces the backend's 404.

``raw_path`` is the full request path exactly as received on the wire (still
percent-encoded), set by the ASGI handler. ``raw_execute_path`` is the
execute-api remainder after the addressing prefix and stage were stripped,
set by the execute-api dispatcher; the v1 gateway segments on it when
present. Contextvars keep this async-task-scoped without widening every
service's handler signature.
"""

import contextvars

raw_path: contextvars.ContextVar[str] = contextvars.ContextVar(
    "raw_path", default=""
)
raw_execute_path: contextvars.ContextVar[str] = contextvars.ContextVar(
    "raw_execute_path", default=""
)
