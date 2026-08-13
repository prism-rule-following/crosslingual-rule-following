#!/usr/bin/env bash
# Starts the gatekeeper on the public port and Label Studio behind it.
set -euo pipefail

: "${ADMIN_KEY:?ADMIN_KEY is not set — the gatekeeper has no key to check against}"

# The gatekeeper takes the public port the host gives us; Label Studio hides
# behind it on a different one. These must never be equal -- Cloud Run sets
# PORT=8080, which is also Label Studio's usual port, and the two would fight
# over the socket and the container would never come up.
export LISTEN_PORT="${PORT:-7860}"
export INTERNAL_PORT="${INTERNAL_PORT:-8080}"
if [ "$LISTEN_PORT" = "$INTERNAL_PORT" ]; then
  export INTERNAL_PORT=$((LISTEN_PORT + 1))
fi
export MIME_TYPES="${MIME_TYPES:-/etc/nginx/mime.types}"

# `logs` is here because nginx opens its compiled-in default error log, relative
# to the prefix, before it ever reads our config -- without it every start-up
# prints a spurious alert.
mkdir -p /tmp/nginx/{logs,body,proxy,fastcgi,uwsgi,scgi}

envsubst '${ADMIN_KEY} ${LISTEN_PORT} ${INTERNAL_PORT} ${MIME_TYPES}' \
  < /app/nginx.conf.template > /tmp/nginx/nginx.conf

nginx -t -c /tmp/nginx/nginx.conf -p /tmp/nginx
nginx -c /tmp/nginx/nginx.conf -p /tmp/nginx

echo "gatekeeper on ${LISTEN_PORT}; label studio on ${INTERNAL_PORT}"

# Label Studio only ever listens on the loopback address: every request from
# outside has to come through the gatekeeper.
exec label-studio start \
  --host "${LABEL_STUDIO_HOST:-}" \
  --internal-host 127.0.0.1 \
  --port "${INTERNAL_PORT}"
