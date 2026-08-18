# Label Studio with a gatekeeper in front of it.
#
# The stock image is untouched apart from adding nginx; all the behaviour lives
# in nginx.conf.template and start.sh.
#
# Pinned deliberately. The `latest` tag on this repository moves with hourly
# development builds, so an unpinned deployment would quietly change Label
# Studio version on its next rebuild. 1.23.0 is the version everything here was
# tested against. To move up, change this line and rebuild.
FROM heartexlabs/label-studio:1.23.0

# The base image runs as uid 1001; we need root to install packages.
USER root

# This image is Alpine-based, so apk rather than apt-get:
#   nginx             the gatekeeper
#   gettext-envsubst  provides envsubst, which fills in the nginx config at
#                     start-up. Alpine splits this out -- the plain `gettext`
#                     package does not contain envsubst.
#   bash              start.sh needs it; Alpine ships only busybox sh, which
#                     has no brace expansion or `set -o pipefail`
RUN apk add --no-cache nginx gettext-envsubst bash \
 && command -v envsubst && command -v nginx && command -v bash

COPY nginx.conf.template /app/nginx.conf.template
COPY start.sh            /app/start.sh

RUN chmod +x /app/start.sh \
 && mkdir -p /tmp/nginx /label-studio/data /var/lib/nginx /var/log/nginx \
 && chmod -R a+rwX /app /tmp/nginx /label-studio/data /var/lib/nginx /var/log/nginx

ENV LABEL_STUDIO_BASE_DATA_DIR=/label-studio/data \
    PORT=7860

EXPOSE 7860

# The base image sets ENTRYPOINT to its own launcher, which would swallow
# anything given as CMD and treat it as an argument. Replace it outright so our
# start-up script is what actually runs.
ENTRYPOINT ["/app/start.sh"]
CMD []
