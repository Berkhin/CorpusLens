# syntax=docker/dockerfile:1
#
# Development image for the React SPA — the Vite dev server with HMR.
#
# Users never build this: the root `Dockerfile` compiles the SPA into the
# production image, and nothing serves it separately there. This exists so a
# contributor gets hot reload without installing Node 24 on the host.
#
# Build context is the repository root (see docker-compose.yml), so the COPY
# paths below are repo-relative.

FROM node:24-alpine

WORKDIR /app

# Manifests only, so `npm ci` is cached independently of source edits. The
# source itself arrives as a bind mount at runtime; node_modules is kept out of
# its way by a named volume (see docker-compose.yml).
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

EXPOSE 5173

# --host binds 0.0.0.0 so the published port is reachable from the host; the
# port itself stays with vite.config.ts, which pins 5173 with `strictPort`
# because the API whitelists exactly that origin for CORS.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
