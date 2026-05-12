FROM node:20-bookworm-slim

# Build deps for native modules that may not have a prebuilt for this
# (Node version, libc, arch) combo:
#   @discordjs/opus → make + g++ + libopus-dev
#   sodium-native (pulled by @discordjs/voice) → libsodium-dev
#   @snazzah/davey (DAVE / voice gateway v8 E2EE) → cmake + g++ for the
#     fallback source build; the npm install will prefer a prebuilt if
#     prebuild-install finds one for linux-x64 Node 20.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        make \
        g++ \
        cmake \
        git \
        libopus-dev \
        libsodium-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

COPY src ./src

# Ensure the socket directory exists at the agreed path.
RUN mkdir -p /tmp/skryba

ENV NODE_ENV=production

CMD ["node", "src/index.js"]
