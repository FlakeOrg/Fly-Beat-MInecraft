# Running a local test server

Used for developing/testing `bot-bridge` and, later, for the agent's
training rollouts (M5+). Everything here is local-only, gitignored, and
disposable — nothing here needs to be shared or committed.

## 1. Java

Mineflayer/bot-bridge don't need Java, but a Paper/vanilla Minecraft
**server** does (Java 21+ for modern versions). If you don't already have a
JDK on PATH, grab a portable (no-installer, no-admin) build:

```
mkdir .toolchain
curl -L -o .toolchain/temurin21.zip "https://api.adoptium.net/v3/binary/latest/21/ga/windows/x64/jdk/hotspot/normal/eclipse?project=jdk"
# unzip it - .toolchain/jdk-21.x.x+y/bin/java.exe is the binary to use below
```

(`.toolchain/` is gitignored.)

## 2. Server jar

Get a Paper build (vanilla-compatible, faster, plugin-capable) for a version
your installed `mineflayer`/`minecraft-data` support — check
`bot-bridge/node_modules/minecraft-data/minecraft-data/data/pc/common/versions.json`
for the newest listed version, then find a **STABLE**-channel build for it
via the PaperMC v3 API:

```
curl -s "https://fill.papermc.io/v3/projects/paper/versions/<version>/builds"
# find the entry with "channel": "STABLE", grab downloads["server:default"].url
mkdir .mc-server
curl -L -o .mc-server/paper.jar "<that url>"
```

(`.mc-server/` is gitignored — it accumulates world data too.)

## 3. Accept the EULA + configure for local bot testing

```
echo "eula=true" > .mc-server/eula.txt
```

`.mc-server/server.properties` (created after the first run, then editable)
should have at least:

```
online-mode=false   # lets an offline-auth Mineflayer bot connect
server-port=25565
```

## 4. Start it

```
cd .mc-server
"<path to java>" -Xms1G -Xmx2G -jar paper.jar --nogui
```

Wait for `Done (...)! For help, type "help"` in the console/log before
connecting a bot.

## 5. Start bot-bridge against it

```
cd bot-bridge
npm install   # first time only
npm start
```

Defaults to `localhost:25565`, offline username `FlyBrain`, WebSocket bridge
on `ws://localhost:8081`. Override with env vars: `MC_HOST`, `MC_PORT`,
`MC_USERNAME`, `MC_VERSION`, `MC_AUTH`, `BRIDGE_PORT`.

## 6. Talk to it from Python

```python
from brain.agent.bridge_client import BridgeClient

with BridgeClient() as client:
    print(client.get_observation())
    client.do_action({"type": "move", "forward": True})
```

Requires `websockets>=12` (for `websockets.sync.client`) in whatever
Python environment you run this from — install it in `brain/`'s own venv,
not globally, so you don't clash with unrelated projects' pinned versions.
