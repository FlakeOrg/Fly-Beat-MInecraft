// Connects a Mineflayer bot to a Minecraft server and exposes it over a
// WebSocket JSON-RPC-ish API so brain/ (Python) never touches the Minecraft
// protocol directly.
//
// Protocol: client sends { id, method, params }, server replies
// { id, result } or { id, error }. Unsolicited { type: "event", event, ... }
// messages are broadcast for lifecycle changes (spawn/death/kicked/end).
//
// Config via env vars: MC_HOST, MC_PORT, MC_USERNAME, MC_VERSION, MC_AUTH,
// BRIDGE_PORT.

import mineflayer from "mineflayer";
import minecraftData from "minecraft-data";
import { WebSocketServer } from "ws";
import { buildObservation } from "./observation.js";
import { performAction } from "./actions.js";

const MC_HOST = process.env.MC_HOST || "localhost";
const MC_PORT = Number(process.env.MC_PORT || 25565);
const MC_USERNAME = process.env.MC_USERNAME || "FlyBrain";
const MC_VERSION = process.env.MC_VERSION || false; // false = auto-detect
const MC_AUTH = process.env.MC_AUTH || "offline";
const BRIDGE_PORT = Number(process.env.BRIDGE_PORT || 8081);

const bot = mineflayer.createBot({
  host: MC_HOST,
  port: MC_PORT,
  username: MC_USERNAME,
  version: MC_VERSION,
  auth: MC_AUTH,
});

let mcData = null;

bot.once("spawn", () => {
  mcData = minecraftData(bot.version);
  console.log(`[bot-bridge] spawned as ${MC_USERNAME} on ${MC_HOST}:${MC_PORT} (mc ${bot.version})`);
  broadcastEvent("spawn", {});
});

bot.on("death", () => broadcastEvent("death", {}));
bot.on("kicked", (reason) => broadcastEvent("kicked", { reason }));
bot.on("end", (reason) => broadcastEvent("end", { reason }));
bot.on("error", (err) => console.error("[bot-bridge] bot error:", err));

const ACTION_TIMEOUT_MS = 10000;

function withTimeout(promise, ms, message) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(message)), ms)),
  ]);
}

const wss = new WebSocketServer({ port: BRIDGE_PORT });
console.log(`[bot-bridge] WebSocket server listening on ws://localhost:${BRIDGE_PORT}`);

const clients = new Set();

function broadcastEvent(event, data) {
  const message = JSON.stringify({ type: "event", event, ...data });
  for (const ws of clients) {
    if (ws.readyState === ws.OPEN) ws.send(message);
  }
}

wss.on("connection", (ws) => {
  clients.add(ws);
  console.log(`[bot-bridge] client connected (${clients.size} total)`);

  ws.on("message", async (raw) => {
    let request;
    try {
      request = JSON.parse(raw.toString());
    } catch {
      ws.send(JSON.stringify({ error: "invalid JSON" }));
      return;
    }

    const { id, method, params } = request;
    try {
      let result;
      if (method === "getObservation") {
        result = buildObservation(bot);
      } else if (method === "doAction") {
        if (!bot.entity) throw new Error("bot has not spawned yet");
        // Blanket safety net so a stuck/never-settling action (mineflayer
        // promises don't always resolve/reject cleanly, e.g. a dig whose
        // target moves out of reach - see actions.js's own dig-specific
        // timeout) can never hang this connection forever.
        result = await withTimeout(
          performAction(bot, mcData, params),
          ACTION_TIMEOUT_MS,
          `action '${params?.type}' timed out after ${ACTION_TIMEOUT_MS}ms`
        );
      } else if (method === "ping") {
        result = { pong: true };
      } else {
        throw new Error(`unknown method '${method}'`);
      }
      ws.send(JSON.stringify({ id, result }));
    } catch (err) {
      ws.send(JSON.stringify({ id, error: err.message }));
    }
  });

  ws.on("close", () => {
    clients.delete(ws);
    console.log(`[bot-bridge] client disconnected (${clients.size} total)`);
  });
});
