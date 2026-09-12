// Translates action commands from the Python brain process into Mineflayer
// calls. Each command is `{ type, ...params }`; unknown/malformed commands
// throw, which index.js turns into a `{ error }` response.

import { Vec3 } from "vec3";

const CONTROL_KEYS = ["forward", "back", "left", "right", "jump", "sprint", "sneak"];

const FACE_VECTORS = {
  up: new Vec3(0, 1, 0),
  down: new Vec3(0, -1, 0),
  north: new Vec3(0, 0, -1),
  south: new Vec3(0, 0, 1),
  east: new Vec3(1, 0, 0),
  west: new Vec3(-1, 0, 0),
};

function requirePosition(command) {
  const { x, y, z } = command;
  if (typeof x !== "number" || typeof y !== "number" || typeof z !== "number") {
    throw new Error(`command missing numeric x/y/z: ${JSON.stringify(command)}`);
  }
  return new Vec3(x, y, z);
}

async function doMove(bot, command) {
  for (const key of CONTROL_KEYS) {
    if (key in command) {
      bot.setControlState(key, Boolean(command[key]));
    }
  }
  return { ok: true };
}

async function doStop(bot) {
  for (const key of CONTROL_KEYS) {
    bot.setControlState(key, false);
  }
  return { ok: true };
}

async function doLook(bot, command) {
  const { yaw, pitch, relative } = command;
  if (typeof yaw !== "number" || typeof pitch !== "number") {
    throw new Error("look requires numeric yaw and pitch (radians)");
  }
  const targetYaw = relative ? bot.entity.yaw + yaw : yaw;
  const targetPitch = relative ? bot.entity.pitch + pitch : pitch;
  await bot.look(targetYaw, targetPitch, true);
  return { ok: true };
}

async function doLookAt(bot, command) {
  const point = requirePosition(command);
  await bot.lookAt(point, true);
  return { ok: true };
}

const DIG_TIMEOUT_MS = 8000;

async function doDig(bot, command) {
  const pos = requirePosition(command);
  const block = bot.blockAt(pos);
  if (!block || block.type === 0) {
    throw new Error(`no diggable block at ${pos}`);
  }
  if (!bot.canDigBlock(block)) {
    throw new Error(`cannot dig block ${block.name} at ${pos}`);
  }

  // bot.dig()'s promise can fail to settle if the target goes out of reach
  // mid-dig (e.g. movement controls are active at the same time) - without
  // this, a single stuck dig hangs the WebSocket connection forever. On
  // timeout, explicitly cancel the in-progress dig rather than just giving
  // up on waiting for it.
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      bot.stopDigging();
      reject(new Error(`dig timed out after ${DIG_TIMEOUT_MS}ms (target may have moved out of reach)`));
    }, DIG_TIMEOUT_MS);
  });
  try {
    await Promise.race([bot.dig(block), timeout]);
  } finally {
    clearTimeout(timer);
  }
  return { ok: true, block: block.name };
}

async function doPlace(bot, command) {
  const pos = requirePosition(command);
  const face = FACE_VECTORS[command.face];
  if (!face) {
    throw new Error(`invalid face '${command.face}', expected one of ${Object.keys(FACE_VECTORS)}`);
  }
  const referenceBlock = bot.blockAt(pos);
  if (!referenceBlock) {
    throw new Error(`no reference block at ${pos}`);
  }
  if (command.item) {
    await doEquip(bot, { item: command.item, destination: "hand" });
  }
  await bot.placeBlock(referenceBlock, face);
  return { ok: true };
}

async function doAttack(bot, command) {
  const { entityId } = command;
  const entity = bot.entities[entityId];
  if (!entity) {
    throw new Error(`no known entity with id ${entityId}`);
  }
  bot.attack(entity);
  return { ok: true };
}

async function doEquip(bot, command, mcData) {
  const { item, destination } = command;
  const items = bot.inventory.items().filter((i) => i.name === item);
  if (items.length === 0) {
    throw new Error(`item '${item}' not found in inventory`);
  }
  await bot.equip(items[0].type, destination || "hand");
  return { ok: true };
}

async function doCraft(bot, command, mcData) {
  const { item, count } = command;
  const itemDef = mcData.itemsByName[item];
  if (!itemDef) {
    throw new Error(`unknown item '${item}'`);
  }

  // 3x3 recipes (most tools) need a placed crafting table within reach -
  // without this, doCraft could only ever make 2x2 recipes (sticks,
  // planks) regardless of what's actually placed nearby.
  const tableId = mcData.blocksByName.crafting_table?.id;
  const craftingTableBlock =
    tableId != null ? bot.findBlock({ matching: tableId, maxDistance: 4 }) : null;

  let recipes = bot.recipesFor(itemDef.id, null, count || 1, craftingTableBlock);
  if (recipes.length === 0 && craftingTableBlock) {
    recipes = bot.recipesFor(itemDef.id, null, count || 1, null); // fall back to tableless recipes
  }
  if (recipes.length === 0) {
    throw new Error(`no available recipe for '${item}' (may need a crafting table nearby)`);
  }
  await bot.craft(recipes[0], count || 1, craftingTableBlock);
  return { ok: true };
}

async function doChat(bot, command) {
  if (typeof command.message !== "string") {
    throw new Error("chat requires a string message");
  }
  bot.chat(command.message);
  return { ok: true };
}

const HANDLERS = {
  move: doMove,
  stop: doStop,
  look: doLook,
  lookAt: doLookAt,
  dig: doDig,
  place: doPlace,
  attack: doAttack,
  equip: doEquip,
  craft: doCraft,
  chat: doChat,
};

export async function performAction(bot, mcData, command) {
  if (!command || typeof command.type !== "string") {
    throw new Error(`command must have a string 'type': ${JSON.stringify(command)}`);
  }
  const handler = HANDLERS[command.type];
  if (!handler) {
    throw new Error(`unknown action type '${command.type}', expected one of ${Object.keys(HANDLERS)}`);
  }
  return handler(bot, command, mcData);
}
