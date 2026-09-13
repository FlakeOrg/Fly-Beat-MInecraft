// Translates action commands from the Python brain process into Mineflayer
// calls. Each command is `{ type, ...params }`; unknown/malformed commands
// throw, which index.js turns into a `{ error }` response.

import { Vec3 } from "vec3";
import pathfinderPkg from "mineflayer-pathfinder";

const { Movements, goals } = pathfinderPkg;

const CONTROL_KEYS = ["forward", "back", "left", "right", "jump", "sprint", "sneak"];

const FACE_VECTORS = {
  up: new Vec3(0, 1, 0),
  down: new Vec3(0, -1, 0),
  north: new Vec3(0, 0, -1),
  south: new Vec3(0, 0, 1),
  east: new Vec3(1, 0, 0),
  west: new Vec3(-1, 0, 0),
};

// Tools the bot should prefer to swing at a hostile, best first.
const WEAPON_PRIORITY = [
  "netherite_sword", "diamond_sword", "iron_sword", "stone_sword", "wooden_sword",
  "netherite_axe", "diamond_axe", "iron_axe", "stone_axe", "wooden_axe",
];

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
  // Cancel any in-progress pathfinding too - otherwise the pathfinder keeps
  // driving the control states straight back on after this clears them,
  // and the bot looks like it's ignoring commands.
  if (bot.pathfinder) bot.pathfinder.setGoal(null);
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

/** Movements tuned for an unattended bot that has to survive its own
 * navigation: water is treated as something to route around rather than
 * swim through (drowning was the single most common death in live runs),
 * and digging is left enabled so the pathfinder can tunnel toward ore
 * instead of only walking over terrain it happens to find traversable. */
function buildMovements(bot, mcData) {
  const movements = new Movements(bot, mcData);
  movements.canDig = true;
  movements.allow1by1towers = true;
  movements.allowParkour = false; // parkour failures = fall damage with no upside for an unattended bot
  if (mcData.blocksByName.water) movements.blocksToAvoid.add(mcData.blocksByName.water.id);
  if (mcData.blocksByName.lava) movements.blocksToAvoid.add(mcData.blocksByName.lava.id);
  return movements;
}

const GOTO_TIMEOUT_MS = 45000;

async function doGoto(bot, command, mcData) {
  const { x, y, z, range } = command;
  if (typeof x !== "number" || typeof z !== "number") {
    throw new Error("goto requires numeric x and z");
  }
  bot.pathfinder.setMovements(buildMovements(bot, mcData));

  // GoalNear when a y is given, GoalNearXZ when it isn't - lets callers say
  // "get to this column" without pretending to know the surface height.
  const goal =
    typeof y === "number"
      ? new goals.GoalNear(x, y, z, typeof range === "number" ? range : 2)
      : new goals.GoalNearXZ(x, z, typeof range === "number" ? range : 2);

  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      bot.pathfinder.setGoal(null);
      reject(new Error(`goto timed out after ${GOTO_TIMEOUT_MS}ms`));
    }, GOTO_TIMEOUT_MS);
  });
  try {
    await Promise.race([bot.pathfinder.goto(goal), timeout]);
  } finally {
    clearTimeout(timer);
  }
  const p = bot.entity.position;
  return { ok: true, position: { x: p.x, y: p.y, z: p.z } };
}

/** Digs down/along toward a target Y using the pathfinder's own tunnelling,
 * which is far safer than hand-rolling a straight-down shaft (it won't drop
 * the bot into lava or a ravine to get there). */
async function doGotoY(bot, command, mcData) {
  const { y } = command;
  if (typeof y !== "number") throw new Error("gotoY requires numeric y");
  bot.pathfinder.setMovements(buildMovements(bot, mcData));

  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      bot.pathfinder.setGoal(null);
      reject(new Error(`gotoY timed out after ${GOTO_TIMEOUT_MS}ms`));
    }, GOTO_TIMEOUT_MS);
  });
  try {
    await Promise.race([bot.pathfinder.goto(new goals.GoalY(y)), timeout]);
  } finally {
    clearTimeout(timer);
  }
  const p = bot.entity.position;
  return { ok: true, position: { x: p.x, y: p.y, z: p.z } };
}

/** Server-side block search. The observation's nearbyBlocks radius is
 * deliberately small (it feeds the fly brain's sensory encoder), which is
 * useless for "where is the nearest iron ore" - this searches the loaded
 * world instead, so the task manager can navigate to things far outside
 * sensory range. */
async function doFindBlocks(bot, command, mcData) {
  const { names, maxDistance, count } = command;
  if (!Array.isArray(names) || names.length === 0) {
    throw new Error("findBlocks requires a non-empty 'names' array");
  }
  const ids = [];
  for (const name of names) {
    const def = mcData.blocksByName[name];
    if (def) ids.push(def.id);
  }
  if (ids.length === 0) {
    return { ok: true, blocks: [] }; // none of those names exist in this version
  }
  const found = bot.findBlocks({
    matching: ids,
    maxDistance: typeof maxDistance === "number" ? maxDistance : 64,
    count: typeof count === "number" ? count : 16,
  });
  const blocks = found.map((pos) => {
    const block = bot.blockAt(pos);
    return {
      x: pos.x,
      y: pos.y,
      z: pos.z,
      name: block ? block.name : "unknown",
      distance: bot.entity.position.distanceTo(pos),
    };
  });
  blocks.sort((a, b) => a.distance - b.distance);
  return { ok: true, blocks };
}

const DIG_TIMEOUT_MS = 20000;

/** Equips the best available tool for a block before digging it.
 * Without this, mining stone/ore bare-handed either takes absurdly long or
 * (for ore and stone) drops nothing at all - which silently blocked the
 * entire stone -> iron progression. */
async function equipBestToolFor(bot, block) {
  if (!bot.pathfinder || typeof bot.pathfinder.bestHarvestTool !== "function") return null;
  const tool = bot.pathfinder.bestHarvestTool(block);
  if (!tool) return null;
  if (bot.heldItem && bot.heldItem.type === tool.type) return tool.name;
  await bot.equip(tool, "hand");
  return tool.name;
}

async function doDig(bot, command) {
  const pos = requirePosition(command);
  const block = bot.blockAt(pos);
  if (!block || block.type === 0) {
    throw new Error(`no diggable block at ${pos}`);
  }

  const equipped = await equipBestToolFor(bot, block);

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
  return { ok: true, block: block.name, tool: equipped };
}

/** Walk to a block, mine it, and give the drop a moment to be picked up.
 * This is the unit of work nearly all resource gathering actually wants;
 * doing it in one bridge call keeps the Python side from having to babysit
 * a multi-step goto/dig/collect dance over three round trips. */
async function doMineBlock(bot, command, mcData) {
  const pos = requirePosition(command);
  const before = countInventory(bot);

  await doGoto(bot, { x: pos.x, y: pos.y, z: pos.z, range: 3 }, mcData);

  const block = bot.blockAt(pos);
  if (!block || block.type === 0) {
    return { ok: true, mined: false, reason: "block already gone" };
  }
  await doDig(bot, { x: pos.x, y: pos.y, z: pos.z });

  // Walk onto the drop so it's actually collected, then let pickup settle.
  try {
    await doGoto(bot, { x: pos.x, y: pos.y, z: pos.z, range: 1 }, mcData);
  } catch {
    // getting exactly onto the block isn't essential - the drop is often
    // already picked up just from being adjacent
  }
  await sleep(600);

  return { ok: true, mined: true, block: block.name, gained: countInventory(bot) - before };
}

function countInventory(bot) {
  return bot.inventory.items().reduce((total, item) => total + item.count, 0);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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

/** Places a block somewhere sane near the bot without the caller having to
 * know which neighbouring face happens to be free. Tries the ground under
 * each nearby standable spot. */
async function doPlaceNearby(bot, command, mcData) {
  const { item } = command;
  if (!item) throw new Error("placeNearby requires an 'item'");
  const held = bot.inventory.items().find((i) => i.name === item);
  if (!held) throw new Error(`item '${item}' not found in inventory`);
  await bot.equip(held.type, "hand");

  const origin = bot.entity.position.floored();
  const offsets = [
    new Vec3(1, 0, 0), new Vec3(-1, 0, 0), new Vec3(0, 0, 1), new Vec3(0, 0, -1),
    new Vec3(2, 0, 0), new Vec3(-2, 0, 0), new Vec3(0, 0, 2), new Vec3(0, 0, -2),
    new Vec3(1, 0, 1), new Vec3(-1, 0, -1), new Vec3(1, 0, -1), new Vec3(-1, 0, 1),
  ];

  for (const offset of offsets) {
    const target = origin.plus(offset);
    const spot = bot.blockAt(target);
    const ground = bot.blockAt(target.offset(0, -1, 0));
    if (!spot || !ground) continue;
    if (spot.type !== 0 && spot.name !== "air") continue; // needs empty space to place into
    if (ground.type === 0 || ground.boundingBox !== "block") continue; // needs something solid to place against
    try {
      await bot.lookAt(target, true);
      await bot.placeBlock(ground, new Vec3(0, 1, 0));
      return { ok: true, position: { x: target.x, y: target.y, z: target.z } };
    } catch {
      continue; // that spot didn't work out; try the next
    }
  }
  throw new Error(`could not find anywhere to place '${item}' nearby`);
}

async function doAttack(bot, command) {
  const { entityId } = command;
  const entity = bot.entities[entityId];
  if (!entity) {
    throw new Error(`no known entity with id ${entityId}`);
  }
  // Swing something sharp if we have one - punching a zombie bare-handed is
  // most of the way to losing the fight.
  const items = bot.inventory.items();
  for (const name of WEAPON_PRIORITY) {
    const weapon = items.find((i) => i.name === name);
    if (weapon) {
      if (!bot.heldItem || bot.heldItem.type !== weapon.type) {
        await bot.equip(weapon.type, "hand");
      }
      break;
    }
  }
  await bot.lookAt(entity.position.offset(0, entity.height * 0.5, 0), true);
  bot.attack(entity);
  return { ok: true };
}

async function doEquip(bot, command) {
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

const SMELT_POLL_MS = 1000;
const SMELT_TIMEOUT_MS = 120000;

/** Full furnace cycle: walk to the furnace, load fuel + input, wait for the
 * smelt to finish, take the output. Smelting is genuinely slow (~10s per
 * item), which is why this has its own much longer timeout than a normal
 * action - see index.js's per-type timeouts. */
async function doSmelt(bot, command, mcData) {
  const { input, fuel, count } = command;
  const wanted = typeof count === "number" ? count : 1;

  const furnaceId = mcData.blocksByName.furnace?.id;
  if (furnaceId == null) throw new Error("this Minecraft version has no furnace block");
  const furnaceBlock = bot.findBlock({ matching: furnaceId, maxDistance: 16 });
  if (!furnaceBlock) throw new Error("no furnace within reach");

  await doGoto(bot, { x: furnaceBlock.position.x, y: furnaceBlock.position.y, z: furnaceBlock.position.z, range: 2 }, mcData);

  const inputItem = bot.inventory.items().find((i) => i.name === input);
  if (!inputItem) throw new Error(`no '${input}' in inventory to smelt`);
  const fuelItem = bot.inventory.items().find((i) => i.name === fuel);
  if (!fuelItem) throw new Error(`no '${fuel}' in inventory to use as fuel`);

  const furnace = await bot.openFurnace(furnaceBlock);
  try {
    await furnace.putFuel(fuelItem.type, null, Math.min(fuelItem.count, Math.max(1, Math.ceil(wanted / 4))));
    await furnace.putInput(inputItem.type, null, Math.min(inputItem.count, wanted));

    const deadline = Date.now() + SMELT_TIMEOUT_MS;
    let taken = 0;
    while (Date.now() < deadline && taken < wanted) {
      await sleep(SMELT_POLL_MS);
      if (furnace.outputItem()) {
        const out = await furnace.takeOutput();
        if (out) taken += out.count;
      }
    }
    return { ok: true, smelted: taken };
  } finally {
    furnace.close();
  }
}

/** Eats whatever is currently held. Starving blocks health regeneration,
 * which is what turns an otherwise-survivable mob fight into a death. */
async function doConsume(bot) {
  if (!bot.heldItem) throw new Error("nothing held to consume");
  await bot.consume();
  return { ok: true, consumed: true };
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
  goto: doGoto,
  gotoY: doGotoY,
  findBlocks: doFindBlocks,
  dig: doDig,
  mineBlock: doMineBlock,
  place: doPlace,
  placeNearby: doPlaceNearby,
  attack: doAttack,
  equip: doEquip,
  craft: doCraft,
  smelt: doSmelt,
  consume: doConsume,
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

// Actions that legitimately take much longer than a normal one - index.js
// uses these instead of its blanket timeout, which would otherwise abort a
// perfectly healthy smelt or long-distance path partway through.
export const ACTION_TIMEOUTS_MS = {
  goto: GOTO_TIMEOUT_MS + 5000,
  gotoY: GOTO_TIMEOUT_MS + 5000,
  mineBlock: GOTO_TIMEOUT_MS * 2 + DIG_TIMEOUT_MS,
  dig: DIG_TIMEOUT_MS + 5000,
  smelt: SMELT_TIMEOUT_MS + 20000,
};
