// Builds a compact JSON snapshot of bot/world state for the Python brain
// process. Kept structured (not pixels) - much easier to map onto a small
// set of sensory channels than raw vision would be.

const RADIUS_XZ = 4;
const RADIUS_Y_DOWN = 2;
const RADIUS_Y_UP = 3;
const ENTITY_RADIUS = 16;

function nearbyBlocks(bot) {
  const origin = bot.entity.position.floored();
  const blocks = [];
  for (let dx = -RADIUS_XZ; dx <= RADIUS_XZ; dx++) {
    for (let dz = -RADIUS_XZ; dz <= RADIUS_XZ; dz++) {
      for (let dy = -RADIUS_Y_DOWN; dy <= RADIUS_Y_UP; dy++) {
        const pos = origin.offset(dx, dy, dz);
        const block = bot.blockAt(pos);
        if (block && block.type !== 0 /* air */) {
          blocks.push({
            x: dx,
            y: dy,
            z: dz,
            name: block.name,
            boundingBox: block.boundingBox,
          });
        }
      }
    }
  }
  return blocks;
}

function nearbyEntities(bot) {
  const entities = [];
  for (const id in bot.entities) {
    const entity = bot.entities[id];
    if (!entity || entity === bot.entity || !entity.position) continue;
    const distance = entity.position.distanceTo(bot.entity.position);
    if (distance > ENTITY_RADIUS) continue;
    entities.push({
      id: entity.id,
      type: entity.type ?? "unknown",
      name: entity.name || entity.username || entity.displayName || "unknown",
      kind: entity.kind ?? "unknown", // undefined would silently vanish from the JSON entirely, not become null
      position: { x: entity.position.x, y: entity.position.y, z: entity.position.z },
      velocity: { x: entity.velocity.x, y: entity.velocity.y, z: entity.velocity.z },
      distance,
      health: entity.health ?? null,
    });
  }
  entities.sort((a, b) => a.distance - b.distance);
  return entities;
}

function inventory(bot) {
  return bot.inventory.items().map((item) => ({
    slot: item.slot,
    name: item.name,
    count: item.count,
  }));
}

export function buildObservation(bot) {
  if (!bot.entity) {
    throw new Error("bot has not spawned yet");
  }
  const pos = bot.entity.position;
  return {
    position: { x: pos.x, y: pos.y, z: pos.z },
    yaw: bot.entity.yaw,
    pitch: bot.entity.pitch,
    onGround: bot.entity.onGround,
    velocity: { x: bot.entity.velocity.x, y: bot.entity.velocity.y, z: bot.entity.velocity.z },
    health: bot.health,
    food: bot.food,
    saturation: bot.foodSaturation,
    experience: bot.experience,
    gameMode: bot.game?.gameMode ?? null,
    timeOfDay: bot.time?.timeOfDay ?? null,
    isRaining: bot.isRaining,
    nearbyBlocks: nearbyBlocks(bot),
    nearbyEntities: nearbyEntities(bot),
    inventory: inventory(bot),
    heldItem: bot.heldItem ? bot.heldItem.name : null,
  };
}
