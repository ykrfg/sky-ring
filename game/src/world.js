// SKY RING world simulation. This module owns every piece of mutable game
// state — the player's flight model, the mission phases, the ring course,
// the enemy aircraft (a standing patrol, mission interceptors and dogfight
// waves), their gunfire, missiles and their smoke, wrecks, particles and
// explosions, plus the sortie's resources: fuel, wind and the game clock —
// and advances it in fixed steps. It knows nothing about rendering or the
// DOM: main.js feeds it an input facade ({ down, consume }) and a sound
// callback, and the renderer only reads what is left in `state`.
//
// Two modes share the one simulation. SORTIE is the phased mission —
// takeoff, the nine rings, a mandatory two-bandit intercept, then landing —
// where objectives, once met, stay met across landings and refuelling.
// DOGFIGHT skips the course: three waves of bandits who shoot back, and the
// player's hull is the score that matters. Tower radio traffic for both
// modes runs through state.radio (the current line) and state.radioQueue.
//
// createWorld(terrain, random)
//   terrain — the height grid built by gltf.js buildTerrainGrid; the world
//             asks it for ground heights via terrain.heightAt(x, z).
//   random  — the seeded generator from random.js. Every random draw the
//             simulation makes (burst spreads, wreck smoke) comes from it,
//             so a seed fully determines a run.

import {
  AIRCRAFT, CLOCK, DOGFIGHT, ENEMY, ENEMY_ROUTE, FLIGHT, FUEL, LANDING,
  MODE, RECOVERY, RING_COUNT, RINGS, WIND, WORLD,
} from './config.js';
import { add, cross, dot, length, normalize, scale, sub } from './math.js';

// --- module-local tuning ---------------------------------------------------

const MISSILE_STOCK = 6;          // rounds on the racks at the start of a run
const RING_RADIUS = 26;           // metres; the pass test uses 92% of this

const PITCH_LIMIT = 0.55;         // radians of nose up/down the stick can ask for
const BANK_LIMIT = 0.95;          // radians of roll, likewise
const SINK_REFERENCE = 120;       // below this speed the airframe settles

const TAKEOFF_SPEED = 64;         // rotation speed on the runway roll
const ROTATION_PITCH = 0.055;     // nose-up needed before the wheels leave
const OVERRUN_SPEED = 8;          // rolling this fast off the strip is a crash
const GROUND_ROLL_SPEED = 150;    // ground speed at full throttle on the roll

const LOCK_COSINE = 0.978;        // the target must sit inside this nose cone
const LOCK_MIN_RANGE = 25;        // metres; closer than this is a fly-through
const LOCK_MAX_RANGE = 1050;      // metres; the seeker cannot see past this
const LOCK_TIME_NEEDED = 0.55;    // seconds of steady tracking before a launch
const LOCK_TIME_FULL = 1.1;       // the lock tone is solid from here on

const MISSILE_SPEED = 245;        // the missile's own cruise speed
const MISSILE_TURN = 3.2;         // guidance blend rate, per second
const MISSILE_HIT_RADIUS = 15;    // metres; proximity fuse
const MISSILE_LIFETIME = 7;       // seconds before an unhit missile dies
const SMOKE_INTERVAL = 0.045;     // seconds between trail puffs

const CONTACT_CLEARANCE = 1.35;   // the gear puts the belly this far off the ground
const LANDING_GRACE = 2.0;        // seconds after liftoff with no contact test
const GEAR_NEEDED = 0.88;         // gear travel fraction that counts as "down"

const BULLET_HIT_RADIUS = 4.6;    // swept-test distance that counts as a hit
const BULLET_LIFE = 2.7;          // seconds before a tracer burns out
const BULLET_CAP = 420;           // oldest tracers are dropped past this
const BURST_INTERVAL = 0.072;     // seconds between rounds in one burst

const RING_SCORE = 500;
const KILL_SCORE = 2000;
const GRADE_BONUSES = { A: 3200, B: 2400, C: 1500, D: 900 };

const AIRSPACE_LIMIT = 2400;      // leaving this box ends the flight
const PARTICLE_CAP = 220;         // oldest particles are dropped past this

// The enemy patrol: a wide circuit over the southern channel. Each aircraft
// flies the same ellipse, offset sideways by its formation slot and started
// at its own phase so the three stay strung out along the route. The patrol
// is the standing roster: interceptors and dogfight waves are these same
// airframes, repositioned and switched to hunting (see spawnCombat).
const PATROL = {
  center: [40, 455, -760],
  sweepX: 720,
  sweepZ: 300,
  bob: 28,
  bobRate: 1.7,
  spacingX: 48,
  spacingZ: 40,
  rate: 0.22,        // phase advance per second
  phaseStep: 2.1,    // phase offset between formation slots
  bankAmount: 0.32,
};

const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));

// Move `value` toward `target` by at most `amount` (gear travel, mainly).
function moveToward(value, target, amount) {
  return value + clamp(target - value, -amount, amount);
}

// Fold an angle into [-PI, PI] so heading errors take the short way round.
function wrapAngle(angle) {
  while (angle > Math.PI) angle -= Math.PI * 2;
  while (angle < -Math.PI) angle += Math.PI * 2;
  return angle;
}

// A stable 0..1 hash of a number. The sortie wind is derived from the run
// seed with this rather than from the random stream, so the weather is a
// property of the seed itself and never shifts with effect draws.
function hash01(n) {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function makeWind(seed) {
  return {
    dir: hash01(seed * 0.731 + 0.17) * Math.PI * 2,
    speed: Math.pow(hash01(seed * 1.913 + 4.21), 0.85) * WIND.maxKnots,
  };
}

// Shortest distance from point p to the segment a-b. Bullet hits are tested
// against the segment a round swept through this step, so a fast tracer can
// never tunnel through the airframe between frames.
function segmentDistance(a, b, p) {
  const ab = sub(b, a);
  const t = clamp(dot(sub(p, a), ab) / (dot(ab, ab) || 1), 0, 1);
  return length(sub(add(a, scale(ab, t)), p));
}

// Score a touchdown from its three measured numbers. Each axis earns up to
// three points — sink and centreline reward a flown approach, speed rewards
// arriving in the slot rather than merely slowly — and the letter falls out
// of the total. The measured values ride along for the result card.
function gradeLanding(sink, center, speed) {
  const sinkScore = sink < 1.5 ? 3 : sink < 3.0 ? 2 : sink < 5.5 ? 1 : 0;
  const centerScore = center < 6 ? 3 : center < 14 ? 2 : center < 26 ? 1 : 0;
  const speedScore = speed >= 58 && speed <= 78 ? 3
    : speed >= 50 && speed <= 86 ? 2 : speed < LANDING.maxSpeed ? 1 : 0;
  const points = sinkScore + centerScore + speedScore;
  return {
    grade: points >= 8 ? 'A' : points >= 6 ? 'B' : points >= 3 ? 'C' : 'D',
    points,
    sink,
    center,
    speed,
    sinkScore,
    centerScore,
    speedScore,
  };
}

export function createWorld(terrain, random) {
  // --- state ---------------------------------------------------------------

  // The flight state at the top of the runway, engine idling. reset() puts
  // exactly this back, so a restart is indistinguishable from a first run.
  // The prev*/evasion fields are the gunnery model's memory: how hard the
  // player is manoeuvring is read from how fast these are changing.
  function freshFlight() {
    return {
      position: [...AIRCRAFT.start],
      heading: AIRCRAFT.startHeading,
      pitch: 0,
      bank: 0,
      speed: 0,
      throttle: 0,
      gear: 1,
      gearTarget: 1,
      grounded: true,
      crashed: false,
      landingScore: '—',
      liftoff: 0,
      verticalSpeed: 0,
      evasion: 0,
      engineOut: false,
    };
  }

  // One patrol airframe on its route waypoint. The combat fields ride along
  // from birth (idle while patrolling) so spawnCombat only has to flip the
  // engaged flag and hand the aircraft a heading, a speed and a grudge.
  function freshEnemy(slot) {
    const point = ENEMY_ROUTE[slot % ENEMY_ROUTE.length];
    return {
      position: [...point],
      origin: [...point],
      alive: true,
      locked: 0,
      phase: slot * PATROL.phaseStep,
      route: 0,
      bank: 0,
      heading: 0,
      pitch: 0,
      speed: ENEMY.sortieSpeed,
      engaged: false,
      dogfight: false,
      aggression: 0,
      slot: slot - 1,
      cooldown: 0,
      burst: 0,
      burstTimer: 0,
      hitFlash: 0,
    };
  }

  const flight = freshFlight();
  const rings = RINGS.map((centre, i) => ({
    position: [...centre],
    radius: RING_RADIUS,
    index: i + 1,
    passed: false,
  }));
  const enemies = ENEMY_ROUTE.map((_, slot) => freshEnemy(slot));

  const state = {
    flight,
    rings,
    enemies,
    bullets: [],
    missiles: [],
    particles: [],
    explosions: [],
    wrecks: [],
    time: 0,
    phase: 'title',
    mode: MODE.SORTIE,
    mission: { phase: 'takeoff', completed: false, tookOff: false, touchdown: false },
    ringIndex: 0,
    score: 0,
    kills: 0,
    interceptKills: 0,
    missileCount: MISSILE_STOCK,
    lock: null,
    lockTime: 0,
    landing: null,
    landings: [],
    crashReason: '',
    seed: 1,
    deterministic: false,
    stage: null,
    best: 0,
    message: '',
    messageTime: 0,
    fuel: FUEL.max,
    hull: 100,
    hullMax: 100,
    wind: makeWind(1),
    clock: CLOCK.start,
    radio: null,
    radioQueue: [],
    damageFlash: 0,
    shake: 0,
    hitRadio: 0,
    fuelCaution: false,
    fuelBingo: false,
    refueling: false,
    dogfight: { wave: 1, waves: DOGFIGHT.waves, waveDelay: 0, victoryDelay: 0, cleared: false },
    result: null,
    onEnemyGun: null,
  };

  // --- tower radio -----------------------------------------------------------

  // Push one line of tower traffic. The renderer shows state.radio; lines
  // wait in the queue rather than talking over each other, and a line that
  // is already up (or already waiting) is never repeated back to back.
  function pushRadio(text, seconds = 4.2) {
    if (!text) return;
    if (state.radio && state.radio.text === text) return;
    if (!state.radio) {
      state.radio = { text, time: seconds, max: seconds };
    } else if (state.radioQueue.length < 4
      && !state.radioQueue.some((queued) => queued.text === text)) {
      state.radioQueue.push({ text, time: seconds });
    }
  }

  function tickRadio(dt) {
    if (state.radio) {
      state.radio.time -= dt;
      if (state.radio.time <= 0) state.radio = null;
    }
    if (!state.radio && state.radioQueue.length) {
      const next = state.radioQueue.shift();
      state.radio = { text: next.text, time: next.time, max: next.time };
    }
  }

  function reset(mode = MODE.SORTIE, seed = state.seed) {
    state.seed = seed >>> 0;
    Object.assign(flight, freshFlight());
    if (mode === MODE.DOGFIGHT) {
      // The dogfight starts where the patrol flies: airborne over the
      // southern channel, gear up, at fighting speed.
      Object.assign(flight, {
        position: [...DOGFIGHT.start],
        heading: DOGFIGHT.heading,
        speed: DOGFIGHT.speed,
        throttle: 0.78,
        gear: 0,
        gearTarget: 0,
        grounded: false,
      });
    }
    for (const ring of rings) ring.passed = false;
    // The roster returns to the three patrol airframes; any extra airframe
    // a dogfight wave borrowed is handed back.
    enemies.forEach((enemy, slot) => Object.assign(enemy, freshEnemy(slot)));
    enemies.length = ENEMY_ROUTE.length;
    state.bullets.length = 0;
    state.missiles.length = 0;
    state.particles.length = 0;
    state.explosions.length = 0;
    state.wrecks.length = 0;
    Object.assign(state, {
      time: 0,
      phase: 'play',
      mode,
      mission: {
        phase: mode === MODE.DOGFIGHT ? 'combat' : 'takeoff',
        completed: false,
        tookOff: mode === MODE.DOGFIGHT,
        touchdown: false,
      },
      ringIndex: 0,
      score: 0,
      kills: 0,
      interceptKills: 0,
      missileCount: MISSILE_STOCK,
      lock: null,
      lockTime: 0,
      landing: null,
      landings: [],
      crashReason: '',
      message: '',
      messageTime: 0,
      fuel: FUEL.max,
      hull: state.hullMax,
      wind: makeWind(state.seed),
      clock: mode === MODE.DOGFIGHT ? CLOCK.dogfightStart : CLOCK.start,
      radio: null,
      radioQueue: [],
      damageFlash: 0,
      shake: 0,
      hitRadio: 0,
      fuelCaution: false,
      fuelBingo: false,
      refueling: false,
      dogfight: { wave: 1, waves: DOGFIGHT.waves, waveDelay: 0, victoryDelay: 0, cleared: false },
      result: null,
    });
    if (mode === MODE.DOGFIGHT) spawnWave(1);
    else pushRadio('TOWER: SKY RING ONE, CLEARED FOR TAKEOFF. RUNWAY 36.', 4.8);
  }

  // --- geography -----------------------------------------------------------

  // The aeroplane's nose direction for its current heading and pitch.
  function forward() {
    const cosPitch = Math.cos(flight.pitch);
    return normalize([
      Math.sin(flight.heading) * cosPitch,
      Math.sin(flight.pitch),
      Math.cos(flight.heading) * cosPitch,
    ]);
  }

  function enemyForward(enemy) {
    const cosPitch = Math.cos(enemy.pitch);
    return [
      Math.sin(enemy.heading) * cosPitch,
      Math.sin(enemy.pitch),
      Math.cos(enemy.heading) * cosPitch,
    ];
  }

  // The runway strip, slightly forgiving at the edges: the centreline margin
  // is shaved so a tyre on the painted edge still counts as on the concrete.
  function onRunway(p) {
    const halfLength = WORLD.runwayLength / 2;
    return Math.abs(p[0] - WORLD.runway[0]) < WORLD.runwayWidth / 2 - 4
      && p[2] > WORLD.runway[1] - halfLength
      && p[2] < WORLD.runway[1] + halfLength;
  }

  function runwayDistance(p) {
    return Math.hypot(p[0] - WORLD.runway[0], p[2] - WORLD.runway[1]);
  }

  // Ground height under a point: the levelled runway pad on the strip, the
  // island's height grid everywhere else.
  function groundAt(p) {
    return onRunway(p) ? WORLD.runwayHeight : terrain.heightAt(p[0], p[2]);
  }

  // The player's velocity through the air mass plus the wind's push — what
  // an enemy gunner has to lead if he wants his tracers to arrive.
  function playerVelocity() {
    const windSpeed = state.wind.speed * 0.514 * 0.55;
    const nose = forward();
    return [
      nose[0] * flight.speed + Math.sin(state.wind.dir) * windSpeed,
      nose[1] * flight.speed,
      nose[2] * flight.speed + Math.cos(state.wind.dir) * windSpeed,
    ];
  }

  function nearestEnemy(from = flight.position) {
    let best = null;
    let bestDistance = Infinity;
    for (const enemy of enemies) {
      if (!enemy.alive) continue;
      const distance = length(sub(enemy.position, from));
      if (distance < bestDistance) {
        bestDistance = distance;
        best = enemy;
      }
    }
    return best;
  }

  // --- effects -------------------------------------------------------------

  // One burst of smoke particles around a point. `black` picks the dark,
  // sinking smoke (fires, wrecks) over the pale, rising kind (missile
  // trails); `power` scales the spread and rise.
  function puff(position, count, black, power = 1) {
    for (let i = 0; i < count; i++) {
      state.particles.push({
        position: [...position],
        velocity: [
          (random() - 0.5) * 26 * power,
          (random() * 0.8 + 0.15) * 18 * power,
          (random() - 0.5) * 26 * power,
        ],
        age: 0,
        life: 1.6 + random() * 1.6,
        black,
        size: 0.8 + random() * 0.9,
      });
    }
  }

  // A fireball plus its smoke: the renderer plays the explosion sheet over
  // `duration` while the dark core and pale fringe drift apart.
  function detonate(position, sound) {
    state.explosions.push({ position: [...position], age: 0, duration: 0.9 });
    puff(position, 18, true, 1.1);
    puff(position, 8, false, 0.7);
    sound('explosion');
  }

  function crash(reason, sound) {
    if (state.phase === 'crashed' || state.phase === 'result') return;
    flight.crashed = true;
    state.phase = 'crashed';
    state.crashReason = reason;
    detonate(flight.position, sound);
  }

  // Shot down rather than crashed: the run ends on the result card with the
  // reason, not behind the crash overlay, and the tower gets the last word.
  function defeat(reason, sound) {
    if (state.phase === 'result') return;
    flight.crashed = true;
    state.crashReason = reason;
    detonate(flight.position, sound);
    state.phase = 'result';
    state.result = {
      victory: false,
      mode: state.mode,
      time: state.time,
      kills: state.kills,
      hull: 0,
      waves: state.dogfight.wave,
      landing: null,
      score: state.score,
      reason,
    };
    pushRadio('TOWER: MAYDAY, MAYDAY — WE LOST YOU.', 7);
  }

  // Age and integrate everything that is only visual: falling wrecks, smoke
  // particles, explosion flashes. Runs in the crashed phase too, so a crash
  // site keeps burning behind the overlay.
  function ageEffects(dt) {
    for (let i = state.wrecks.length - 1; i >= 0; i--) {
      const wreck = state.wrecks[i];
      wreck.age += dt;
      wreck.velocity[1] -= 18 * dt;
      wreck.position = add(wreck.position, scale(wreck.velocity, dt));
      wreck.spin += dt * 2.4;
      if (wreck.age < 3.8 && random() < dt * 24) {
        state.particles.push({
          position: [...wreck.position],
          velocity: [(random() - 0.5) * 5, 2, (random() - 0.5) * 5],
          age: 0,
          life: 2.4,
          black: true,
          size: 1.2 + random(),
        });
      }
      // One ground lookup per wreck per step serves both tests below: the
      // wreck is removed once it is old or has reached the ground, and it
      // only bursts when the removal happens close to the surface (a wreck
      // that times out at altitude just despawns).
      const wreckGround = groundAt(wreck.position);
      if (wreck.age > 4.5 || wreck.position[1] < wreckGround + 2) {
        if (wreck.position[1] < wreckGround + 4) detonate(wreck.position, () => {});
        state.wrecks.splice(i, 1);
      }
    }

    for (let i = state.particles.length - 1; i >= 0; i--) {
      const p = state.particles[i];
      p.age += dt;
      p.position = add(p.position, scale(p.velocity, dt));
      // Dark smoke settles, pale smoke climbs; both bleed sideways speed.
      p.velocity[1] += p.black ? -1.2 * dt : 1.6 * dt;
      p.velocity[0] *= 1 - 0.4 * dt;
      p.velocity[2] *= 1 - 0.4 * dt;
      if (p.age > p.life) state.particles.splice(i, 1);
    }
    if (state.particles.length > PARTICLE_CAP) {
      state.particles.splice(0, state.particles.length - PARTICLE_CAP);
    }

    for (let i = state.explosions.length - 1; i >= 0; i--) {
      state.explosions[i].age += dt;
      if (state.explosions[i].age >= state.explosions[i].duration) {
        state.explosions.splice(i, 1);
      }
    }
  }

  // --- the enemy patrol ------------------------------------------------------

  // The standing formation, flown exactly as it always was — but only by
  // airframes that are not currently hunting. An engaged bandit is flown by
  // huntEnemies instead and never touches its ellipse.
  function flyFormation(dt) {
    for (let slot = 0; slot < enemies.length; slot++) {
      const enemy = enemies[slot];
      if (!enemy.alive || enemy.engaged) continue;
      enemy.phase += dt * PATROL.rate;
      const a = enemy.phase;
      enemy.position[0] = PATROL.center[0] + Math.cos(a) * PATROL.sweepX + (slot - 1) * PATROL.spacingX;
      enemy.position[1] = PATROL.center[1] + Math.sin(a * PATROL.bobRate + slot) * PATROL.bob;
      enemy.position[2] = PATROL.center[2] + Math.sin(a) * PATROL.sweepZ + (slot - 1) * PATROL.spacingZ;
      enemy.bank = Math.sin(a) * PATROL.bankAmount;
    }
  }

  // --- combat: spawning, hunting, guns ---------------------------------------

  // Turn `count` roster airframes into hunters placed ahead of the player,
  // in lanes across his nose, and ground the rest of the roster — a wave or
  // an intercept is exactly the bandits it says it is. A wave larger than
  // the standing roster borrows extra airframes, which reset() hands back.
  function spawnCombat(count, dogfight) {
    const nose = forward();
    const right = normalize([nose[2], 0, -nose[0]]);
    while (enemies.length < count) enemies.push(freshEnemy(enemies.length));
    enemies.forEach((enemy, i) => {
      if (i >= count) {
        enemy.alive = false;
        enemy.engaged = false;
        return;
      }
      const lane = i - (count - 1) / 2;
      const reach = dogfight ? 520 + (i % 3) * 110 : 540 + (i % 2) * 90;
      const position = add(
        add(flight.position, scale(nose, reach)),
        scale(right, lane * 145),
      );
      position[1] = Math.max(
        flight.position[1] + (dogfight ? (i % 2 ? 55 : -35) : 30),
        terrain.heightAt(position[0], position[2]) + 140,
      );
      Object.assign(enemy, {
        position,
        origin: [...position],
        heading: Math.atan2(flight.position[0] - position[0], flight.position[2] - position[2]),
        pitch: 0,
        bank: 0,
        speed: dogfight ? ENEMY.dogfightSpeed : ENEMY.sortieSpeed,
        alive: true,
        locked: 0,
        engaged: true,
        dogfight,
        aggression: dogfight ? 1 : 0.42,
        slot: lane,
        cooldown: dogfight ? 1.0 + random() * 1.2 : 2.6 + random() * 1.8,
        burst: 0,
        burstTimer: 0,
        hitFlash: 0,
      });
    });
  }

  function spawnWave(wave) {
    const counts = DOGFIGHT.waves;
    state.dogfight.wave = clamp(wave, 1, counts.length);
    state.dogfight.waveDelay = 0;
    state.dogfight.victoryDelay = 0;
    state.dogfight.cleared = false;
    // A fresh wave means a fresh rack: the three waves field nine bandits
    // against six rails, so without a restock between waves the last wave
    // could never be cleared and the mode could never be won.
    if (wave > 1) state.missileCount = MISSILE_STOCK;
    spawnCombat(counts[state.dogfight.wave - 1], true);
    pushRadio(`TOWER: WAVE ${state.dogfight.wave} — ${counts[state.dogfight.wave - 1]} BANDITS INBOUND.`, 4.6);
  }

  function spawnInterceptors() {
    state.mission.phase = 'intercept';
    spawnCombat(2, false);
    pushRadio('TOWER: TWO BANDITS NORTH OF THE CHANNEL. WEAPONS FREE.', 5.4);
  }

  // One tracer round from a bandit's guns. The aim leads the player by most
  // of the bullet's flight time, then a spread is scattered over it — and
  // the spread is where evasion lives: a player who is banking steeply,
  // wrenching the stick or chopping the throttle is mostly shooting-area,
  // not target. Damage is rolled per round as it leaves the barrel.
  function fireBullet(enemy, distance) {
    const evasion = clamp(flight.evasion, 0, 1);
    const spread = (enemy.dogfight ? 0.008 : 0.006)
      + evasion * (enemy.dogfight ? 0.125 : 0.075);
    const lead = distance / DOGFIGHT.bulletSpeed * 0.85;
    const aimPoint = add(flight.position, scale(playerVelocity(), lead));
    let dir = normalize(sub(aimPoint, enemy.position));
    const right = normalize(cross(dir, [0, 1, 0]));
    const up = cross(right, dir);
    dir = normalize(add(dir, add(
      scale(right, (random() - 0.5) * 2 * spread),
      scale(up, (random() - 0.5) * 2 * spread),
    )));
    const [low, high] = enemy.dogfight ? ENEMY.dogfightDamage : ENEMY.sortieDamage;
    state.bullets.push({
      position: add(enemy.position, scale(enemyForward(enemy), 13)),
      velocity: scale(dir, DOGFIGHT.bulletSpeed),
      age: 0,
      life: BULLET_LIFE,
      damage: low + random() * (high - low),
    });
  }

  // Fly every engaged bandit: work toward a firing slot behind the player,
  // keep off the terrain, and shoot in bursts when the nose, the range and
  // the cooldown all agree. A bandit too far from its slot stops chasing
  // the slot and simply comes at the player until it is back in the fight.
  function huntEnemies(dt, sound) {
    const nose = forward();
    const right = normalize([nose[2], 0, -nose[0]]);
    for (const enemy of enemies) {
      if (!enemy.alive || !enemy.engaged) continue;
      enemy.hitFlash = Math.max(0, enemy.hitFlash - dt * 3);

      const slotDistance = enemy.dogfight ? 235 : 155;
      const slot = add(
        add(flight.position, scale(nose, -slotDistance)),
        scale(right, (enemy.slot || 0) * 82),
      );
      slot[1] += Math.sin(state.time * 0.9 + enemy.phase) * (enemy.dogfight ? 46 : 22)
        + (enemy.dogfight ? 18 : 43);
      const toSlot = sub(slot, enemy.position);
      const aim = length(toSlot) > 1150 ? sub(flight.position, enemy.position) : toSlot;

      const headingError = wrapAngle(Math.atan2(aim[0], aim[2]) - enemy.heading);
      const turnLimit = (enemy.dogfight ? ENEMY.dogfightTurn : ENEMY.sortieTurn) * dt;
      enemy.heading += clamp(headingError, -turnLimit, turnLimit);
      const aimPitch = clamp(
        Math.atan2(aim[1], Math.hypot(aim[0], aim[2]) || 1), -0.48, 0.62,
      );
      enemy.pitch += clamp(aimPitch - enemy.pitch, -0.72 * dt, 0.72 * dt);
      // The island is not a firing position: a bandit that finds itself low
      // over the hills climbs out before it does anything clever.
      if (enemy.position[1] - terrain.heightAt(enemy.position[0], enemy.position[2]) < 130) {
        enemy.pitch = Math.max(enemy.pitch, 0.3);
      }
      enemy.position = add(enemy.position, scale(enemyForward(enemy), enemy.speed * dt));
      if (Math.abs(enemy.position[0]) > 2650 || Math.abs(enemy.position[2]) > 2650) {
        enemy.heading = Math.atan2(
          flight.position[0] - enemy.position[0],
          flight.position[2] - enemy.position[2],
        );
      }
      enemy.bank += (clamp(headingError * 1.7, -0.82, 0.82) - enemy.bank) * Math.min(1, dt * 3.4);

      // Guns. The cone is the honest part of the model: a bandit whose nose
      // is not on the player does not fire, however close he is.
      const toPlayer = sub(flight.position, enemy.position);
      const distance = length(toPlayer);
      const cosine = dot(normalize(toPlayer), enemyForward(enemy));
      const cone = enemy.dogfight ? 0.968 : 0.935;
      const range = enemy.dogfight ? ENEMY.dogfightRange : ENEMY.sortieRange;
      enemy.cooldown -= dt;
      if (state.hull > 0 && distance > 55 && distance < range
        && cosine > cone && enemy.cooldown <= 0 && enemy.burst <= 0) {
        enemy.burst = enemy.dogfight ? 6 + Math.floor(random() * 4) : 4 + Math.floor(random() * 3);
        enemy.cooldown = enemy.dogfight ? 1.5 + random() * 1.2 : 3.0 + random() * 1.8;
        enemy.burstTimer = 0;
        if (state.onEnemyGun) state.onEnemyGun(enemy);
        else sound('enemyGun');
      }
      if (enemy.burst > 0) {
        enemy.burstTimer -= dt;
        if (enemy.burstTimer <= 0) {
          fireBullet(enemy, distance);
          enemy.burstTimer = BURST_INTERVAL;
          enemy.burst -= 1;
        }
      }
    }
  }

  // A tracer finds the player. The flash and the shake are the renderer's
  // cue; the tower's hit calls are throttled so a long burst reads as one
  // warning, not a stutter.
  function hitPlayer(damage, sound) {
    if (state.hull <= 0) return;
    state.hull = Math.max(0, state.hull - damage);
    state.damageFlash = Math.min(1, state.damageFlash + 0.7);
    state.shake = Math.min(1.2, state.shake + 0.6);
    if (state.hull <= 0) {
      defeat('AIRFRAME DESTROYED', sound);
      return;
    }
    if (state.hitRadio <= 0) {
      state.hitRadio = 3.2;
      pushRadio(`TOWER: YOU ARE HIT — HULL ${Math.round(state.hull)} PERCENT.`, 3.2);
      if (state.hull <= 25) pushRadio('TOWER: HULL CRITICAL. DISENGAGE OR LAND NOW.', 4);
    }
  }

  function updateBullets(dt, sound) {
    for (let i = state.bullets.length - 1; i >= 0; i--) {
      const bullet = state.bullets[i];
      bullet.age += dt;
      const previous = bullet.position;
      bullet.position = add(bullet.position, scale(bullet.velocity, dt));
      let dead = bullet.age > bullet.life;
      if (!dead && segmentDistance(previous, bullet.position, flight.position) < BULLET_HIT_RADIUS) {
        hitPlayer(bullet.damage, sound);
        dead = true;
      }
      if (!dead) {
        const ground = Math.max(terrain.heightAt(bullet.position[0], bullet.position[2]), WORLD.seaLevel);
        if (bullet.position[1] < ground) dead = true;
      }
      if (dead) state.bullets.splice(i, 1);
    }
    if (state.bullets.length > BULLET_CAP) {
      state.bullets.splice(0, state.bullets.length - BULLET_CAP);
    }
  }

  // --- the seeker and the missiles -------------------------------------------

  // The live enemy sitting closest to the nose inside the seeker's cone and
  // range window, or null. The cone test is a cosine against the nose vector,
  // so a target dead ahead always wins over one at the edge of vision.
  function acquireLock() {
    const nose = forward();
    let target = null;
    let bestCosine = LOCK_COSINE;
    for (const enemy of enemies) {
      if (!enemy.alive) continue;
      const offset = sub(enemy.position, flight.position);
      const dist = length(offset);
      if (dist > LOCK_MAX_RANGE || dist < LOCK_MIN_RANGE) continue;
      const cosine = dot(offset, nose) / dist;
      if (cosine > bestCosine) {
        bestCosine = cosine;
        target = enemy;
      }
    }
    return target;
  }

  // Launch one missile at the locked target, if the racks and the lock allow
  // it; otherwise explain on the message line why nothing left the rail.
  function tryLaunch(nose, sound) {
    if (state.missileCount <= 0) {
      state.message = 'MISSILE RACKS EMPTY';
      state.messageTime = 1.2;
      return;
    }
    const target = state.lock;
    if (!target || !target.alive || state.lockTime <= LOCK_TIME_NEEDED) {
      state.message = 'NO LOCK — HOLD TARGET IN SIGHT';
      state.messageTime = 1.2;
      return;
    }
    state.missileCount -= 1;
    state.missiles.push({
      position: add(flight.position, scale(nose, 11)),
      velocity: scale(nose, flight.speed + 60),
      target,
      age: 0,
      life: 0,
      smoke: 0,
    });
    sound('missile');
    state.message = 'MISSILE AWAY — TRACKING';
    state.messageTime = 1.2;
  }

  // Steer every missile in flight toward its target, lay its smoke trail,
  // and resolve hits: a kill scores, detonates, and throws the target's
  // airframe into the wreck list to tumble out of the sky. Kills made while
  // the intercept is the live objective count toward it, and the tower
  // calls the splashes so the player knows the objective is draining.
  function guideMissiles(dt, sound) {
    for (let i = state.missiles.length - 1; i >= 0; i--) {
      const missile = state.missiles[i];
      missile.age += dt;
      if (missile.target && missile.target.alive) {
        const aim = normalize(sub(missile.target.position, missile.position));
        const blend = Math.min(1, MISSILE_TURN * dt);
        missile.velocity = add(
          scale(missile.velocity, 1 - blend),
          scale(scale(aim, MISSILE_SPEED), blend),
        );
      }
      missile.position = add(missile.position, scale(missile.velocity, dt));

      missile.smoke += dt;
      if (missile.smoke > SMOKE_INTERVAL) {
        missile.smoke = 0;
        state.particles.push({
          position: [...missile.position],
          velocity: [(random() - 0.5) * 3, 1.5 + random() * 2, (random() - 0.5) * 3],
          age: 0,
          life: 1.15,
          black: false,
          size: 0.55 + random() * 0.35,
        });
      }

      const target = missile.target;
      if (target && target.alive && length(sub(missile.position, target.position)) < MISSILE_HIT_RADIUS) {
        target.alive = false;
        state.kills += 1;
        state.score += KILL_SCORE;
        state.message = 'TARGET DESTROYED  +2000';
        state.messageTime = 2;
        detonate(missile.position, sound);
        state.wrecks.push({
          position: [...target.position],
          velocity: [missile.velocity[0] * 0.18, 2, missile.velocity[2] * 0.18],
          age: 0,
          spin: 0,
        });
        state.missiles.splice(i, 1);
        if (state.mode === MODE.SORTIE && state.mission.phase === 'intercept') {
          state.interceptKills += 1;
          pushRadio(state.interceptKills >= 2
            ? 'TOWER: SPLASH TWO. AIRSPACE CLEAR.'
            : 'TOWER: SPLASH ONE. ONE BANDIT REMAINING.', 4);
        }
      } else if (missile.age > MISSILE_LIFETIME) {
        state.missiles.splice(i, 1);
      }
    }
  }

  // --- recovery --------------------------------------------------------------

  // Put a lost or stalled player back on a sensible line, at the cost of a
  // time penalty: in a fight that means a firing position behind the
  // nearest bandit, on the return leg a fresh final approach, on the course
  // a run-in at the next unpassed ring.
  function recover() {
    if (state.phase !== 'play' && state.phase !== 'return') return false;
    state.time += RECOVERY.timePenalty;
    state.lock = null;
    state.lockTime = 0;
    const bandit = nearestEnemy();
    if ((state.mode === MODE.DOGFIGHT || state.mission.phase === 'intercept') && bandit) {
      const behind = enemyForward(bandit);
      const x = bandit.position[0] - behind[0] * RECOVERY.distance;
      const z = bandit.position[2] - behind[2] * RECOVERY.distance;
      const y = Math.max(bandit.position[1] + 10, terrain.heightAt(x, z) + 70);
      Object.assign(flight, {
        position: [x, y, z],
        heading: Math.atan2(bandit.position[0] - x, bandit.position[2] - z),
        pitch: 0,
        bank: 0,
        speed: 150,
        throttle: 0.72,
        gear: 0,
        gearTarget: 0,
        grounded: false,
        liftoff: state.time,
      });
      state.message = 'BOGEY RECOVERED — TARGET AHEAD  (+5s)';
    } else if (state.phase === 'return' || state.mission.phase === 'landing') {
      Object.assign(flight, {
        position: [WORLD.runway[0], WORLD.runwayHeight + 45, WORLD.runway[1] - 760],
        heading: WORLD.runwayHeading,
        pitch: -0.025,
        bank: 0,
        speed: 82,
        throttle: 0.28,
        gear: 1,
        gearTarget: 1,
        grounded: false,
        liftoff: state.time,
      });
      state.message = 'FINAL APPROACH RECOVERED  (+5s)';
    } else {
      const target = rings[state.ringIndex];
      const previous = state.ringIndex > 0
        ? rings[state.ringIndex - 1].position
        : [WORLD.runway[0], WORLD.runwayHeight, WORLD.runway[1] + 200];
      const line = normalize(sub(target.position, previous));
      const x = target.position[0] - line[0] * RECOVERY.distance;
      const z = target.position[2] - line[2] * RECOVERY.distance;
      const y = Math.max(target.position[1] + 4, terrain.heightAt(x, z) + 55);
      Object.assign(flight, {
        position: [x, y, z],
        heading: Math.atan2(line[0], line[2]),
        pitch: 0,
        bank: 0,
        speed: 128,
        throttle: 0.62,
        gear: 0,
        gearTarget: 0,
        grounded: false,
        liftoff: state.time,
      });
      state.message = `COURSE RECOVERED — RING ${target.index}  (+5s)`;
    }
    state.messageTime = 2.6;
    pushRadio('TOWER: RECOVERY COMPLETE. REJOIN THE FIGHT.', 3.4);
    return true;
  }

  // --- fuel --------------------------------------------------------------------

  // Burn, warnings and the hose. The warnings latch — the tower says bingo
  // once, not every step below the line — and a flameout is final in the
  // air: the only restart is a full tank on the ground, which is the whole
  // point of coming home. Refuelling needs the full parked picture: on the
  // strip, gear down, nearly stopped.
  function updateFuel(dt) {
    if (!flight.engineOut) {
      const burn = FUEL.idleBurn + flight.throttle * flight.throttle * FUEL.throttleBurn;
      state.fuel = Math.max(0, state.fuel - burn * dt);
      if (!state.fuelCaution && state.fuel <= FUEL.caution) {
        state.fuelCaution = true;
        pushRadio('TOWER: FUEL AT 30 PERCENT. WATCH THE GAUGE.', 4);
      }
      if (!state.fuelBingo && state.fuel <= FUEL.bingo) {
        state.fuelBingo = true;
        pushRadio('TOWER: BINGO FUEL. RETURN TO RUNWAY NOW.', 5);
      }
      if (state.fuel <= 0) {
        flight.engineOut = true;
        flight.throttle = 0;
        pushRadio('TOWER: FLAMEOUT. YOU ARE GLIDING — NO AIR RESTART.', 5.4);
      }
    }

    const hoseOn = flight.grounded && onRunway(flight.position)
      && flight.gear > 0.85 && flight.speed < FUEL.refuelSpeed
      && state.fuel < FUEL.max - 0.4;
    if (hoseOn) {
      if (!state.refueling) {
        state.refueling = true;
        pushRadio('TOWER: REFUELING. HOLD POSITION.', 3);
      }
      state.fuel = Math.min(FUEL.max, state.fuel + FUEL.refuelRate * dt);
      if (state.fuel >= FUEL.max - 0.01) {
        state.refueling = false;
        state.fuel = FUEL.max;
        flight.engineOut = false;
        // The ground crew rearms while the hose is on: a sortie that spent
        // its racks and came home is not done flying the intercept.
        state.missileCount = MISSILE_STOCK;
        pushRadio('TOWER: FUEL FULL. READY WHEN YOU ARE.', 3.2);
      }
    } else {
      state.refueling = false;
    }
  }

  // --- the mission -------------------------------------------------------------

  // Phase transitions for the sortie. Objectives are checked off here and
  // nowhere else, and nothing here ever un-checks one: a touchdown in the
  // middle of the course is logged by completeLanding and the mission
  // simply continues after the next takeoff.
  function updateMission(sound) {
    if (state.mode !== MODE.SORTIE) return;
    if (!state.mission.tookOff && !flight.grounded) {
      state.mission.tookOff = true;
      if (state.mission.phase === 'takeoff') state.mission.phase = 'rings';
      pushRadio('TOWER: AIRBORNE. GEAR UP, CLIMB TO COURSE.', 3.6);
    }
    if (state.mission.phase === 'rings' && state.ringIndex >= RING_COUNT) {
      state.phase = 'return';
      spawnInterceptors();
      state.message = 'COURSE COMPLETE — INTERCEPT ORDERED';
      state.messageTime = 3;
    }
    if (state.mission.phase === 'intercept' && state.interceptKills >= 2) {
      state.mission.phase = 'landing';
      flight.gearTarget = 1;
      pushRadio('TOWER: AIRSPACE CLEAR. CLEARED TO LAND, GEAR DOWN.', 5.2);
      state.message = 'RETURN TO RUNWAY — FINAL APPROACH';
      state.messageTime = 3;
    }
  }

  function objectivesRemaining() {
    let count = 0;
    if (state.ringIndex < RING_COUNT) count += RING_COUNT - state.ringIndex;
    if (state.interceptKills < 2) count += 2 - state.interceptKills;
    return count;
  }

  // Settle a good touchdown: grade it, score it, park the aeroplane on the
  // strip — and only end the run if the mission is actually finished. A
  // touchdown with objectives outstanding is a pit stop, not a result.
  function completeLanding(sink, speed, center, sound) {
    const grade = gradeLanding(sink, center, speed);
    state.landing = grade;
    state.landings.push(grade);
    flight.landingScore = grade.grade;
    flight.grounded = true;
    flight.pitch = 0;
    flight.bank = 0;
    flight.position[1] = groundAt(flight.position) + 0.35;
    flight.speed = Math.min(flight.speed, 58);
    flight.gearTarget = 1;
    flight.liftoff = state.time;
    state.score += GRADE_BONUSES[grade.grade];
    if (sink > 7) pushRadio('TOWER: HARD LANDING — CHECK AIRFRAME.', 4);

    const done = state.ringIndex >= RING_COUNT && state.interceptKills >= 2;
    if (state.mode === MODE.SORTIE && done) {
      state.mission.completed = true;
      state.mission.phase = 'complete';
      state.phase = 'result';
      state.result = {
        victory: true,
        mode: state.mode,
        time: state.time,
        kills: state.kills,
        hull: state.hull,
        waves: state.dogfight.wave,
        landing: grade,
        score: state.score,
        reason: '',
        rings: state.ringIndex,
      };
      pushRadio('TOWER: WELCOME HOME, SKY RING ONE. MISSION COMPLETE.', 7);
      sound('jingle');
    } else {
      state.mission.touchdown = true;
      const remaining = objectivesRemaining();
      pushRadio(`TOWER: TOUCHDOWN LOGGED. ${remaining} OBJECTIVE${remaining === 1 ? '' : 'S'} REMAIN.`, 5);
      state.message = 'TOUCHDOWN — OBJECTIVES REMAIN';
      state.messageTime = 3;
    }
  }

  // Judge a ground contact that was not a takeoff roll. A landing counts
  // when it is on the strip, gear down, slow, flat and gentle; anything
  // else is a crash, and the reason names the first test it failed.
  function judgeContact(sound) {
    const sink = Math.max(0, -flight.verticalSpeed);
    const center = Math.abs(flight.position[0] - WORLD.runway[0]);
    const onStrip = onRunway(flight.position);
    if (onStrip
      && flight.gear > GEAR_NEEDED
      && flight.speed < LANDING.maxSpeed
      && Math.abs(flight.pitch) < LANDING.maxPitch
      && Math.abs(flight.bank) < LANDING.maxBank
      && sink < LANDING.maxSink) {
      completeLanding(sink, flight.speed, center, sound);
      return;
    }
    const reason = !onStrip ? 'OUTSIDE RUNWAY'
      : flight.gear < GEAR_NEEDED ? 'GEAR RETRACTED'
        : flight.speed >= LANDING.maxSpeed ? 'TOO FAST'
          : sink >= LANDING.maxSink ? 'HARD IMPACT'
            : Math.abs(flight.pitch) >= LANDING.maxPitch ? 'STEEP IMPACT' : 'SINK RATE';
    crash(reason, sound);
  }

  // --- the step --------------------------------------------------------------

  function step(dt, input, sound) {
    // Staged screenshot scenes freeze the world exactly as staged.
    if (state.deterministic) return;
    state.shake = Math.max(0, state.shake - dt * 1.9);
    state.damageFlash = Math.max(0, state.damageFlash - dt * 1.7);
    state.hitRadio = Math.max(-1, state.hitRadio - dt);
    tickRadio(dt);
    if (state.phase === 'crashed') {
      state.time += dt;
      ageEffects(dt);
      return;
    }
    if (state.phase !== 'play' && state.phase !== 'return') return;
    state.time += dt;
    state.clock += CLOCK.rate * dt;
    if (state.clock >= 24) state.clock -= 24;
    state.messageTime = Math.max(0, state.messageTime - dt);

    const startBank = flight.bank;
    const startThrottle = flight.throttle;
    const startY = flight.position[1];

    // Throttle and airspeed: the engine spools toward the speed the lever
    // asks for, and a climb bleeds speed while a dive buys it back. A dead
    // engine asks for nothing: the lever dies in the hand and the airframe
    // trades height for speed on the glide.
    const throttleInput = (input.down('KeyW') ? 1 : 0) - (input.down('KeyS') ? 1 : 0);
    if (flight.engineOut) {
      flight.throttle = Math.max(0, flight.throttle - dt * 0.5);
    } else {
      flight.throttle = clamp(flight.throttle + throttleInput * FLIGHT.throttleRate * dt, 0, 1);
    }
    if (flight.grounded) {
      // The ground roll has its own model: the old airborne spool could
      // never bring the aeroplane below its minimum flight speed, which
      // made stopping — and therefore refuelling — impossible. On the
      // concrete the speed chases the lever directly, brakes harder than
      // it accelerates, and can reach a full stop at idle.
      const target = flight.throttle * GROUND_ROLL_SPEED;
      const rate = target > flight.speed ? 0.16 : 0.55;
      flight.speed += (target - flight.speed) * Math.min(1, dt * rate * 60);
      if (flight.throttle < 0.06 && flight.speed < 8) {
        flight.speed = Math.max(0, flight.speed - 8 * dt);
      }
    } else if (flight.engineOut) {
      flight.speed -= (10 + flight.speed * 0.06) * dt;
      flight.speed += -Math.sin(flight.pitch) * 22 * dt;
      flight.speed = clamp(flight.speed, 0, FLIGHT.maxSpeed);
    } else {
      const targetSpeed = FLIGHT.minSpeed + flight.throttle * (FLIGHT.maxSpeed - FLIGHT.minSpeed);
      flight.speed += (targetSpeed - flight.speed) * dt * 0.7 - Math.sin(flight.pitch) * 16 * dt;
      flight.speed = clamp(flight.speed, 0, FLIGHT.maxSpeed);
    }

    // Stick: pitch and bank ease back to neutral when the player lets go,
    // and the turn comes from the bank, faster the faster you fly.
    const pitchInput = (input.down('ArrowUp') ? 1 : 0) - (input.down('ArrowDown') ? 1 : 0);
    const bankInput = (input.down('ArrowLeft') ? 1 : 0) - (input.down('ArrowRight') ? 1 : 0);
    flight.pitch += pitchInput * FLIGHT.pitchRate * dt;
    if (pitchInput === 0) flight.pitch *= Math.exp(-1.6 * dt);
    flight.pitch = clamp(flight.pitch, -PITCH_LIMIT, PITCH_LIMIT);
    flight.bank += bankInput * FLIGHT.bankRate * dt;
    if (bankInput === 0) flight.bank *= Math.exp(-1.9 * dt);
    flight.bank = clamp(flight.bank, -BANK_LIMIT, BANK_LIMIT);
    const turnFactor = Math.min(1.9, FLIGHT.turnRate * Math.max(0.4, flight.speed / 140));
    flight.heading += Math.sin(flight.bank) * turnFactor * dt;

    // Evasion, for the enemy gunnery model: how much of the sky the player
    // is sweeping through right now. Steep bank counts most, then how fast
    // the bank is changing, then throttle violence, then pitch.
    const bankRate = (flight.bank - startBank) / Math.max(0.0001, dt);
    const throttleChange = (flight.throttle - startThrottle) / Math.max(0.0001, dt);
    const evasionTarget = clamp(
      Math.abs(flight.bank) / BANK_LIMIT * 0.5
      + Math.min(1, Math.abs(bankRate) * 0.55) * 0.25
      + Math.min(1, Math.abs(throttleChange) * 2.6) * 0.2
      + Math.abs(flight.pitch) * 0.35,
      0, 1,
    );
    flight.evasion += (evasionTarget - flight.evasion) * Math.min(1, dt * 9);

    // Gear: G cycles it in the air; on the final leg it comes down by
    // itself once the runway is close.
    if (input.consume('KeyG') && !flight.grounded) {
      flight.gearTarget = flight.gearTarget > 0.5 ? 0 : 1;
    }
    if (state.phase === 'return' && state.mission.phase === 'landing'
      && runwayDistance(flight.position) < 950) {
      flight.gearTarget = 1;
    }
    flight.gear = moveToward(flight.gear, flight.gearTarget, dt * 1.7);

    const nose = forward();
    if (flight.grounded) {
      // Runway roll: pinned to the strip until rotation speed with the nose
      // up; running out of concrete at speed ends the flight instead.
      const ground = groundAt(flight.position);
      flight.position[0] += nose[0] * flight.speed * dt;
      flight.position[2] += nose[2] * flight.speed * dt;
      flight.position[1] = ground + 0.35;
      if (flight.speed > TAKEOFF_SPEED && flight.pitch > ROTATION_PITCH && !flight.engineOut) {
        flight.grounded = false;
        flight.liftoff = state.time;
        flight.gearTarget = 0;
        state.message = 'AIRBORNE — GEAR UP';
        state.messageTime = 1.6;
      } else if (!onRunway(flight.position) && flight.speed > OVERRUN_SPEED) {
        crash('RUNWAY OVERRUN', sound);
        return;
      }
    } else {
      // In the air a slow aeroplane settles: below the reference speed the
      // sink term grows and has to be held off with pitch or power. The
      // wind rides on top of the nose vector — the airframe flies its
      // heading, the air mass decides where that actually goes.
      const sink = Math.max(0, SINK_REFERENCE - flight.speed) * FLIGHT.cruiseSink
        + (flight.engineOut ? 3.2 : 0);
      const windSpeed = state.wind.speed * 0.514 * 0.55;
      flight.position[0] += nose[0] * flight.speed * dt + Math.sin(state.wind.dir) * windSpeed * dt;
      flight.position[1] += (nose[1] * flight.speed - sink) * dt;
      flight.position[2] += nose[2] * flight.speed * dt + Math.cos(state.wind.dir) * windSpeed * dt;
    }
    flight.verticalSpeed = (flight.position[1] - startY) / Math.max(0.0001, dt);

    // Rings are flown in order. The test is a sphere around the centre —
    // generous, so a fast pass through the hoop's plane is never robbed by
    // a frame boundary.
    const nextRing = rings[state.ringIndex];
    if (nextRing && length(sub(nextRing.position, flight.position)) < nextRing.radius * 0.92) {
      nextRing.passed = true;
      state.ringIndex += 1;
      state.score += RING_SCORE;
      state.message = `RING ${nextRing.index} CLEAR  +500`;
      state.messageTime = 1.4;
      sound('ring');
      const left = RING_COUNT - state.ringIndex;
      if (left === 3 || left === 1) {
        pushRadio(`TOWER: ${left} RING${left === 1 ? '' : 'S'} REMAINING.`, 3);
      }
    }

    updateMission(sound);

    // The patrol, the hunters, their guns, and the seeker.
    flyFormation(dt);
    huntEnemies(dt, sound);
    updateBullets(dt, sound);
    const found = acquireLock();
    if (found !== state.lock) {
      state.lock = found;
      state.lockTime = 0;
      if (found) sound('lock');
    }
    if (found) {
      state.lockTime = Math.min(LOCK_TIME_FULL, state.lockTime + dt);
      found.locked = state.lockTime;
    } else {
      state.lockTime = 0;
      for (const enemy of enemies) enemy.locked = 0;
    }
    if (input.consume('KeyJ') || input.consume('KeyF')) tryLaunch(nose, sound);

    guideMissiles(dt, sound);
    ageEffects(dt);
    updateFuel(dt);

    // Ground contact is tested at every altitude, not just near the strip:
    // a hillside at clearance height is a crash like any other. The judge
    // is exempt for the first two seconds after liftoff so the roll itself
    // never trips it — but the ground is not exempt: during the grace the
    // airframe is still clamped to its parked clearance, so a botched
    // rotation settles back onto the surface instead of sinking through
    // it (it reached -37 m before this clamp existed).
    if (!flight.grounded) {
      const height = groundAt(flight.position);
      const above = flight.position[1] - height;
      if (above <= CONTACT_CLEARANCE) {
        const inGrace = state.time - flight.liftoff <= LANDING_GRACE;
        flight.position[1] = Math.max(
          flight.position[1],
          height + (inGrace ? 0.35 : CONTACT_CLEARANCE),
        );
        if (!inGrace) {
          judgeContact(sound);
          return;
        }
      }
    }
    // Over water the height grid reads as the sea itself (0), not the
    // surface, so the ditch test keys off the surface itself: without
    // this, an aircraft could level off a few metres under the visible
    // sea — below seaLevel yet above the old seaLevel-6 threshold — and
    // fly on submerged. The grid's own convention is that 0 is the sea
    // reading; low land (a coastal shelf on final approach) reads above
    // it and must not trip the test, so the water test is exact.
    const overWater = terrain.heightAt(flight.position[0], flight.position[2]) <= 0;
    if ((overWater && flight.position[1] < WORLD.seaLevel + CONTACT_CLEARANCE)
      || flight.position[1] < WORLD.seaLevel - 6) {
      crash('DITCHED', sound);
    }
    if (Math.abs(flight.position[0]) > AIRSPACE_LIMIT || Math.abs(flight.position[2]) > AIRSPACE_LIMIT) {
      crash('LEFT AIRSPACE', sound);
    }

    // Dogfight waves: a cleared sky starts the countdown to the next wave,
    // and a cleared final wave ends the hunt on the victory card.
    if (state.mode === MODE.DOGFIGHT && state.phase === 'play') {
      const alive = enemies.reduce((count, enemy) => count + (enemy.alive ? 1 : 0), 0);
      if (alive === 0) {
        if (state.dogfight.wave >= DOGFIGHT.waves.length) {
          if (!state.dogfight.cleared) {
            state.dogfight.cleared = true;
            state.dogfight.victoryDelay = DOGFIGHT.victoryDelay;
            pushRadio('TOWER: ALL BANDITS DOWN. GOOD HUNTING.', 5);
          }
          state.dogfight.victoryDelay -= dt;
          if (state.dogfight.victoryDelay <= 0) {
            state.phase = 'result';
            state.result = {
              victory: true,
              mode: state.mode,
              time: state.time,
              kills: state.kills,
              hull: state.hull,
              waves: state.dogfight.wave,
              landing: null,
              score: state.score,
              reason: '',
            };
            sound('jingle');
          }
        } else if (state.dogfight.waveDelay > 0) {
          state.dogfight.waveDelay -= dt;
          if (state.dogfight.waveDelay <= 0) spawnWave(state.dogfight.wave + 1);
        } else {
          state.dogfight.waveDelay = DOGFIGHT.waveDelay;
          pushRadio('TOWER: WAVE CLEAR. NEXT WAVE INBOUND.', 4);
        }
      }
    }
  }

  return {
    state,
    terrain,
    rings,
    enemies,
    reset,
    step,
    recover,
    forward,
    onRunway,
    runwayDistance,
    groundAt,
    spawnWave,
    nearestEnemy,
  };
}
