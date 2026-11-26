// SKY RING world simulation. This module owns every piece of mutable game
// state — the player's flight model, the ring course, the enemy patrol,
// missiles and their smoke, wrecks, particles and explosions — and advances
// it in fixed steps. It knows nothing about rendering or the DOM: main.js
// feeds it an input facade ({ down, consume }) and a sound callback, and the
// renderer only reads what is left in `state`.
//
// createWorld(terrain, random)
//   terrain — the height grid built by gltf.js buildTerrainGrid; the world
//             asks it for ground heights via terrain.heightAt(x, z).
//   random  — the seeded generator from random.js. Every random draw the
//             simulation makes (burst spreads, wreck smoke) comes from it,
//             so a seed fully determines a run.

import { AIRCRAFT, ENEMY_ROUTE, FLIGHT, RECOVERY, RING_COUNT, RINGS, WORLD } from './config.js';
import { add, dot, length, normalize, scale, sub } from './math.js';

// --- module-local tuning ---------------------------------------------------

const MISSILE_STOCK = 6;          // rounds on the racks at the start of a run
const RING_RADIUS = 26;           // metres; the pass test uses 92% of this

const PITCH_LIMIT = 0.55;         // radians of nose up/down the stick can ask for
const BANK_LIMIT = 0.95;          // radians of roll, likewise
const SINK_REFERENCE = 120;       // below this speed the airframe settles

const TAKEOFF_SPEED = 64;         // rotation speed on the runway roll
const ROTATION_PITCH = 0.055;     // nose-up needed before the wheels leave
const OVERRUN_SPEED = 20;         // rolling this fast off the strip is a crash

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
const MAX_TOUCH_SPEED = 92;       // faster arrivals break the aeroplane
const PERFECT_SPEED = 63;         // touchdown at or below this is a greaser
const CLEAN_SPEED = 78;           // ... at or below this it is still smooth
const MAX_TOUCH_PITCH = 0.17;
const MAX_TOUCH_BANK = 0.2;

const RING_SCORE = 500;
const KILL_SCORE = 2000;
const LANDING_SCORES = [3000, 2200, 1500]; // PERFECT / CLEAN / FIRM

const AIRSPACE_LIMIT = 2400;      // leaving this box ends the flight
const PARTICLE_CAP = 220;         // oldest particles are dropped past this

// The enemy patrol: a wide circuit over the southern channel. Each aircraft
// flies the same ellipse, offset sideways by its formation slot and started
// at its own phase so the three stay strung out along the route.
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

export function createWorld(terrain, random) {
  // --- state ---------------------------------------------------------------

  // The flight state at the top of the runway, engine idling. reset() puts
  // exactly this back, so a restart is indistinguishable from a first run.
  function freshFlight() {
    return {
      position: [...AIRCRAFT.start],
      heading: AIRCRAFT.startHeading,
      pitch: 0,
      bank: 0,
      speed: 0,
      throttle: 0.25,
      gear: 1,
      gearTarget: 1,
      grounded: true,
      crashed: false,
      landingScore: '—',
      liftoff: 0,
    };
  }

  function freshEnemy(slot) {
    const point = ENEMY_ROUTE[slot];
    return {
      position: [...point],
      origin: [...point],
      alive: true,
      locked: 0,
      phase: slot * PATROL.phaseStep,
      route: 0,
      bank: 0,
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
    missiles: [],
    particles: [],
    explosions: [],
    wrecks: [],
    time: 0,
    phase: 'title',
    ringIndex: 0,
    score: 0,
    kills: 0,
    missileCount: MISSILE_STOCK,
    lock: null,
    lockTime: 0,
    landing: null,
    crashReason: '',
    seed: 1,
    deterministic: false,
    stage: null,
    best: 0,
    message: '',
    messageTime: 0,
  };

  function reset() {
    Object.assign(flight, freshFlight());
    for (const ring of rings) ring.passed = false;
    enemies.forEach((enemy, slot) => Object.assign(enemy, freshEnemy(slot)));
    state.missiles.length = 0;
    state.particles.length = 0;
    state.explosions.length = 0;
    state.wrecks.length = 0;
    Object.assign(state, {
      time: 0,
      phase: 'play',
      ringIndex: 0,
      score: 0,
      kills: 0,
      missileCount: MISSILE_STOCK,
      lock: null,
      lockTime: 0,
      landing: null,
      crashReason: '',
      message: '',
      messageTime: 0,
    });
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

  // --- the enemy patrol and the seeker --------------------------------------

  function flyFormation(dt) {
    for (let slot = 0; slot < enemies.length; slot++) {
      const enemy = enemies[slot];
      if (!enemy.alive) continue;
      enemy.phase += dt * PATROL.rate;
      const a = enemy.phase;
      enemy.position[0] = PATROL.center[0] + Math.cos(a) * PATROL.sweepX + (slot - 1) * PATROL.spacingX;
      enemy.position[1] = PATROL.center[1] + Math.sin(a * PATROL.bobRate + slot) * PATROL.bob;
      enemy.position[2] = PATROL.center[2] + Math.sin(a) * PATROL.sweepZ + (slot - 1) * PATROL.spacingZ;
      enemy.bank = Math.sin(a) * PATROL.bankAmount;
    }
  }

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
  // airframe into the wreck list to tumble out of the sky.
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
      } else if (missile.age > MISSILE_LIFETIME) {
        state.missiles.splice(i, 1);
      }
    }
  }

  // --- recovery --------------------------------------------------------------

  // Put a lost or stalled player back on a sensible line, at the cost of a
  // time penalty: on the return leg that means a fresh final approach, on
  // the course it means a run-in at the next unpassed ring.
  function recover() {
    if (state.phase !== 'play' && state.phase !== 'return') return false;
    state.time += RECOVERY.timePenalty;
    state.lock = null;
    state.lockTime = 0;
    if (state.phase === 'return') {
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
    return true;
  }

  // --- the step --------------------------------------------------------------

  // Judge a ground contact that was not a takeoff roll. A landing counts
  // when it is on the strip, gear down, slow, and flat; anything else is a
  // crash, and the reason names the first test it failed.
  function judgeTouchdown(sound) {
    const onStrip = onRunway(flight.position);
    if (onStrip
      && flight.gear > GEAR_NEEDED
      && flight.speed < MAX_TOUCH_SPEED
      && Math.abs(flight.pitch) < MAX_TOUCH_PITCH
      && Math.abs(flight.bank) < MAX_TOUCH_BANK) {
      const grade = flight.speed < PERFECT_SPEED ? 0 : flight.speed < CLEAN_SPEED ? 1 : 2;
      state.phase = 'result';
      flight.landingScore = ['PERFECT', 'CLEAN', 'FIRM'][grade];
      state.score += LANDING_SCORES[grade];
      state.landing = {
        speed: flight.speed,
        center: Math.abs(flight.position[0] - WORLD.runway[0]),
      };
      sound('jingle');
      return;
    }
    const reason = !onStrip ? 'OUTSIDE RUNWAY'
      : flight.gear < GEAR_NEEDED ? 'GEAR RETRACTED'
        : flight.speed >= MAX_TOUCH_SPEED ? 'TOO FAST'
          : Math.abs(flight.pitch) >= MAX_TOUCH_PITCH ? 'STEEP IMPACT' : 'SINK RATE';
    crash(reason, sound);
  }

  function step(dt, input, sound) {
    // Staged screenshot scenes freeze the world exactly as staged.
    if (state.deterministic) return;
    if (state.phase === 'crashed') {
      state.time += dt;
      ageEffects(dt);
      return;
    }
    if (state.phase !== 'play' && state.phase !== 'return') return;
    state.time += dt;
    state.messageTime = Math.max(0, state.messageTime - dt);

    // Throttle and airspeed: the engine spools toward the speed the lever
    // asks for, and a climb bleeds speed while a dive buys it back.
    const throttleInput = (input.down('KeyW') ? 1 : 0) - (input.down('KeyS') ? 1 : 0);
    flight.throttle = clamp(flight.throttle + throttleInput * FLIGHT.throttleRate * dt, 0, 1);
    const targetSpeed = FLIGHT.minSpeed + flight.throttle * (FLIGHT.maxSpeed - FLIGHT.minSpeed);
    flight.speed += (targetSpeed - flight.speed) * dt * 0.7 - Math.sin(flight.pitch) * 16 * dt;
    flight.speed = clamp(flight.speed, 0, FLIGHT.maxSpeed);

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

    // Gear: G cycles it in the air; on the return leg it comes down by
    // itself once the runway is close.
    if (input.consume('KeyG') && !flight.grounded) {
      flight.gearTarget = flight.gearTarget > 0.5 ? 0 : 1;
    }
    if (state.phase === 'return' && runwayDistance(flight.position) < 950) {
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
      if (flight.speed > TAKEOFF_SPEED && flight.pitch > ROTATION_PITCH) {
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
      // sink term grows and has to be held off with pitch or power.
      const sink = Math.max(0, SINK_REFERENCE - flight.speed) * FLIGHT.cruiseSink;
      flight.position[0] += nose[0] * flight.speed * dt;
      flight.position[1] += (nose[1] * flight.speed - sink) * dt;
      flight.position[2] += nose[2] * flight.speed * dt;
    }

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
      if (state.ringIndex >= RING_COUNT) {
        state.phase = 'return';
        state.message = 'COURSE COMPLETE — RETURN TO RUNWAY';
        state.messageTime = 3;
        flight.gearTarget = 1;
      }
    }

    // The patrol, the seeker, and the trigger.
    flyFormation(dt);
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

    // Ground contact is tested at every altitude, not just near the strip:
    // a hillside at clearance height is a crash like any other. The first
    // two seconds after liftoff are exempt so the roll itself never trips it.
    if (!flight.grounded && state.time - flight.liftoff > LANDING_GRACE) {
      const height = groundAt(flight.position);
      const above = flight.position[1] - height;
      if (above <= CONTACT_CLEARANCE) {
        flight.position[1] = Math.max(flight.position[1], height + CONTACT_CLEARANCE);
        judgeTouchdown(sound);
        return;
      }
    }
    if (flight.position[1] < WORLD.seaLevel - 6) crash('DITCHED', sound);
    if (Math.abs(flight.position[0]) > AIRSPACE_LIMIT || Math.abs(flight.position[2]) > AIRSPACE_LIMIT) {
      crash('LEFT AIRSPACE', sound);
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
  };
}
