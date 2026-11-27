// Screenshot staging: frozen, reproducible frames for the game's screenshots.
//
// Open game/index.html with ?scene=<name> (and optionally &seed=<n>) and the
// world is arranged into that scene and never simulated — world.step() refuses
// to run while state.deterministic is set — so the frame on screen is the same
// on every load, whenever it is opened. Scene mode also pins the renderer to
// the internal 1280x720 frame (see main.js).
//
//   title      the title screen, parked in the title pose over the island (day)
//   takeoff    wheels just up, gear halfway through its retract (day)
//   ring       on final for ring 1 (afternoon)
//   blur       top speed in a bank — the speed-blur check (late afternoon)
//   climb      steep climb; checks the chase camera keeps the aircraft framed
//   range      facing the mountains — terrain normals and baked shading
//   behind     flying away from the next ring — edge marker, recovery prompt
//   missile    a missile just away, laying its white smoke trail (dusk)
//   lock       lock held on the enemy formation (dusk)
//   dogfight   wave 2 of DOGFIGHT: tracers in the air, hull down, the tower's
//              hit call pinned on the radio line (dusk)
//   touchdown  a breath above the runway — contact shadow check (night)
//   landing    final approach, course complete, gear down (night, runway lit)
//   refuel     parked on the strip mid-mission, hose on, refuelling (night)
//   result     the result card, with a graded landing on it
//   crash      the failed-landing overlay, fire and black smoke (dusk)
//   credits    the credits / licences page
//   settings   the BGM / SE / blur settings panel
//
// Every scene also pins state.clock, so each shot keeps its own sky — the
// renderer takes the sun, sky, fog and lamps from the clock alone — and sets
// whatever mission, radio and resource state the HUD now reads, so the
// checklist, radio line, fuel and hull all tell the same story as the frame.
//
// &seed=N fixes both the seeded world generation and every staged clock.

import { CLOCK, MODE, TITLE_POSE } from './config.js';
import { createRandom } from './random.js';

// Every scene name the staging understands, in the order they are documented
// above — kept as data so tooling (and the smoke tests) can enumerate the
// shots instead of scraping this file's comments.
export const SCENE_NAMES = [
  'title', 'takeoff', 'ring', 'blur', 'climb', 'range', 'behind', 'missile',
  'lock', 'dogfight', 'touchdown', 'landing', 'refuel', 'result', 'crash',
  'credits', 'settings',
];

// Assign a whole flight pose at once; scenes describe the aircraft, not a
// series of mutations.
function pose(state, values) {
  Object.assign(state.flight, values);
}

// The menu screens all share the title pose and a stopped clock; only the
// phase (and, for settings, the panel contents) differs. The backdrop sits
// at CLOCK.title, the hour the live title screen flies at.
function park(state, phase) {
  pose(state, { ...TITLE_POSE, position: [...TITLE_POSE.position], liftoff: 0 });
  state.phase = phase;
  state.time = 0;
  state.clock = CLOCK.title;
}

// Pin the tower's current line. A frozen scene never ages the line out, so
// whatever a scene pins here is what the frame shows; the queue is emptied
// so nothing is waiting behind it.
function radio(state, text, seconds = 4.2) {
  state.radio = { text, time: seconds, max: seconds };
  state.radioQueue = [];
}

// A bandit placed by hand: position and origin together (the renderer faces
// it at the player itself), flagged for the fight it belongs to.
function bandit(enemy, position, { dogfight = false, bank = 0 } = {}) {
  enemy.position = [...position];
  enemy.origin = [...position];
  enemy.alive = true;
  enemy.engaged = true;
  enemy.dogfight = dogfight;
  enemy.aggression = dogfight ? 1 : 0.42;
  enemy.speed = dogfight ? 178 : 162;
  enemy.bank = bank;
  enemy.locked = 0;
}

export function stageScene(name, world, seed = 1) {
  const s = world.state;
  world.reset(name === 'dogfight' ? MODE.DOGFIGHT : MODE.SORTIE, seed);

  // Common ground: a fresh run with every ring unclaimed, every enemy up,
  // no effects in flight, and a clock derived from the seed alone.
  s.seed = seed;
  s.deterministic = true;
  s.stage = name;
  s.random = createRandom(seed >>> 0);
  s.time = 14 + ((seed >>> 0) % 97) * 0.01;
  for (const ring of s.rings) ring.passed = false;
  for (const enemy of s.enemies) enemy.alive = true;
  s.missiles.length = 0;
  s.bullets.length = 0;
  s.particles.length = 0;
  s.explosions.length = 0;
  s.wrecks.length = 0;
  s.phase = 'play';
  // reset() leaves its own opening call on the radio (cleared for takeoff,
  // or the first wave inbound); a staged frame keeps only the line its own
  // scene pins, and starts with a steady camera and unmarked edges.
  s.radio = null;
  s.radioQueue = [];
  s.damageFlash = 0;
  s.shake = 0;

  switch (name) {
    case 'title':
      park(s, 'title');
      break;

    case 'credits':
      park(s, 'credits');
      break;

    case 'settings':
      park(s, 'settings');
      s.settings = { bgm: 0.45, se: 0.8, blur: 0.45 };
      s.settingsSelected = 0;
      break;

    case 'takeoff':
      pose(s, {
        position: [1150, 96, -470], heading: 0.03, pitch: 0.14, bank: 0.03,
        speed: 86, throttle: 0.68, gear: 0.55, gearTarget: 0, grounded: false,
      });
      s.mission.tookOff = true;
      s.mission.phase = 'rings';
      s.clock = 14.2;
      s.time = 8.4;
      radio(s, 'TOWER: AIRBORNE. GEAR UP, CLIMB TO COURSE.', 3.6);
      break;

    case 'ring':
      pose(s, {
        position: [1150, 185, -130], heading: 0.01, pitch: 0.03, bank: 0.02,
        speed: 132, throttle: 0.58, gear: 0, gearTarget: 0, grounded: false,
      });
      s.mission.tookOff = true;
      s.mission.phase = 'rings';
      s.clock = 15.7;
      s.time = 16.2;
      radio(s, 'TOWER: FIRST GATE AHEAD. HOLD YOUR LINE.');
      break;

    case 'range':
      pose(s, {
        position: [600, 360, -300], heading: 4.2, pitch: 0.01, bank: 0,
        speed: 150, throttle: 0.7, gear: 0, gearTarget: 0, grounded: false,
      });
      s.ringIndex = 3;
      s.mission.tookOff = true;
      s.mission.phase = 'rings';
      s.clock = 15.0;
      s.time = 24.8;
      break;

    case 'blur':
      // The range framing at full throttle, with the blur slider pinned.
      pose(s, {
        position: [600, 360, -300], heading: 4.2, pitch: 0.01, bank: 0.5,
        speed: 245, throttle: 1, gear: 0, gearTarget: 0, grounded: false,
      });
      s.ringIndex = 3;
      s.mission.tookOff = true;
      s.mission.phase = 'rings';
      s.settings = { bgm: 0.45, se: 0.8, blur: 1 };
      s.clock = 16.2;
      s.time = 24.8;
      break;

    case 'behind':
      pose(s, {
        position: [1150, 300, 300], heading: 0, pitch: 0.02, bank: 0,
        speed: 150, throttle: 0.7, gear: 0, gearTarget: 0, grounded: false,
      });
      s.mission.tookOff = true;
      s.mission.phase = 'rings';
      s.clock = 16.0;
      s.time = 33.4;
      break;

    case 'climb':
      pose(s, {
        position: [1150, 400, -300], heading: 0.1, pitch: 0.5, bank: 0.24,
        speed: 155, throttle: 0.8, gear: 0, gearTarget: 0, grounded: false,
      });
      s.mission.tookOff = true;
      s.mission.phase = 'rings';
      s.clock = 15.4;
      s.time = 30.2;
      break;

    case 'dogfight': {
      // Wave 2 of the dogfight, a minute in: one bandit of the wave is
      // already down (its wreck is tumbling below the fight), a second is
      // dead ahead inside lock range, and the third is behind the player,
      // mid-burst — its tracers are the rounds strung along the line past
      // the player's shoulder. The hull has paid for the kills so far.
      pose(s, {
        position: [420, 470, -760], heading: 0.12, pitch: 0.02, bank: 0.3,
        speed: 186, throttle: 0.86, gear: 0, gearTarget: 0, grounded: false,
      });
      s.time = 51.6;
      s.clock = 18.0;
      s.dogfight.wave = 2;
      s.kills = 3;
      s.score = 6000;
      s.hull = 65;
      s.fuel = 71;
      s.missileCount = 4;
      s.damageFlash = 0.34;
      s.shake = 0.3;

      const target = s.enemies[0];
      bandit(target, [441, 473, -591], { dogfight: true, bank: -0.2 });
      s.lock = target;
      s.lockTime = 1.0;
      target.locked = s.lockTime;

      const shooter = s.enemies[1];
      bandit(shooter, [322, 452, -1005], { dogfight: true, bank: 0.35 });

      const down = s.enemies[2];
      down.alive = false;
      down.engaged = false;
      s.wrecks.push({
        position: [520, 330, -430], velocity: [12, -6, 20], age: 1.1, spin: 0.8,
      });

      // The burst in flight: rounds spaced along one line from the
      // shooter's nose to a point just off the player's shoulder, all at
      // tracer speed. The string runs past the player (some 250 m along
      // the line), so the last rounds are already ahead of the aircraft,
      // streaming through the camera's view.
      const muzzle = [327, 453, -993];
      const past = [447, 477, -747];
      const dx = past[0] - muzzle[0];
      const dy = past[1] - muzzle[1];
      const dz = past[2] - muzzle[2];
      const len = Math.hypot(dx, dy, dz);
      const dir = [dx / len, dy / len, dz / len];
      for (let i = 0; i < 9; i++) {
        const d = 30 + i * 34;
        s.bullets.push({
          position: [muzzle[0] + dir[0] * d, muzzle[1] + dir[1] * d, muzzle[2] + dir[2] * d],
          velocity: [dir[0] * 620, dir[1] * 620, dir[2] * 620],
          age: 0.1 + i * 0.02, life: 2.7, damage: 6,
        });
      }
      radio(s, 'TOWER: YOU ARE HIT — HULL 65 PERCENT.', 3.2);
      break;
    }

    case 'touchdown':
      pose(s, {
        position: [1150, 68, -560], heading: 0, pitch: -0.02, bank: 0,
        speed: 45, throttle: 0.2, gear: 1, gearTarget: 1, grounded: false,
      });
      s.ringIndex = 9;
      s.interceptKills = 2;
      s.kills = 2;
      s.mission.tookOff = true;
      s.mission.phase = 'landing';
      for (const ring of s.rings) ring.passed = true;
      s.clock = 21.2;
      s.time = 12.4;
      radio(s, 'TOWER: GEAR DOWN. EASE THE SINK RATE.', 5);
      break;

    case 'landing':
      pose(s, {
        position: [1150, 76, -850], heading: 0, pitch: -0.012, bank: 0,
        speed: 75, throttle: 0.3, gear: 1, gearTarget: 1, grounded: false,
      });
      s.ringIndex = 9;
      s.phase = 'return';
      s.interceptKills = 2;
      s.kills = 2;
      s.mission.tookOff = true;
      s.mission.phase = 'landing';
      s.score = 8500;
      s.clock = 21.4;
      s.time = 104.8;
      for (const ring of s.rings) ring.passed = true;
      radio(s, 'TOWER: CLEARED TO LAND, RUNWAY 36.', 5);
      break;

    case 'refuel': {
      // A pit stop, not the end of the sortie: the course is flown and one
      // of the two intercept bandits is down, but the second is still up —
      // so the aircraft sits on the strip at night with the hose on, the
      // checklist holding its place (takeoff and rings ticked, intercept
      // current) and the touchdown it just made already graded.
      pose(s, {
        position: [1150, 64.4, -300], heading: 0, pitch: 0, bank: 0,
        speed: 0, throttle: 0, gear: 1, gearTarget: 1, grounded: true,
        landingScore: 'B',
      });
      s.time = 232.6;
      s.clock = 21.5;
      s.ringIndex = 9;
      for (const ring of s.rings) ring.passed = true;
      s.interceptKills = 1;
      s.kills = 1;
      s.score = 8900;
      s.mission.tookOff = true;
      s.mission.touchdown = true;
      s.mission.phase = 'intercept';
      const grade = {
        grade: 'B', points: 7, sink: 2.6, center: 5.0, speed: 84,
        sinkScore: 2, centerScore: 3, speedScore: 2,
      };
      s.landing = grade;
      s.landings = [grade];
      s.fuel = 52;
      s.refueling = true;
      bandit(s.enemies[0], [610, 505, -930], { bank: 0.15 });
      s.enemies[1].alive = false;
      s.enemies[2].alive = false;
      radio(s, 'TOWER: REFUELING. HOLD POSITION.', 3);
      break;
    }

    case 'result': {
      pose(s, {
        position: [1150, 64.4, -240], heading: 0, pitch: 0, bank: 0,
        speed: 44, throttle: 0.2, gear: 1, gearTarget: 1, grounded: true,
        landingScore: 'B',
      });
      s.phase = 'result';
      s.ringIndex = 9;
      s.interceptKills = 2;
      s.kills = 2;
      s.score = 10900;
      s.hull = 88;
      s.clock = 19.1;
      s.time = 96.4;
      for (const ring of s.rings) ring.passed = true;
      s.mission.tookOff = true;
      s.mission.touchdown = true;
      s.mission.phase = 'complete';
      s.mission.completed = true;
      // The touchdown on the card, measured the way world.js grades one:
      // a firm arrival, close to the centreline, a touch fast — a B.
      const landing = {
        grade: 'B', points: 7, sink: 2.2, center: 4.1, speed: 82,
        sinkScore: 2, centerScore: 3, speedScore: 2,
      };
      s.landing = landing;
      s.landings = [landing];
      s.result = {
        victory: true, mode: MODE.SORTIE, time: s.time, kills: 2, hull: 88,
        waves: 1, landing, score: 10900, reason: '', rings: 9,
      };
      break;
    }

    case 'crash': {
      pose(s, {
        position: [1150, 78, -610], heading: 0, pitch: -0.2, bank: 0.25,
        speed: 150, throttle: 0.7, gear: 0, gearTarget: 0, grounded: false,
      });
      s.phase = 'crashed';
      s.crashReason = 'TOO FAST';
      s.ringIndex = 9;
      s.interceptKills = 2;
      s.mission.tookOff = true;
      s.mission.phase = 'landing';
      for (const ring of s.rings) ring.passed = true;
      s.clock = 18.7;
      s.time = 41.2;
      s.explosions.push({ position: [1150, 74, -604], age: 0.34, duration: 0.9 });
      // A grid of rising black smoke over the wreck, aged in a fixed pattern.
      for (let i = 0; i < 24; i++) {
        s.particles.push({
          position: [1150 + (i % 6) * 2, 76 + (i % 5), -610 + Math.floor(i / 6) * 2],
          velocity: [0, 2, 0],
          age: 0.2 + (i % 7) * 0.08,
          life: 2.4, black: true, size: 1.4,
        });
      }
      break;
    }

    default: {
      // 'lock' and 'missile' share one framing over the southern channel,
      // late in the sortie: the course is flown and the intercept is live,
      // with the second bandit up behind. The missile scene starts a beat
      // earlier in the engagement and puts a round in the air. Any
      // unrecognised name lands here too, framed as the lock scene.
      pose(s, {
        position: [430, 472, -760], heading: 0.09, pitch: 0.025, bank: -0.06,
        speed: 158, throttle: 0.72, gear: 0, gearTarget: 0, grounded: false,
      });
      s.phase = 'return';
      s.ringIndex = 9;
      s.mission.tookOff = true;
      s.mission.phase = 'intercept';
      for (const ring of s.rings) ring.passed = true;
      s.clock = name === 'lock' ? 18.0 : 18.2;
      s.time = 58.6;
      const target = s.enemies[0];
      bandit(target, name === 'lock' ? [442, 470, -700] : [470, 468, -620]);
      bandit(s.enemies[1], [610, 500, -930], { bank: 0.1 });
      s.enemies[2].alive = false;
      s.lock = target;
      s.lockTime = name === 'lock' ? 1.05 : 0.82;
      target.locked = s.lockTime;
      s.missileCount = name === 'missile' ? 5 : 6;
      radio(s, 'TOWER: TWO BANDITS NORTH OF THE CHANNEL. WEAPONS FREE.', 5.4);
      if (name === 'missile') {
        s.missiles.push({
          position: [510, 472, -690], velocity: [150, 1, 190],
          target, age: 0.42, life: 0, smoke: 0.04,
        });
        // The trail the round has laid so far, oldest at the tail.
        for (let i = 0; i < 30; i++) {
          const t = i / 30;
          s.particles.push({
            position: [
              444 + t * 66 + Math.sin(i * 1.7) * 0.8,
              472.2 + t * 1.5,
              -744 + t * 52,
            ],
            velocity: [0, 1.4, 0],
            age: 0.06 + t * 0.6, life: 1.45, black: false,
            size: 0.55 + t * 0.95,
          });
        }
      }
      break;
    }
  }

  s.deterministic = true;
  return s;
}
