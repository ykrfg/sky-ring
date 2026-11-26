// Screenshot staging: frozen, reproducible frames for the game's screenshots.
//
// Open game/index.html with ?scene=<name> (and optionally &seed=<n>) and the
// world is arranged into that scene and never simulated — world.step() refuses
// to run while state.deterministic is set — so the frame on screen is the same
// on every load, whenever it is opened. Scene mode also pins the renderer to
// the internal 1280x720 frame (see main.js).
//
//   title      the title screen, parked in the title pose over the island
//   takeoff    wheels just up, gear halfway through its retract
//   ring       on final for ring 1
//   blur       top speed in a bank — the speed-blur check
//   climb      steep climb; checks the chase camera keeps the aircraft framed
//   range      facing the mountains — terrain normals and baked shading
//   behind     flying away from the next ring — edge marker, recovery prompt
//   missile    a missile just away, laying its white smoke trail
//   lock       lock held on the enemy formation
//   touchdown  a breath above the runway — contact shadow check
//   landing    final approach, course complete, gear down
//   result     the result card with a representative score line
//   crash      the failed-landing overlay, fire and black smoke
//   credits    the credits / licences page
//   settings   the BGM / SE / blur settings panel
//
// &seed=N fixes both the seeded world generation and every staged clock.

import { TITLE_POSE } from './config.js';
import { createRandom } from './random.js';

// Every scene name the staging understands, in the order they are documented
// above — kept as data so tooling (and the smoke tests) can enumerate the
// shots instead of scraping this file's comments.
export const SCENE_NAMES = [
  'title', 'takeoff', 'ring', 'blur', 'climb', 'range', 'behind', 'missile',
  'lock', 'touchdown', 'landing', 'result', 'crash', 'credits', 'settings',
];

// Assign a whole flight pose at once; scenes describe the aircraft, not a
// series of mutations.
function pose(state, values) {
  Object.assign(state.flight, values);
}

// The menu screens all share the title pose and a stopped clock; only the
// phase (and, for settings, the panel contents) differs.
function park(state, phase) {
  pose(state, { ...TITLE_POSE, position: [...TITLE_POSE.position], liftoff: 0 });
  state.phase = phase;
  state.time = 0;
}

export function stageScene(name, world, seed = 1) {
  const s = world.state;
  world.reset();

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
  s.particles.length = 0;
  s.explosions.length = 0;
  s.wrecks.length = 0;
  s.phase = 'play';

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
      s.time = 8.4;
      break;

    case 'ring':
      pose(s, {
        position: [1150, 185, -130], heading: 0.01, pitch: 0.03, bank: 0.02,
        speed: 132, throttle: 0.58, gear: 0, gearTarget: 0, grounded: false,
      });
      s.time = 16.2;
      break;

    case 'range':
      pose(s, {
        position: [600, 360, -300], heading: 4.2, pitch: 0.01, bank: 0,
        speed: 150, throttle: 0.7, gear: 0, gearTarget: 0, grounded: false,
      });
      s.ringIndex = 3;
      s.time = 24.8;
      break;

    case 'blur':
      // The range framing at full throttle, with the blur slider pinned.
      pose(s, {
        position: [600, 360, -300], heading: 4.2, pitch: 0.01, bank: 0.5,
        speed: 245, throttle: 1, gear: 0, gearTarget: 0, grounded: false,
      });
      s.ringIndex = 3;
      s.settings = { bgm: 0.45, se: 0.8, blur: 1 };
      s.time = 24.8;
      break;

    case 'behind':
      pose(s, {
        position: [1150, 300, 300], heading: 0, pitch: 0.02, bank: 0,
        speed: 150, throttle: 0.7, gear: 0, gearTarget: 0, grounded: false,
      });
      s.time = 33.4;
      break;

    case 'climb':
      pose(s, {
        position: [1150, 400, -300], heading: 0.1, pitch: 0.5, bank: 0.24,
        speed: 155, throttle: 0.8, gear: 0, gearTarget: 0, grounded: false,
      });
      s.time = 30.2;
      break;

    case 'touchdown':
      pose(s, {
        position: [1150, 68, -560], heading: 0, pitch: -0.02, bank: 0,
        speed: 45, throttle: 0.2, gear: 1, gearTarget: 1, grounded: false,
      });
      s.time = 12.4;
      break;

    case 'landing':
      pose(s, {
        position: [1150, 76, -850], heading: 0, pitch: -0.012, bank: 0,
        speed: 75, throttle: 0.3, gear: 1, gearTarget: 1, grounded: false,
      });
      s.ringIndex = 9;
      s.phase = 'return';
      s.score = 4500;
      s.time = 104.8;
      for (const ring of s.rings) ring.passed = true;
      break;

    case 'result':
      pose(s, {
        position: [1150, 64.4, -240], heading: 0, pitch: 0, bank: 0,
        speed: 44, throttle: 0.2, gear: 1, gearTarget: 1, grounded: true,
        landingScore: 'PERFECT',
      });
      s.phase = 'result';
      s.ringIndex = 9;
      s.kills = 2;
      s.score = 10800;
      s.time = 96.4;
      for (const ring of s.rings) ring.passed = true;
      break;

    case 'crash': {
      pose(s, {
        position: [1150, 78, -610], heading: 0, pitch: -0.2, bank: 0.25,
        speed: 150, throttle: 0.7, gear: 0, gearTarget: 0, grounded: false,
      });
      s.phase = 'crashed';
      s.crashReason = 'TOO FAST';
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
      // 'lock' and 'missile' share one framing over the southern channel;
      // the missile scene starts a beat earlier in the engagement and puts
      // a round in the air. Any unrecognised name lands here too, framed
      // as the lock scene.
      pose(s, {
        position: [430, 472, -760], heading: 0.09, pitch: 0.025, bank: -0.06,
        speed: 158, throttle: 0.72, gear: 0, gearTarget: 0, grounded: false,
      });
      s.time = 58.6;
      const target = s.enemies[0];
      target.position = name === 'lock' ? [442, 470, -700] : [470, 468, -620];
      target.origin = [...target.position];
      target.alive = true;
      s.lock = target;
      s.lockTime = name === 'lock' ? 1.05 : 0.82;
      target.locked = s.lockTime;
      s.missileCount = name === 'missile' ? 5 : 6;
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
