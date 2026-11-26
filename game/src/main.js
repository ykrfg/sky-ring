// SKY RING entry point: boot, the frame loop, and everything that glues the
// modules together — DOM handles, persisted settings, the sound effects, the
// two music tracks, screen routing and the fixed-step simulation drive.
//
// The world advances in fixed STEP slices drained from an accumulator, so
// the simulation rate never depends on the display's refresh rate. Rendering
// is the renderer's business; this file only decides what state it is in.
//
// ?scene=<name> hands control to shots.js instead of the loop: the staged
// world is painted twice, window.__shotReady is set, and the same frozen
// frame is repainted for as long as the page stays open.

import { AUDIO, BGM, RING_COUNT, STEP, SUN_DIR, TITLE_POSE, WORLD } from './config.js';
import { buildTerrainAO, buildTerrainGrid, loadGLB, prepareTerrain } from './gltf.js';
import { createInput } from './input.js';
import { createRandom } from './random.js';
import { createRenderer } from './render.js';
import { stageScene } from './shots.js';
import { createWorld } from './world.js';

// --- URL parameters ---------------------------------------------------------

const params = new URLSearchParams(location.search);
const sceneName = params.get('scene');
const seedParam = params.get('seed');
const seed = seedParam === null ? 90210 : (Number.parseInt(seedParam, 10) || 1);

// --- DOM handles --------------------------------------------------------------

const glCanvas = document.getElementById('view');
const hudCanvas = document.getElementById('hud');
const statusEl = document.getElementById('status');
const bootEl = document.getElementById('boot');
const input = createInput(hudCanvas);

// --- Settings -----------------------------------------------------------------

const SETTINGS_KEY = 'sky-ring.audio';
const SETTING_KEYS = ['bgm', 'se', 'blur'];
const clamp01 = (value) => Math.max(0, Math.min(1, value));

function loadSettings() {
  const settings = { bgm: 0.45, se: 0.8, blur: 0.45 };
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
    for (const key of SETTING_KEYS) {
      if (Number.isFinite(saved[key])) settings[key] = clamp01(saved[key]);
    }
  } catch { /* storage unavailable — fly with the defaults */ }
  return settings;
}
const settings = loadSettings();
function saveSettings() {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch { /* private mode */ }
}

// --- Sound effects --------------------------------------------------------------

// One <audio> element per entry in the AUDIO map. The engine is the only
// loop; it starts silent and the frame loop drives its rate and volume from
// the throttle. Everything else is a one-shot restarted by playSound().
const sfx = {};
for (const [key, path] of Object.entries(AUDIO)) {
  const element = new Audio(path);
  element.preload = 'auto';
  element.loop = key === 'engine';
  element.volume = 0;
  sfx[key] = element;
}
function playSound(name) {
  const element = sfx[name];
  if (!element || name === 'engine') return;
  try {
    element.volume = settings.se;
    element.currentTime = 0;
    element.play().catch(() => {});
  } catch { /* autoplay policy — the next trigger retries */ }
}

// --- Music ----------------------------------------------------------------------

// Two looping tracks, one per family of screens: menus (title, credits,
// settings) get the title track; a sortie and its outcome (play, return,
// result, crashed) get the flight track. syncMusic is the only place either
// is started or stopped, so the two can never overlap.
const MUSIC_GAIN = 0.55; // the music sits under the effects on its slider
const music = {};
for (const [name, path] of Object.entries(BGM)) {
  const element = new Audio(path);
  element.preload = 'auto';
  element.loop = true;
  element.volume = settings.bgm * MUSIC_GAIN;
  music[name] = element;
}
function musicFamily(phase) {
  return phase === 'title' || phase === 'credits' || phase === 'settings' ? 'title' : 'flight';
}
let phaseNow = 'title'; // what the loop is showing; 'title' until it starts
function syncMusic(phase) {
  if (!audioUnlocked) return;
  const want = musicFamily(phase);
  for (const [name, track] of Object.entries(music)) {
    if (name === want) {
      if (track.paused) track.play().catch(() => {});
    } else if (!track.paused) {
      track.pause();
    }
  }
}

function applyVolumes() {
  for (const track of Object.values(music)) track.volume = settings.bgm * MUSIC_GAIN;
  for (const [key, element] of Object.entries(sfx)) {
    if (key !== 'engine') element.volume = settings.se;
  }
}
applyVolumes();

// Browsers keep audio suspended until the player gestures. The first key or
// pointer press primes every one-shot (a muted play/pause round-trip) and
// starts the music for whatever screen is up.
let audioUnlocked = false;
function unlockAudio() {
  if (audioUnlocked) return;
  audioUnlocked = true;
  for (const [key, element] of Object.entries(sfx)) {
    if (key === 'engine') continue;
    element.muted = true;
    element.play()
      .then(() => { element.pause(); element.currentTime = 0; element.muted = false; })
      .catch(() => { element.muted = false; });
  }
  syncMusic(phaseNow);
}
window.addEventListener('keydown', unlockAudio);
window.addEventListener('pointerdown', unlockAudio);

// --- Boot -----------------------------------------------------------------------

function touchCapable() {
  return navigator.maxTouchPoints > 0 || 'ontouchstart' in window;
}

async function boot() {
  statusEl.textContent = 'Loading the aircraft and the island…';
  const [aircraft, island] = await Promise.all([
    loadGLB('models/fx27_raven_gear.glb'),
    loadGLB('models/terrain.glb'),
  ]);

  // Level the coastal shelf in the terrain mesh itself, so the separate
  // runway strip has flat ground under it. These numbers belong with
  // WORLD.runway / WORLD.runwayHeight — change them together.
  const terrainMesh = prepareTerrain(island, WORLD.terrainOffset, WORLD.terrainScale, {
    centerX: WORLD.runway[0], centerZ: WORLD.runway[1],
    halfWidth: 72, halfLength: 430, height: WORLD.runwayHeight - 1.2, blend: 150,
  });
  const grid = buildTerrainGrid(terrainMesh, WORLD.terrainOffset, WORLD.terrainScale, 160);
  terrainMesh.ao = buildTerrainAO(terrainMesh, grid, SUN_DIR);

  const world = createWorld(grid, createRandom(seed));
  const state = world.state;
  state.seed = seed;
  state.settings = settings;
  state.settingsSelected = 0;
  state.touchUI = touchCapable();

  const renderer = createRenderer(glCanvas, hudCanvas, world, { fixedSize: Boolean(sceneName) });
  renderer.setTerrain(terrainMesh);
  renderer.setAircraft(aircraft);
  await renderer.ready();
  bootEl.hidden = true;

  // A handle for debugging and for the screenshot harness.
  window.__skyring = { world, state, renderer, aircraft, island, sfx, music, settings };

  if (sceneName) {
    stageShot(world, renderer);
    return;
  }
  runGame(world, renderer);
}

// Screenshot mode: stage the scene, paint it twice, flag readiness, then
// keep repainting the identical frozen frame.
function stageShot(world, renderer) {
  stageScene(sceneName, world, seed);
  renderer.render(world.state);
  renderer.render(world.state);
  window.__shotReady = true;
  statusEl.textContent = `Staged scene: ${sceneName}.`;
  const repaint = () => {
    renderer.render(world.state);
    requestAnimationFrame(repaint);
  };
  requestAnimationFrame(repaint);
}

// --- The live game ---------------------------------------------------------------

function describeState(state) {
  const feet = Math.max(0, Math.round((state.flight.position[1] - WORLD.seaLevel) * 3.28));
  const knots = Math.round(state.flight.speed * 1.94);
  return `${state.phase}. Altitude ${feet} feet. Speed ${knots} knots. `
    + `Rings ${state.ringIndex} of ${RING_COUNT}. Missiles ${state.missileCount}.`;
}

function runGame(world, renderer) {
  const state = world.state;
  let accumulator = 0;
  let engineLive = false;

  // The title screen flies the parked TITLE_POSE; the live game and the
  // ?scene=title shot share it so the two can never drift apart.
  function parkOnTitle() {
    Object.assign(state.flight, TITLE_POSE, { position: [...TITLE_POSE.position], liftoff: 0 });
    state.phase = 'title';
    state.time = 0;
    accumulator = 0;
  }

  function beginRun() {
    world.reset();
    accumulator = 0;
    engineLive = true;
    sfx.engine.volume = 0.12;
    sfx.engine.play().catch(() => {});
    // Every sortie hears the flight track from its top, not from wherever
    // the last landing or crash left it looping.
    try { music.flight.currentTime = 0; } catch { /* metadata not in yet */ }
  }

  parkOnTitle();
  phaseNow = 'title';
  statusEl.textContent = 'Ready. Press Enter or Space to begin.';

  let last = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;

    // This frame's presses, drained once. down() also answers for a press
    // that landed between frames; consume() spends a press exactly once.
    const pressed = input.take();
    const frameInput = {
      down: (key) => input.down(key) || pressed.has(key),
      consume: (key) => pressed.delete(key),
    };

    // Screen routing: Escape backs out first, then at most one transition.
    if (pressed.has('Escape')) {
      if (state.phase === 'settings' || state.phase === 'credits') state.phase = 'title';
      else if (state.phase !== 'title') parkOnTitle();
    }
    if (state.phase === 'title' && (pressed.has('KeyC') || pressed.has('PointerCredits'))) {
      state.phase = 'credits';
    } else if (state.phase === 'title' && (pressed.has('KeyO') || pressed.has('PointerAudio'))) {
      state.phase = 'settings';
      state.settingsSelected = 0;
    } else if (state.phase === 'settings' && pressed.has('KeyO')) {
      state.phase = 'title';
    } else if (state.phase === 'credits' && pressed.has('KeyC')) {
      state.phase = 'title';
    } else if ((state.phase === 'title' || state.phase === 'credits')
      && (pressed.has('Enter') || pressed.has('Space') || pressed.has('PointerStart'))) {
      beginRun();
    } else if ((state.phase === 'result' || state.phase === 'crashed')
      && (pressed.has('Enter') || pressed.has('Space') || pressed.has('PointerStart'))) {
      beginRun();
    }

    // The settings panel: up/down chooses a row, holding left/right slides it.
    if (state.phase === 'settings') {
      if (pressed.has('ArrowUp')) state.settingsSelected = (state.settingsSelected + 2) % 3;
      else if (pressed.has('ArrowDown')) state.settingsSelected = (state.settingsSelected + 1) % 3;
      const direction = (input.down('ArrowRight') ? 1 : 0) - (input.down('ArrowLeft') ? 1 : 0);
      if (direction !== 0) {
        const key = SETTING_KEYS[state.settingsSelected];
        settings[key] = clamp01(settings[key] + direction * dt * 0.55);
        if (key !== 'blur') applyVolumes();
        saveSettings();
      }
    }

    if ((state.phase === 'play' || state.phase === 'return') && pressed.has('KeyR')) {
      world.recover();
    }

    // Fixed-step simulation. Menu phases never step; a step that ends the
    // sortie (landing, crash) stops the drain for this frame.
    if (state.phase === 'play' || state.phase === 'return' || state.phase === 'crashed') {
      accumulator += dt;
      while (accumulator >= STEP) {
        world.step(STEP, frameInput, playSound);
        accumulator -= STEP;
        if (state.phase !== 'play' && state.phase !== 'return' && state.phase !== 'crashed') break;
      }
    } else {
      accumulator = 0;
    }

    // The engine loop runs only while a sortie is live, pitched and gained
    // by the throttle; it is silenced on every other screen.
    const engine = sfx.engine;
    if (engineLive && (state.phase === 'play' || state.phase === 'return')) {
      if (engine.paused) engine.play().catch(() => {});
      const throttle = state.flight.throttle;
      engine.playbackRate = 0.72 + throttle * 0.78;
      engine.volume = (0.07 + throttle * 0.5) * settings.se;
    } else if (!engine.paused) {
      engine.pause();
    }

    phaseNow = state.phase;
    syncMusic(state.phase);
    renderer.render(state, dt);
    statusEl.textContent = describeState(state);
  }
  requestAnimationFrame(frame);
}

boot().catch((error) => {
  console.error(error);
  statusEl.textContent = `Unable to start: ${error.message}`;
  bootEl.hidden = true;
  const ctx = hudCanvas.getContext('2d');
  ctx.fillStyle = '#ff8178';
  ctx.font = '18px ui-monospace, monospace';
  ctx.fillText(error.message, 34, 54);
});
