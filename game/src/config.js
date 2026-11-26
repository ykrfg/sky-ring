// SKY RING tuning sheet. Everything that might need adjusting during integration
// lives here so the rest of the code stays free of magic numbers. The game renders
// into a fixed 1280x720 logical frame that is scaled to the window, so none of the
// values below depend on the player's display.

/** Logical frame size and the fixed simulation step (60 updates per second). */
export const VIEW = { width: 1280, height: 720 };
export const STEP = 1 / 60;

/** Backbuffer ceiling: the logical frame, at most 1.5x device pixel ratio, never past 1080p. */
export const RENDER = { maxWidth: 1920, maxHeight: 1080, maxDpr: 1.5 };

// Terrain hookup. The island GLB is placed with terrainScale/terrainOffset on load and
// the runway is a separate flat strip laid over the coast at (runway[0], runway[1]).
// These are the numbers to revisit if the source model is ever swapped.
export const WORLD = {
  terrainScale: 1,
  terrainOffset: [0, 0, 0],
  runway: [1150, -390],
  runwayHeading: 0,
  runwayLength: 620,
  runwayWidth: 110,
  runwayHeight: 64,
  seaLevel: 34,
  fog: [0.47, 0.68, 0.76],
};

// Player aircraft hookup. The Raven GLB faces +Z; yawOffset exists in case a future
// model does not, and scale is sized against the roughly 21 m airframe.
export const AIRCRAFT = {
  scale: 1,
  yawOffset: 0,
  start: [1150, 64.4, -560],
  startHeading: 0,
};

// The parked-in-the-sky pose the title screen flies. The live game and the
// ?scene=title screenshot share this object so the two can never drift apart.
export const TITLE_POSE = {
  position: [1150, 230, -1000],
  heading: 0.32,
  pitch: 0.035,
  bank: 0.12,
  speed: 150,
  throttle: 0.5,
  gear: 0,
  gearTarget: 0,
  grounded: false,
};

/** Flight model limits and control rates, per second unless noted. */
export const FLIGHT = {
  minSpeed: 46,
  maxSpeed: 245,
  throttleRate: 0.42,
  pitchRate: 0.52,
  bankRate: 1.05,
  turnRate: 1.55,
  cruiseSink: 0.13,
};

// The ring course, in the order the player must thread them: [x, y, z] centres.
export const RINGS = [
  [1150, 200, 100],
  [1000, 255, 500],
  [500, 320, 850],
  [-100, 400, 750],
  [-650, 600, 400],
  [-1050, 510, -100],
  [-800, 520, -650],
  [-250, 420, -900],
  [450, 350, -1050],
];
// Derived, not restated: the HUD and the world both count rings from here.
export const RING_COUNT = RINGS.length;

// Recovery puts a lost player back on a safe approach, at the cost of time.
export const RECOVERY = { distance: 300, timePenalty: 5 };

// Waypoints of the enemy patrol the player can intercept.
export const ENEMY_ROUTE = [
  [-760, 470, -820],
  [-640, 500, -1010],
  [-880, 440, -1060],
];

// Sun direction shared by the shader and the baked terrain shading pass.
export const SUN_DIR = [-0.42, 0.80, -0.44];

export const PALETTE = {
  skyTop: '#1e5f95',
  skyHorizon: '#b8dce6',
  water: '#0b3142',
  haze: '#9cc7d2',
  fog: [0.55, 0.71, 0.77],
};

// Sound files under game/audio. BGM is delivered separately and may be missing
// while developing, so it gets its own path constant instead of an AUDIO entry.
export const BGM_PATH = 'audio/bgm.ogg';
export const AUDIO = {
  engine: 'audio/engine-loop.ogg',
  missile: 'audio/missile.ogg',
  explosion: 'audio/explosion.ogg',
  lock: 'audio/lockon.ogg',
  ring: 'audio/ring.ogg',
  jingle: 'audio/jingle.ogg',
};
