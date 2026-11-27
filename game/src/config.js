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

// Sound files under game/audio. The music lives outside the AUDIO map because
// there are two tracks and main.js crossfades between them by phase: the title
// track under the menu screens, the flight track once the sortie starts.
export const BGM = {
  title: 'audio/bgm_title.ogg',
  flight: 'audio/bgm_ingame.ogg',
};
export const AUDIO = {
  engine: 'audio/engine-loop.ogg',
  missile: 'audio/missile.ogg',
  explosion: 'audio/explosion.ogg',
  lock: 'audio/lockon.ogg',
  ring: 'audio/ring.ogg',
  jingle: 'audio/jingle.ogg',
};

// --- Modes, resources and the clock -----------------------------------------

/** The two ways to fly: the phased sortie and the pure air-combat mode. */
export const MODE = { SORTIE: 'sortie', DOGFIGHT: 'dogfight' };

// Fuel, in tank percent. The burn is deliberately generous: a clean lap of
// the nine rings, the two-bandit intercept and the return leg uses roughly
// half a tank, so the pressure only shows when a player keeps chasing or
// takes a long detour. Refuelling happens on the runway (see world.js).
export const FUEL = {
  max: 100,
  idleBurn: 0.040,      // percent per second at idle throttle
  throttleBurn: 0.300,  // extra percent per second at full throttle (throttle^2)
  caution: 30,          // percent: first tower warning
  bingo: 15,            // percent: bingo call — come home now
  refuelRate: 26,       // percent per second with the hose on
  refuelSpeed: 5,       // below this ground speed the hose stays connected
};

// Per-sortie wind, knots. The direction is hashed from the run seed, so a
// seed always flies in the same weather.
export const WIND = { maxKnots: 15 };

// The in-game clock, in hours. It runs at one game minute per real second,
// so a sortie that leaves in the afternoon is landing near dusk and a long
// dogfight runs into the night. `title` is where the menu backdrop sits.
export const CLOCK = {
  start: 14.0,          // SORTIE wheels-up time: mid-afternoon
  dogfightStart: 17.35, // DOGFIGHT starts in the early evening
  rate: 1 / 60,         // hours of game time per real second
  title: 13.4,
};

// Air combat. Waves arrive in the listed sizes; the player starts airborne
// over the southern channel, already at fighting speed.
export const DOGFIGHT = {
  waves: [2, 3, 4],
  start: [420, 470, -760],
  heading: 0.12,
  speed: 172,
  waveDelay: 3.2,     // seconds between a wave clearing and the next arriving
  victoryDelay: 2.4,  // a beat of quiet sky before the victory card
  bulletSpeed: 620,
  maxRange: 720,
};

// What a touchdown may look like and still count as a landing. Anything
// outside these is a crash, and the grade (world.js) scores what is inside.
export const LANDING = {
  maxSpeed: 92,
  maxPitch: 0.17,
  maxBank: 0.2,
  maxSink: 12,
};

// Enemy behaviour, split by how hard the bandits are trying: the dogfight
// waves press home, the sortie interceptors are a gentler introduction to
// being shot at. Damage is rolled per bullet at the moment it is fired.
export const ENEMY = {
  dogfightSpeed: 178,
  sortieSpeed: 162,
  dogfightTurn: 1.05,   // radians per second of heading change
  sortieTurn: 0.72,
  dogfightRange: 720,   // metres; guns open inside this
  sortieRange: 620,
  dogfightDamage: [4.5, 9.5],
  sortieDamage: [3.0, 7.0],
};
