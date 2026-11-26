// Keyboard and pointer input for the flight game.
//
// Two kinds of signal come out of here:
//   down(key)    — is this key held right now (steering, throttle)
//   take()       — the set of one-shot presses since the last call, drained by
//                  the caller so a press between frames is never lost or
//                  counted twice. Pointer taps on the menu buttons join the
//                  same queue as pseudo-keys ('PointerStart', ...).
// consume(key) is take() for a single key.
//
// Touch flight controls ride the same two signals: render.js exports the
// button layout as TOUCH_ZONES, and a pointer held inside a zone presses the
// keyboard code that button stands in for — down() for as long as the finger
// stays down, plus one queued press so one-shot actions (gear, fire) fire.

import { VIEW } from './config.js';
import { TOUCH_ZONES } from './render.js';

// Keys the game reacts to. Anything else keeps its default browser behaviour;
// these are preventDefault()ed so arrows and space cannot scroll the page.
const GAME_KEYS = new Set([
  'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
  'KeyW', 'KeyS', 'KeyG', 'KeyJ', 'KeyF', 'KeyC', 'KeyO', 'KeyR',
  'Space', 'Enter', 'Escape',
]);

export function createInput(canvas) {
  const held = new Set();
  let queued = new Set();

  // Touch pointers must not turn into scroll or zoom gestures mid-flight.
  canvas.style.touchAction = 'none';

  window.addEventListener('keydown', (event) => {
    if (!GAME_KEYS.has(event.code)) return;
    event.preventDefault();
    held.add(event.code);
    if (!event.repeat) queued.add(event.code);
  });
  window.addEventListener('keyup', (event) => {
    held.delete(event.code);
  });
  // Pointers held by touch zones right now, pointerId -> keyboard code. A
  // code leaves `held` only when the last pointer standing in for it lifts.
  const zoneHolds = new Map();
  function releaseZone(pointerId) {
    const code = zoneHolds.get(pointerId);
    if (code === undefined) return;
    zoneHolds.delete(pointerId);
    for (const other of zoneHolds.values()) if (other === code) return;
    held.delete(code);
  }

  // A key released while the window is unfocused never fires keyup here, and
  // a touch interrupted the same way never fires pointerup either.
  window.addEventListener('blur', () => {
    held.clear();
    queued.clear();
    zoneHolds.clear();
  });

  // Menu buttons are drawn on the HUD canvas in logical coordinates; map the
  // tap back into that space and queue the matching pseudo-key. The boxes are
  // [left, top, right, bottom] and must stay in step with the HUD layout.
  const HIT_BOXES = [
    ['PointerStart', 398, 512, 632, 564],
    ['PointerCredits', 648, 512, 882, 564],
    ['PointerAudio', 548, 578, 732, 614],
  ];
  function toLogical(event) {
    const rect = canvas.getBoundingClientRect();
    return [
      ((event.clientX - rect.left) / rect.width) * VIEW.width,
      ((event.clientY - rect.top) / rect.height) * VIEW.height,
    ];
  }
  window.addEventListener('pointerdown', (event) => {
    const [x, y] = toLogical(event);
    for (const [key, left, top, right, bottom] of HIT_BOXES) {
      if (x >= left && x <= right && y >= top && y <= bottom) {
        queued.add(key);
        break;
      }
    }
    // A press inside a touch zone holds its code until this pointer lifts,
    // and queues one press so one-shot actions trigger on the tap itself.
    for (const zone of TOUCH_ZONES) {
      const dx = x - zone.x, dy = y - zone.y;
      if (dx * dx + dy * dy <= zone.r * zone.r) {
        zoneHolds.set(event.pointerId, zone.code);
        held.add(zone.code);
        queued.add(zone.code);
        break;
      }
    }
  });
  window.addEventListener('pointerup', (event) => releaseZone(event.pointerId));
  window.addEventListener('pointercancel', (event) => releaseZone(event.pointerId));

  return {
    down: (key) => held.has(key),
    consume: (key) => {
      if (!queued.has(key)) return false;
      queued.delete(key);
      return true;
    },
    take: () => {
      const presses = queued;
      queued = new Set();
      return presses;
    },
  };
}
