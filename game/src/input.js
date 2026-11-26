// Keyboard and pointer input for the flight game.
//
// Two kinds of signal come out of here:
//   down(key)    — is this key held right now (steering, throttle)
//   take()       — the set of one-shot presses since the last call, drained by
//                  the caller so a press between frames is never lost or
//                  counted twice. Pointer taps on the menu buttons join the
//                  same queue as pseudo-keys ('PointerStart', ...).
// consume(key) is take() for a single key.

import { VIEW } from './config.js';

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

  window.addEventListener('keydown', (event) => {
    if (!GAME_KEYS.has(event.code)) return;
    event.preventDefault();
    held.add(event.code);
    if (!event.repeat) queued.add(event.code);
  });
  window.addEventListener('keyup', (event) => {
    held.delete(event.code);
  });
  // A key released while the window is unfocused never fires keyup here.
  window.addEventListener('blur', () => {
    held.clear();
    queued.clear();
  });

  // Menu buttons are drawn on the HUD canvas in logical coordinates; map the
  // tap back into that space and queue the matching pseudo-key. The boxes are
  // [left, top, right, bottom] and must stay in step with the HUD layout.
  const HIT_BOXES = [
    ['PointerStart', 398, 512, 632, 564],
    ['PointerCredits', 648, 512, 882, 564],
    ['PointerAudio', 548, 578, 732, 614],
  ];
  window.addEventListener('pointerdown', (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * VIEW.width;
    const y = ((event.clientY - rect.top) / rect.height) * VIEW.height;
    for (const [key, left, top, right, bottom] of HIT_BOXES) {
      if (x >= left && x <= right && y >= top && y <= bottom) {
        queued.add(key);
        break;
      }
    }
  });

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
