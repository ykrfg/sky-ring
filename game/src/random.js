// Seeded randomness (mulberry32). World generation and the staged screenshot
// scenes both draw from this generator, so one seed must always produce the
// same stream — never swap the algorithm without re-shooting the screenshots.

/**
 * Create a generator for the given seed. Each call to the returned function
 * yields the next float in [0, 1).
 */
export function createRandom(seed) {
  let state = seed | 0;
  function nextRandom() {
    state = (state + 0x6d2b79f5) | 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  // Convenience draws on the same stream, for placement code that wants a
  // span directly: range(lo, hi) is a float in [lo, hi), int(lo, hi) an
  // integer with both ends included.
  nextRandom.range = (lo, hi) => lo + nextRandom() * (hi - lo);
  nextRandom.int = (lo, hi) => lo + Math.floor(nextRandom() * (hi - lo + 1));
  return nextRandom;
}
