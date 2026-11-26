// SKY RING renderer: everything the player sees, in two layers.
//
//   #view — a hand-written WebGL scene. One shader does the whole world:
//           GGX/Smith/Schlick PBR lit by a directional sun and a hemisphere
//           ambient, an image-free sky gradient with a sun disc for the
//           fresnel environment sheen, procedural animated water, distance
//           fog, and an ACES filmic curve before gamma. A radial speed-blur
//           post pass runs over an off-screen copy of the frame.
//   #hud  — a 2D canvas on top: flight readouts, the attitude indicator, the
//           tactical radar, the cockpit panel with its gauges, and the
//           title / credits / settings / result / crash screens.
//
// Determinism is a hard rule here: the only clock the renderer reads is
// state.time, and the only randomness is a fixed-seed generator used once
// while the scenery is built. A staged screenshot therefore comes out
// identical on every load.

import { AIRCRAFT, PALETTE, RENDER, RING_COUNT, SUN_DIR, VIEW, WORLD } from './config.js';
import {
  compose, cross, dot, identity, lookAt, multiply, normalize, perspective,
  quatMatrix, scaling, sub, translation,
} from './math.js';

// ---------------------------------------------------------------------------
// Touch flight controls (drawn only while state.touchUI is set)
// ---------------------------------------------------------------------------
//
// The zones live in this one table: drawTouchControls paints them, and the
// input layer mirrors the same circles for hit-testing, so the two can never
// drift apart. Coordinates are logical 1280x720 HUD space. The pad sits on
// the left, clear of the attitude indicator below it and the readouts above;
// the right cluster sits under the radar and above the cockpit panel slope.
// `code` is the keyboard control each button stands in for.

export const TOUCH_ZONES = [
  { name: 'bankLeft', code: 'ArrowLeft', label: '◀', x: 52, y: 316, r: 33 },
  { name: 'bankRight', code: 'ArrowRight', label: '▶', x: 156, y: 316, r: 33 },
  { name: 'pitchUp', code: 'ArrowUp', label: '▲', x: 104, y: 264, r: 33 },
  { name: 'pitchDown', code: 'ArrowDown', label: '▼', x: 104, y: 368, r: 33 },
  { name: 'throttleUp', code: 'KeyW', label: 'THR+', x: 1096, y: 396, r: 34 },
  { name: 'throttleDown', code: 'KeyS', label: 'THR−', x: 1096, y: 486, r: 34 },
  { name: 'fire', code: 'KeyJ', label: 'FIRE', x: 1196, y: 486, r: 44 },
  { name: 'gear', code: 'KeyG', label: 'GEAR', x: 1196, y: 386, r: 28 },
];

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

// One vertex shader for every mesh in the scene: positions, normals and UVs
// in, world-space position/normal plus the baked AO factor out.
const SCENE_VERT = `
attribute vec3 aPosition;
attribute vec3 aNormal;
attribute vec2 aUV;
attribute float aAO;
uniform mat4 uMVP;
uniform mat4 uModel;
varying vec3 vWorld;
varying vec3 vNormal;
varying vec2 vUV;
varying float vAO;
void main() {
  vec4 world = uModel * vec4(aPosition, 1.0);
  vWorld = world.xyz;
  vNormal = mat3(uModel) * aNormal;
  vUV = aUV;
  vAO = aAO;
  gl_Position = uMVP * vec4(aPosition, 1.0);
}`;

// The fragment shader branches on three material kinds — sky dome, water,
// terrain — over a shared PBR core for everything else. All colour work is
// done in linear space and finished with ACES + gamma at the very end.
const SCENE_FRAG = `
precision highp float;
varying vec3 vWorld;
varying vec3 vNormal;
varying vec2 vUV;
varying float vAO;
uniform vec3 uCamera;
uniform vec3 uTint;
uniform vec3 uFogColor;
uniform vec3 uSunDir;
uniform vec3 uSunOnSky;
uniform float uMetallic;
uniform float uRoughness;
uniform float uOpacity;
uniform float uEmissive;
uniform float uTime;
uniform float uFogNear;
uniform float uFogFar;
uniform sampler2D uBaseMap;
uniform sampler2D uOrmMap;
uniform sampler2D uNormalMap;
uniform int uHasBase;
uniform int uHasOrm;
uniform int uHasNormal;
uniform int uKindWater;
uniform int uKindSky;
uniform int uKindTerrain;
uniform int uBlended;

const float PI = 3.14159265;

vec3 toLinear(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }
vec3 filmic(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
vec3 finish(vec3 linear) { return pow(filmic(linear), vec3(1.0 / 2.2)); }

// The whole sky in one function: a height gradient, a warm band at the
// horizon, the sun disc with its halo. The PBR core reuses it as the
// environment map for the fresnel sheen, so reflections always agree with
// the sky the player actually sees.
vec3 skyLight(vec3 dir) {
  float h = clamp(dir.y, -1.0, 1.0);
  vec3 zenith = toLinear(vec3(0.045, 0.20, 0.44));
  vec3 horizon = toLinear(vec3(0.62, 0.81, 0.90));
  vec3 below = toLinear(vec3(0.55, 0.72, 0.78));
  vec3 c = mix(horizon, zenith, pow(max(h, 0.0), 0.38));
  c = mix(below, c, smoothstep(-0.12, 0.03, h));
  float sunAmount = max(dot(dir, uSunOnSky), 0.0);
  c += vec3(1.0, 0.82, 0.58) * (pow(sunAmount, 420.0) * 1.7 + pow(sunAmount, 18.0) * 0.11);
  return c;
}

// GGX normal distribution, Schlick-GGX Smith visibility, Schlick Fresnel —
// the standard direct-lighting PBR trio.
float ggx(float noH, float alpha) {
  float a2 = alpha * alpha;
  float d = noH * noH * (a2 - 1.0) + 1.0;
  return a2 / max(0.0001, PI * d * d);
}
float smith(float noV, float noL, float alpha) {
  float k = (alpha + 1.0) * (alpha + 1.0) / 8.0;
  return (noV / (noV * (1.0 - k) + k)) * (noL / (noL * (1.0 - k) + k));
}
vec3 schlick(float cosT, vec3 f0) { return f0 + (1.0 - f0) * pow(1.0 - cosT, 5.0); }

void main() {
  vec3 viewDir = normalize(uCamera - vWorld);

  if (uKindSky == 1) {
    gl_FragColor = vec4(finish(skyLight(normalize(vWorld - uCamera)) * 1.12), 1.0);
    return;
  }

  vec4 sampled = uHasBase == 1 ? texture2D(uBaseMap, vUV) : vec4(1.0);
  vec3 albedo = toLinear(sampled.rgb) * toLinear(uTint);
  float roughness = clamp(uRoughness, 0.045, 1.0);
  float metallic = clamp(uMetallic, 0.0, 1.0);
  if (uHasOrm == 1) {
    vec3 orm = texture2D(uOrmMap, vUV).rgb;
    roughness = clamp(roughness * orm.g, 0.045, 1.0);
    metallic = clamp(metallic * orm.b * 1.35, 0.0, 1.0);
  }
  vec3 normal = normalize(vNormal);
  if (uHasNormal == 1) {
    // No tangent frame is carried, so the map is applied as a gentle
    // perturbation of the geometric normal — enough to break up the light
    // on the airframe without inventing a direction it does not have.
    vec3 mapped = texture2D(uNormalMap, vUV).xyz * 2.0 - 1.0;
    normal = normalize(normal + mapped * 0.38);
  }
  if (!gl_FrontFacing) normal = -normal;
  float ao = clamp(vAO, 0.55, 1.0);

  if (uKindTerrain == 1) {
    // The island is coloured by rule, not by texture: grass on the flats,
    // rock where the slope steepens or the altitude grows, snow only on
    // high, gentle ground. Two slow sine fields keep the bands from looking
    // stamped on, and the source texture survives only as luminance detail
    // (its baked snow tiles would otherwise punch white holes in the mesh).
    float height = vWorld.y;
    float field1 = sin(vWorld.x * 0.021 + sin(vWorld.z * 0.017) * 1.7) * 0.5 + 0.5;
    float field2 = sin(vWorld.z * 0.033 + vWorld.x * 0.012) * 0.5 + 0.5;
    float slope = 1.0 - clamp(normal.y, 0.0, 1.0);
    vec3 grass = mix(toLinear(vec3(0.10, 0.28, 0.16)), toLinear(vec3(0.29, 0.43, 0.22)), field1 * 0.7 + field2 * 0.3);
    vec3 rock = mix(toLinear(vec3(0.24, 0.24, 0.22)), toLinear(vec3(0.43, 0.41, 0.36)), field2);
    vec3 snow = toLinear(vec3(0.86, 0.90, 0.93));
    float rockMix = smoothstep(0.20, 0.58, slope) + smoothstep(370.0, 540.0, height) * 0.24;
    float snowMix = smoothstep(480.0, 640.0, height) * smoothstep(0.62, 0.30, slope);
    albedo = mix(grass, rock, clamp(rockMix, 0.0, 1.0));
    albedo = mix(albedo, snow, clamp(snowMix, 0.0, 1.0));
    if (uHasBase == 1) {
      float detail = dot(texture2D(uBaseMap, vUV).rgb, vec3(0.30, 0.55, 0.15));
      albedo *= 0.88 + detail * 0.24;
    }
  }

  vec3 sunDir = normalize(uSunDir);
  vec3 halfDir = normalize(viewDir + sunDir);
  float noL = max(dot(normal, sunDir), 0.0);
  float noV = max(dot(normal, viewDir), 0.001);
  float noH = max(dot(normal, halfDir), 0.0);
  float voH = max(dot(viewDir, halfDir), 0.0);
  float alpha = roughness * roughness;
  vec3 f0 = mix(vec3(0.04), albedo, metallic);
  vec3 fresnel = schlick(voH, f0);
  vec3 specular = ggx(noH, alpha) * smith(noV, noL, alpha) * fresnel
    / max(0.001, 4.0 * noV * noL + 0.0001);

  // Hemisphere ambient: sky colour from above, sea-dark bounce from below.
  vec3 hemi = mix(toLinear(vec3(0.12, 0.18, 0.25)), toLinear(vec3(0.58, 0.72, 0.80)),
    clamp(normal.y * 0.5 + 0.5, 0.0, 1.0));
  vec3 sunColour = vec3(1.0, 0.88, 0.72) * 2.35;
  float hemiWeight = uKindTerrain == 1 ? 1.65 : 0.75;

  vec3 colour = albedo * (1.0 - metallic) * (hemi * hemiWeight + sunColour * noL / PI) * ao;
  colour += specular * sunColour * mix(0.55, 1.0, ao);
  colour += fresnel * hemi * 0.22 * ao;
  colour += albedo * metallic * hemi * 0.16 * ao;
  colour += albedo * uEmissive * 1.8;
  colour += albedo * 0.08;
  if (uKindTerrain == 1) colour += albedo * 0.15;

  // Fresnel environment sheen: reflect the view ray into the same sky the
  // dome draws and weight it by Fresnel and surface smoothness, so polished
  // metal picks up the horizon at grazing angles while rough paint barely
  // notices it.
  vec3 env = skyLight(reflect(-viewDir, normal)) * mix(1.0, 0.35, roughness);
  colour += env * fresnel * (0.30 + 0.55 * (1.0 - roughness));

  // A cool fill from low over the sea keeps shadow sides readable.
  vec3 fillDir = normalize(vec3(0.55, 0.35, 0.75));
  colour += albedo * max(dot(normal, fillDir), 0.0) * vec3(0.26, 0.36, 0.46) * 1.25 * ao;
  colour += vec3(0.35, 0.58, 0.75) * pow(1.0 - noV, 3.0) * 0.30;

  if (uKindWater == 1) {
    // Animated water: three travelling sine waves perturb the normal (the
    // perturbation fades with distance so the far sea stays calm), then the
    // surface is mostly a Fresnel blend of deep colour and reflected sky,
    // with a tight sun glint on top. uTime is state.time — the same staged
    // frame always shows the same waves.
    vec2 q = vWorld.xz;
    float camDist = length(uCamera - vWorld);
    float detail = 1.0 - smoothstep(40.0, 260.0, camDist);
    float w1 = sin(q.x * 0.041 + q.y * 0.017 + uTime * 0.7);
    float w2 = sin(q.x * 0.093 - q.y * 0.071 - uTime * 1.1);
    float w3 = sin(q.x * 0.21 + q.y * 0.17 + uTime * 1.9);
    vec3 wNormal = normalize(normal + vec3(
      w1 * 0.0008 + w2 * 0.0005, 0.0, w2 * 0.0007 + w3 * 0.0003) * detail);
    vec3 reflected = reflect(-viewDir, wNormal);
    float waterFresnel = 0.025 + 0.975 * pow(1.0 - max(dot(wNormal, viewDir), 0.0), 5.0);
    vec3 deep = toLinear(vec3(0.018, 0.10, 0.145));
    colour = mix(deep, skyLight(reflected), clamp(0.10 + waterFresnel * 0.52, 0.0, 0.86));
    // The glint sparkles with the fastest wave up close and settles to a
    // steady sheen far away, where the normal detail has faded out anyway.
    float sparkle = 1.6 + 0.9 * (w3 * 0.5 + 0.5) * detail;
    colour += vec3(1.0, 0.83, 0.60) * pow(max(dot(reflected, sunDir), 0.0), 260.0) * sparkle;
    colour += vec3(0.004, 0.012, 0.016) * ((w1 * 0.55 + w2 * 0.45) * 0.5 + 0.5);
  }

  float dist = length(uCamera - vWorld);
  float fogged = smoothstep(uFogNear, uFogFar, dist);
  colour = mix(colour, toLinear(uFogColor), fogged * 0.92);
  float outAlpha = uBlended == 1 ? sampled.a * uOpacity : 1.0;
  gl_FragColor = vec4(finish(colour * 1.05), outAlpha);
}`;

// Radial speed blur: the frame is re-sampled a few times along the ray from
// the screen centre, weighted toward the sharp centre sample. Strength 0 is
// an exact pass-through.
const POST_VERT = `
attribute vec2 aPos;
varying vec2 vUV;
void main() {
  vUV = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const POST_FRAG = `
precision mediump float;
varying vec2 vUV;
uniform sampler2D uScene;
uniform float uStrength;
uniform vec2 uCentre;
void main() {
  if (uStrength < 0.0001) {
    gl_FragColor = texture2D(uScene, vUV);
    return;
  }
  vec2 dir = vUV - uCentre;
  float mask = smoothstep(0.035, 0.55, length(dir));
  vec4 sum = vec4(0.0);
  float total = 0.0;
  for (int i = 0; i < 4; i++) {
    float t = float(i) / 3.0;
    float w = 1.0 - t * 0.55;
    sum += texture2D(uScene, vUV - dir * uStrength * mask * t) * w;
    total += w;
  }
  gl_FragColor = sum / total;
}`;

// ---------------------------------------------------------------------------
// Small WebGL helpers
// ---------------------------------------------------------------------------

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

function linkProgram(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  return program;
}

// ---------------------------------------------------------------------------
// Geometry kit
// ---------------------------------------------------------------------------
//
// Scenery is built once into plain-array accumulators and merged, so the
// whole static world costs a handful of draw calls. finalize() wraps an
// accumulator in the same accessor shape the glTF loader produces
// ({ position: { values, count, components }, ... }), which is what the
// buffer cache below consumes for every mesh, loaded or built.

function newBuilder() {
  return { positions: [], normals: [], uvs: [], indices: [] };
}

function pushVertex(b, p, n, uv) {
  b.positions.push(p[0], p[1], p[2]);
  b.normals.push(n[0], n[1], n[2]);
  b.uvs.push(uv[0], uv[1]);
}

// One quad as two triangles; corners run a-b-c-d around the face.
function pushQuad(b, a, c, d, e, normal, uvs = null) {
  const base = b.positions.length / 3;
  const corners = [a, c, d, e];
  const cornerUVs = uvs || [[0, 0], [1, 0], [1, 1], [0, 1]];
  for (let i = 0; i < 4; i++) pushVertex(b, corners[i], normal, cornerUVs[i]);
  b.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
}

// One triangle with a computed face normal.
function pushTri(b, a, c, d) {
  const ux = c[0] - a[0], uy = c[1] - a[1], uz = c[2] - a[2];
  const vx = d[0] - a[0], vy = d[1] - a[1], vz = d[2] - a[2];
  let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
  const len = Math.hypot(nx, ny, nz) || 1;
  nx /= len; ny /= len; nz /= len;
  const base = b.positions.length / 3;
  for (const p of [a, c, d]) pushVertex(b, p, [nx, ny, nz], [0, 0]);
  b.indices.push(base, base + 1, base + 2);
}

function pushBox(b, cx, cy, cz, sx, sy, sz) {
  const x0 = cx - sx / 2, x1 = cx + sx / 2;
  const y0 = cy - sy / 2, y1 = cy + sy / 2;
  const z0 = cz - sz / 2, z1 = cz + sz / 2;
  pushQuad(b, [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1], [0, 1, 0]);
  pushQuad(b, [x0, y0, z1], [x1, y0, z1], [x1, y0, z0], [x0, y0, z0], [0, -1, 0]);
  pushQuad(b, [x0, y0, z1], [x0, y1, z1], [x1, y1, z1], [x1, y0, z1], [0, 0, 1]);
  pushQuad(b, [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z0], [0, 0, -1]);
  pushQuad(b, [x1, y0, z1], [x1, y1, z1], [x1, y1, z0], [x1, y0, z0], [1, 0, 0]);
  pushQuad(b, [x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1], [-1, 0, 0]);
}

function pushCone(b, cx, cy, cz, radius, height, sides = 6) {
  const apex = [cx, cy + height, cz];
  for (let i = 0; i < sides; i++) {
    const a0 = (i / sides) * Math.PI * 2, a1 = ((i + 1) / sides) * Math.PI * 2;
    pushTri(b,
      [cx + Math.cos(a0) * radius, cy, cz + Math.sin(a0) * radius],
      [cx + Math.cos(a1) * radius, cy, cz + Math.sin(a1) * radius],
      apex);
  }
}

function pushCylinder(b, cx, cy, cz, radius, height, sides = 6) {
  for (let i = 0; i < sides; i++) {
    const a0 = (i / sides) * Math.PI * 2, a1 = ((i + 1) / sides) * Math.PI * 2;
    const p0 = [cx + Math.cos(a0) * radius, cy, cz + Math.sin(a0) * radius];
    const p1 = [cx + Math.cos(a1) * radius, cy, cz + Math.sin(a1) * radius];
    const mid = (a0 + a1) / 2;
    pushQuad(b, p0, p1, [p1[0], p1[1] + height, p1[2]], [p0[0], p0[1] + height, p0[2]],
      [Math.cos(mid), 0, Math.sin(mid)]);
  }
}

function pushSphere(b, cx, cy, cz, rx, ry, rz, lat = 5, lon = 7) {
  const start = b.positions.length / 3;
  for (let y = 0; y <= lat; y++) {
    const v = (y / lat) * Math.PI;
    for (let x = 0; x <= lon; x++) {
      const u = (x / lon) * Math.PI * 2;
      const nx = Math.sin(v) * Math.cos(u), ny = Math.cos(v), nz = Math.sin(v) * Math.sin(u);
      pushVertex(b, [cx + nx * rx, cy + ny * ry, cz + nz * rz], [nx, ny, nz], [x / lon, y / lat]);
    }
  }
  for (let y = 0; y < lat; y++) {
    for (let x = 0; x < lon; x++) {
      const a = start + y * (lon + 1) + x, below = a + lon + 1;
      b.indices.push(a, a + 1, below + 1, a, below + 1, below);
    }
  }
}

// A ring gate: a torus standing in the vertical plane that faces `yaw`,
// built straight into world space around its centre.
function pushTorus(b, centre, radius, tube, yaw, segments = 56, sides = 6) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const place = (p) => [p[0] * cy + p[2] * sy, p[1], -p[0] * sy + p[2] * cy];
  const start = b.positions.length / 3;
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2, ca = Math.cos(a), sa = Math.sin(a);
    for (let j = 0; j <= sides; j++) {
      const t = (j / sides) * Math.PI * 2, cb = Math.cos(t), sb = Math.sin(t);
      const ring = radius + tube * cb;
      const p = place([ring * ca, ring * sa, tube * sb]);
      const n = place([cb * ca, cb * sa, sb]);
      pushVertex(b, [centre[0] + p[0], centre[1] + p[1], centre[2] + p[2]], n, [i / segments, j / sides]);
    }
  }
  for (let i = 0; i < segments; i++) {
    for (let j = 0; j < sides; j++) {
      const a = start + i * (sides + 1) + j, c = a + sides + 1;
      b.indices.push(a, a + 1, c + 1, a, c + 1, c);
    }
  }
}

function finalize(b) {
  const count = b.positions.length / 3;
  return {
    position: { values: new Float32Array(b.positions), count, components: 3 },
    normal: { values: new Float32Array(b.normals), count, components: 3 },
    uv: { values: new Float32Array(b.uvs), count, components: 2 },
    indices: { values: new Uint32Array(b.indices), count: b.indices.length, components: 1 },
  };
}

// A flat horizontal quad centred on the origin — the sea, the blob shadow.
function flatPlane(width, depth, y = 0) {
  const b = newBuilder();
  const w = width / 2, d = depth / 2;
  pushQuad(b, [-w, y, -d], [-w, y, d], [w, y, d], [w, y, -d], [0, 1, 0]);
  return finalize(b);
}

// A unit quad in the XY plane facing +Z — smoke puffs and explosion frames.
function unitQuad(uvs = [0, 0, 1, 0, 1, 1, 0, 1]) {
  const b = newBuilder();
  pushQuad(b, [-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0], [0, 0, 1],
    [[uvs[0], uvs[1]], [uvs[2], uvs[3]], [uvs[4], uvs[5]], [uvs[6], uvs[7]]]);
  return finalize(b);
}

// One cell of the explosion flipbook: an 8x4 sheet, frames read left to
// right, top row first (the sheet is uploaded with FLIP_Y, hence the
// inverted v range).
function explosionFrame(index) {
  const col = index % 8, row = Math.floor(index / 8);
  const x0 = col / 8, x1 = (col + 1) / 8;
  const y0 = 1 - (row + 1) / 4, y1 = 1 - row / 4;
  return unitQuad([x0, y0, x1, y0, x1, y1, x0, y1]);
}

// ---------------------------------------------------------------------------
// The renderer
// ---------------------------------------------------------------------------

export function createRenderer(glCanvas, hudCanvas, world, options = {}) {
  const gl = glCanvas.getContext('webgl', {
    antialias: true, alpha: false, depth: true, powerPreference: 'high-performance',
  });
  if (!gl) throw new Error('WebGL is unavailable. Enable hardware acceleration in your browser.');
  const hud = hudCanvas.getContext('2d');

  const program = linkProgram(gl, SCENE_VERT, SCENE_FRAG);
  gl.useProgram(program);
  const UNIFORM_NAMES = [
    'uMVP', 'uModel', 'uCamera', 'uTint', 'uFogColor', 'uSunDir', 'uSunOnSky',
    'uMetallic', 'uRoughness', 'uOpacity', 'uEmissive', 'uTime', 'uFogNear', 'uFogFar',
    'uBaseMap', 'uOrmMap', 'uNormalMap', 'uHasBase', 'uHasOrm', 'uHasNormal',
    'uKindWater', 'uKindSky', 'uKindTerrain', 'uBlended',
  ];
  const uni = {};
  for (const name of UNIFORM_NAMES) uni[name] = gl.getUniformLocation(program, name);
  const att = {
    position: gl.getAttribLocation(program, 'aPosition'),
    normal: gl.getAttribLocation(program, 'aNormal'),
    uv: gl.getAttribLocation(program, 'aUV'),
    ao: gl.getAttribLocation(program, 'aAO'),
  };
  gl.getExtension('OES_element_index_uint');

  // The sun the shader lights with, and the (different) direction the sky
  // gradient paints the disc at — the disc sits low over the sea in the
  // title framing while the light itself comes from higher.
  const sunDir = SUN_DIR;
  const sunOnSky = [-0.55, 0.09, 0.82];

  // --- post pass state -----------------------------------------------------

  const postProgram = linkProgram(gl, POST_VERT, POST_FRAG);
  const postUni = {
    scene: gl.getUniformLocation(postProgram, 'uScene'),
    strength: gl.getUniformLocation(postProgram, 'uStrength'),
    centre: gl.getUniformLocation(postProgram, 'uCentre'),
  };
  const postAtt = gl.getAttribLocation(postProgram, 'aPos');
  const postQuadBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, postQuadBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  let sceneTarget = null;

  function rebuildSceneTarget(width, height) {
    if (sceneTarget) {
      gl.deleteFramebuffer(sceneTarget.fbo);
      gl.deleteTexture(sceneTarget.colour);
      gl.deleteRenderbuffer(sceneTarget.depth);
      sceneTarget = null;
    }
    const colour = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, colour);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const depth = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, width, height);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, colour, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
    const complete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (complete) sceneTarget = { fbo, colour, depth };
  }

  function presentScene(strength) {
    if (!sceneTarget) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, glCanvas.width, glCanvas.height);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
    gl.depthMask(true);
    gl.useProgram(postProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneTarget.colour);
    gl.uniform1i(postUni.scene, 0);
    gl.uniform1f(postUni.strength, strength);
    gl.uniform2f(postUni.centre, 0.5, 0.52);
    gl.bindBuffer(gl.ARRAY_BUFFER, postQuadBuffer);
    gl.enableVertexAttribArray(postAtt);
    gl.vertexAttribPointer(postAtt, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.useProgram(program);
  }

  // --- texture and buffer caches -------------------------------------------
  // Keyed by the source object itself, so a geometry or image is uploaded
  // exactly once no matter how many frames draw it.

  const textureCache = new WeakMap();
  function textureFor(source) {
    if (!source) return null;
    let tex = textureCache.get(source);
    if (tex) return tex;
    tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    const powerOfTwo = (source.width & (source.width - 1)) === 0
      && (source.height & (source.height - 1)) === 0;
    if (powerOfTwo) {
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    } else {
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    textureCache.set(source, tex);
    return tex;
  }

  const bufferCache = new WeakMap();
  function buffersFor(geo) {
    let entry = bufferCache.get(geo);
    if (entry) return entry;
    const upload = (data, target = gl.ARRAY_BUFFER) => {
      const id = gl.createBuffer();
      gl.bindBuffer(target, id);
      gl.bufferData(target, data, gl.STATIC_DRAW);
      return id;
    };
    const wide = geo.position.count > 65535;
    entry = {
      position: upload(geo.position.values),
      normal: upload(geo.normal ? geo.normal.values : new Float32Array(geo.position.count * 3)),
      uv: upload(geo.uv ? geo.uv.values : new Float32Array(geo.position.count * 2)),
      ao: geo.ao ? upload(geo.ao.values) : null,
      index: upload(
        wide ? new Uint32Array(geo.indices.values) : new Uint16Array(geo.indices.values),
        gl.ELEMENT_ARRAY_BUFFER),
      indexType: wide ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,
      count: geo.indices.count,
      mode: geo.mode !== undefined && geo.mode !== null ? geo.mode : gl.TRIANGLES,
    };
    bufferCache.set(geo, entry);
    return entry;
  }

  // --- the one draw routine --------------------------------------------------
  // material: { color, metallic, roughness, base, orm, normal, opacity,
  //             emissive } — every field optional.
  // flags:    { water, sky, terrain, blended, doubleSided }

  function drawMesh(geo, material, model, vp, camera, time, flags = {}) {
    if (!geo) return;
    const m = material || {};
    gl.uniformMatrix4fv(uni.uModel, false, model);
    gl.uniformMatrix4fv(uni.uMVP, false, multiply(vp, model));
    gl.uniform3fv(uni.uCamera, camera);
    const tint = m.color || [1, 1, 1, 1];
    gl.uniform3f(uni.uTint, tint[0], tint[1], tint[2]);
    gl.uniform1f(uni.uMetallic, m.metallic !== undefined && m.metallic !== null ? m.metallic : 0);
    gl.uniform1f(uni.uRoughness, m.roughness !== undefined && m.roughness !== null ? m.roughness : 0.7);
    gl.uniform1f(uni.uOpacity, m.opacity !== undefined && m.opacity !== null ? m.opacity : 1);
    gl.uniform1f(uni.uEmissive, m.emissive || 0);
    gl.uniform1f(uni.uTime, time);
    gl.uniform3fv(uni.uFogColor, PALETTE.fog);
    gl.uniform1f(uni.uFogNear, 350);
    gl.uniform1f(uni.uFogFar, 3900);
    gl.uniform3fv(uni.uSunDir, sunDir);
    gl.uniform3fv(uni.uSunOnSky, sunOnSky);
    gl.uniform1i(uni.uHasBase, m.base ? 1 : 0);
    gl.uniform1i(uni.uHasOrm, m.orm ? 1 : 0);
    gl.uniform1i(uni.uHasNormal, m.normal ? 1 : 0);
    gl.uniform1i(uni.uKindWater, flags.water ? 1 : 0);
    gl.uniform1i(uni.uKindSky, flags.sky ? 1 : 0);
    gl.uniform1i(uni.uKindTerrain, flags.terrain ? 1 : 0);
    gl.uniform1i(uni.uBlended, flags.blended ? 1 : 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textureFor(m.base));
    gl.uniform1i(uni.uBaseMap, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, textureFor(m.orm));
    gl.uniform1i(uni.uOrmMap, 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, textureFor(m.normal));
    gl.uniform1i(uni.uNormalMap, 2);
    if (flags.doubleSided) gl.disable(gl.CULL_FACE);
    else gl.enable(gl.CULL_FACE);

    const b = buffersFor(geo);
    gl.bindBuffer(gl.ARRAY_BUFFER, b.position);
    gl.enableVertexAttribArray(att.position);
    gl.vertexAttribPointer(att.position, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, b.normal);
    gl.enableVertexAttribArray(att.normal);
    gl.vertexAttribPointer(att.normal, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, b.uv);
    gl.enableVertexAttribArray(att.uv);
    gl.vertexAttribPointer(att.uv, 2, gl.FLOAT, false, 0, 0);
    if (b.ao) {
      gl.bindBuffer(gl.ARRAY_BUFFER, b.ao);
      gl.enableVertexAttribArray(att.ao);
      gl.vertexAttribPointer(att.ao, 1, gl.FLOAT, false, 0, 0);
    } else {
      gl.disableVertexAttribArray(att.ao);
      gl.vertexAttrib1f(att.ao, 1);
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b.index);
    gl.drawElements(b.mode, b.count, b.indexType, 0);
  }

  // --- static scenery --------------------------------------------------------
  // Built once, in world space, around the runway the config names. The
  // runway meshes are built around the origin and placed with a model
  // matrix; everything else is baked at its final coordinates.

  const runwayX = WORLD.runway[0];
  const runwayZ = WORLD.runway[1];
  const runwayModel = translation(runwayX, 0, runwayZ);

  const runwaySlab = (() => {
    const b = newBuilder();
    const w = WORLD.runwayWidth / 2, l = WORLD.runwayLength / 2, y = WORLD.runwayHeight;
    pushQuad(b, [-w, y, -l], [-w, y, l], [w, y, l], [w, y, -l], [0, 1, 0]);
    pushQuad(b, [-w, y - 6, l], [-w, y, l], [w, y, l], [w, y - 6, l], [0, 0, 1]);
    pushQuad(b, [w, y - 6, -l], [w, y, -l], [-w, y, -l], [-w, y - 6, -l], [0, 0, -1]);
    pushQuad(b, [w, y - 6, l], [w, y, l], [w, y, -l], [w, y - 6, -l], [1, 0, 0]);
    pushQuad(b, [-w, y - 6, -l], [-w, y, -l], [-w, y, l], [-w, y - 6, l], [-1, 0, 0]);
    pushQuad(b, [-w, y, l], [-w, y, -l], [w, y, -l], [w, y, l], [0, -1, 0]);
    return finalize(b);
  })();

  const runwayPaint = (() => {
    const b = newBuilder();
    const y = WORLD.runwayHeight + 0.06;
    const stripe = (cx, cz, w, l) => pushQuad(b,
      [cx - w / 2, y, cz - l / 2], [cx - w / 2, y, cz + l / 2],
      [cx + w / 2, y, cz + l / 2], [cx + w / 2, y, cz - l / 2], [0, 1, 0]);
    for (let z = -275; z < 275; z += 34) stripe(0, z + 8, 1.5, 15);
    for (const end of [-1, 1]) {
      for (let i = 0; i < 8; i++) stripe(-30 + i * 8.6, end * 292, 4.4, 34);
      stripe(0, end * 240, 10, 42);
    }
    for (const side of [-1, 1]) stripe(side * 52, 0, 1.2, 585);
    return finalize(b);
  })();

  const runwayLights = (() => {
    const b = newBuilder();
    const y = WORLD.runwayHeight + 0.5;
    for (let z = -300; z <= 300; z += 50) {
      for (const side of [-1, 1]) pushBox(b, side * 54, y, z, 1.0, 1.6, 1.0);
    }
    return finalize(b);
  })();

  // Trees: scattered by a private fixed-seed generator over the island's
  // mid slopes — never on the runway shelf, never in the sea or the snow.
  const treeSpots = (() => {
    let s = 987654321;
    const next = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
    const spots = [];
    for (let i = 0; i < 300 && spots.length < 150; i++) {
      const x = -1350 + next() * 2700;
      const z = -1350 + next() * 2700;
      const h = world.terrain.heightAt(x, z);
      if (h < 55 || h > 470) continue;
      if (Math.abs(x - runwayX) < 95 && Math.abs(z - runwayZ) < 420) continue;
      spots.push({ x, z, h, height: 12 + next() * 15, radius: 4.0 + next() * 4.2 });
    }
    return spots;
  })();
  const treeCanopy = (() => {
    const b = newBuilder();
    for (const t of treeSpots) pushCone(b, t.x, t.h + 2.5, t.z, t.radius, t.height, 6);
    return finalize(b);
  })();
  const treeTrunks = (() => {
    const b = newBuilder();
    for (const t of treeSpots) pushCylinder(b, t.x, t.h, t.z, 0.55, 4.2, 5);
    return finalize(b);
  })();

  // The radar's island picture: the height grid painted once into a small
  // canvas, brightened with altitude.
  const radarPicture = (() => {
    const c = document.createElement('canvas');
    c.width = c.height = 128;
    const g = c.getContext('2d');
    const t = world.terrain;
    for (let iy = 0; iy < 128; iy++) {
      for (let ix = 0; ix < 128; ix++) {
        const x = t.minX + ((ix + 0.5) / 128) * (t.maxX - t.minX);
        const z = t.maxZ - ((iy + 0.5) / 128) * (t.maxZ - t.minZ);
        const h = t.heightAt(x, z);
        if (h > WORLD.seaLevel + 2) {
          const high = Math.min(1, Math.max(0, (h - 60) / 500));
          g.fillStyle = `rgba(${Math.round(52 + high * 72)},${Math.round(112 + high * 58)},${Math.round(92 + high * 48)},0.96)`;
          g.fillRect(ix, iy, 1, 1);
        }
      }
    }
    return c;
  })();

  // A small octahedron that hovers over the current target.
  const navDiamond = (() => {
    const b = newBuilder();
    const top = [0, 4, 0], bottom = [0, -4, 0];
    const right = [3, 0, 0], left = [-3, 0, 0], front = [0, 0, 3], back = [0, 0, -3];
    pushTri(b, top, front, right);
    pushTri(b, top, left, front);
    pushTri(b, top, back, left);
    pushTri(b, top, right, back);
    pushTri(b, bottom, right, front);
    pushTri(b, bottom, front, left);
    pushTri(b, bottom, left, back);
    pushTri(b, bottom, back, right);
    return finalize(b);
  })();

  const cloudBank = (() => {
    const b = newBuilder();
    const spots = [
      [-900, 620, 900], [350, 680, 1150], [1180, 720, 700], [-1250, 660, -350],
      [900, 690, -1150], [-300, 640, -1350], [1250, 600, -500],
    ];
    spots.forEach(([x, y, z], i) => {
      pushSphere(b, x, y, z, 130 + i * 12, 34, 80);
      pushSphere(b, x + 110, y - 8, z + 30, 80, 26, 62);
      pushSphere(b, x - 100, y - 4, z - 24, 72, 24, 55);
    });
    return finalize(b);
  })();

  // Each ring gate faces along the course: toward the next ring, or away
  // from the previous one for the last gate.
  const ringGeometries = world.rings.map((ring, i) => {
    const rings = world.rings;
    const prev = rings[Math.max(0, i - 1)];
    const next = rings[Math.min(i + 1, rings.length - 1)];
    const d = i === rings.length - 1 ? sub(ring.position, prev.position) : sub(next.position, ring.position);
    const b = newBuilder();
    pushTorus(b, ring.position, ring.radius, 2.6, Math.atan2(d[0], d[2]));
    return finalize(b);
  });

  const missileBody = (() => {
    const b = newBuilder();
    pushBox(b, 0, 0, 0, 0.85, 0.85, 3.4);
    const nose = [0, 0, 2.9];
    for (let i = 0; i < 4; i++) {
      const a0 = (i / 4) * Math.PI * 2, a1 = ((i + 1) / 4) * Math.PI * 2;
      pushTri(b,
        [Math.cos(a0) * 0.55, Math.sin(a0) * 0.55, 1.7],
        [Math.cos(a1) * 0.55, Math.sin(a1) * 0.55, 1.7],
        nose);
    }
    for (let i = 0; i < 4; i++) {
      const a = (i / 4) * Math.PI * 2 + Math.PI / 4;
      const dx = Math.cos(a), dy = Math.sin(a);
      pushQuad(b,
        [dx * 0.4, dy * 0.4, -1.2], [dx * 1.6, dy * 1.6, -2.1],
        [dx * 1.6, dy * 1.6, -1.5], [dx * 0.4, dy * 0.4, -0.7], [0, 0, 1]);
    }
    return finalize(b);
  })();

  // The sky dome: a unit sphere drawn around the camera, normals flipped so
  // the inside faces shade.
  const skyDome = (() => {
    const b = newBuilder();
    const lat = 14, lon = 24;
    const start = 0;
    for (let y = 0; y <= lat; y++) {
      const v = (y / lat) * Math.PI;
      for (let x = 0; x <= lon; x++) {
        const u = (x / lon) * Math.PI * 2;
        const nx = Math.sin(v) * Math.cos(u), ny = Math.cos(v), nz = Math.sin(v) * Math.sin(u);
        b.positions.push(nx, ny, nz);
        b.normals.push(-nx, -ny, -nz);
        b.uvs.push(x / lon, y / lat);
      }
    }
    for (let y = 0; y < lat; y++) {
      for (let x = 0; x < lon; x++) {
        const a = start + y * (lon + 1) + x, below = a + lon + 1;
        b.indices.push(a, a + 1, below + 1, a, below + 1, below);
      }
    }
    return finalize(b);
  })();

  const puffQuad = unitQuad();
  const explosionFrames = Array.from({ length: 32 }, (_, i) => explosionFrame(i));
  const shadowQuadGeo = flatPlane(1, 1, 0);
  const shadowTexture = (() => {
    const c = document.createElement('canvas');
    c.width = c.height = 128;
    const g = c.getContext('2d');
    const grad = g.createRadialGradient(64, 64, 4, 64, 64, 62);
    grad.addColorStop(0, 'rgba(0,0,0,0.72)');
    grad.addColorStop(0.55, 'rgba(0,0,0,0.34)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 128, 128);
    return c;
  })();
  const seaPlane = flatPlane(14000, 14000, WORLD.seaLevel);

  // --- models handed in after construction -----------------------------------

  let aircraftModel = null;
  let terrainGeometry = null;
  const fxTextures = {};

  function setAircraft(model) {
    aircraftModel = model;
    // The source materials are highly metallic; a slight paint-like trim
    // and a lifted base colour keep the airframe readable against the
    // bright sea while the ORM and normal maps still do their work.
    for (const prims of model.meshes) {
      for (const prim of prims) {
        if (prim.material && prim.material.name && prim.material.name.includes('tripo')) {
          prim.material = { ...prim.material, color: [1.55, 1.6, 1.7, 1], metallic: 0.66, roughness: 0.44 };
        }
      }
    }
  }

  function setTerrain(geo) {
    terrainGeometry = geo;
  }

  // --- sizing ------------------------------------------------------------------

  function resize() {
    let width, height;
    if (options.fixedSize) {
      width = VIEW.width;
      height = VIEW.height;
    } else {
      const ratio = Math.min(window.devicePixelRatio || 1, RENDER.maxDpr);
      width = Math.min(RENDER.maxWidth, Math.round(window.innerWidth * ratio));
      height = Math.min(RENDER.maxHeight, Math.round(window.innerHeight * ratio));
    }
    glCanvas.width = width;
    hudCanvas.width = width;
    glCanvas.height = height;
    hudCanvas.height = height;
    if (!options.fixedSize) {
      glCanvas.style.width = `${window.innerWidth}px`;
      hudCanvas.style.width = `${window.innerWidth}px`;
      glCanvas.style.height = `${window.innerHeight}px`;
      hudCanvas.style.height = `${window.innerHeight}px`;
    }
    gl.viewport(0, 0, width, height);
    rebuildSceneTarget(width, height);
  }
  window.addEventListener('resize', resize);
  resize();

  // --- aircraft, enemies, wrecks ---------------------------------------------
  // One routine draws every airframe: the player's Raven with its own
  // materials and animated gear, the patrol with a red override material
  // (gear bays skipped — a tinted model reads better without them), and
  // wrecks tumbling in dark metal.

  const rotationAbout = (axis, angle) => {
    const s = Math.sin(angle / 2), c = Math.cos(angle / 2);
    return quatMatrix([axis[0] * s, axis[1] * s, axis[2] * s, c]);
  };

  // Sample the GearDeploy clip's rotation track for one node at a deploy
  // fraction of 0..1: the fraction is mapped onto the clip's own time range
  // and the bracketing keys are blended.
  function gearQuaternion(model, nodeIndex, deploy) {
    const clip = (model.animations || []).find((a) => a.name === 'GearDeploy');
    if (!clip) return null;
    const channel = clip.channels.find((c) => c.node === nodeIndex && c.path === 'rotation');
    if (!channel) return null;
    const times = channel.sampler.input.values;
    const quats = channel.sampler.output.values;
    const at = times[0] + Math.max(0, Math.min(1, deploy)) * (times[times.length - 1] - times[0]);
    let k = 0;
    while (k < times.length - 2 && times[k + 1] < at) k++;
    const span = Math.max(0.00001, times[k + 1] - times[k]);
    const t = Math.max(0, Math.min(1, (at - times[k]) / span));
    const q = [0, 1, 2, 3].map((j) => quats[k * 4 + j] * (1 - t) + quats[(k + 1) * 4 + j] * t);
    const len = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    return q.map((v) => v / len);
  }

  const ENEMY_PAINT = { color: [0.63, 0.17, 0.13], metallic: 0.45, roughness: 0.52 };
  const WRECK_PAINT = { color: [0.18, 0.17, 0.16], metallic: 0.5, roughness: 0.75 };

  function drawAirframe(model, pose, vp, camera, time, gear, override, scaleFactor = 1) {
    if (!model) return;
    const s = AIRCRAFT.scale * scaleFactor;
    const root = multiply(
      translation(pose.position[0], pose.position[1], pose.position[2]),
      multiply(
        rotationAbout([0, 1, 0], pose.heading + AIRCRAFT.yawOffset),
        multiply(
          rotationAbout([1, 0, 0], pose.pitch),
          multiply(rotationAbout([0, 0, 1], -pose.bank), scaling(s, s, s)))));
    for (let i = 0; i < model.nodes.length; i++) {
      const node = model.nodes[i];
      if (node.mesh < 0) continue;
      const isGear = node.name.startsWith('Gear_');
      if (override && isGear) continue;
      let local = node.local;
      if (!override && isGear) {
        const q = gearQuaternion(model, i, gear);
        if (q) local = compose(node.translation, q, node.scale);
      }
      const worldMatrix = multiply(root, local);
      for (const prim of model.meshes[node.mesh]) {
        const material = override
          ? { ...override, base: null, orm: null, normal: null }
          : prim.material;
        drawMesh(prim, material, worldMatrix, vp, camera, time, { doubleSided: true });
      }
    }
  }

  function drawEnemy(enemy, vp, camera, time) {
    const p = enemy.position;
    const dx = world.state.flight.position[0] - p[0];
    const dz = world.state.flight.position[2] - p[2];
    drawAirframe(aircraftModel,
      { position: p, heading: Math.atan2(dx, dz), pitch: 0, bank: enemy.bank },
      vp, camera, time, 0, ENEMY_PAINT, 1.0);
  }

  function drawWreck(wreck, vp, camera, time) {
    drawAirframe(aircraftModel,
      { position: wreck.position, heading: wreck.spin, pitch: wreck.spin * 0.7, bank: wreck.spin * 1.3 },
      vp, camera, time, 0, WRECK_PAINT, 0.9);
  }

  // --- billboards --------------------------------------------------------------

  // A camera-facing quad matrix: the third axis points from the quad toward
  // the camera, the first is kept level with the horizon.
  function billboardMatrix(position, size, camera) {
    const dir = normalize([
      camera[0] - position[0], camera[1] - position[1], camera[2] - position[2],
    ]);
    let right = normalize([dir[2], 0, -dir[0]]);
    if (!Number.isFinite(right[0])) right = [1, 0, 0];
    const up = cross(right, dir);
    return new Float32Array([
      right[0] * size, right[1] * size, right[2] * size, 0,
      up[0] * size, up[1] * size, up[2] * size, 0,
      dir[0] * size, dir[1] * size, dir[2] * size, 0,
      position[0], position[1], position[2], 1,
    ]);
  }

  // An oriented quad matrix for the missile: local +Z along its velocity.
  function headingMatrix(position, velocity, size) {
    const f = normalize(velocity);
    let right = normalize([-f[2], 0, f[0]]);
    if (!Number.isFinite(right[0])) right = [1, 0, 0];
    const up = cross(right, f);
    return new Float32Array([
      right[0] * size, right[1] * size, right[2] * size, 0,
      up[0] * size, up[1] * size, up[2] * size, 0,
      f[0] * size, f[1] * size, f[2] * size, 0,
      position[0], position[1], position[2], 1,
    ]);
  }

  // --- HUD drawing ---------------------------------------------------------------

  const HUD_FONT = 'ui-monospace, "Cascadia Mono", "Segoe UI Mono", monospace';
  let pointer = { x: -1, y: -1 };

  function trackPointer(event) {
    const rect = hudCanvas.getBoundingClientRect();
    pointer = {
      x: ((event.clientX - rect.left) / rect.width) * VIEW.width,
      y: ((event.clientY - rect.top) / rect.height) * VIEW.height,
    };
  }
  if (!options.fixedSize) {
    window.addEventListener('pointermove', trackPointer);
    window.addEventListener('pointerdown', trackPointer);
  }

  // Rounded panel path; the caller sets the fill style first.
  function panelPath(x, y, w, h, r = 12) {
    hud.beginPath();
    hud.moveTo(x + r, y);
    hud.lineTo(x + w - r, y);
    hud.quadraticCurveTo(x + w, y, x + w, y + r);
    hud.lineTo(x + w, y + h - r);
    hud.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    hud.lineTo(x + r, y + h);
    hud.quadraticCurveTo(x, y + h, x, y + h - r);
    hud.lineTo(x, y + r);
    hud.quadraticCurveTo(x, y, x + r, y);
    hud.closePath();
    hud.fill();
  }

  function label(value, x, y, size, colour, align = 'left', weight = 600, glow = null) {
    hud.font = `${weight} ${size}px ${HUD_FONT}`;
    hud.textAlign = align;
    hud.textBaseline = 'alphabetic';
    if (glow) {
      hud.shadowColor = glow;
      hud.shadowBlur = size * 0.7;
    }
    hud.fillStyle = colour;
    hud.fillText(value, x, y);
    if (glow) hud.shadowBlur = 0;
  }

  // One round cockpit gauge: a glowing rim, ten ticks (the last three
  // amber), a needle for the 0..1 value, and the name under the hub.
  function drawGauge(cx, cy, radius, value, name, unit) {
    hud.save();
    hud.translate(cx, cy);
    hud.shadowColor = '#4de6ff';
    hud.shadowBlur = 16;
    hud.fillStyle = '#07141f';
    hud.beginPath();
    hud.arc(0, 0, radius, 0, Math.PI * 2);
    hud.fill();
    hud.shadowBlur = 0;
    hud.strokeStyle = 'rgba(120,216,230,.55)';
    hud.lineWidth = 2;
    hud.stroke();
    for (let i = 0; i <= 10; i++) {
      const ang = Math.PI * (0.78 + (i * 1.44) / 10);
      hud.strokeStyle = i > 7 ? 'rgba(255,174,105,.7)' : 'rgba(170,238,246,.55)';
      hud.lineWidth = i % 5 === 0 ? 2 : 1;
      hud.beginPath();
      hud.moveTo(Math.cos(ang) * (radius - 8), Math.sin(ang) * (radius - 8));
      hud.lineTo(Math.cos(ang) * (radius - 2), Math.sin(ang) * (radius - 2));
      hud.stroke();
    }
    const clamped = Math.max(0, Math.min(1, value));
    const ang = Math.PI * (0.78 + clamped * 1.44);
    hud.strokeStyle = '#ffcf83';
    hud.shadowColor = '#ffc266';
    hud.shadowBlur = 10;
    hud.lineWidth = 2.5;
    hud.beginPath();
    hud.moveTo(Math.cos(ang) * -6, Math.sin(ang) * -6);
    hud.lineTo(Math.cos(ang) * (radius - 12), Math.sin(ang) * (radius - 12));
    hud.stroke();
    hud.shadowBlur = 0;
    hud.fillStyle = '#b8f6ff';
    hud.beginPath();
    hud.arc(0, 0, 3, 0, Math.PI * 2);
    hud.fill();
    label(name, 0, radius * 0.42, 11, '#93dce7', 'center', 700);
    label(unit, 0, radius * 0.62, 8, '#5f8d98', 'center', 500);
    hud.restore();
  }

  // The attitude indicator: a clipped circle whose sky/ground card rolls
  // with the bank and slides with the pitch, a fixed aircraft symbol over
  // it, and a bank pointer riding the rim.
  function drawHorizon(cx, cy, radius, pitch, bank) {
    hud.save();
    hud.beginPath();
    hud.arc(cx, cy, radius, 0, Math.PI * 2);
    hud.clip();
    hud.translate(cx, cy);
    hud.rotate(-bank);
    hud.translate(0, (pitch / 0.5) * radius * 0.8);
    hud.fillStyle = '#184e6b';
    hud.fillRect(-radius * 2, -radius * 4, radius * 4, radius * 4);
    hud.fillStyle = '#6d4a2c';
    hud.fillRect(-radius * 2, 0, radius * 4, radius * 4);
    hud.strokeStyle = 'rgba(230,250,255,.8)';
    hud.lineWidth = 1.5;
    hud.beginPath();
    hud.moveTo(-radius * 2, 0);
    hud.lineTo(radius * 2, 0);
    hud.stroke();
    for (let d = -40; d <= 40; d += 10) {
      if (d === 0) continue;
      const y = (d / 0.5) * radius * 0.8;
      const len = Math.abs(d) === 10 ? 28 : Math.abs(d) === 20 ? 22 : 16;
      hud.strokeStyle = 'rgba(220,245,250,.48)';
      hud.lineWidth = 1;
      hud.beginPath();
      hud.moveTo(-len, y);
      hud.lineTo(len, y);
      hud.stroke();
      if (Math.abs(d) % 20 === 0) label(String(d), len + 8, y + 4, 9, 'rgba(225,248,255,.6)', 'left', 500);
    }
    hud.restore();

    hud.save();
    hud.translate(cx, cy);
    hud.strokeStyle = 'rgba(124,238,252,.85)';
    hud.lineWidth = 2;
    hud.beginPath();
    hud.arc(0, 0, radius, 0, Math.PI * 2);
    hud.stroke();
    hud.strokeStyle = '#ffe3a1';
    hud.shadowColor = '#ffcf73';
    hud.shadowBlur = 8;
    hud.beginPath();
    hud.moveTo(-42, 0);
    hud.lineTo(-14, 0);
    hud.moveTo(14, 0);
    hud.lineTo(42, 0);
    hud.moveTo(-14, 0);
    hud.lineTo(-6, -8);
    hud.moveTo(14, 0);
    hud.lineTo(6, -8);
    hud.stroke();
    hud.beginPath();
    hud.arc(0, 0, 4, 0, Math.PI * 2);
    hud.stroke();
    hud.shadowBlur = 0;
    hud.strokeStyle = 'rgba(220,250,255,.8)';
    hud.lineWidth = 2;
    hud.beginPath();
    hud.arc(0, 0, radius + 5, Math.PI * 1.2, Math.PI * 1.8);
    hud.stroke();
    const bx = -35 + (bank / 0.9) * 35;
    hud.fillStyle = '#ffd889';
    hud.beginPath();
    hud.moveTo(bx, -radius - 10);
    hud.lineTo(bx - 7, -radius - 20);
    hud.lineTo(bx + 7, -radius - 20);
    hud.closePath();
    hud.fill();
    hud.restore();
  }

  // World point → logical HUD point, or null when behind the camera.
  function project(point, vp) {
    const x = vp[0] * point[0] + vp[4] * point[1] + vp[8] * point[2] + vp[12];
    const y = vp[1] * point[0] + vp[5] * point[1] + vp[9] * point[2] + vp[13];
    const w = vp[3] * point[0] + vp[7] * point[1] + vp[11] * point[2] + vp[15];
    if (w <= 0) return null;
    return {
      x: ((x / w) * 0.5 + 0.5) * VIEW.width,
      y: (1 - ((y / w) * 0.5 + 0.5)) * VIEW.height,
    };
  }

  function radarPoint(point, cx, cy, r) {
    const t = world.terrain;
    const nx = (point[0] - (t.minX + t.maxX) / 2) / (t.maxX - t.minX);
    const nz = (point[2] - (t.minZ + t.maxZ) / 2) / (t.maxZ - t.minZ);
    return [cx + nx * r * 2, cy - nz * r * 2];
  }

  function drawRadar(state, cx, cy, r) {
    const f = state.flight;
    hud.save();
    hud.beginPath();
    hud.arc(cx, cy, r, 0, Math.PI * 2);
    hud.fillStyle = 'rgba(3,15,22,.84)';
    hud.fill();
    hud.save();
    hud.clip();
    hud.drawImage(radarPicture, cx - r, cy - r, r * 2, r * 2);
    hud.strokeStyle = 'rgba(120,220,235,.12)';
    hud.lineWidth = 1;
    for (let i = 1; i <= 3; i++) {
      hud.beginPath();
      hud.arc(cx, cy, (r * i) / 3, 0, Math.PI * 2);
      hud.stroke();
    }
    hud.beginPath();
    hud.moveTo(cx - r, cy);
    hud.lineTo(cx + r, cy);
    hud.moveTo(cx, cy - r);
    hud.lineTo(cx, cy + r);
    hud.stroke();

    const stripA = radarPoint([runwayX, 0, runwayZ - WORLD.runwayLength / 2], cx, cy, r);
    const stripB = radarPoint([runwayX, 0, runwayZ + WORLD.runwayLength / 2], cx, cy, r);
    hud.strokeStyle = 'rgba(255,210,120,.8)';
    hud.lineWidth = 2;
    hud.beginPath();
    hud.moveTo(stripA[0], stripA[1]);
    hud.lineTo(stripB[0], stripB[1]);
    hud.stroke();

    const returning = state.phase === 'return';
    const target = returning
      ? [runwayX, WORLD.runwayHeight + 4, runwayZ]
      : (state.rings[state.ringIndex] ? state.rings[state.ringIndex].position : null);
    const playerPoint = radarPoint(f.position, cx, cy, r);
    if (target) {
      const tp = radarPoint(target, cx, cy, r);
      hud.strokeStyle = '#ffd277';
      hud.lineWidth = 1.2;
      hud.setLineDash([3, 3]);
      hud.beginPath();
      hud.moveTo(playerPoint[0], playerPoint[1]);
      hud.lineTo(tp[0], tp[1]);
      hud.stroke();
      hud.setLineDash([]);
    }
    for (const ring of state.rings) {
      if (ring.passed) continue;
      const rp = radarPoint(ring.position, cx, cy, r);
      const isNext = ring.index - 1 === state.ringIndex;
      hud.fillStyle = isNext ? '#ffd777' : 'rgba(105,232,245,.75)';
      hud.beginPath();
      hud.arc(rp[0], rp[1], isNext ? 3.6 : 2.2, 0, Math.PI * 2);
      hud.fill();
      if (isNext) {
        hud.strokeStyle = 'rgba(255,215,120,.8)';
        hud.lineWidth = 1;
        hud.beginPath();
        hud.arc(rp[0], rp[1], 7, 0, Math.PI * 2);
        hud.stroke();
      }
    }
    for (const enemy of state.enemies) {
      if (!enemy.alive) continue;
      const ep = radarPoint(enemy.position, cx, cy, r);
      hud.fillStyle = '#ff7768';
      hud.beginPath();
      hud.arc(ep[0], ep[1], 2.5, 0, Math.PI * 2);
      hud.fill();
    }
    let px = playerPoint[0], py = playerPoint[1];
    const pdx = px - cx, pdy = py - cy;
    const plen = Math.hypot(pdx, pdy);
    if (plen > r - 7) {
      px = cx + (pdx / plen) * (r - 7);
      py = cy + (pdy / plen) * (r - 7);
    }
    hud.save();
    hud.translate(px, py);
    hud.rotate(f.heading);
    hud.fillStyle = '#ecfeff';
    hud.strokeStyle = 'rgba(2,10,16,.8)';
    hud.lineWidth = 1.5;
    hud.beginPath();
    hud.moveTo(0, -7);
    hud.lineTo(-5, 6);
    hud.lineTo(0, 3);
    hud.lineTo(5, 6);
    hud.closePath();
    hud.fill();
    hud.stroke();
    hud.restore();
    hud.restore();
    hud.strokeStyle = 'rgba(140,238,250,.7)';
    hud.lineWidth = 1.5;
    hud.beginPath();
    hud.arc(cx, cy, r, 0, Math.PI * 2);
    hud.stroke();
    label('N', cx, cy - r + 13, 9, '#a9d6df', 'center', 700);
    label('TACTICAL', cx, cy + r + 14, 9, '#a8d4dc', 'center', 600);
    hud.restore();
  }

  // The destination marker: a bracket around the target while it is on
  // screen, an arrow pinned to the screen edge when it is not, so the next
  // ring (or the runway) can always be found.
  function drawNavMarker(vp, camera, cam, point, text, colour, isRunway = false) {
    const p = project(point, vp);
    if (p && p.x > 66 && p.x < 1214 && p.y > 86 && p.y < 536) {
      const size = isRunway ? 20 : 25;
      hud.strokeStyle = colour;
      hud.lineWidth = 1.5;
      hud.strokeRect(p.x - size, p.y - size, size * 2, size * 2);
      hud.globalAlpha = 0.35;
      hud.beginPath();
      hud.moveTo(p.x - size - 12, p.y);
      hud.lineTo(p.x - size - 4, p.y);
      hud.moveTo(p.x + size + 4, p.y);
      hud.lineTo(p.x + size + 12, p.y);
      hud.stroke();
      hud.globalAlpha = 1;
      label(text, p.x, p.y + size + 18, 11, colour, 'center', 600, 'rgba(2,10,16,.8)');
      return;
    }
    const to = normalize(sub(point, camera));
    const viewDir = normalize(sub(cam.target, camera));
    const right = normalize(cross(viewDir, cam.up));
    const up = normalize(cross(right, viewDir));
    let dx = dot(to, right);
    let dy = -dot(to, up);
    const dlen = Math.hypot(dx, dy);
    if (dlen < 0.001) { dx = 0; dy = 1; } else { dx /= dlen; dy /= dlen; }
    const x = Math.max(78, Math.min(1202, 640 + dx * 520));
    const y = Math.max(110, Math.min(520, 360 + dy * 275));
    hud.save();
    hud.translate(x, y);
    hud.rotate(Math.atan2(dy, dx));
    hud.fillStyle = colour;
    hud.strokeStyle = 'rgba(2,10,16,.8)';
    hud.lineWidth = 1.5;
    hud.beginPath();
    hud.moveTo(13, 0);
    hud.lineTo(-8, -8);
    hud.lineTo(-3, 0);
    hud.lineTo(-8, 8);
    hud.closePath();
    hud.fill();
    hud.stroke();
    hud.restore();
    label(text, x + dx * 20, y + dy * 20 + (dy < 0 ? -10 : 18), 11, colour, 'center', 600, 'rgba(2,10,16,.85)');
  }

  // Touch flight controls — painted only in touch mode (state.touchUI), on
  // top of the play HUD. The zones come from TOUCH_ZONES; the input layer
  // hit-tests the same circles. A button lights up when the flight state
  // shows its control is in effect: bank/pitch from the airframe's actual
  // attitude, the throttle buttons at the ends of the lever's travel, GEAR
  // while the gear is down or travelling down, FIRE while a lock is solid.
  function touchZoneActive(zone, state) {
    const f = state.flight;
    switch (zone.name) {
      case 'bankLeft': return f.bank > 0.04;
      case 'bankRight': return f.bank < -0.04;
      case 'pitchUp': return f.pitch > 0.03;
      case 'pitchDown': return f.pitch < -0.03;
      case 'throttleUp': return f.throttle > 0.98;
      case 'throttleDown': return f.throttle < 0.02;
      case 'gear': return f.gearTarget > 0.5;
      case 'fire': return Boolean(state.lock && state.lock.alive) && state.lockTime > 0.55;
      default: return false;
    }
  }

  function drawTouchControls(state) {
    for (const zone of TOUCH_ZONES) {
      const active = touchZoneActive(zone, state);
      hud.beginPath();
      hud.arc(zone.x, zone.y, zone.r, 0, Math.PI * 2);
      hud.fillStyle = active ? 'rgba(96,224,246,.34)' : 'rgba(6,20,30,.38)';
      hud.fill();
      hud.strokeStyle = active ? 'rgba(170,246,255,.9)' : 'rgba(140,220,235,.45)';
      hud.lineWidth = active ? 2 : 1.5;
      hud.stroke();
      const fontSize = zone.label.length > 2 ? 13 : 17;
      label(zone.label, zone.x, zone.y + fontSize * 0.36, fontSize,
        active ? '#eafcff' : 'rgba(205,240,246,.85)', 'center', 700,
        active ? 'rgba(90,220,250,.6)' : null);
    }
  }

  // --- screens -------------------------------------------------------------------

  function drawFlightHud(state, vp, camera, cam) {
    const f = state.flight;
    const topShade = hud.createLinearGradient(0, 0, 0, 205);
    topShade.addColorStop(0, 'rgba(2,10,18,.86)');
    topShade.addColorStop(0.6, 'rgba(2,10,18,.5)');
    topShade.addColorStop(1, 'rgba(2,10,18,0)');
    hud.fillStyle = topShade;
    hud.fillRect(0, 0, VIEW.width, 205);

    const feet = Math.max(0, Math.round((f.position[1] - WORLD.seaLevel) * 3.28));
    const knots = Math.round(f.speed * 1.94);
    const headingDeg = Math.round((((f.heading * 180) / Math.PI) % 360 + 360) % 360);
    label(`ALT ${String(feet).padStart(4, '0')} ft`, 36, 44, 17, '#d9fbff', 'left', 600, 'rgba(76,230,255,.9)');
    label(`SPD ${String(knots).padStart(3, '0')} kt`, 36, 71, 17, '#d9fbff', 'left', 600, 'rgba(76,230,255,.9)');
    label(`HDG ${String(headingDeg).padStart(3, '0')}°`, 36, 98, 17, '#d9fbff', 'left', 600, 'rgba(76,230,255,.9)');
    label(`RWY ${Math.round(world.runwayDistance(f.position))} m`, 36, 125, 12, '#b9eef6', 'left', 600, 'rgba(70,200,220,.5)');

    label(`TIME ${state.time.toFixed(1)}s`, 1244, 44, 17, '#d9fbff', 'right', 600, 'rgba(76,230,255,.9)');
    label(`RINGS ${Math.min(state.ringIndex, RING_COUNT)}/${RING_COUNT}`, 1244, 71, 17, '#d9fbff', 'right', 600, 'rgba(76,230,255,.9)');
    label(`MSL ${state.missileCount}`, 1244, 98, 17, '#d9fbff', 'right', 600, 'rgba(76,230,255,.9)');
    label(`LEFT ${Math.max(0, RING_COUNT - state.ringIndex)}`, 1244, 150, 12, '#b9eef6', 'right', 600, 'rgba(70,200,220,.5)');
    for (let i = 0; i < RING_COUNT; i++) {
      hud.fillStyle = i < state.ringIndex ? '#ffc45f'
        : i === state.ringIndex ? '#8ff7ff' : 'rgba(120,220,235,.24)';
      hud.fillRect(1128 + i * 13, 112, 8, 4);
    }

    drawHorizon(152, 464, 62, f.pitch, f.bank);
    label(String(knots).padStart(3, '0'), 238, 458, 16, '#ffe3a1', 'left', 700, 'rgba(255,192,95,.4)');
    label('KT', 292, 458, 9, '#8a9d92', 'left', 500);
    label(String(feet).padStart(4, '0'), 238, 482, 16, '#ffe3a1', 'left', 700, 'rgba(255,192,95,.4)');
    label('FT', 298, 482, 9, '#8a9d92', 'left', 500);

    drawRadar(state, 1146, 226, 62);

    const returning = state.phase === 'return';
    const nextRing = state.rings[state.ringIndex];
    const navPoint = returning ? [runwayX, WORLD.runwayHeight + 5, runwayZ] : (nextRing ? nextRing.position : null);
    const navDist = navPoint ? Math.round(Math.hypot(
      navPoint[0] - f.position[0], navPoint[1] - f.position[1], navPoint[2] - f.position[2])) : 0;
    if (navPoint) {
      drawNavMarker(vp, camera, cam, navPoint,
        returning ? `RWY  ${navDist}m` : `RING ${nextRing.index}  ${navDist}m`,
        '#ffd777', returning);
    }
    if (returning) {
      label(`RETURN TO RUNWAY  ${navDist}m`, 640, 150, 16, '#ffe1a5', 'center', 700, 'rgba(255,195,98,.45)');
      label('GEAR AUTO — SLOW TO 120 KT • SHALLOW DESCENT', 640, 174, 11, '#d5b98b', 'center', 500);
    }
    if (navPoint) {
      const ahead = dot(sub(navPoint, f.position), world.forward());
      if (ahead < 0 || navDist > 900) {
        label('R — RECOVER TO COURSE   (+5s)', 640, 200, 12, '#ffe6b0', 'center', 700, 'rgba(2,10,16,.92)');
      }
    }

    if (state.lock && state.lock.alive) {
      const p = project(state.lock.position, vp);
      if (p) {
        const locked = state.lockTime > 0.55;
        const s = locked ? 25 : 34;
        hud.strokeStyle = locked ? '#ff6b5c' : 'rgba(255,196,110,.8)';
        hud.lineWidth = 2;
        hud.strokeRect(p.x - s, p.y - s, s * 2, s * 2);
        hud.strokeStyle = locked ? 'rgba(255,107,92,.5)' : 'rgba(255,196,110,.4)';
        hud.beginPath();
        hud.arc(p.x, p.y, s + 8, 0, Math.PI * 2);
        hud.stroke();
        label(locked ? 'LOCK' : 'SEEK', p.x, p.y - s - 10, 12,
          locked ? '#ff9488' : '#ffd08c', 'center', 700, locked ? 'rgba(255,90,70,.6)' : null);
      }
    }

    if (state.messageTime > 0 && state.message) {
      hud.globalAlpha = Math.min(1, state.messageTime * 2.2);
      label(state.message, 640, 128, 14, '#d8fbff', 'center', 700, 'rgba(90,234,255,.55)');
      hud.globalAlpha = 1;
    }

    // The cockpit panel: a dark trapezoid across the bottom carrying the
    // four gauges, with the control reminder in the middle.
    const panelGrad = hud.createLinearGradient(0, 596, 0, VIEW.height);
    panelGrad.addColorStop(0, 'rgba(15,33,45,.96)');
    panelGrad.addColorStop(0.12, 'rgba(7,18,27,.99)');
    panelGrad.addColorStop(1, '#040b12');
    hud.fillStyle = panelGrad;
    hud.beginPath();
    hud.moveTo(0, 628);
    hud.lineTo(194, 596);
    hud.lineTo(1086, 596);
    hud.lineTo(1280, 628);
    hud.lineTo(1280, 720);
    hud.lineTo(0, 720);
    hud.closePath();
    hud.fill();
    hud.strokeStyle = 'rgba(150,206,213,.42)';
    hud.lineWidth = 1.5;
    hud.stroke();
    hud.strokeStyle = 'rgba(84,228,244,.2)';
    hud.lineWidth = 1;
    hud.beginPath();
    hud.moveTo(194, 604);
    hud.lineTo(1086, 604);
    hud.stroke();
    hud.fillStyle = 'rgba(160,235,245,.055)';
    hud.beginPath();
    hud.moveTo(0, 628);
    hud.lineTo(194, 596);
    hud.lineTo(400, 596);
    hud.lineTo(120, 720);
    hud.lineTo(0, 720);
    hud.closePath();
    hud.fill();

    drawGauge(112, 660, 42, f.speed / 245, 'SPD', 'KNOTS');
    drawGauge(1168, 660, 42, (f.position[1] - WORLD.seaLevel) / 700, 'ALT', 'FEET');
    drawGauge(232, 664, 31, f.throttle, 'THR', '%');
    drawGauge(1048, 664, 31, f.gear, 'GEAR', f.gear > 0.88 ? 'DOWN' : f.gear < 0.12 ? 'UP' : 'MOVING');
    label('SKY RING', 640, 628, 16, '#d6f7fc', 'center', 700, 'rgba(83,230,248,.5)');
    label(returning ? 'RETURN LEG' : 'COURSE ACTIVE', 640, 647, 10, '#7fa6ad', 'center', 500);
    label('↑ CLIMB   ↓ DIVE   ←→ BANK / TURN   W/S THROTTLE   G GEAR   J/F FIRE   R RECOVER',
      640, 690, 11, '#8fb2b8', 'center', 500);
    if (state.lock && state.lock.alive && state.lockTime > 0.55) {
      label('MISSILE LOCK — FIRE', 830, 660, 11, '#ff8e80', 'center', 700, 'rgba(255,90,70,.5)');
    } else {
      label(state.missileCount > 0 ? 'ARMED' : 'EMPTY', 830, 660, 11, '#7fa6ad', 'center', 600);
    }

    if (state.touchUI) drawTouchControls(state);
  }

  function drawTitleScreen() {
    hud.fillStyle = 'rgba(2,9,16,.18)';
    hud.fillRect(0, 0, VIEW.width, VIEW.height);
    const vignette = hud.createRadialGradient(640, 330, 80, 640, 330, 620);
    vignette.addColorStop(0, 'rgba(4,17,28,.02)');
    vignette.addColorStop(1, 'rgba(2,9,16,.6)');
    hud.fillStyle = vignette;
    hud.fillRect(0, 0, VIEW.width, VIEW.height);
    const topGrad = hud.createLinearGradient(0, 0, 0, 330);
    topGrad.addColorStop(0, 'rgba(2,9,16,.08)');
    topGrad.addColorStop(0.38, 'rgba(2,9,16,.42)');
    topGrad.addColorStop(0.75, 'rgba(2,9,16,.22)');
    topGrad.addColorStop(1, 'rgba(2,9,16,0)');
    hud.fillStyle = topGrad;
    hud.fillRect(0, 0, VIEW.width, 330);

    hud.save();
    hud.translate(640, 162);
    hud.scale(1.01, 1);
    label('S K Y   R I N G', 0, 0, 66, '#effdff', 'center', 800, 'rgba(84,224,246,.7)');
    hud.restore();
    hud.strokeStyle = 'rgba(132,237,252,.5)';
    hud.lineWidth = 1;
    hud.beginPath();
    hud.moveTo(438, 194);
    hud.lineTo(842, 194);
    hud.stroke();
    label('AN ORIGINAL STYLIZED FLIGHT', 640, 222, 13, '#e2fbff', 'center', 600, 'rgba(1,8,14,.85)');
    label('FLY THE NINE RINGS.  RETURN TO THE RUNWAY.', 640, 250, 12, '#d6f2f8', 'center', 500, 'rgba(1,8,14,.85)');
    label('CHASE VIEW  •  PBR SEA & METAL  •  OFFLINE', 640, 286, 10, '#b9dce4', 'center', 500, 'rgba(1,8,14,.85)');

    // The two main buttons must stay inside the input layer's hit boxes:
    // start x 398..632, credits x 648..882, both y 512..564.
    const by = 512;
    const menuButton = (text, x, w, hovered) => {
      hud.fillStyle = hovered ? 'rgba(75,213,236,.20)' : 'rgba(7,24,35,.78)';
      panelPath(x, by, w, 52, 8);
      hud.strokeStyle = hovered ? '#9ff7ff' : 'rgba(120,215,230,.5)';
      hud.lineWidth = 1.5;
      hud.beginPath();
      hud.moveTo(x + 8, by);
      hud.lineTo(x + w - 8, by);
      hud.lineTo(x + w - 8, by + 52);
      hud.lineTo(x + 8, by + 52);
      hud.closePath();
      hud.stroke();
      label(text, x + w / 2, by + 33, 14, hovered ? '#ecfeff' : '#b9e9ef', 'center', 700,
        hovered ? 'rgba(100,240,255,.55)' : null);
    };
    menuButton('ENTER  —  SORTIE', 398, 234,
      pointer.x > 398 && pointer.x < 632 && pointer.y > by && pointer.y < by + 52);
    menuButton('C  —  CREDITS', 648, 234,
      pointer.x > 648 && pointer.x < 882 && pointer.y > by && pointer.y < by + 52);

    const audioHovered = pointer.x > 548 && pointer.x < 732 && pointer.y > 578 && pointer.y < 614;
    hud.fillStyle = audioHovered ? 'rgba(75,213,236,.22)' : 'rgba(7,24,35,.72)';
    panelPath(548, 578, 184, 36, 6);
    hud.strokeStyle = audioHovered ? '#9ff7ff' : 'rgba(120,215,230,.4)';
    hud.lineWidth = 1;
    hud.beginPath();
    hud.moveTo(556, 578);
    hud.lineTo(724, 578);
    hud.lineTo(724, 614);
    hud.lineTo(556, 614);
    hud.closePath();
    hud.stroke();
    label('O  —  AUDIO', 640, 602, 11, '#c9edf3', 'center', 700);
    label('↑ climb   ↓ dive   ←→ bank   W/S throttle   G gear   J/F missile   R recover',
      640, 678, 10, '#c3e0e5', 'center', 500, 'rgba(2,10,16,.8)');
  }

  function drawCreditsScreen() {
    hud.fillStyle = 'rgba(2,9,16,.88)';
    hud.fillRect(0, 0, VIEW.width, VIEW.height);
    hud.fillStyle = 'rgba(7,22,33,.9)';
    panelPath(126, 70, 1028, 588, 10);
    hud.strokeStyle = 'rgba(110,225,242,.35)';
    hud.lineWidth = 1.5;
    hud.beginPath();
    hud.moveTo(126, 130);
    hud.lineTo(1154, 130);
    hud.stroke();
    label('CREDITS', 640, 112, 22, '#e8fdff', 'center', 700, 'rgba(84,224,246,.55)');

    // The MODELS and MUSIC sections are the author's own work — the two
    // models were made by the author in Tripo, the music generated by the
    // author with Suno — and are listed apart from the third-party assets
    // below them, which carry their licenses.
    const rows = [
      ['MODELS', ''],
      ['FX-27 Raven', 'original model by the author, created with Tripo'],
      ['Island terrain', 'original model by the author, created with Tripo'],
      ['MUSIC', ''],
      ['Title & flight BGM', 'generated with Suno'],
      ['AUDIO', ''],
      ['G08-25_4-E Jet Constant Noise', 'by craigsmith (CC0) — https://freesound.org/people/craigsmith/sounds/437888/'],
      ['Rocket Launch', 'by gracenew (CC0) — https://freesound.org/people/gracenew/sounds/857959/'],
      ['Sci-Fi_Explosion_2', 'by Anomaex (CC0) — https://freesound.org/people/Anomaex/sounds/490266/'],
      ['Lock on (Sci-Fi UI SFX Free)', 'by heyheytheree (CC BY 4.0) — https://freesound.org/people/heyheytheree/sounds/872714/'],
      ['', 'license: https://creativecommons.org/licenses/by/4.0/'],
      ['', 'CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/'],
      ['Ring chime, landing jingle', 'by Kenney (CC0) — https://kenney.nl'],
      ['VISUAL EFFECTS', ''],
      ['Explosion sprite (orange recolor)', 'by Aj_ (CC0) — https://opengameart.org/content/explosion-sprite'],
      ['Smoke particles', 'by Kenney (CC0) — https://kenney.nl'],
    ];
    let y = 162;
    for (const [left, right] of rows) {
      const isHeader = left !== '' && right === '';
      if (isHeader) {
        label(left, 190, y, 13, '#76e9fb', 'left', 700);
        y += 30;
      } else {
        if (left) label(left, 190, y, 12, '#dcf3f7', 'left', 600);
        if (right) label(right, 610, y, 11.5, '#a9cdd4', 'left', 500);
        y += 24;
      }
    }
    label('Third-party assets are used under the licenses shown; everything else is the author’s own work.',
      640, 600, 10, '#7fa6ae', 'center', 500);
    label('[ ESC / C ]  BACK', 640, 632, 12, '#8ff3ff', 'center', 700);
  }

  function drawSettingsScreen(state) {
    const s = state.settings || { bgm: 0.45, se: 0.8, blur: 0.45 };
    const selected = state.settingsSelected || 0;
    hud.fillStyle = 'rgba(2,9,16,.84)';
    hud.fillRect(0, 0, VIEW.width, VIEW.height);
    hud.fillStyle = 'rgba(6,20,31,.94)';
    panelPath(352, 136, 576, 500, 12);
    hud.strokeStyle = 'rgba(110,225,242,.35)';
    hud.lineWidth = 1.5;
    hud.beginPath();
    hud.moveTo(352, 198);
    hud.lineTo(928, 198);
    hud.stroke();
    label('AUDIO & VISUALS', 640, 181, 26, '#e8fdff', 'center', 700, 'rgba(84,224,246,.55)');
    label('SOUND AND SPEED BLUR', 640, 222, 10, '#7faab4', 'center', 500);

    const sliderRow = (name, value, rowY, active) => {
      if (active) {
        hud.fillStyle = 'rgba(83,220,240,.08)';
        panelPath(390, rowY - 34, 500, 62, 6);
        hud.strokeStyle = 'rgba(120,235,250,.55)';
        hud.lineWidth = 1;
        hud.stroke();
        label('▶', 408, rowY + 6, 12, '#8ff3ff', 'center', 700);
      }
      label(name, 430, rowY + 6, 13, active ? '#e8fdff' : '#aecbd1', 'left', 700);
      const x0 = 548, x1 = 800;
      const t = Math.max(0, Math.min(1, value));
      hud.fillStyle = 'rgba(6,20,28,.9)';
      panelPath(x0, rowY - 7, x1 - x0, 14, 7);
      hud.fillStyle = active ? '#68e8f7' : 'rgba(88,186,200,.7)';
      hud.fillRect(x0 + 3, rowY - 4, (x1 - x0 - 6) * t, 8);
      hud.fillStyle = active ? '#f2feff' : '#b7d8dd';
      hud.beginPath();
      hud.arc(x0 + 3 + (x1 - x0 - 6) * t, rowY, 7, 0, Math.PI * 2);
      hud.fill();
      label(`${Math.round(t * 100)}`, 830, rowY + 6, 13, active ? '#ffe3a1' : '#8fb2b8', 'right', 700);
    };
    sliderRow('BGM', s.bgm, 282, selected === 0);
    sliderRow('SE', s.se, 350, selected === 1);
    sliderRow('BLUR', s.blur !== undefined && s.blur !== null ? s.blur : 0.45, 418, selected === 2);
    hud.strokeStyle = 'rgba(120,210,225,.18)';
    hud.beginPath();
    hud.moveTo(390, 462);
    hud.lineTo(890, 462);
    hud.stroke();
    label('↑ / ↓   SELECT        ← / →   ADJUST', 640, 492, 12, '#b9e9ef', 'center', 600);
    label('SE covers the engine, weapon, ring and landing sounds.', 640, 518, 10, '#7fa6ae', 'center', 500);
    label('BLUR is a radial speed blur; 0 turns it off.', 640, 540, 10, '#7fa6ae', 'center', 500);
    label('BGM: title and flight themes, generated with Suno.', 640, 562, 10, '#7fa6ae', 'center', 500);
    label('[ ESC / O ]  BACK', 640, 596, 12, '#8ff3ff', 'center', 700);
  }

  function drawResultScreen(state) {
    hud.fillStyle = 'rgba(2,9,16,.76)';
    hud.fillRect(0, 0, VIEW.width, VIEW.height);
    hud.fillStyle = 'rgba(6,20,31,.92)';
    panelPath(250, 92, 780, 530, 12);
    hud.strokeStyle = 'rgba(255,205,122,.55)';
    hud.lineWidth = 2;
    hud.beginPath();
    hud.moveTo(250, 168);
    hud.lineTo(1030, 168);
    hud.stroke();
    label('FLIGHT COMPLETE', 640, 150, 38, '#ffedc9', 'center', 800, 'rgba(255,180,70,.45)');
    label('RUNWAY SECURED  •  AIRCRAFT RECOVERED', 640, 196, 11, '#9dc4ca', 'center', 600);
    const rows = [
      ['TIME', `${state.time.toFixed(1)} s`],
      ['RINGS', `${state.ringIndex} / ${RING_COUNT}`],
      ['KILLS', String(state.kills)],
      ['LANDING', state.flight.landingScore],
      ['SCORE', String(state.score)],
    ];
    rows.forEach(([key, value], i) => {
      const y = 264 + i * 46;
      const isScore = i === rows.length - 1;
      label(key, 430, y, 15, '#85b8c1', 'left', 600);
      label(value, 840, y, isScore ? 27 : 22, isScore ? '#ffe19b' : '#eafcff', 'right', 700,
        isScore ? 'rgba(255,190,80,.4)' : null);
    });
    hud.strokeStyle = 'rgba(120,210,225,.2)';
    hud.beginPath();
    hud.moveTo(390, 480);
    hud.lineTo(890, 480);
    hud.stroke();
    label('ENTER  —  FLY AGAIN', 640, 520, 14, '#8ff3ff', 'center', 700, 'rgba(100,230,250,.5)');
    label('ESC  —  TITLE', 640, 548, 10, '#6f959e', 'center', 500);
  }

  function drawCrashScreen(state) {
    hud.fillStyle = 'rgba(18,5,4,.55)';
    hud.fillRect(0, 0, VIEW.width, VIEW.height);
    const glow = hud.createRadialGradient(640, 330, 80, 640, 330, 600);
    glow.addColorStop(0, 'rgba(20,7,5,.12)');
    glow.addColorStop(1, 'rgba(10,3,2,.9)');
    hud.fillStyle = glow;
    hud.fillRect(0, 0, VIEW.width, VIEW.height);
    label('FLIGHT LOST', 640, 250, 52, '#ffd8cb', 'center', 800, 'rgba(255,90,60,.55)');
    label(state.crashReason, 640, 304, 18, '#ff9e88', 'center', 700, 'rgba(255,80,50,.4)');
    label('ENTER  —  TRY AGAIN', 640, 584, 15, '#ffd2a2', 'center', 700, 'rgba(2,5,8,.9)');
    label('ESC  —  TITLE', 640, 612, 10, '#c9a092', 'center', 500, 'rgba(2,5,8,.9)');
  }

  function drawHud(state, vp, camera, cam) {
    const width = hudCanvas.width, height = hudCanvas.height;
    hud.setTransform(1, 0, 0, 1, 0, 0);
    hud.clearRect(0, 0, width, height);
    hud.save();
    hud.scale(width / VIEW.width, height / VIEW.height);
    const overlay = state.phase === 'title' || state.phase === 'credits'
      || state.phase === 'settings' || state.phase === 'result' || state.phase === 'crashed';
    if (!overlay) drawFlightHud(state, vp, camera, cam);
    if (state.phase === 'title') drawTitleScreen();
    else if (state.phase === 'credits') drawCreditsScreen();
    else if (state.phase === 'settings') drawSettingsScreen(state);
    else if (state.phase === 'result') drawResultScreen(state);
    else if (state.phase === 'crashed') drawCrashScreen(state);
    hud.restore();
  }

  // --- camera ----------------------------------------------------------------------

  // Title-family screens park the camera off the aircraft's shoulder; in
  // flight it chases from behind and above, rolling part-way with the bank
  // so a turn leans the horizon without losing the airframe.
  function cameraFor(state) {
    const f = state.flight;
    if (state.phase === 'title' || state.phase === 'credits' || state.phase === 'settings') {
      return {
        position: [f.position[0] + 34, f.position[1] + 14, f.position[2] - 44],
        target: [f.position[0], f.position[1] + 2, f.position[2] + 40],
        up: [0, 1, 0.04],
      };
    }
    const fwd = world.forward();
    const distance = 46 + Math.min(18, f.speed * 0.045);
    return {
      position: [
        f.position[0] - fwd[0] * distance,
        f.position[1] + 10 + fwd[1] * 4,
        f.position[2] - fwd[2] * distance,
      ],
      target: [
        f.position[0] + fwd[0] * 58,
        f.position[1] + 2 + fwd[1] * 4,
        f.position[2] + fwd[2] * 58,
      ],
      up: [Math.sin(f.bank) * 0.55, Math.cos(f.bank) * 0.55, 0.04],
    };
  }

  // --- the frame ---------------------------------------------------------------------

  function render(state, dt = 0) {
    const width = glCanvas.width, height = glCanvas.height;
    const cam = cameraFor(state);
    const camera = cam.position;
    const proj = perspective(Math.PI / 3.1, width / height, 0.8, 9000);
    const view = lookAt(camera, cam.target, cam.up);
    const vp = multiply(proj, view);
    const time = state.time;
    const f = state.flight;

    gl.useProgram(program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, sceneTarget ? sceneTarget.fbo : null);
    gl.disable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(true);
    gl.clearColor(PALETTE.fog[0], PALETTE.fog[1], PALETTE.fog[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // The sky dome follows the camera and neither reads nor writes depth.
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    drawMesh(skyDome, { color: [1, 1, 1], roughness: 1 },
      multiply(translation(camera[0], camera[1], camera[2]), scaling(6500, 6500, 6500)),
      vp, camera, time, { sky: true, doubleSided: true });
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);

    // Sea, island, and the built scenery.
    drawMesh(seaPlane, { color: [0.16, 0.52, 0.62], metallic: 0.72, roughness: 0.16 },
      identity(), vp, camera, time, { water: true });
    if (terrainGeometry) {
      drawMesh(terrainGeometry,
        { ...terrainGeometry.material, color: [0.82, 0.9, 0.78], metallic: 0.02, roughness: 0.94 },
        identity(), vp, camera, time, { terrain: true, doubleSided: true });
    }
    drawMesh(runwaySlab, { color: [0.11, 0.13, 0.15], metallic: 0.22, roughness: 0.68 },
      runwayModel, vp, camera, time);
    drawMesh(runwayPaint, { color: [0.94, 0.86, 0.63], metallic: 0.08, roughness: 0.5, emissive: 0.06 },
      runwayModel, vp, camera, time);
    drawMesh(runwayLights, { color: [1.0, 0.62, 0.24], metallic: 0, roughness: 0.4, emissive: 1.6 },
      runwayModel, vp, camera, time);
    drawMesh(treeTrunks, { color: [0.22, 0.16, 0.11], metallic: 0, roughness: 0.95 },
      identity(), vp, camera, time);
    drawMesh(treeCanopy, { color: [0.24, 0.5, 0.3], metallic: 0, roughness: 0.9, emissive: 0.1 },
      identity(), vp, camera, time);

    // The ring course; the next gate pulses gold, the rest stay cyan.
    for (let i = 0; i < world.rings.length; i++) {
      if (state.rings[i].passed || state.phase === 'title') continue;
      const isNext = i === state.ringIndex;
      const pulse = isNext ? 0.82 + Math.sin(time * 4.2) * 0.18 : 0.5;
      const base = isNext ? [1.0, 0.78, 0.34] : [0.22, 0.78, 0.86];
      drawMesh(ringGeometries[i],
        { color: base.map((c) => c * pulse), metallic: 0.68, roughness: 0.22, emissive: isNext ? 0.55 : 0.2 },
        identity(), vp, camera, time, { doubleSided: true });
    }

    // Airframes.
    if (state.phase !== 'crashed') {
      drawAirframe(aircraftModel, state.flight, vp, camera, time, state.flight.gear, null);
    }
    for (const enemy of state.enemies) {
      if (enemy.alive && state.phase !== 'title') drawEnemy(enemy, vp, camera, time);
    }
    for (const wreck of state.wrecks) drawWreck(wreck, vp, camera, time);

    // The hovering target diamond, drawn through the terrain so the goal is
    // never hidden behind a ridge.
    const navPoint = state.phase === 'return'
      ? [runwayX, WORLD.runwayHeight + 30, runwayZ]
      : (state.rings[state.ringIndex] ? state.rings[state.ringIndex].position : null);
    const navScreen = navPoint ? project(navPoint, vp) : null;
    if (navPoint && navScreen
      && navScreen.x > -80 && navScreen.x < VIEW.width + 80
      && navScreen.y > -80 && navScreen.y < VIEW.height + 80
      && (state.phase === 'play' || state.phase === 'return')) {
      const bob = Math.sin(time * 3) * 1.8;
      gl.disable(gl.DEPTH_TEST);
      drawMesh(navDiamond,
        { color: [1.0, 0.80, 0.36], metallic: 0.4, roughness: 0.3, emissive: 1.0 },
        multiply(translation(navPoint[0], navPoint[1] + 36 + bob, navPoint[2]), scaling(1.5, 1.5, 1.5)),
        vp, camera, time, { doubleSided: true });
      gl.enable(gl.DEPTH_TEST);
    }

    // Missiles in flight.
    for (const m of state.missiles) {
      drawMesh(missileBody, { color: [0.82, 0.80, 0.76], metallic: 0.7, roughness: 0.28, emissive: 0.05 },
        headingMatrix(m.position, m.velocity, 1.15), vp, camera, time);
    }

    // Blended layer: the contact shadow, clouds, smoke, missile glare and
    // explosion flipbooks, back to front in list order, depth write off.
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);

    const altitude = f.position[1] - world.groundAt(f.position);
    const overGround = world.onRunway(f.position) || world.groundAt(f.position) < WORLD.seaLevel + 1;
    if (overGround && altitude < 420 && altitude > 0.8) {
      const groundY = Math.max(world.groundAt(f.position),
        world.onRunway(f.position) ? WORLD.runwayHeight : WORLD.seaLevel);
      const size = 30 + altitude * 0.09;
      const shadowModel = multiply(
        translation(f.position[0], groundY + 0.25, f.position[2]),
        multiply(rotationAbout([0, 1, 0], f.heading), scaling(size, 1, size * 1.5)));
      drawMesh(shadowQuadGeo,
        { base: shadowTexture, color: [1, 1, 1], opacity: Math.max(0, 0.5 - altitude / 900), metallic: 0, roughness: 1 },
        shadowModel, vp, camera, time, { blended: true });
    }

    drawMesh(cloudBank, { color: [0.96, 0.98, 1.0], metallic: 0, roughness: 1, opacity: 0.17, emissive: 0.3 },
      identity(), vp, camera, time, { blended: true });

    for (const p of state.particles) {
      const fade = Math.max(0, Math.min(0.78, 0.78 * (1 - p.age / p.life)));
      const size = (p.black ? 10 : 6.5) * (p.size || 1) * (0.55 + p.age * 0.75);
      drawMesh(puffQuad,
        {
          base: p.black ? fxTextures.smokeBlack : fxTextures.smokeWhite,
          color: p.black ? [0.10, 0.10, 0.11] : [0.96, 0.98, 0.98],
          opacity: fade, metallic: 0, roughness: 1,
        },
        billboardMatrix(p.position, size, camera), vp, camera, time,
        { blended: true, doubleSided: true });
    }
    for (const m of state.missiles) {
      drawMesh(puffQuad,
        { base: fxTextures.smokeWhite, color: [1.0, 0.62, 0.24], opacity: 0.95, metallic: 0, roughness: 1 },
        billboardMatrix(m.position, 10, camera), vp, camera, time,
        { blended: true, doubleSided: true });
    }
    for (const ex of state.explosions) {
      const t = Math.min(0.999, ex.age / ex.duration);
      const frame = Math.min(31, Math.floor(t * 32));
      drawMesh(explosionFrames[frame],
        { base: fxTextures.explosion, color: [1, 0.86, 0.66], opacity: 1, metallic: 0, roughness: 1 },
        billboardMatrix(ex.position, 26 + t * 68, camera), vp, camera, time,
        { blended: true, doubleSided: true });
    }
    gl.depthMask(true);
    gl.disable(gl.BLEND);

    // Speed blur for the post pass, from the settings slider and the
    // airspeed over the blur threshold.
    const blurSetting = state.settings && state.settings.blur !== undefined && state.settings.blur !== null
      ? state.settings.blur : 0.45;
    const speedFactor = Math.max(0, Math.min(1, (f.speed - 88) / 157));
    const blurStrength = (state.phase === 'play' || state.phase === 'return')
      ? blurSetting * speedFactor * 0.026 * (1 + Math.abs(f.bank) * 0.35)
      : 0;
    presentScene(blurStrength);

    drawHud(state, vp, camera, cam);
  }

  // Loads the three effect textures (explosion flipbook sheet, white and
  // black smoke) as canvases the texture cache can upload. Called once by
  // main.js before the first frame.
  async function ready() {
    const sources = {
      explosion: 'fx/explosion-sheet.png',
      smokeWhite: 'fx/smoke-white.png',
      smokeBlack: 'fx/smoke-black.png',
    };
    await Promise.all(Object.entries(sources).map(async ([key, path]) => {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`Could not load ${path}`);
      const bitmap = await createImageBitmap(await response.blob());
      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext('2d').drawImage(bitmap, 0, 0);
      if (bitmap.close) bitmap.close();
      fxTextures[key] = canvas;
    }));
  }

  return { render, ready, setAircraft, setTerrain, cameraFor, gl };
}
