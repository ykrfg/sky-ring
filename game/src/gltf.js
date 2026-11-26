// A small glTF 2.0 binary (GLB) loader, written for the two models this game
// ships: the FX-27 aircraft and the island terrain. It covers exactly what a
// static scene needs — chunk parsing, accessor decoding (interleaved and
// normalized data included), the node tree with TRS transforms, PBR material
// factors with their texture images decoded to canvases, and animation
// channels with readable sampler tracks. Skins, cameras, lights, extensions
// and external (non-embedded) resources are out of scope by design.
//
// Data shapes handed to the renderer:
//   model     { url, doc, nodes, meshes, materials, textures, animations }
//   accessor  { values, count, components } — values is a typed array
//   geometry  { mode, material, position, normal, uv, indices }
//             (normal / uv / indices are accessors or null)
//   material  { name, color, metallic, roughness, base, orm, normal, alphaMode }
//             (base / orm / normal are canvases or null)
//   node      { name, mesh, children, translation, rotation, scale, local, world }
//             (mesh is a mesh index or -1; local / world are column-major
//              Float32Array(16), matching math.js)
//   animation { name, channels: [{ node, path, sampler }] } with
//             sampler = { interpolation, input, output } as accessors
//
// The terrain helpers at the bottom prepare the island mesh for play: they
// bake its placement into the vertices, level a shelf for the runway, and
// derive a height grid plus baked ambient occlusion from the result.

import { compose, identity, multiply } from './math.js';

// ---------------------------------------------------------------------------
// Accessor decoding
// ---------------------------------------------------------------------------

// One row per glTF componentType: byte width, the DataView reader, and the
// typed array a tightly packed, non-normalized accessor decodes into.
const COMPONENT_TYPES = {
  5120: { bytes: 1, read: 'getInt8', Array: Int8Array },
  5121: { bytes: 1, read: 'getUint8', Array: Uint8Array },
  5122: { bytes: 2, read: 'getInt16', Array: Int16Array },
  5123: { bytes: 2, read: 'getUint16', Array: Uint16Array },
  5125: { bytes: 4, read: 'getUint32', Array: Uint32Array },
  5126: { bytes: 4, read: 'getFloat32', Array: Float32Array },
};

const TYPE_COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };

// Signed normalized integers bottom out at -1, not at -max (glTF spec).
function denormalize(value, componentType) {
  switch (componentType) {
    case 5120: return Math.max(value / 127, -1);
    case 5121: return value / 255;
    case 5122: return Math.max(value / 32767, -1);
    case 5123: return value / 65535;
    default: return value;
  }
}

// Reads accessor `index` out of the binary chunk that starts at absolute byte
// `binStart` inside `buffer`. Elements are visited one by one so interleaved
// bufferViews (byteStride larger than the packed size) decode correctly. Two
// storage corners are handled as well: an accessor with no bufferView reads
// as all zeros, and a sparse accessor overlays its stored elements on top of
// the base data.
function readAccessor(doc, buffer, binStart, index) {
  const accessor = doc.accessors[index];
  const kind = COMPONENT_TYPES[accessor.componentType];
  const components = TYPE_COMPONENTS[accessor.type];
  const floats = accessor.normalized && accessor.componentType !== 5126;
  const values = new (floats ? Float32Array : kind.Array)(accessor.count * components);

  if (accessor.bufferView !== undefined) {
    const view = doc.bufferViews[accessor.bufferView];
    const elementBytes = kind.bytes * components;
    const stride = view.byteStride || elementBytes;
    const base = binStart + (view.byteOffset || 0) + (accessor.byteOffset || 0);
    const reader = new DataView(buffer, base, accessor.count * stride - (stride - elementBytes));
    for (let i = 0; i < accessor.count; i++) {
      const at = i * stride;
      for (let c = 0; c < components; c++) {
        const raw = reader[kind.read](at + c * kind.bytes, true);
        values[i * components + c] = floats ? denormalize(raw, accessor.componentType) : raw;
      }
    }
  }

  if (accessor.sparse) {
    const spots = readAccessor(doc, buffer, binStart, accessor.sparse.indices);
    const edits = readAccessor(doc, buffer, binStart, accessor.sparse.values);
    for (let i = 0; i < accessor.sparse.count; i++) {
      const target = spots.values[i] * components;
      for (let c = 0; c < components; c++) values[target + c] = edits.values[i * components + c];
    }
  }
  return { values, count: accessor.count, components };
}

// ---------------------------------------------------------------------------
// Texture images
// ---------------------------------------------------------------------------

// Embedded image bytes become a canvas the renderer can upload as a texture.
// createImageBitmap accepts the Blob directly, so no object URL round-trip.
async function decodeImage(bytes, mimeType) {
  const bitmap = await createImageBitmap(new Blob([bytes], { type: mimeType || 'image/png' }));
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext('2d').drawImage(bitmap, 0, 0);
  if (bitmap.close) bitmap.close();
  return canvas;
}

// ---------------------------------------------------------------------------
// The loader
// ---------------------------------------------------------------------------

const GLB_MAGIC = 0x46546c67; // 'glTF'
const CHUNK_JSON = 0x4e4f534a; // 'JSON'
const CHUNK_BIN = 0x004e4942; // 'BIN\0'

const FALLBACK_MATERIAL = {
  name: '', color: [0.72, 0.72, 0.72, 1], metallic: 0, roughness: 0.8,
  base: null, orm: null, normal: null, alphaMode: 'OPAQUE',
};

export async function loadGLB(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`loadGLB: ${url} answered ${response.status}`);
  const buffer = await response.arrayBuffer();

  // Walk the container: a 12-byte header, then length/type-prefixed chunks.
  const header = new DataView(buffer);
  if (header.getUint32(0, true) !== GLB_MAGIC || header.getUint32(4, true) !== 2) {
    throw new Error(`loadGLB: ${url} is not a glTF 2.0 binary`);
  }
  let doc = null;
  let binStart = -1;
  let binLength = 0;
  for (let at = 12; at + 8 <= buffer.byteLength;) {
    const length = header.getUint32(at, true);
    const type = header.getUint32(at + 4, true);
    if (type === CHUNK_JSON) {
      const text = new TextDecoder().decode(new Uint8Array(buffer, at + 8, length));
      doc = JSON.parse(text.replace(/\0+$/, '').trim());
    } else if (type === CHUNK_BIN) {
      binStart = at + 8;
      binLength = length;
    }
    at += 8 + length;
  }
  if (!doc) throw new Error(`loadGLB: ${url} has no JSON chunk`);
  if (binStart < 0) throw new Error(`loadGLB: ${url} has no binary chunk`);

  const read = (i) => readAccessor(doc, buffer, binStart, i);
  const chunkBytes = (viewIndex) => {
    const view = doc.bufferViews[viewIndex];
    const start = binStart + (view.byteOffset || 0);
    return new Uint8Array(buffer, start, view.byteLength);
  };

  // Textures: decode every embedded image first, then let materials point at
  // the canvas for their texture's source image.
  const images = await Promise.all((doc.images || []).map((image) => {
    if (image.bufferView === undefined) {
      throw new Error(`loadGLB: ${url} references a non-embedded image`);
    }
    return decodeImage(chunkBytes(image.bufferView), image.mimeType);
  }));
  const textureCanvas = (texInfo) => {
    if (!texInfo) return null;
    const texture = doc.textures[texInfo.index];
    return images[texture.source] || null;
  };

  const materials = (doc.materials || []).map((m) => {
    const pbr = m.pbrMetallicRoughness || {};
    return {
      name: m.name || '',
      color: pbr.baseColorFactor || [1, 1, 1, 1],
      metallic: pbr.metallicFactor !== undefined ? pbr.metallicFactor : 1,
      roughness: pbr.roughnessFactor !== undefined ? pbr.roughnessFactor : 1,
      base: textureCanvas(pbr.baseColorTexture),
      orm: textureCanvas(pbr.metallicRoughnessTexture),
      normal: textureCanvas(m.normalTexture),
      alphaMode: m.alphaMode || 'OPAQUE',
    };
  });

  const meshes = (doc.meshes || []).map((mesh) => (mesh.primitives || []).map((prim) => ({
    mode: prim.mode !== undefined ? prim.mode : 4, // 4 = TRIANGLES
    material: prim.material !== undefined ? materials[prim.material] : FALLBACK_MATERIAL,
    position: read(prim.attributes.POSITION),
    normal: prim.attributes.NORMAL !== undefined ? read(prim.attributes.NORMAL) : null,
    uv: prim.attributes.TEXCOORD_0 !== undefined ? read(prim.attributes.TEXCOORD_0) : null,
    indices: prim.indices !== undefined ? read(prim.indices) : null,
  })));

  const animations = (doc.animations || []).map((clip) => ({
    name: clip.name || '',
    channels: (clip.channels || []).map((channel) => {
      const sampler = clip.samplers[channel.sampler];
      return {
        node: channel.target.node,
        path: channel.target.path,
        sampler: {
          interpolation: sampler.interpolation || 'LINEAR',
          input: read(sampler.input),
          output: read(sampler.output),
        },
      };
    }),
  }));

  // Nodes keep both their own TRS matrix (`local`) and the scene-composed
  // `world`, filled in by a walk from the scene roots below.
  const nodes = (doc.nodes || []).map((n) => ({
    name: n.name || '',
    mesh: n.mesh !== undefined ? n.mesh : -1,
    children: n.children || [],
    translation: n.translation || [0, 0, 0],
    rotation: n.rotation || [0, 0, 0, 1],
    scale: n.scale || [1, 1, 1],
    local: n.matrix ? Float32Array.from(n.matrix) : compose(n.translation, n.rotation, n.scale),
    world: identity(),
  }));
  const scene = (doc.scenes || [])[doc.scene || 0] || { nodes: [] };
  const visit = (index, parentWorld) => {
    const node = nodes[index];
    node.world = multiply(parentWorld, node.local);
    for (const child of node.children) visit(child, node.world);
  };
  for (const root of scene.nodes) visit(root, identity());

  return { url, doc, nodes, meshes, materials, textures: images, animations };
}

// ---------------------------------------------------------------------------
// Terrain preparation
// ---------------------------------------------------------------------------

// Rebuilds a geometry's normals from its triangles. Vertices that share a
// position (within a centimetre) are welded first, so UV seams do not turn
// into shading seams. With `original` + `weights`, each rebuilt normal is
// blended back toward the authored one by the vertex's weight — the runway
// shelf gets smooth shading while the untouched island keeps its look.
export function recalculateNormals(geometry, options = {}) {
  const pos = geometry.position.values;
  const count = geometry.position.count;
  const index = geometry.indices ? geometry.indices.values : null;

  const buckets = new Map();
  const keyOf = (i) => `${Math.round(pos[i * 3] * 100)},${Math.round(pos[i * 3 + 1] * 100)},${Math.round(pos[i * 3 + 2] * 100)}`;
  const accumulate = (i, nx, ny, nz) => {
    const key = keyOf(i);
    let sum = buckets.get(key);
    if (!sum) { sum = [0, 0, 0]; buckets.set(key, sum); }
    sum[0] += nx; sum[1] += ny; sum[2] += nz;
  };
  const face = (a, b, c) => {
    const ux = pos[b * 3] - pos[a * 3], uy = pos[b * 3 + 1] - pos[a * 3 + 1], uz = pos[b * 3 + 2] - pos[a * 3 + 2];
    const vx = pos[c * 3] - pos[a * 3], vy = pos[c * 3 + 1] - pos[a * 3 + 1], vz = pos[c * 3 + 2] - pos[a * 3 + 2];
    // The unnormalized cross product doubles as an area weight.
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    accumulate(a, nx, ny, nz); accumulate(b, nx, ny, nz); accumulate(c, nx, ny, nz);
  };
  if (index) for (let i = 0; i < index.length; i += 3) face(index[i], index[i + 1], index[i + 2]);
  else for (let i = 0; i + 2 < count; i += 3) face(i, i + 1, i + 2);

  const normals = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const sum = buckets.get(keyOf(i));
    let nx = sum ? sum[0] : 0, ny = sum ? sum[1] : 1, nz = sum ? sum[2] : 0;
    if (options.original && options.weights) {
      const w = options.weights[i];
      nx = options.original[i * 3] * (1 - w) + nx * w;
      ny = options.original[i * 3 + 1] * (1 - w) + ny * w;
      nz = options.original[i * 3 + 2] * (1 - w) + nz * w;
    }
    const len = Math.hypot(nx, ny, nz) || 1;
    normals[i * 3] = nx / len; normals[i * 3 + 1] = ny / len; normals[i * 3 + 2] = nz / len;
  }
  geometry.normal = { values: normals, count, components: 3 };
  return geometry;
}

// Bakes placement (scale, then offset) into the terrain's vertices and levels
// a rectangular shelf for the runway: inside the rectangle the ground sits at
// cut.height, and a smoothstep falloff of cut.blend metres blends the shelf
// back into the natural slope. `cut` is { centerX, centerZ, halfWidth,
// halfLength, height, blend } in world coordinates. Mutates and returns the
// terrain's first primitive.
export function prepareTerrain(model, offset, scale, cut) {
  const mesh = model.meshes[0][0];
  const pos = mesh.position.values;
  const authored = mesh.normal ? Float32Array.from(mesh.normal.values) : null;
  const weights = new Float32Array(mesh.position.count);
  for (let i = 0; i < mesh.position.count; i++) {
    const x = pos[i * 3] * scale + offset[0];
    const y = pos[i * 3 + 1] * scale + offset[1];
    const z = pos[i * 3 + 2] * scale + offset[2];
    // Distance outside the shelf rectangle along each axis, eased to 0..1.
    const edgeX = Math.max(0, Math.min(1, (cut.halfWidth + cut.blend - Math.abs(x - cut.centerX)) / cut.blend));
    const edgeZ = Math.max(0, Math.min(1, (cut.halfLength + cut.blend - Math.abs(z - cut.centerZ)) / cut.blend));
    const w = Math.min(edgeX, edgeZ);
    const eased = w * w * (3 - 2 * w);
    weights[i] = eased;
    pos[i * 3] = x;
    pos[i * 3 + 1] = y + (cut.height - y) * eased;
    pos[i * 3 + 2] = z;
  }
  if (authored) recalculateNormals(mesh, { original: authored, weights });
  else recalculateNormals(mesh);
  return mesh;
}

// Rasterizes a (prepared) terrain mesh into a square height grid by stamping
// every triangle's interpolated height into the cells it covers, keeping the
// highest stamp. Positions are expected in world space already — `offset` and
// `scale` are accepted for symmetry with prepareTerrain, and only offset[1]
// is used, as the height empty cells start from.
export function buildTerrainGrid(mesh, offset = [0, 0, 0], scale = 1, size = 160) {
  const pos = mesh.position.values;
  const count = mesh.position.count;
  const index = mesh.indices ? mesh.indices.values : null;

  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (let i = 0; i < count; i++) {
    if (pos[i * 3] < minX) minX = pos[i * 3];
    if (pos[i * 3] > maxX) maxX = pos[i * 3];
    if (pos[i * 3 + 2] < minZ) minZ = pos[i * 3 + 2];
    if (pos[i * 3 + 2] > maxZ) maxZ = pos[i * 3 + 2];
  }
  const cellX = (maxX - minX) / (size - 1);
  const cellZ = (maxZ - minZ) / (size - 1);
  const heights = new Float32Array(size * size).fill(offset[1]);

  const stamp = (a, b, c) => {
    const ax = pos[a * 3], ay = pos[a * 3 + 1], az = pos[a * 3 + 2];
    const bx = pos[b * 3], by = pos[b * 3 + 1], bz = pos[b * 3 + 2];
    const cx = pos[c * 3], cy = pos[c * 3 + 1], cz = pos[c * 3 + 2];
    const denom = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz);
    if (Math.abs(denom) < 1e-8) return; // vertical in plan view: no footprint
    const gx0 = Math.max(0, Math.floor((Math.min(ax, bx, cx) - minX) / cellX));
    const gx1 = Math.min(size - 1, Math.ceil((Math.max(ax, bx, cx) - minX) / cellX));
    const gz0 = Math.max(0, Math.floor((Math.min(az, bz, cz) - minZ) / cellZ));
    const gz1 = Math.min(size - 1, Math.ceil((Math.max(az, bz, cz) - minZ) / cellZ));
    for (let gz = gz0; gz <= gz1; gz++) {
      for (let gx = gx0; gx <= gx1; gx++) {
        const px = minX + gx * cellX, pz = minZ + gz * cellZ;
        const u = ((bz - cz) * (px - cx) + (cx - bx) * (pz - cz)) / denom;
        const v = ((cz - az) * (px - cx) + (ax - cx) * (pz - cz)) / denom;
        const w = 1 - u - v;
        if (u < -1e-4 || v < -1e-4 || w < -1e-4) continue;
        const h = u * ay + v * by + w * cy;
        const cell = gz * size + gx;
        if (h > heights[cell]) heights[cell] = h;
      }
    }
  };
  if (index) for (let i = 0; i < index.length; i += 3) stamp(index[i], index[i + 1], index[i + 2]);
  else for (let i = 0; i + 2 < count; i += 3) stamp(i, i + 1, i + 2);

  return {
    size, heights, minX, minZ, maxX, maxZ, cellX, cellZ,
    // Bilinear height lookup; off-island queries read as sea level (0).
    heightAt(x, z) {
      const fx = (x - minX) / cellX, fz = (z - minZ) / cellZ;
      if (fx < 0 || fz < 0 || fx > size - 1 || fz > size - 1) return 0;
      const ix = Math.min(size - 2, Math.floor(fx)), iz = Math.min(size - 2, Math.floor(fz));
      const tx = fx - ix, tz = fz - iz;
      const a = heights[iz * size + ix], b = heights[iz * size + ix + 1];
      const c = heights[(iz + 1) * size + ix], d = heights[(iz + 1) * size + ix + 1];
      return (a * (1 - tx) + b * tx) * (1 - tz) + (c * (1 - tx) + d * tx) * tz;
    },
  };
}

// Bakes a per-vertex light/visibility factor for the terrain: a coarse march
// toward the sun against the height grid (a ridge between the vertex and the
// sun dims it), multiplied by a horizon term from eight compass directions.
// The result rides along as a 1-component attribute the shader multiplies in.
export function buildTerrainAO(mesh, grid, sunDir) {
  const pos = mesh.position.values;
  const count = mesh.position.count;
  const values = new Float32Array(count);
  const sunLen = Math.hypot(sunDir[0], sunDir[1], sunDir[2]) || 1;
  const sx = sunDir[0] / sunLen, sy = sunDir[1] / sunLen, sz = sunDir[2] / sunLen;
  for (let i = 0; i < count; i++) {
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
    let sun = 1;
    for (let t = 18; t <= 420; t += 26) {
      if (grid.heightAt(x + sx * t, z + sz * t) > y + sy * t + 2) { sun = 0.64; break; }
    }
    let blocked = 0, samples = 0;
    for (let k = 0; k < 8; k++) {
      const angle = (k / 8) * Math.PI * 2;
      for (const radius of [32, 76]) {
        const h = grid.heightAt(x + Math.cos(angle) * radius, z + Math.sin(angle) * radius);
        blocked += Math.max(0, Math.min(1, (h - y) / 58));
        samples++;
      }
    }
    const sky = 1 - (blocked / Math.max(1, samples)) * 0.55;
    values[i] = Math.max(0.55, Math.min(1, sun * sky));
  }
  return { values, count, components: 1 };
}
