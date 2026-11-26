// Small linear-algebra kit for the renderer and the glTF loader.
// Vectors are plain [x, y, z] arrays. Matrices are column-major Float32Array(16),
// the layout WebGL and glTF both expect, so elements 12..14 hold the translation.

// ---- vectors ---------------------------------------------------------------

export const v3 = (x = 0, y = 0, z = 0) => [x, y, z];

export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];

export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

export const length = (a) => Math.hypot(a[0], a[1], a[2]);

export function normalize(a) {
  const len = length(a) || 1; // a zero vector normalizes to itself rather than NaNs
  return scale(a, 1 / len);
}

/** Rotate a point around the Y axis by `angle` radians. */
export function rotateY(p, angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c];
}

// ---- matrices --------------------------------------------------------------

export function identity() {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

/** Matrix product a * b (b is applied to vectors first). Rows unrolled: this
 *  runs for every scene node on every frame, so it earns the verbosity. */
export function multiply(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    const b0 = b[col * 4];
    const b1 = b[col * 4 + 1];
    const b2 = b[col * 4 + 2];
    const b3 = b[col * 4 + 3];
    out[col * 4] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
    out[col * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
    out[col * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
    out[col * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
  }
  return out;
}

export function translation(x, y, z) {
  const m = identity();
  m[12] = x;
  m[13] = y;
  m[14] = z;
  return m;
}

export function scaling(x, y, z) {
  const m = identity();
  m[0] = x;
  m[5] = y;
  m[10] = z;
  return m;
}

/** Rotation matrix for a quaternion given as [x, y, z, w]. */
export function quatMatrix(q) {
  const [x, y, z, w] = q;
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  return new Float32Array([
    1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy), 0,
    2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx), 0,
    2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy), 0,
    0, 0, 0, 1,
  ]);
}

/** Build a TRS matrix: translation * rotation * scale. */
export function compose(t = [0, 0, 0], q = [0, 0, 0, 1], s = [1, 1, 1]) {
  return multiply(translation(...t), multiply(quatMatrix(q), scaling(...s)));
}

export function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  const depth = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * depth;
  m[11] = -1;
  m[14] = 2 * far * near * depth;
  return m;
}

export function lookAt(eye, center, up) {
  const zAxis = normalize(sub(eye, center)); // camera looks down its -Z
  const xAxis = normalize(cross(up, zAxis));
  const yAxis = cross(zAxis, xAxis);
  return new Float32Array([
    xAxis[0], yAxis[0], zAxis[0], 0,
    xAxis[1], yAxis[1], zAxis[1], 0,
    xAxis[2], yAxis[2], zAxis[2], 0,
    -dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1,
  ]);
}

/** Apply matrix m to point p (assumes the w row is 0,0,0,1). */
export function transformPoint(m, p) {
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
  ];
}
