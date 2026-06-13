#!/usr/bin/env node
/**
 * generate-chair.mjs
 * Produces chair.glb (GLTF 2.0 binary) from scratch — zero npm dependencies.
 * Run with: node generate-chair.mjs
 */
import { writeFileSync } from 'fs';

// ── Quaternion from Euler angles (intrinsic XYZ, radians) ──────────────────
function euler(rx, ry, rz) {
  const cx = Math.cos(rx / 2), sx = Math.sin(rx / 2);
  const cy = Math.cos(ry / 2), sy = Math.sin(ry / 2);
  const cz = Math.cos(rz / 2), sz = Math.sin(rz / 2);
  return [
    sx * cy * cz + cx * sy * sz,
    cx * sy * cz - sx * cy * sz,
    cx * cy * sz + sx * sy * cz,
    cx * cy * cz - sx * sy * sz,
  ]; // [x, y, z, w]
}

const PI2 = Math.PI / 2;
const IDENT = [0, 0, 0, 1];

// ── Box geometry (Y-up, centered at origin) ────────────────────────────────
function box(w, h, d) {
  const hw = w / 2, hh = h / 2, hd = d / 2;
  // 6 faces × 4 vertices (unique per-face normals)
  const pos = [
    // +Z face (0-3)
    -hw, -hh, hd,   hw, -hh, hd,   hw,  hh, hd,  -hw,  hh, hd,
    // -Z face (4-7)
     hw, -hh,-hd,  -hw, -hh,-hd,  -hw,  hh,-hd,   hw,  hh,-hd,
    // -X face (8-11)
    -hw, -hh,-hd,  -hw, -hh, hd,  -hw,  hh, hd,  -hw,  hh,-hd,
    // +X face (12-15)
     hw, -hh, hd,   hw, -hh,-hd,   hw,  hh,-hd,   hw,  hh, hd,
    // +Y face (16-19)
    -hw,  hh, hd,   hw,  hh, hd,   hw,  hh,-hd,  -hw,  hh,-hd,
    // -Y face (20-23)
    -hw, -hh,-hd,   hw, -hh,-hd,   hw, -hh, hd,  -hw, -hh, hd,
  ];
  const faceNormals = [[0,0,1],[0,0,-1],[-1,0,0],[1,0,0],[0,1,0],[0,-1,0]];
  const nor = [];
  faceNormals.forEach(n => { for (let i = 0; i < 4; i++) nor.push(...n); });
  const idx = [];
  // CCW winding (verified per face)
  for (let f = 0; f < 6; f++) {
    const b = f * 4;
    idx.push(b, b + 1, b + 2,  b, b + 2, b + 3);
  }
  return {
    pos: new Float32Array(pos),
    nor: new Float32Array(nor),
    idx: new Uint16Array(idx),
  };
}

// ── Cylinder geometry (axis = Y, centered at origin) ──────────────────────
function cylinder(rTop, rBot, h, segs = 10) {
  const pos = [], nor = [], idx = [];
  const hh = h / 2;
  const slope = (rBot - rTop) / h;
  const nY   = -slope / Math.sqrt(1 + slope * slope);
  const nXZ  =      1 / Math.sqrt(1 + slope * slope);

  // Side vertices: pairs (top, bottom) for each angle step
  for (let i = 0; i <= segs; i++) {
    const a = (i / segs) * Math.PI * 2;
    const c = Math.cos(a), s = Math.sin(a);
    pos.push(rTop * c, hh, rTop * s);   // top ring
    pos.push(rBot * c, -hh, rBot * s);  // bottom ring
    nor.push(nXZ * c, nY, nXZ * s);
    nor.push(nXZ * c, nY, nXZ * s);
  }
  // Side indices (CCW from outside)
  for (let i = 0; i < segs; i++) {
    const b = i * 2;
    idx.push(b, b + 2, b + 1,   b + 1, b + 2, b + 3);
  }

  // Top cap (normal = +Y)
  const tc = pos.length / 3;
  pos.push(0, hh, 0); nor.push(0, 1, 0);
  for (let i = 0; i < segs; i++) {
    const a = (i / segs) * Math.PI * 2;
    pos.push(rTop * Math.cos(a), hh, rTop * Math.sin(a));
    nor.push(0, 1, 0);
  }
  // CCW from +Y: reverse order
  for (let i = 0; i < segs; i++) {
    idx.push(tc, tc + 1 + (i + 1) % segs, tc + 1 + i);
  }

  // Bottom cap (normal = -Y)
  const bc = pos.length / 3;
  pos.push(0, -hh, 0); nor.push(0, -1, 0);
  for (let i = 0; i < segs; i++) {
    const a = (i / segs) * Math.PI * 2;
    pos.push(rBot * Math.cos(a), -hh, rBot * Math.sin(a));
    nor.push(0, -1, 0);
  }
  // CCW from -Y
  for (let i = 0; i < segs; i++) {
    idx.push(bc, bc + 1 + i, bc + 1 + (i + 1) % segs);
  }

  return {
    pos: new Float32Array(pos),
    nor: new Float32Array(nor),
    idx: new Uint16Array(idx),
  };
}

// ── GLB builder ────────────────────────────────────────────────────────────
class GLBBuilder {
  constructor() {
    this._mats     = [];
    this._meshes   = [];
    this._nodes    = [];
    this._accs     = [];
    this._views    = [];
    this._bufParts = [];
    this._byteOff  = 0;
  }

  addMaterial(name, color, roughness = 0.8, metallic = 0.0) {
    this._mats.push({
      name,
      pbrMetallicRoughness: {
        baseColorFactor: color,
        roughnessFactor:  roughness,
        metallicFactor:   metallic,
      },
    });
    return this._mats.length - 1;
  }

  // Push raw bytes; pad to 4-byte boundary; record bufferView
  _push(typedArray, target) {
    const bytes = new Uint8Array(
      typedArray.buffer,
      typedArray.byteOffset,
      typedArray.byteLength
    );
    const pad = (4 - bytes.byteLength % 4) % 4;
    const buf = new Uint8Array(bytes.byteLength + pad);
    buf.set(bytes);

    const bv = { byteOffset: this._byteOff, byteLength: bytes.byteLength, target };
    this._views.push(bv);
    this._bufParts.push(buf);
    this._byteOff += buf.byteLength;
    return this._views.length - 1;
  }

  addNode(geo, matIdx, translation = [0, 0, 0], rotation = IDENT) {
    const pvBV = this._push(geo.pos, 34962); // ARRAY_BUFFER
    const nvBV = this._push(geo.nor, 34962);
    const iBV  = this._push(geo.idx, 34963); // ELEMENT_ARRAY_BUFFER

    // Bounding box for POSITION (required by spec)
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < geo.pos.length; i += 3) {
      for (let j = 0; j < 3; j++) {
        if (geo.pos[i + j] < min[j]) min[j] = geo.pos[i + j];
        if (geo.pos[i + j] > max[j]) max[j] = geo.pos[i + j];
      }
    }

    const posAcc = this._accs.length;
    this._accs.push({
      bufferView: pvBV, componentType: 5126 /* FLOAT */,
      count: geo.pos.length / 3, type: 'VEC3', min, max,
    });
    this._accs.push({
      bufferView: nvBV, componentType: 5126,
      count: geo.nor.length / 3, type: 'VEC3',
    });
    this._accs.push({
      bufferView: iBV, componentType: 5123 /* UNSIGNED_SHORT */,
      count: geo.idx.length, type: 'SCALAR',
    });

    const meshIdx = this._meshes.length;
    this._meshes.push({
      primitives: [{
        attributes: { POSITION: posAcc, NORMAL: posAcc + 1 },
        indices: posAcc + 2,
        material: matIdx,
      }],
    });

    const node = { mesh: meshIdx, translation };
    if (rotation !== IDENT) node.rotation = rotation;
    this._nodes.push(node);
  }

  build() {
    // Assemble binary buffer
    const totalBin = this._byteOff;
    const binBuf = new Uint8Array(totalBin);
    let off = 0;
    for (const part of this._bufParts) { binBuf.set(part, off); off += part.byteLength; }

    const json = {
      asset: { version: '2.0', generator: 'AR Chair Builder' },
      scene: 0,
      scenes: [{ name: 'Chair', nodes: this._nodes.map((_, i) => i) }],
      nodes: this._nodes,
      meshes: this._meshes,
      materials: this._mats,
      accessors: this._accs,
      bufferViews: this._views.map(v => ({
        byteOffset: v.byteOffset,
        byteLength: v.byteLength,
        target: v.target,
      })),
      buffers: [{ byteLength: totalBin }],
    };

    // JSON chunk (padded with spaces to 4-byte boundary)
    const jsonStr  = JSON.stringify(json);
    const jsonPad  = (4 - jsonStr.length % 4) % 4;
    const jsonFull = jsonStr + ' '.repeat(jsonPad);
    const jsonBuf  = Buffer.from(jsonFull, 'utf8');

    const totalLen = 12 + 8 + jsonBuf.length + 8 + totalBin;
    const out = Buffer.alloc(totalLen);
    let p = 0;

    // GLB Header
    out.writeUInt32LE(0x46546C67, p); p += 4; // magic "glTF"
    out.writeUInt32LE(2,          p); p += 4; // version 2
    out.writeUInt32LE(totalLen,   p); p += 4;

    // JSON chunk
    out.writeUInt32LE(jsonBuf.length, p); p += 4;
    out.writeUInt32LE(0x4E4F534A,     p); p += 4; // "JSON"
    jsonBuf.copy(out, p);  p += jsonBuf.length;

    // BIN chunk
    out.writeUInt32LE(totalBin,   p); p += 4;
    out.writeUInt32LE(0x004E4942, p); p += 4; // "BIN\0"
    Buffer.from(binBuf).copy(out, p);

    return out;
  }
}

// ── Chair assembly ─────────────────────────────────────────────────────────
const b    = new GLBBuilder();
const WOOD = b.addMaterial('wood',   [0.484, 0.290, 0.118, 1.0], 0.80, 0.0);
const FABR = b.addMaterial('fabric', [0.173, 0.227, 0.290, 1.0], 0.95, 0.0);

// Seat frame + cushion
b.addNode(box(0.46, 0.050, 0.46),  WOOD, [0, 0.440, 0]);
b.addNode(box(0.42, 0.065, 0.42),  FABR, [0, 0.505, 0]);

// Backrest frame + cushion
b.addNode(box(0.46, 0.500, 0.045), WOOD, [0,  0.730, -0.210]);
b.addNode(box(0.41, 0.440, 0.040), FABR, [0,  0.730, -0.190]);

// Armrests + supports
b.addNode(box(0.040, 0.038, 0.38), WOOD, [-0.235, 0.600, 0.020]);
b.addNode(box(0.040, 0.038, 0.38), WOOD, [ 0.235, 0.600, 0.020]);
b.addNode(cylinder(0.018, 0.018, 0.14), WOOD, [-0.235, 0.535, 0.190]);
b.addNode(cylinder(0.018, 0.018, 0.14), WOOD, [ 0.235, 0.535, 0.190]);

// Legs
const LEG = cylinder(0.026, 0.020, 0.44);
b.addNode(LEG, WOOD, [-0.190, 0.220, -0.190]);
b.addNode(LEG, WOOD, [ 0.190, 0.220, -0.190]);
b.addNode(LEG, WOOD, [-0.190, 0.220,  0.190]);
b.addNode(LEG, WOOD, [ 0.190, 0.220,  0.190]);

// Stretchers along Z (rotate cylinder 90° around X so it lies along Z)
const rotX = euler(PI2, 0, 0);
const STRZ = cylinder(0.018, 0.018, 0.37);
b.addNode(STRZ, WOOD, [-0.190, 0.120, 0], rotX);
b.addNode(STRZ, WOOD, [ 0.190, 0.120, 0], rotX);

// Stretchers along X (rotate cylinder 90° around Z so it lies along X)
const rotZ = euler(0, 0, PI2);
const STRX = cylinder(0.018, 0.018, 0.37);
b.addNode(STRX, WOOD, [0, 0.120, -0.190], rotZ);
b.addNode(STRX, WOOD, [0, 0.120,  0.190], rotZ);

writeFileSync('chair.glb', b.build());
console.log('✓ chair.glb written');
