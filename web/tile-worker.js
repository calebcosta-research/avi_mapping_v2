// ============================================================
//  tile-worker.js — Avalanche tile colorizer (Web Worker)
//
//  Receives identity tiles from R2 (R=zone_index, G=cell_id,
//  A=255 if avi terrain) and applies a pre-built LUT to produce
//  colored PNG tiles. Runs off the main thread so tile loading
//  never blocks the UI.
//
//  Messages IN:
//    { type: 'INIT_LUTS', danger, problem, likelihood }
//      — Uint32Array buffers (transferred), one per color mode.
//        Each array is 256*256 entries: lut[zoneIndex*256+cellId]
//        = packed RGBA Uint32 (little-endian: R at byte 0).
//
//    { type: 'FETCH_TILE', requestId, url, mode }
//      — Fetch the identity PNG at `url`, apply lut for `mode`,
//        return colored PNG bytes as ArrayBuffer.
//
//  Messages OUT:
//    { type: 'TILE_DONE',  requestId, buffer } — ArrayBuffer (transferred, PNG bytes)
//    { type: 'TILE_ERROR', requestId, error }  — string
// ============================================================

'use strict';

// ── STATE ──────────────────────────────────────────────────
const luts = {
  danger:     null,   // Uint32Array[65536]
  problem:    null,
  likelihood: null,
};

// ── MESSAGE HANDLER ────────────────────────────────────────
self.onmessage = async function (e) {
  const msg = e.data;

  if (msg.type === 'INIT_LUTS') {
    luts.danger     = msg.danger;       // Uint32Array (transferred)
    luts.problem    = msg.problem;
    luts.likelihood = msg.likelihood;
    return;
  }

  if (msg.type === 'FETCH_TILE') {
    const { requestId, url, mode } = msg;
    try {
      const buffer = await fetchAndColor(url, mode);
      self.postMessage({ type: 'TILE_DONE', requestId, buffer }, [buffer]);
    } catch (err) {
      self.postMessage({ type: 'TILE_ERROR', requestId, error: String(err) });
    }
  }
};

// ── CORE ───────────────────────────────────────────────────
async function fetchAndColor(url, mode) {
  const lut = luts[mode];

  // Fetch identity tile PNG from R2
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);

  const blob = await resp.blob();
  const bmp  = await createImageBitmap(blob);

  // Save dimensions BEFORE closing the bitmap
  const W = bmp.width;
  const H = bmp.height;

  const canvas = new OffscreenCanvas(W, H);
  const ctx    = canvas.getContext('2d');
  ctx.drawImage(bmp, 0, 0);
  bmp.close();  // free GPU memory

  const src  = ctx.getImageData(0, 0, W, H).data;   // Uint8ClampedArray
  const out  = new Uint8ClampedArray(W * H * 4);     // output pixels
  const u32  = new Uint32Array(out.buffer);           // Uint32 view for fast writes

  if (lut) {
    for (let i = 0, n = W * H; i < n; i++) {
      const base  = i * 4;
      if (src[base + 3] === 0) continue;          // transparent → leave 0
      const zone = src[base + 0];                 // R = zone_index
      const cell = src[base + 1];                 // G = cell_id
      u32[i] = lut[zone * 256 + cell];            // single LUT lookup
    }
  }
  // If lut not ready: return transparent tile (shouldn't happen with correct init order)

  // Feather edges at max zoom (z10) — 2-pass box blur smooths jagged polygon boundaries
  const zoom = extractZoom(url);
  if (zoom >= 10) {
    boxBlur(out, W, H);
    boxBlur(out, W, H);
  }

  // Encode as PNG and return ArrayBuffer (MapLibre addProtocol expects PNG/JPEG bytes)
  const outCanvas = new OffscreenCanvas(W, H);
  outCanvas.getContext('2d').putImageData(new ImageData(out, W, H), 0, 0);
  const pngBlob = await outCanvas.convertToBlob({ type: 'image/png' });
  return pngBlob.arrayBuffer();
}

// ── HELPERS ────────────────────────────────────────────────

function extractZoom(url) {
  // https://domain/tiles/identity/10/203/390.png  → 10
  const m = url.match(/\/(\d+)\/\d+\/\d+\.png$/);
  return m ? parseInt(m[1], 10) : 0;
}

function boxBlur(data, W, H) {
  const src = new Uint8ClampedArray(data);  // snapshot before modifying
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      let r = 0, g = 0, b = 0, a = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
          const j = (ny * W + nx) * 4;
          r += src[j]; g += src[j + 1]; b += src[j + 2]; a += src[j + 3];
          n++;
        }
      }
      data[i]     = r / n;
      data[i + 1] = g / n;
      data[i + 2] = b / n;
      data[i + 3] = a / n;
    }
  }
}
