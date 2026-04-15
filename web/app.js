// ============================================================
//  app.js — Avalanche Forecast Terrain Map
//  Mapbox GL JS v2  |  Identity tile decoder  |  forecast.json
// ============================================================

// ── CONFIG ──────────────────────────────────────────────────
const MAPBOX_TOKEN  = 'pk.eyJ1IjoiY2FsZWJjb3N0YTEiLCJhIjoiY21uNms3djVrMDZneDMwcHJ6dm5jZTlsMSJ9.LbSImS4YvC4iTcoRNTFbGg';

const BASE_STYLES = {
  'mapbox-satellite': {
    tiles:       [`https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.jpg?access_token=${MAPBOX_TOKEN}`],
    tileSize:    512,
    attribution: '© Mapbox © Maxar',
  },
  'esri-satellite': {
    tiles:       ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
    tileSize:    256,
    attribution: '© Esri, Maxar, Earthstar Geographics',
  },
  'open-topo': {
    tiles:       ['https://tile.opentopomap.org/{z}/{x}/{y}.png'],
    tileSize:    256,
    attribution: '© OpenTopoMap contributors',
  },
  'osm': {
    tiles:       ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
    tileSize:    256,
    attribution: '© OpenStreetMap contributors',
  },
};
const R2_BASE       = 'https://pub-741e47c836714225a4613e29b6cf4d22.r2.dev';
const FORECAST_URL  = `${R2_BASE}/data/forecast.json`;
const IDENTITY_BASE = `${R2_BASE}/tiles/identity`;

function forecastUrlForDate(date) {
  return `${R2_BASE}/data/forecasts/${date}.json`;
}

const INITIAL_CENTER  = [-110.8, 43.5];
const INITIAL_ZOOM    = 8;
const INITIAL_PITCH   = 55;
const INITIAL_BEARING = -20;

// ── GAP ZONES ────────────────────────────────────────────────
const GAP_ZONES = [
  {
    id: 'lassen-ca',
    name: 'Lassen Volcanic NP',
    flyTo: { center: [-121.43, 40.47], zoom: 10, pitch: 50, bearing: 0 },
  },
  {
    id: 'yosemite-high-country',
    name: 'Yosemite National Park',
    flyTo: { center: [-119.56, 37.85], zoom: 9, pitch: 50, bearing: 0 },
  },
];

function predictionsUrlForDate(date) {
  return `${R2_BASE}/data/predictions/${date}.json`;
}

// ── COLOR PALETTES ──────────────────────────────────────────
const DANGER_COLORS = {
  0: '#888888',
  1: '#5db85c',
  2: '#fff200',
  3: '#f5a623',
  4: '#d0021b',
  5: '#000000',
};

const PROBLEM_COLORS = {
  'Wind Slab':            '#4fc3f7',
  'Storm Slab':           '#7986cb',
  'Wet Slab':             '#f06292',
  'Persistent Slab':      '#ff8a65',
  'Deep Persistent Slab': '#a1887f',
  'Cornice':              '#80cbc4',
  'Glide Avalanche':      '#fff176',
  'Wet Loose':            '#ce93d8',
  'Dry Loose':            '#b0bec5',
  'None':                 '#333333',
};

const LIKELIHOOD_COLORS = {
  'unlikely':       '#5db85c',
  'possible':       '#fff200',
  'likely':         '#f5a623',
  'very likely':    '#d0021b',
  'almost certain': '#6a0000',
};

const LEGEND_CONFIGS = {
  danger: {
    title: 'Danger Level',
    items: [
      { label: 'Low',          color: DANGER_COLORS[1] },
      { label: 'Moderate',     color: DANGER_COLORS[2] },
      { label: 'Considerable', color: DANGER_COLORS[3] },
      { label: 'High',         color: DANGER_COLORS[4] },
      { label: 'Extreme',      color: DANGER_COLORS[5] },
      { label: 'No Rating',    color: DANGER_COLORS[0] },
    ],
  },
  problem: {
    title: 'Problem Type',
    items: Object.entries(PROBLEM_COLORS).map(([k, v]) => ({ label: k, color: v })),
  },
  likelihood: {
    title: 'Likelihood',
    items: [
      { label: 'Unlikely',       color: LIKELIHOOD_COLORS['unlikely'] },
      { label: 'Possible',       color: LIKELIHOOD_COLORS['possible'] },
      { label: 'Likely',         color: LIKELIHOOD_COLORS['likely'] },
      { label: 'Very Likely',    color: LIKELIHOOD_COLORS['very likely'] },
      { label: 'Almost Certain', color: LIKELIHOOD_COLORS['almost certain'] },
    ],
  },
};

// Decode hex color → {r, g, b}
function hexToRgb(hex) {
  return {
    r: parseInt(hex.slice(1, 3), 16),
    g: parseInt(hex.slice(3, 5), 16),
    b: parseInt(hex.slice(5, 7), 16),
  };
}

// Pack RGBA as little-endian Uint32 (byte layout: R, G, B, A in memory)
// Matches ImageData layout on all x86/ARM platforms.
function packRgba(r, g, b, a) {
  return ((a * 0x1000000) + (b * 0x10000) + (g * 0x100) + r) >>> 0;
}

// ── LUT BUILDER ───────────────────────────────────────────
// Converts forecast.json into three Uint32Array LUTs (one per color mode).
// lut[zoneIndex * 256 + cellId] = packed RGBA for that cell.
function buildLuts(forecast) {
  const SIZE  = 256 * 256;
  const ALPHA_ON  = 200;   // avi terrain with data
  const ALPHA_OFF = 55;    // avi terrain, no active problem (faint)

  const danger     = new Uint32Array(SIZE);
  const problem    = new Uint32Array(SIZE);
  const likelihood = new Uint32Array(SIZE);

  const LH_ORDER = ['unlikely', 'possible', 'likely', 'very likely', 'almost certain'];

  for (const [ziStr, zone] of Object.entries(forecast.zones)) {
    const zi = parseInt(ziStr, 10);
    if (isNaN(zi) || zi < 1 || zi > 255) continue;

    for (const [ciStr, cell] of Object.entries(zone)) {
      const ci = parseInt(ciStr, 10);
      if (isNaN(ci) || !cell || typeof cell !== 'object') continue;

      const idx      = zi * 256 + ci;
      const problems = cell.problems || [];
      const hasProb  = problems.length > 0;

      // ── danger ──
      const dc = hexToRgb(DANGER_COLORS[cell.danger] || DANGER_COLORS[0]);
      danger[idx] = packRgba(dc.r, dc.g, dc.b, hasProb ? ALPHA_ON : ALPHA_OFF);

      // ── problem (primary problem type) ──
      if (hasProb) {
        const pHex = PROBLEM_COLORS[problems[0].type] || PROBLEM_COLORS['None'];
        const pc   = hexToRgb(pHex);
        problem[idx] = packRgba(pc.r, pc.g, pc.b, ALPHA_ON);
      } else {
        problem[idx] = packRgba(dc.r, dc.g, dc.b, ALPHA_OFF);
      }

      // ── likelihood (highest across all problems) ──
      if (hasProb) {
        let maxIdx = -1;
        for (const p of problems) {
          const i = LH_ORDER.indexOf((p.likelihood || '').toLowerCase());
          if (i > maxIdx) maxIdx = i;
        }
        if (maxIdx >= 0) {
          const lhHex = LIKELIHOOD_COLORS[LH_ORDER[maxIdx]];
          const lc    = hexToRgb(lhHex);
          likelihood[idx] = packRgba(lc.r, lc.g, lc.b, ALPHA_ON);
        } else {
          likelihood[idx] = packRgba(dc.r, dc.g, dc.b, ALPHA_OFF);
        }
      } else {
        likelihood[idx] = packRgba(dc.r, dc.g, dc.b, ALPHA_OFF);
      }
    }
  }

  return { danger, problem, likelihood };
}

// ── WORKER POOL ───────────────────────────────────────────
const NUM_WORKERS = 4;
const workers     = [];
const pending     = new Map();   // requestId → { resolve, reject }
let   nextWorker  = 0;
let   nextReqId   = 0;

function initWorkerPool() {
  for (let i = 0; i < NUM_WORKERS; i++) {
    const w = new Worker('./tile-worker.js');
    w.onmessage = onWorkerMessage;
    workers.push(w);
  }
}

function sendLutsToWorkers(luts) {
  workers.forEach(w => {
    // Slice creates a copy — each worker gets its own transferable buffer
    const d = luts.danger.slice();
    const p = luts.problem.slice();
    const l = luts.likelihood.slice();
    w.postMessage({ type: 'INIT_LUTS', danger: d, problem: p, likelihood: l },
                  [d.buffer, p.buffer, l.buffer]);
  });
}


// ── STATE ─────────────────────────────────────────────────
let colorMode          = 'likelihood';
let activeDate         = '';      // 'YYYY-MM-DD' of currently loaded forecast
let forecastData       = null;
let predictionsData    = null;
let predictionsVisible = false;
let activeGridZone     = 'lassen-ca';
let popup             = null;

// ── DECODE HELPERS ────────────────────────────────────────
const ASPECTS    = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
const ELEV_NAMES = { 1: 'Below Treeline', 2: 'Treeline', 3: 'Alpine' };

function decodeCellId(cellId) {
  return {
    aspect:   ASPECTS[Math.floor(cellId / 10)] || '—',
    elevBand: ELEV_NAMES[cellId % 10]           || '—',
  };
}

// ── CUSTOM TILE PROTOCOL ──────────────────────────────────
// MapLibre GL JS v3 uses callback-style addProtocol (not Promise/AbortController).
// Handler: (params, callback) => Cancelable
// callback signature: (error, data, cacheControl, expires)
mapboxgl.addProtocol('avicast', (params, callback) => {
  // Parse "avicast://2026-03-26/likelihood/10/168/384"
  // [0]=date  [1]=mode  [2..4]=z/x/y
  const parts    = params.url.slice('avicast://'.length).split('/');
  const mode     = parts[1];
  const zxy      = parts.slice(2).join('/');
  const identity = `${IDENTITY_BASE}/${zxy}.png`;

  const id = nextReqId++;
  let cancelled = false;

  pending.set(id, {
    resolve: (buffer) => { if (!cancelled) callback(null, buffer, null, null); },
    reject:  (err)    => { if (!cancelled) callback(err); },
  });

  workers[nextWorker++ % NUM_WORKERS]
    .postMessage({ type: 'FETCH_TILE', requestId: id, url: identity, mode });

  return {
    cancel: () => {
      cancelled = true;
      pending.delete(id);
    },
  };
});

// ── WORKER RESULT HANDLER ─────────────────────────────────
function onWorkerMessage(e) {
  const { type, requestId, buffer, error } = e.data;
  const entry = pending.get(requestId);
  if (!entry) return;
  pending.delete(requestId);
  if (type === 'TILE_DONE') {
    entry.resolve(buffer);
  } else {
    entry.reject(new Error(error || 'worker error'));
  }
}

// ── TILE URL ──────────────────────────────────────────────
function tileUrl() {
  return `avicast://${activeDate}/${colorMode}/{z}/{x}/{y}`;
}

function refreshTiles() {
  if (map.getSource('avi-terrain')) {
    map.getSource('avi-terrain').setTiles([tileUrl()]);
  }
}

// ── ORBIT CONTROL ─────────────────────────────────────────
class OrbitControl {
  onAdd(map) {
    this._map     = map;
    this._active  = false;
    this._dragging = false;

    this._container = document.createElement('div');
    this._container.className = 'maplibregl-ctrl maplibregl-ctrl-group';

    this._btn = document.createElement('button');
    this._btn.className = 'orbit-btn';
    this._btn.title = 'Orbit mode — drag to rotate freely';
    this._btn.innerHTML = `<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="9" ry="4"/><line x1="12" y1="3" x2="12" y2="21"/><polyline points="16 8 20 5 20 9"/></svg>`;
    this._btn.addEventListener('click', () => this._toggle());
    this._container.appendChild(this._btn);

    this._onDown = this._mouseDown.bind(this);
    this._onMove = this._mouseMove.bind(this);
    this._onUp   = this._mouseUp.bind(this);

    return this._container;
  }

  onRemove() {
    this._container.parentNode.removeChild(this._container);
  }

  _toggle() {
    this._active = !this._active;
    this._btn.classList.toggle('active', this._active);
    const canvas = this._map.getCanvas();
    if (this._active) {
      this._map.dragPan.disable();
      canvas.addEventListener('mousedown', this._onDown);
      canvas.style.cursor = 'grab';
    } else {
      this._map.dragPan.enable();
      canvas.removeEventListener('mousedown', this._onDown);
      window.removeEventListener('mousemove', this._onMove);
      window.removeEventListener('mouseup',   this._onUp);
      canvas.style.cursor = 'crosshair';
    }
  }

  _mouseDown(e) {
    if (e.button !== 0) return;
    this._lastX   = e.clientX;
    this._lastY   = e.clientY;
    this._dragging = true;
    this._map.getCanvas().style.cursor = 'grabbing';
    window.addEventListener('mousemove', this._onMove);
    window.addEventListener('mouseup',   this._onUp);
    e.preventDefault();
  }

  _mouseMove(e) {
    if (!this._dragging) return;
    const dx = e.clientX - this._lastX;
    const dy = e.clientY - this._lastY;
    this._lastX = e.clientX;
    this._lastY = e.clientY;
    this._map.setBearing(this._map.getBearing() - dx * 0.4);
    this._map.setPitch(Math.max(0, Math.min(85, this._map.getPitch() - dy * 0.3)));
  }

  _mouseUp() {
    this._dragging = false;
    this._map.getCanvas().style.cursor = 'grab';
    window.removeEventListener('mousemove', this._onMove);
    window.removeEventListener('mouseup',   this._onUp);
  }
}

let orbitControl;

// ── INIT MAP ──────────────────────────────────────────────
const map = new mapboxgl.Map({
  container: 'map',
  style: {
    version: 8,
    sources: {
      'base-tiles': {
        type:        'raster',
        tiles:       BASE_STYLES['esri-satellite'].tiles,
        tileSize:    BASE_STYLES['esri-satellite'].tileSize,
        attribution: BASE_STYLES['esri-satellite'].attribution,
      },
    },
    layers: [{ id: 'base-tiles', type: 'raster', source: 'base-tiles' }],
    glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
  },
  center:    INITIAL_CENTER,
  zoom:      INITIAL_ZOOM,
  pitch:     INITIAL_PITCH,
  bearing:   INITIAL_BEARING,
  antialias: true,
});

map.addControl(new mapboxgl.NavigationControl(), 'bottom-right');
map.addControl(new mapboxgl.ScaleControl({ unit: 'imperial' }), 'bottom-right');
orbitControl = new OrbitControl();
map.addControl(orbitControl, 'bottom-right');

// ── MAP LOAD ──────────────────────────────────────────────
map.on('load', async () => {
  initWorkerPool();

  // 3D terrain
  map.addSource('mapbox-dem', {
    type:     'raster-dem',
    tiles:    ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
    tileSize: 256,
    maxzoom:  15,
    encoding: 'terrarium',
  });
  map.setTerrain({ source: 'mapbox-dem', exaggeration: 1.3 });

  // Init date picker: today as default and max value
  const todayStr = new Date().toISOString().slice(0, 10);
  const dateInput = document.getElementById('date-input');
  dateInput.max   = todayStr;
  dateInput.value = todayStr;
  dateInput.addEventListener('change', async (e) => {
    const date = e.target.value;
    if (!date) return;
    await loadForecastForDate(date);
  });

  // Load today's forecast, build LUTs, add layer
  await loadForecastForDate(null);   // null → load from data/forecast.json

  addPredictionLayer();
  setupInteractions();
  buildCenterSelector();

  // Prediction overlay toggle
  document.getElementById('pred-toggle').addEventListener('change', (e) => {
    const on = e.target.checked;
    togglePredictions(on);
    document.getElementById('gap-zone-select').classList.toggle('visible', on);
    // Auto-fly to the selected zone when enabling
    if (on) {
      const zone = GAP_ZONES.find(z => z.id === activeGridZone);
      if (zone?.flyTo) map.flyTo({ ...zone.flyTo, duration: 1200 });
    }
  });

  // Gap zone selector — fly to zone and repaint
  document.getElementById('gap-zone-select').addEventListener('change', (e) => {
    activeGridZone = e.target.value;
    map.triggerRepaint();
    const zone = GAP_ZONES.find(z => z.id === activeGridZone);
    if (zone?.flyTo) map.flyTo({ ...zone.flyTo, duration: 1800 });
  });
});

// ── FORECAST LOADER ───────────────────────────────────────
// Fetches forecast JSON (today's or historical), rebuilds LUTs,
// refreshes tiles.  Pass null to load the current forecast.json.
async function loadForecastForDate(date) {
  const badge = document.getElementById('date-badge');
  badge.textContent = 'Loading…';

  const url = date ? forecastUrlForDate(date) : FORECAST_URL;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    forecastData = await res.json();
    activeDate   = forecastData.date || date || new Date().toISOString().slice(0, 10);

    // Keep date input in sync with loaded date
    document.getElementById('date-input').value = activeDate;

    // Human-readable label e.g. "Mar 26, 2026"
    const label = new Date(activeDate + 'T12:00:00').toLocaleDateString(
      'en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    badge.textContent = label;

    // Push new LUTs to workers
    sendLutsToWorkers(buildLuts(forecastData));

    // Load interpolated predictions for the same date (fails gracefully if not yet available)
    loadPredictions(activeDate);

    if (map.getSource('avi-terrain')) {
      // Source already exists — just swap tile URLs (busts MapLibre's tile cache)
      refreshTiles();
    } else {
      // First load — add source + layer
      map.addSource('avi-terrain', {
        type:     'raster',
        tiles:    [tileUrl()],
        tileSize: 256,
        minzoom:  6,
        maxzoom:  13,
      });
      map.addLayer({
        id:     'avi-raster',
        type:   'raster',
        source: 'avi-terrain',
        layout: { visibility: 'visible' },
        paint: {
          'raster-opacity':       1.0,
          'raster-resampling':    'linear',
          'raster-fade-duration': 100,
        },
      });
    }
  } catch (err) {
    console.error('Failed to load forecast:', err);
    badge.textContent = date ? `No data for ${date}` : 'Unavailable';
    // Restore date input to last successful date
    if (activeDate) document.getElementById('date-input').value = activeDate;
  }
}

// ── INTERPOLATION OVERLAY — Canvas 2D approach ────────────
//
// MapLibre GL v3 WebGL fill/circle layers silently fail to render when
// terrain (DEM exaggeration) is active.  Instead we draw the grid on a
// plain <canvas> element positioned on top of the map and re-project
// geographic coordinates to screen pixels with map.project() on every
// render frame.  This bypasses WebGL entirely and always works.

const GRID_CANVAS = document.getElementById('grid-canvas');

function syncCanvasSize() {
  const mapCanvas = map.getCanvas();
  GRID_CANVAS.width  = mapCanvas.width;
  GRID_CANVAS.height = mapCanvas.height;
  GRID_CANVAS.style.width  = mapCanvas.style.width  || mapCanvas.width  + 'px';
  GRID_CANVAS.style.height = mapCanvas.style.height || mapCanvas.height + 'px';
}

function renderGridCanvas() {
  if (!GRID_CANVAS) return;
  const ctx = GRID_CANVAS.getContext('2d');
  ctx.clearRect(0, 0, GRID_CANVAS.width, GRID_CANVAS.height);

  if (!predictionsVisible || !predictionsData) return;
  const pred = predictionsData.predictions?.[activeGridZone];
  if (!pred?.spatial_grid?.features?.length) return;

  ctx.globalAlpha = 0.82;

  for (const f of pred.spatial_grid.features) {
    const d     = f.properties.danger_display || 0;
    const color = DANGER_COLORS[d] ?? DANGER_COLORS[0];
    const ring  = f.geometry.coordinates[0];

    // Project each corner to screen pixels
    const pts = ring.map(([lng, lat]) => map.project([lng, lat]));

    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.4)';
    ctx.lineWidth   = 0.5;
    ctx.stroke();
  }

  ctx.globalAlpha = 1.0;
}

async function loadPredictions(date) {
  try {
    const res = await fetch(predictionsUrlForDate(date));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    predictionsData = await res.json();
  } catch (err) {
    console.warn('[predictions] fetch failed for', date, err.message);
    predictionsData = null;
  }
  if (predictionsVisible) map.triggerRepaint();
}

function addPredictionLayer() {
  // Size canvas to match map and hook into the render loop.
  syncCanvasSize();
  map.on('resize', syncCanvasSize);
  map.on('render', renderGridCanvas);
}

function updatePredictionLayer() {
  if (predictionsVisible) map.triggerRepaint();
}

function togglePredictions(visible) {
  predictionsVisible = visible;
  GRID_CANVAS.style.display = visible ? 'block' : 'none';
  map.triggerRepaint();
}

function showPredictionPopup(lngLat, props) {
  if (popup) popup.remove();
  const names = ['—', 'Low', 'Moderate', 'Considerable', 'High', 'Extreme'];
  const dot = (d) =>
    `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;
      background:${DANGER_COLORS[d]||'#888'};margin-right:5px;vertical-align:middle"></span>`;

  popup = new mapboxgl.Popup({
    closeButton: true, maxWidth: '300px',
    anchor: popupAnchor(lngLat), offset: 12,
  }).setLngLat(lngLat).setHTML(`
    <div class="popup-header">
      <div class="popup-center">Interpolated</div>
      <div class="popup-zone">${props.zone_name}</div>
    </div>
    <div class="popup-body">
      <table class="pred-table">
        <tr><th>Band</th><th>Danger</th><th>±</th></tr>
        <tr><td>Alpine</td>
          <td>${dot(props.danger_alpine)}${names[props.danger_alpine]||'—'}</td>
          <td>±${Number(props.unc_alpine).toFixed(1)}</td></tr>
        <tr><td>Treeline</td>
          <td>${dot(props.danger_treeline)}${names[props.danger_treeline]||'—'}</td>
          <td>—</td></tr>
        <tr><td>Below</td>
          <td>${dot(props.danger_below)}${names[props.danger_below]||'—'}</td>
          <td>—</td></tr>
      </table>
      <p class="popup-model-note">GP baseline · experimental</p>
    </div>
  `).addTo(map);
}

// ── CLICK → DECODE IDENTITY TILE ─────────────────────────
function lngLatToTile(lng, lat, z) {
  const n = 1 << z;
  const x = Math.floor((lng + 180) / 360 * n);
  const latRad = lat * Math.PI / 180;
  const y = Math.floor((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * n);
  return { x, y };
}

function lngLatToPixel(lng, lat, z, tx, ty) {
  const n = 1 << z;
  const px = Math.floor(((lng + 180) / 360 * n - tx) * 256);
  const latRad = lat * Math.PI / 180;
  const py = Math.floor(((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * n - ty) * 256);
  return { px: Math.max(0, Math.min(255, px)), py: Math.max(0, Math.min(255, py)) };
}

async function decodeClickPoint(lng, lat) {
  if (!forecastData) return null;
  const z = Math.min(13, Math.max(6, Math.round(map.getZoom())));
  const { x, y } = lngLatToTile(lng, lat, z);
  const { px, py } = lngLatToPixel(lng, lat, z, x, y);

  try {
    const res = await fetch(`${IDENTITY_BASE}/${z}/${x}/${y}.png`);
    if (!res.ok) return null;

    const bitmap = await createImageBitmap(await res.blob());
    const canvas = new OffscreenCanvas(256, 256);
    const ctx    = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);
    const pixel = ctx.getImageData(px, py, 1, 1).data;

    const zoneIndex = pixel[0];
    const cellId    = pixel[1];
    const alpha     = pixel[3];
    if (alpha === 0 || zoneIndex === 0) return null;

    const zone = forecastData.zones?.[String(zoneIndex)];
    if (!zone) return null;
    const cell = zone[String(cellId)];

    const { aspect, elevBand } = decodeCellId(cellId);

    return {
      center_id:      zone.center_id,
      zone_name:      zone.zone_name,
      aspect,
      elev_band_name: elevBand,
      danger_level:   cell?.danger       || 0,
      danger_label:   cell?.danger_label || 'No Rating',
      danger_color:   cell?.danger_color || '#888888',
      problems:       cell?.problems     || [],
    };
  } catch (err) {
    console.error('Click decode error:', err);
    return null;
  }
}

// ── INTERACTIONS ──────────────────────────────────────────
function setupInteractions() {
  map.getCanvas().style.cursor = 'crosshair';
  map.on('click', async (e) => {
    if (orbitControl?._active) return;
    const info = await decodeClickPoint(e.lngLat.lng, e.lngLat.lat);
    if (info) showPopup(e.lngLat, info);
    else if (popup) { popup.remove(); popup = null; }
  });
}

// ── POPUP ─────────────────────────────────────────────────
function likelihoodClass(lh) {
  if (!lh) return '';
  return `lh-${lh.toLowerCase().replace(/\s+/g, '-')}`;
}

function buildProblemCards(problems) {
  if (!problems || !problems.length) {
    return '<p class="popup-no-problem">No active avalanche problems on this terrain face.</p>';
  }
  return problems.map(p => `
    <div class="problem-card">
      <div class="problem-type">${p.type || '—'}</div>
      <div class="problem-meta">
        <div class="meta-item">
          Likelihood: <span class="${likelihoodClass(p.likelihood)}">${capitalize(p.likelihood || '—')}</span>
        </div>
        <div class="meta-item">
          Size: <span>D${p.size_min ?? '?'}–D${p.size_max ?? '?'}</span>
        </div>
      </div>
    </div>
  `).join('');
}

function popupAnchor(lngLat) {
  const pt = map.project(lngLat);
  const c  = map.getCanvas();
  const topHalf  = pt.y < c.offsetHeight * 0.5;
  const leftHalf = pt.x < c.offsetWidth  * 0.5;
  if ( topHalf &&  leftHalf) return 'top-left';
  if ( topHalf && !leftHalf) return 'top-right';
  if (!topHalf &&  leftHalf) return 'bottom-left';
  return 'bottom-right';
}

function dangerTextColor(hex) {
  return hex === '#fff200' ? '#b8a800' : hex;
}

function showPopup(lngLat, info) {
  if (popup) popup.remove();

  const dangerColor = info.danger_color || DANGER_COLORS[info.danger_level] || '#888';

  const html = `
    <div class="popup-header">
      <div class="popup-center">${info.center_id || ''}</div>
      <div class="popup-zone">${info.zone_name || '—'}</div>
    </div>

    <div class="popup-danger-banner">
      <div class="danger-pill" style="background:${hexWithAlpha(dangerColor, 0.2)};border:1px solid ${dangerColor}88">
        <div class="danger-dot" style="background:${dangerColor}"></div>
        <span style="color:${dangerTextColor(dangerColor)}">${info.danger_label}</span>
      </div>
      <div class="popup-elev-aspect">${info.elev_band_name} · ${info.aspect}-facing</div>
    </div>

    <div class="popup-body">
      ${buildProblemCards(info.problems)}
    </div>
  `;

  popup = new mapboxgl.Popup({
    closeButton: true,
    maxWidth:    '340px',
    anchor:      popupAnchor(lngLat),
    offset:      12,
  })
    .setLngLat(lngLat)
    .setHTML(html)
    .addTo(map);
}

// ── LEGEND always visible ─────────────────────────────────
document.getElementById('legend').classList.add('visible');
updateLegend(colorMode);

// ── COLOR MODE SWITCH ─────────────────────────────────────
document.getElementById('color-select').addEventListener('change', (e) => {
  colorMode = e.target.value;
  refreshTiles();
  updateLegend(colorMode);
});

// ── LEGEND ────────────────────────────────────────────────
function updateLegend(mode) {
  const cfg = LEGEND_CONFIGS[mode];
  document.getElementById('legend-title').textContent = cfg.title;
  document.getElementById('legend-items').innerHTML = cfg.items.map(item => `
    <div class="legend-row">
      <div class="legend-swatch" style="background:${item.color};
        ${item.color === '#fff200' ? 'border:1px solid #aaa8' : ''}"></div>
      ${item.label}
    </div>
  `).join('');
}

// ── AVALANCHE CENTER SELECTOR ─────────────────────────────
// bounds: [[west, south], [east, north]]
const CENTER_BOUNDS = {
  BAC:    { name: 'Bridgeport (BAC)',             bounds: [[-119.5, 37.8], [-118.7, 38.8]] },
  BTAC:   { name: 'Bridger-Teton (BTAC)',         bounds: [[-111.5, 42.5], [-109.5, 44.5]] },
  CAAC:   { name: 'Coastal AK / Juneau (CAAC)',   bounds: [[-135.5, 57.8], [-133.5, 58.8]] },
  CAC:    { name: 'Cordova (CAC)',                bounds: [[-146.5, 60.0], [-144.5, 61.5]] },
  CAIC:   { name: 'Colorado (CAIC)',              bounds: [[-109.5, 36.8], [-104.5, 41.0]] },
  CNFAIC: { name: 'Chugach NF (CNFAIC)',          bounds: [[-150.5, 60.0], [-147.5, 62.0]] },
  COAA:   { name: 'Central Oregon (COAA)',        bounds: [[-122.5, 43.5], [-121.0, 45.0]] },
  EARAC:  { name: 'Eastern AK Range (EARAC)',     bounds: [[-147.0, 62.5], [-143.0, 64.5]] },
  ESAC:   { name: 'Eastern Sierra (ESAC)',        bounds: [[-119.5, 36.5], [-117.5, 38.5]] },
  EWYAIX: { name: 'Eastern Wyoming (EWYAIX)',     bounds: [[-107.5, 41.0], [-104.5, 43.5]] },
  FAC:    { name: 'Flathead (FAC)',               bounds: [[-115.0, 47.0], [-113.0, 49.0]] },
  GNFAC:  { name: 'Gallatin NF (GNFAC)',          bounds: [[-112.0, 44.5], [-109.5, 46.0]] },
  HAC:    { name: 'Haines (HAC)',                 bounds: [[-137.0, 59.0], [-134.5, 60.5]] },
  HPAC:   { name: 'Hatcher Pass (HPAC)',          bounds: [[-149.5, 61.5], [-148.2, 62.1]] },
  IPAC:   { name: 'Idaho Panhandle (IPAC)',       bounds: [[-117.5, 47.0], [-115.0, 48.5]] },
  KPAC:   { name: 'Kachina Peaks / Flagstaff (KPAC)', bounds: [[-111.8, 35.1], [-111.3, 35.6]] },
  MSAC:   { name: 'Mt. Shasta (MSAC)',            bounds: [[-122.5, 41.0], [-121.8, 41.8]] },
  MWAC:   { name: 'Mt. Washington (MWAC)',        bounds: [[-71.5,  43.5], [-70.5,  44.8]] },
  NWAC:   { name: 'Northwest (NWAC)',             bounds: [[-122.5, 45.0], [-119.5, 49.0]] },
  PAC:    { name: 'Payette (PAC)',                bounds: [[-116.5, 43.5], [-114.5, 45.5]] },
  SAC:    { name: 'Sierra (SAC)',                 bounds: [[-120.5, 37.5], [-119.0, 39.0]] },
  SNFAC:  { name: 'Sawtooth (SNFAC)',             bounds: [[-115.5, 43.0], [-113.5, 44.5]] },
  SOAIX:  { name: 'Southern Oregon (SOAIX)',      bounds: [[-123.0, 42.0], [-121.5, 43.5]] },
  TAC:    { name: 'Taos / Northern NM (TAC)',     bounds: [[-106.0, 35.5], [-104.5, 37.5]] },
  UAC:    { name: 'Utah (UAC)',                   bounds: [[-114.0, 37.0], [-110.5, 42.0]] },
  VAC:    { name: 'Valdez (VAC)',                 bounds: [[-147.5, 60.5], [-145.0, 62.0]] },
  WAC:    { name: 'Wallowa (WAC)',                bounds: [[-118.5, 44.5], [-116.5, 46.5]] },
  WCMAC:  { name: 'W. Central Montana (WCMAC)',   bounds: [[-115.0, 46.0], [-112.5, 47.5]] },
};

function buildCenterSelector() {
  const select = document.getElementById('center-select');
  Object.entries(CENTER_BOUNDS)
    .sort((a, b) => a[1].name.localeCompare(b[1].name))
    .forEach(([id, info]) => {
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = info.name;
      select.appendChild(opt);
    });

  select.addEventListener('change', (e) => {
    const id = e.target.value;
    if (!id) return;
    const info = CENTER_BOUNDS[id];
    if (!info) return;
    map.fitBounds(info.bounds, { padding: 40, pitch: INITIAL_PITCH, bearing: INITIAL_BEARING, duration: 1200 });
    // Reset dropdown so same center can be re-selected
    setTimeout(() => { select.value = ''; }, 1300);
  });
}

// ── BASE LAYER SWITCHER ───────────────────────────────────
document.getElementById('base-layer-select').addEventListener('change', (e) => {
  const style = BASE_STYLES[e.target.value];
  if (!style) return;
  const src = map.getSource('base-tiles');
  if (src) src.setTiles(style.tiles);
});

// ── UTILS ─────────────────────────────────────────────────
function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function hexWithAlpha(hex, alpha) {
  const { r, g, b } = hexToRgb(hex);
  return `rgba(${r},${g},${b},${alpha})`;
}
