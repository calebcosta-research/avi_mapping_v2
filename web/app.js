// ============================================================
//  app.js — Avalanche Forecast Terrain Map
//  Mapbox GL JS v3  |  3D terrain  |  forecast_layer.geojson
// ============================================================

// ── CONFIG ──────────────────────────────────────────────────
const MAPBOX_TOKEN = 'pk.eyJ1IjoiY2FsZWJjb3N0YTEiLCJhIjoiY2tmMnloNHk1MTZvcDJ4cDdleGpsZzZpeiJ9.6ZopP_G4jX479NsAgbJ16Q';

// Cloudflare R2 public base URL
const R2_BASE      = 'https://pub-741e47c836714225a4613e29b6cf4d22.r2.dev';

// Data sources — fall back to local if R2 not available
const FORECAST_URL    = `${R2_BASE}/data/forecast.json`;
const IDENTITY_TILES  = `${R2_BASE}/tiles/identity/{z}/{x}/{y}.png`;
// Legacy vector layer (still used for click interaction until identity tiles are live)
const DATA_URL        = `${R2_BASE}/data/forecast_layer.geojson`;

const INITIAL_CENTER  = [-110.8, 43.5];           // BTAC area
const INITIAL_ZOOM    = 8;
const INITIAL_PITCH   = 55;
const INITIAL_BEARING = -20;

// ── COLOR PALETTES ──────────────────────────────────────────
const DANGER_COLORS = {
  0: '#888888',   // No Rating
  1: '#5db85c',   // Low
  2: '#fff200',   // Moderate
  3: '#f5a623',   // Considerable
  4: '#d0021b',   // High
  5: '#000000',   // Extreme
};

const PROBLEM_COLORS = {
  'Wind Slab':            '#4fc3f7',
  'Storm Slab':           '#7986cb',
  'Wet Slab':             '#f06292',
  'Persistent Slab':      '#ff8a65',
  'Deep Persistent Slab': '#a1887f',
  'Cornice':              '#80cbc4',
  'Glide Avalanche':      '#fff176',
  'Loose Wet':            '#ce93d8',
  'Loose Dry':            '#b0bec5',
  'None':                 '#333333',
};

const LIKELIHOOD_COLORS = {
  'unlikely':        '#5db85c',
  'possible':        '#fff200',
  'likely':          '#f5a623',
  'very likely':     '#d0021b',
  'almost certain':  '#6a0000',
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

// ── STATE ────────────────────────────────────────────────────
let layerVisible     = false;
let problemsOnly     = false;   // filter to only show terrain with active problems
let colorMode        = 'danger';
let forecastData     = null;
let popup            = null;

// ── INIT MAP ─────────────────────────────────────────────────
mapboxgl.accessToken = MAPBOX_TOKEN;

const map = new mapboxgl.Map({
  container: 'map',
  style:     'mapbox://styles/mapbox/satellite-streets-v12',
  center:    INITIAL_CENTER,
  zoom:      INITIAL_ZOOM,
  pitch:     INITIAL_PITCH,
  bearing:   INITIAL_BEARING,
  antialias: true,
});

map.addControl(new mapboxgl.NavigationControl(), 'bottom-right');
map.addControl(new mapboxgl.ScaleControl({ unit: 'imperial' }), 'bottom-right');

// ── 3D TERRAIN ───────────────────────────────────────────────
map.on('load', () => {
  // Mapbox terrain DEM source
  map.addSource('mapbox-dem', {
    type:        'raster-dem',
    url:         'mapbox://mapbox.mapbox-terrain-dem-v1',
    tileSize:    512,
    maxzoom:     14,
  });
  map.setTerrain({ source: 'mapbox-dem', exaggeration: 1.3 });

  // Sky layer for atmosphere effect
  map.addLayer({
    id:   'sky',
    type: 'sky',
    paint: {
      'sky-type':             'atmosphere',
      'sky-atmosphere-sun':   [0.0, 90.0],
      'sky-atmosphere-sun-intensity': 15,
    },
  });

  // Load forecast data
  loadForecastLayer();
});

// ── LOAD FORECAST LAYER ──────────────────────────────────────
async function loadForecastLayer() {
  try {
    const res = await fetch(DATA_URL);
    forecastData = await res.json();

    // Extract forecast date from first feature
    const firstDate = forecastData.features.find(f => f.properties.forecast_date)
                        ?.properties?.forecast_date;
    if (firstDate) {
      document.getElementById('date-badge').textContent = `Forecast: ${firstDate}`;
    }

    // ── Raster tile sources — one per color mode ───────────────
    const TILE_MODES = ['danger', 'problem', 'likelihood'];
    const origin = window.location.origin;

    TILE_MODES.forEach(mode => {
      map.addSource(`avi-tiles-${mode}`, {
        type:        'raster',
        tiles:       [`${origin}/tiles/forecast/${mode}/{z}/{x}/{y}.png`],
        tileSize:    256,
        minzoom:     6,
        maxzoom:     13,
        attribution: 'Avalanche forecast data: avalanche.org',
      });

      map.addLayer({
        id:      `avi-raster-${mode}`,
        type:    'raster',
        source:  `avi-tiles-${mode}`,
        layout:  { visibility: 'none' },
        paint: {
          'raster-opacity':       0.85,
          'raster-resampling':    'nearest',
          'raster-fade-duration': 150,
        },
      });
    });

    // ── GeoJSON source (invisible hit targets for click popups) ─
    map.addSource('avi-terrain', {
      type: 'geojson',
      data: forecastData,
    });

    // Transparent fill — click target only, raster handles visuals
    map.addLayer({
      id:     'avi-fill',
      type:   'fill',
      source: 'avi-terrain',
      layout: { visibility: 'none' },
      paint: {
        'fill-color':   buildFillColor('danger'),
        'fill-opacity': 0,   // invisible — raster layer handles display
      },
    });

    // Hover highlight (semi-transparent white overlay on hover)
    map.addLayer({
      id:     'avi-hover',
      type:   'fill',
      source: 'avi-terrain',
      layout: { visibility: 'none' },
      paint: {
        'fill-color':   'rgba(255,255,255,0.2)',
        'fill-opacity': [
          'case',
          ['boolean', ['feature-state', 'hover'], false], 1,
          0,
        ],
      },
    });

    // Outline on hover only
    map.addLayer({
      id:     'avi-outline',
      type:   'line',
      source: 'avi-terrain',
      layout: { visibility: 'none' },
      paint: {
        'line-color': [
          'case',
          ['boolean', ['feature-state', 'hover'], false],
          'rgba(255,255,255,0.7)',
          'rgba(255,255,255,0)',
        ],
        'line-width': 1.5,
      },
    });

    setupInteractions();
    updateLegend('danger');

  } catch (err) {
    console.error('Failed to load forecast layer:', err);
  }
}

// ── COLOR EXPRESSIONS ────────────────────────────────────────
function buildFillColor(mode) {
  if (mode === 'danger') {
    return [
      'match', ['get', 'danger_level'],
      1, DANGER_COLORS[1],
      2, DANGER_COLORS[2],
      3, DANGER_COLORS[3],
      4, DANGER_COLORS[4],
      5, DANGER_COLORS[5],
      DANGER_COLORS[0],
    ];
  }

  if (mode === 'problem') {
    const expr = ['match', ['get', 'primary_problem']];
    Object.entries(PROBLEM_COLORS).forEach(([k, v]) => { expr.push(k, v); });
    expr.push(PROBLEM_COLORS['None']);
    return expr;
  }

  if (mode === 'likelihood') {
    const expr = ['match', ['get', 'primary_likelihood']];
    Object.entries(LIKELIHOOD_COLORS).forEach(([k, v]) => { expr.push(k, v); });
    expr.push('#444444');
    return expr;
  }

  return '#888888';
}

// ── HOVER INTERACTIONS ───────────────────────────────────────
let hoveredId = null;

function setupInteractions() {
  // Hover highlight
  map.on('mousemove', 'avi-fill', (e) => {
    if (!layerVisible) return;
    map.getCanvas().style.cursor = 'pointer';
    if (e.features.length > 0) {
      if (hoveredId !== null) {
        map.setFeatureState({ source: 'avi-terrain', id: hoveredId }, { hover: false });
      }
      hoveredId = e.features[0].id;
      map.setFeatureState({ source: 'avi-terrain', id: hoveredId }, { hover: true });
    }
  });

  map.on('mouseleave', 'avi-fill', () => {
    map.getCanvas().style.cursor = '';
    if (hoveredId !== null) {
      map.setFeatureState({ source: 'avi-terrain', id: hoveredId }, { hover: false });
    }
    hoveredId = null;
  });

  // Click → popup
  map.on('click', 'avi-fill', (e) => {
    if (!layerVisible || !e.features.length) return;
    const feat = e.features[0];
    showPopup(e.lngLat, feat.properties);
  });
}

// ── POPUP ────────────────────────────────────────────────────
function likelihoodClass(lh) {
  if (!lh) return '';
  const key = lh.toLowerCase().replace(/\s+/g, '-');
  return `lh-${key}`;
}

function buildProblemCards(propsJson) {
  let problems = [];
  try { problems = JSON.parse(propsJson || '[]'); } catch {}

  if (!problems.length) {
    return '<p class="popup-no-problem">No active avalanche problems on this terrain face.</p>';
  }

  return problems.map(p => `
    <div class="problem-card">
      <div class="problem-type">${p.problem_type || '—'}</div>
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
  // Choose popup anchor so it never clips off the viewport edges
  const pt     = map.project(lngLat);
  const canvas = map.getCanvas();
  const w = canvas.offsetWidth;
  const h = canvas.offsetHeight;
  const topHalf  = pt.y < h * 0.5;
  const leftHalf = pt.x < w * 0.5;
  if (topHalf  && leftHalf)  return 'top-left';
  if (topHalf  && !leftHalf) return 'top-right';
  if (!topHalf && leftHalf)  return 'bottom-left';
  return 'bottom-right';
}

function dangerTextColor(hex) {
  // Return readable text color for danger pills — yellow needs darkening
  if (hex === '#fff200') return '#b8a800';
  return hex;
}

function showPopup(lngLat, props) {
  if (popup) popup.remove();

  const dangerColor = DANGER_COLORS[props.danger_level] || '#888';
  const dangerLabel = props.danger_label || 'No Rating';
  const elevLabel   = capitalize(props.elev_band_name?.replace(/_/g, ' ') || '');
  const aspect      = props.aspect || '—';

  const bottomLine = props.bottom_line
    ? `<div class="popup-bottom-line">${props.bottom_line.substring(0, 220)}${props.bottom_line.length > 220 ? '…' : ''}</div>`
    : '';

  const html = `
    <div class="popup-header">
      <div class="popup-center">${props.center_id || ''}</div>
      <div class="popup-zone">${props.zone_name || '—'}</div>
    </div>

    <div class="popup-danger-banner">
      <div class="danger-pill" style="background:${hexWithAlpha(dangerColor, 0.2)}; border:1px solid ${dangerColor}88">
        <div class="danger-dot" style="background:${dangerColor}"></div>
        <span style="color:${dangerTextColor(dangerColor)}">${dangerLabel}</span>
      </div>
      <div class="popup-elev-aspect">${elevLabel} · ${aspect}-facing</div>
    </div>

    <div class="popup-body">
      ${buildProblemCards(props.all_problems_json)}
    </div>

    ${bottomLine}
  `;

  popup = new mapboxgl.Popup({
    closeButton:  true,
    maxWidth:     '340px',
    anchor:       popupAnchor(lngLat),
    offset:       12,
  })
    .setLngLat(lngLat)
    .setHTML(html)
    .addTo(map);
}

// ── LAYER TOGGLE ─────────────────────────────────────────────
const toggleBtn      = document.getElementById('layer-toggle');
const problemsToggle = document.getElementById('problems-only-toggle');
const legend         = document.getElementById('legend');

function applyLayerFilter() {
  // null = show all; ['==', ['get', 'has_problem'], true] = only active problems
  const filter = problemsOnly ? ['==', ['get', 'has_problem'], true] : null;
  ['avi-fill', 'avi-outline', 'avi-hover'].forEach(id => {
    if (map.getLayer(id)) map.setFilter(id, filter);
  });
}

toggleBtn.addEventListener('click', () => {
  layerVisible = !layerVisible;
  const vis = layerVisible ? 'visible' : 'none';

  ['danger', 'problem', 'likelihood'].forEach(m => {
    const id = `avi-raster-${m}`;
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', m === colorMode ? vis : 'none');
  });
  ['avi-fill', 'avi-outline', 'avi-hover'].forEach(id => {
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', vis);
  });

  toggleBtn.classList.toggle('active', layerVisible);
  problemsToggle.style.display = layerVisible ? 'flex' : 'none';
  legend.classList.toggle('visible', layerVisible);

  if (!layerVisible && popup) popup.remove();
});

problemsToggle.addEventListener('click', () => {
  problemsOnly = !problemsOnly;
  problemsToggle.classList.toggle('active', problemsOnly);
  problemsToggle.querySelector('.toggle-dot').style.background = problemsOnly ? '#ff5533' : '#555';
  applyLayerFilter();
  if (popup) popup.remove();
});

// ── COLOR MODE SWITCH ─────────────────────────────────────────
document.getElementById('color-select').addEventListener('change', (e) => {
  colorMode = e.target.value;

  // Swap which raster layer is visible (only when advisory layer is on)
  if (layerVisible) {
    ['danger', 'problem', 'likelihood'].forEach(m => {
      const id = `avi-raster-${m}`;
      if (map.getLayer(id))
        map.setLayoutProperty(id, 'visibility', m === colorMode ? 'visible' : 'none');
    });
  }

  // Keep vector fill color in sync (used for hover highlight tinting)
  if (map.getLayer('avi-fill')) {
    map.setPaintProperty('avi-fill', 'fill-color', buildFillColor(colorMode));
  }

  updateLegend(colorMode);
});

// ── LEGEND ───────────────────────────────────────────────────
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

// ── UTILS ─────────────────────────────────────────────────────
function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function hexWithAlpha(hex, alpha) {
  // Convert hex + alpha to rgba string
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
