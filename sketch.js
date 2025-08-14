let photoTable;

const CATEGORY_DISPLAY_ORDER = [
  'Standing on all fours',
  'Standing up on hind legs',
  'Loaf',
  'Curled up',
  'Sitting on human',
  'Belly up',
  'Pretzel',
  'Laying on side',
  'Other'
];

let categoryCounts = {};
let countsArray = [];

let categoryColorsByIndex = [];
let categoryCenterAngles = [];

let mosaicTiles = [];
let mosaicTileSize = 10;
let mosaicComputedFor = { w: -1, h: -1 };

// Added: per-category items and selfie stats, label hitboxes, and interaction/animation state
let itemsByCategory = [];
let selfieStatsByCategory = [];
let labelHitboxes = [];
let interactionState = {
  mode: 'radial', // 'radial' | 'grid'
  selectedCategoryIndex: -1,
  animating: false,
  animationStartMs: 0,
  animationDurationMs: 1200,
  animationDirection: 0, // 1 to grid, -1 to radial
  progress: 0
};
let tileMappingByIndex = {}; // tileIndex -> { toX, toY, toSize, targetHex }
let gridLayout = null;      // { left, top, cols, rows, tileSize, bounds:{x1,y1,x2,y2}, title:string }

// Added: canvas sizing and positioning helpers
let currentFaceRadius = 0;
let radialCenterXCurrent = 0;
let radialCenterYCurrent = 0;
function computeCanvasWidth() {
  return windowWidth;
}
function computeCanvasHeight() {
  return windowHeight;
}
function getRadialCenterX() { return radialCenterXCurrent; }
function getRadialCenterY() { return radialCenterYCurrent; }
function computeGridScreenLeft() { return getRadialCenterX() + currentFaceRadius * 1.55 + 48; }

// Added: overlay layout constants/helpers
const OVERLAY_TITLE_Y = 32;
const OVERLAY_BLOCK_SPACING = 12;
const OVERLAY_TITLE_OFFSET = 34; // gap below title to first stat line
const OVERLAY_LINE_HEIGHT = 26;
const OVERLAY_LINES = 3;
const OVERLAY_AFTER_TEXT_PADDING = 32;
function overlayBottomY() {
  return OVERLAY_TITLE_Y + OVERLAY_TITLE_OFFSET + OVERLAY_BLOCK_SPACING + OVERLAY_LINE_HEIGHT * OVERLAY_LINES;
}

function preload() {
  photoTable = loadTable('cat_analysis_noref_human.csv', 'csv', 'header');
}

function setup() {
  createCanvas(computeCanvasWidth(), computeCanvasHeight());
  pixelDensity(2);
  textFont('sans-serif');
  countCategories();
  noLoop();
}

function windowResized() {
  resizeCanvas(computeCanvasWidth(), computeCanvasHeight());

  mosaicComputedFor = { w: -1, h: -1 };

  if (interactionState && interactionState.selectedCategoryIndex >= 0) {
    prepareGridLayoutForCategory(interactionState.selectedCategoryIndex);
  }

  redraw();
}

function countCategories() {
  categoryCounts = {};

  categoryColorsByIndex = CATEGORY_DISPLAY_ORDER.map(() => []);

  for (let label of CATEGORY_DISPLAY_ORDER) {
    categoryCounts[label] = 0;
  }
  if (photoTable) {
    for (let r = 0; r < photoTable.getRowCount(); r++) {

      const rawLabel = photoTable.getString(r, 'body_position');
      const normalized = normalizeBodyPosition(rawLabel);
      let hex = photoTable.getString(r, 'average_hex_color');
      hex = sanitizeHex(hex);
      if (Object.prototype.hasOwnProperty.call(categoryCounts, normalized)) {
        categoryCounts[normalized] += 1;
        const idx = CATEGORY_DISPLAY_ORDER.indexOf(normalized);
        if (idx >= 0) categoryColorsByIndex[idx].push(hex);
      } else {
        categoryCounts['other'] += 1;
        const idx = CATEGORY_DISPLAY_ORDER.indexOf('other');
        if (idx >= 0) categoryColorsByIndex[idx].push(hex);

      }
    }
  }
  countsArray = CATEGORY_DISPLAY_ORDER.map(label => categoryCounts[label] || 0);


  // Shuffle color arrays for nicer mixing
  for (let i = 0; i < categoryColorsByIndex.length; i++) {
    shuffleArrayInPlace(categoryColorsByIndex[i]);
  }

  // Precompute category center angles used for grouping
  categoryCenterAngles = [];
  for (let i = 0; i < CATEGORY_DISPLAY_ORDER.length; i++) {
    categoryCenterAngles.push(-HALF_PI + (i * TWO_PI) / CATEGORY_DISPLAY_ORDER.length);
  }

  // Added: build per-category items and selfie stats for later grid view
  computeCategoryItemsAndSelfieStats();
}

function sanitizeHex(value) {
  const s = (value || '').toString().trim();
  if (/^#?[0-9a-fA-F]{6}$/.test(s)) {
    return s[0] === '#' ? s : '#' + s;
  }
  return '#cccccc';
}

function shuffleArrayInPlace(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
  }
}

function normalizeBodyPosition(value) {
  const s = (value || '').toString().trim().toLowerCase();
  if (!s) return 'Other';
  if (s.includes('standing') && s.includes('hind')) return 'Standing up on hind legs';
  if (s.includes('standing') && (s.includes('all fours') || s.includes('fours'))) return 'Standing on all fours';
  if (s.includes('loaf')) return 'Loaf';
  if (s.includes('curl')) return 'Curled up';
  if (s.includes('sitting') && s.includes('human')) return 'Sitting on human';
  if (s.includes('belly')) return 'Belly up';
  if (s.includes('pretzel')) return 'Pretzel';
  if ((s.includes('laying') && s.includes('side')) || s.includes('on side')) return 'Laying on side';
  if (s === 'other') return 'Other';
  return 'Other';
}

function draw() {
  background(253, 250, 246);

  // Added: update animation progress if needed
  updateAnimationState();

  // Compute face radius first, then place radial center toward the left
  const faceRadius = Math.min(windowWidth, windowHeight) * 0.28;
  currentFaceRadius = faceRadius;
  radialCenterXCurrent = Math.ceil(24 + faceRadius * 1.5);
  radialCenterYCurrent = Math.floor(windowHeight / 2);

  translate(radialCenterXCurrent, radialCenterYCurrent);

  ensureMosaicComputed(faceRadius);

  drawCatFaceBase(faceRadius);

  const tickRadius = faceRadius * 0.95;
  const innerRadius = faceRadius * 0.28;
  const maxAdditional = faceRadius * 0.55;

  drawTicksAndLabels(tickRadius);
  drawRadialSpikes(innerRadius, maxAdditional);
  drawCatFeatures(faceRadius);

  // Added: overlay for grid view (titles/counts)
  drawGridOverlay();
}


function ensureMosaicComputed(r) {
  if (mosaicComputedFor.w === width && mosaicComputedFor.h === height && mosaicTiles.length > 0) return;
  computeMosaicTiles(r);
  mosaicComputedFor = { w: width, h: height };
}

function computeMosaicTiles(r) {
  mosaicTiles = [];

  // Estimate a tile size so that we have at least as many cells as images
  const totalCount = countsArray.reduce((a, b) => a + b, 0);
  const circleArea = Math.PI * r * r;
  mosaicTileSize = Math.max(4, Math.min(Math.sqrt(circleArea / Math.max(1, totalCount)) * 0.9, r * 0.12));
  const step = mosaicTileSize; // grid step

  // Precompute ear triangles (world coordinates)
  const ears = computeEarWorldTriangles(r);

  // Build candidate grid inside face (circle + ears)
  const bounds = r * 1.85;
  const jitter = step * 0.25;
  const candidates = [];
  let idxCounter = 0;
  for (let y = -bounds; y <= bounds; y += step) {
    for (let x = -bounds; x <= bounds; x += step) {
      const jx = x + (Math.random() * 2 - 1) * jitter;
      const jy = y + (Math.random() * 2 - 1) * jitter;
      if (!isRectInsideFace(jx, jy, step, r, ears)) continue;
      const phi = Math.atan2(jy, jx);
      const dists = categoryCenterAngles.map(a => angularDistance(phi, a));
      let nearestCat = 0;
      let nearestDist = dists[0];
      for (let k = 1; k < dists.length; k++) {
        if (dists[k] < nearestDist) {
          nearestDist = dists[k];
          nearestCat = k;
        }
      }
      candidates.push({ idx: idxCounter++, x: jx, y: jy, phi, dists, nearestDist, nearestCat });
    }
  }

  if (candidates.length === 0) return;

  // Partition by nearest category and sort by closeness
  const byCat = CATEGORY_DISPLAY_ORDER.map(() => []);
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i].nearestCat;
    byCat[c].push(candidates[i]);
  }
  for (let c = 0; c < byCat.length; c++) {
    byCat[c].sort((a, b) => a.nearestDist - b.nearestDist);
  }

  // Remaining needs per category
  const needs = countsArray.slice();
  const selectedByCat = CATEGORY_DISPLAY_ORDER.map(() => []);
  const taken = new Array(candidates.length).fill(false);

  // Step 1: Fill using cells whose nearest category is this one
  for (let c = 0; c < CATEGORY_DISPLAY_ORDER.length; c++) {
    const want = needs[c];
    if (want <= 0) continue;
    const arr = byCat[c];
    let p = 0;
    while (selectedByCat[c].length < want && p < arr.length) {
      const cand = arr[p++];
      if (!taken[cand.idx]) {
        taken[cand.idx] = true;
        selectedByCat[c].push(cand);
      }
    }
  }

  // Step 2: If still short, fill with closest remaining cells by angular distance to that category
  for (let c = 0; c < CATEGORY_DISPLAY_ORDER.length; c++) {
    const want = needs[c];
    if (want <= selectedByCat[c].length) continue;
    const remaining = [];
    for (let i = 0; i < candidates.length; i++) {
      const cand = candidates[i];
      if (!taken[cand.idx]) remaining.push(cand);
    }
    remaining.sort((a, b) => a.dists[c] - b.dists[c]);
    let p = 0;
    while (selectedByCat[c].length < want && p < remaining.length) {
      const cand = remaining[p++];
      if (!taken[cand.idx]) {
        taken[cand.idx] = true;
        selectedByCat[c].push(cand);
      }
    }
  }

  // Build tiles with category colors
  const colorIdx = CATEGORY_DISPLAY_ORDER.map(() => 0);
  for (let c = 0; c < CATEGORY_DISPLAY_ORDER.length; c++) {
    const arr = selectedByCat[c];
    for (let i = 0; i < arr.length; i++) {
      const idx = colorIdx[c] % Math.max(1, categoryColorsByIndex[c].length);
      const colorHex = categoryColorsByIndex[c][idx] || '#cccccc';
      colorIdx[c]++;
      mosaicTiles.push({ x: arr[i].x, y: arr[i].y, size: mosaicTileSize, hex: colorHex, catIndex: c });
    }
  }
}


function drawCatFaceBase(r) {
  /*
  push();
  noStroke();
  fill(0, 0, 0, 20);
  ellipse(8, 18, r * 2.06, r * 2.06);

  pop();
  */

  // Overdraw ear outlines to keep silhouette readable above mosaic
  const earOffsetX = r * 0.62;
  const earOffsetY = -r * 0.65;
  const earSize = r * 0.9;
  drawEarOutline(-earOffsetX, earOffsetY, earSize, false);
  drawEarOutline(earOffsetX, earOffsetY, earSize, true);
  
  drawMosaicTiles();
}

function drawMosaicTiles() {
  if (!mosaicTiles || mosaicTiles.length === 0) return;
  push();
  noStroke();
  for (let i = 0; i < mosaicTiles.length; i++) {
    const t = mosaicTiles[i];
    // Added: compute animated rendering params for selected category
    const render = getTileRenderParams(i, t);
    if (!render.visible) continue;
    fill(render.hex);
    rectMode(CENTER);
    rect(render.x, render.y, render.size, render.size, render.size * 0.15);
  }
  pop();
}

function computeEarWorldTriangles(r) {
  const earOffsetX = r * 0.62;
  const earOffsetY = -r * 0.65;
  const earSize = r * 0.9;
  const h = earSize * 0.9;
  const w = earSize * 0.7;

  const localA = createVector(-w * 0.5, h * 0.5);
  const localB = createVector(0, -h * 0.5);
  const localC = createVector(w * 0.5, h * 0.5);

  function transform(pt, rot, tx, ty) {
    const x = pt.x * Math.cos(rot) - pt.y * Math.sin(rot) + tx;
    const y = pt.x * Math.sin(rot) + pt.y * Math.cos(rot) + ty;
    return { x, y };
  }

  const leftRot = -0.2;
  const rightRot = 0.2;
  const leftTx = -earOffsetX;
  const rightTx = earOffsetX;
  const ty = earOffsetY;

  const left = [
    transform(localA, leftRot, leftTx, ty),
    transform(localB, leftRot, leftTx, ty),
    transform(localC, leftRot, leftTx, ty)
  ];
  const right = [
    transform(localA, rightRot, rightTx, ty),
    transform(localB, rightRot, rightTx, ty),
    transform(localC, rightRot, rightTx, ty)
  ];

  return { left, right };
}

function isInsideFace(x, y, r, ears) {
  // Inside head circle
  if (x * x + y * y <= r * r) return true;
  // Inside ears triangles
  const L = ears.left;
  const R = ears.right;
  if (pointInTriangle({ x, y }, L[0], L[1], L[2])) return true;
  if (pointInTriangle({ x, y }, R[0], R[1], R[2])) return true;
  return false;
}

function isRectInsideFace(cx, cy, size, r, ears) {
  const half = size / 2;
  const x1 = cx - half, y1 = cy - half;
  const x2 = cx + half, y2 = cy - half;
  const x3 = cx + half, y3 = cy + half;
  const x4 = cx - half, y4 = cy + half;
  if (!isInsideFace(x1, y1, r, ears)) return false;
  if (!isInsideFace(x2, y2, r, ears)) return false;
  if (!isInsideFace(x3, y3, r, ears)) return false;
  if (!isInsideFace(x4, y4, r, ears)) return false;
  return true;
}

function pointInTriangle(p, a, b, c) {
  function sign(p1, p2, p3) {
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
  }
  const b1 = sign(p, a, b) < 0.0;
  const b2 = sign(p, b, c) < 0.0;
  const b3 = sign(p, c, a) < 0.0;
  return (b1 === b2) && (b2 === b3);
}

function angularDistance(a, b) {
  let d = Math.abs(a - b) % TWO_PI;
  return d > PI ? TWO_PI - d : d;
}

function drawEar(x, y, size, flip) {
  push();
  translate(x, y);
  rotate(flip ? 0.2 : -0.2);
  noStroke();
  fill(255, 229, 200);
  const h = size * 0.9;
  const w = size * 0.7;
  triangle(-w * 0.5, h * 0.5, 0, -h * 0.5, w * 0.5, h * 0.5);
  fill(255, 190, 220);
  triangle(-w * 0.3, h * 0.3, 0, -h * 0.35, w * 0.3, h * 0.3);
  pop();
}


function drawEarOutline(x, y, size, flip) {
  push();
  translate(x, y);
  rotate(flip ? 0.2 : -0.2);
  fill(40, 40, 40, 140);
  stroke(40, 40, 40, 140);
  //strokeWeight(Math.max(1, size * 0.04));
  const h = size * 0.9;
  const w = size * 0.7;
  triangle(-w * 0.5, h * 0.5, 0, -h * 0.5, w * 0.5, h * 0.5);
  noStroke();
  fill(255, 190, 220, 160);
  triangle(-w * 0.3, h * 0.3, 0, -h * 0.35, w * 0.3, h * 0.3);
  pop();
}


function drawCatFeatures(r) {
  push();
  noStroke();
  fill(180, 190, 129, 200);
  const eyeY = -r * 0.1;
  const eyeX = r * 0.5;
  const eyeW = r * 0.18;
  const eyeH = r * 0.22;
  ellipse(-eyeX, eyeY, eyeW, eyeH);
  ellipse(eyeX, eyeY, eyeW, eyeH);

  fill(0, 0, 0, 180);
  ellipse(-eyeX - eyeW * 0.05, eyeY - eyeH * 0.05, eyeW * 0.28, eyeH * 0.68);
  ellipse(eyeX - eyeW * 0.05, eyeY - eyeH * 0.05, eyeW * 0.28, eyeH * 0.68);

  fill(240, 120, 140);
  const noseW = r * 0.14;
  const noseY = r * 0.05;
  push();
  strokeJoin(ROUND);
  triangle(-noseW * 0.5, noseY, 0, noseY + noseW * 0.65, noseW * 0.5, noseY);
  pop();

  noFill();
  stroke(80);
  strokeWeight(Math.max(1, r * 0.02));
  const mouthY = noseY + noseW * 0.5;
  const mouthR = r * 0.1;
  arc(-noseW * 0.2, mouthY, mouthR, mouthR * 0.8, 0.75, PI - 0.15);
  arc(noseW * 0.2, mouthY, mouthR, mouthR * 0.8, 0.15, PI - 0.75);

  /*
  stroke(255, 100);
  strokeWeight(Math.max(0.5, r * 0.015));
  const whiskerY = noseY + noseW * 0.2;
  for (let i = 0; i < 3; i++) {
    const dy = whiskerY + i * r * 0.08;
    line(-r * 0.15, dy, -r * 0.65, dy - r * (0.06 + i * 0.02));
    line(-r * 0.15, dy, -r * 0.65, dy + r * (0.01 + i * 0.03));
    line(r * 0.15, dy, r * 0.65, dy - r * (0.01 + i * 0.03));
    line(r * 0.15, dy, r * 0.65, dy + r * (0.06 + i * 0.02));
  }
  */
  pop();
}

function drawTicksAndLabels(radius) {
  push();
  stroke(60, 60, 60, 90);
  strokeWeight(1.5);
  fill(30);
  textSize(Math.max(10, radius * 0.09));
  textAlign(CENTER, CENTER);
  const labelRadius = radius * 1.1;
  // Added: reset hitboxes for this frame
  labelHitboxes = CATEGORY_DISPLAY_ORDER.map(() => null);
  for (let i = 0; i < CATEGORY_DISPLAY_ORDER.length; i++) {
    const angle = -HALF_PI + (i * TWO_PI) / CATEGORY_DISPLAY_ORDER.length;
    const x1 = Math.cos(angle) * (radius - 6);
    const y1 = Math.sin(angle) * (radius - 6);
    const x2 = Math.cos(angle) * (radius + 6);
    const y2 = Math.sin(angle) * (radius + 6);
    line(x1, y1, x2*1.2, y2*1.2);

    const lx = Math.cos(angle) * labelRadius;
    const ly = Math.sin(angle) * labelRadius;
    push();
    translate(lx*1.33, ly*1.33);
    //rotate(angle);
    const label = CATEGORY_DISPLAY_ORDER[i];
    noStroke();
    fill(0);
    textFont("Futura");
    text(shortenLabel(label), 0, 0);
    pop();

    // Added: compute a conservative hitbox around rendered text (centered at lx*1.33, ly*1.33)
    const ts = Math.max(10, radius * 0.09);
    const shortLabel = shortenLabel(CATEGORY_DISPLAY_ORDER[i]);
    const lines = shortLabel.split('\n');
    let maxW = 0;
    for (let k = 0; k < lines.length; k++) {
      maxW = Math.max(maxW, textWidth(lines[k]));
    }
    const h = lines.length * ts * 1.15;
    const w = maxW;
    labelHitboxes[i] = { x: lx*1.33, y: ly*1.33, w: w + 10, h: h + 8 };
  }
  pop();
}

function shortenLabel(label) {
  const words = label.split(' ');
  if (words.length <= 2) return label;
  const mid = Math.ceil(words.length / 2);
  return words.slice(0, mid).join(' ') + '\n' + words.slice(mid).join(' ');
}

function drawRadialSpikes(innerRadius, maxAdditional) {
  const maxCount = countsArray.length ? Math.max(...countsArray) : 1;
  const n = CATEGORY_DISPLAY_ORDER.length;

  push();
  noFill();
  stroke(120, 80);
  strokeWeight(1);
  const rings = 4;
  for (let r = 0; r <= rings; r++) {
    const t = r / rings;
    const rad = innerRadius + maxAdditional * t;
    ellipse(0, 0, rad * 2, rad * 2);
  }

  noStroke();
  noFill();
  beginShape();
  for (let i = 0; i < n; i++) {
    const count = countsArray[i] || 0;
    const t = maxCount > 0 ? count / maxCount : 0;
    const radius = innerRadius + maxAdditional * t;
    const angle = -HALF_PI + (i * TWO_PI) / n;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    vertex(x, y);
  }
  endShape(CLOSE);

  fill(255,255,255,180);
  stroke(255, 255, 255);
  strokeWeight(2);
  beginShape();
  for (let i = 0; i < n; i++) {
    const count = countsArray[i] || 0;
    const t = maxCount > 0 ? count / maxCount : 0;
    const radius = innerRadius + maxAdditional * t;
    const angle = -HALF_PI + (i * TWO_PI) / n;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    curveVertex(x, y);
  }
  for (let i = 0; i < 3; i++) {
    const count = countsArray[i] || 0;
    const t = maxCount > 0 ? count / maxCount : 0;
    const radius = innerRadius + maxAdditional * t;
    const angle = -HALF_PI + (i * TWO_PI) / n;
    curveVertex(Math.cos(angle) * radius, Math.sin(angle) * radius);
  }
  endShape();

  
  fill(35);
  noStroke();
  textSize(Math.max(10, innerRadius * 0.25));
  for (let i = 0; i < n; i++) {
    const count = countsArray[i] || 0;
    const t = maxCount > 0 ? count / maxCount : 0;
    const radius = innerRadius + maxAdditional * t;
    const angle = -HALF_PI + (i * TWO_PI) / n;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    fill(35);
    ellipse(x, y, innerRadius * 0.08, innerRadius * 0.08);
    fill(0, 190, 190);
    textFont("Futura");
    textSize(18);
    textStyle(BOLD);
    text(count, x, y - innerRadius * 0.12);
  }
  pop();
  
}

// Added: Build per-category items and selfie stats for grid view
function computeCategoryItemsAndSelfieStats() {
  itemsByCategory = CATEGORY_DISPLAY_ORDER.map(() => []);
  selfieStatsByCategory = CATEGORY_DISPLAY_ORDER.map(() => ({ yes: 0, no: 0, unknown: 0, total: 0 }));
  if (!photoTable) return;
  for (let r = 0; r < photoTable.getRowCount(); r++) {
    const rawLabel = photoTable.getString(r, 'body_position');
    const normalized = normalizeBodyPosition(rawLabel);
    const idx = CATEGORY_DISPLAY_ORDER.indexOf(normalized);
    if (idx < 0) continue;
    const hex = sanitizeHex(photoTable.getString(r, 'average_hex_color'));
    const selfieRaw = (photoTable.getString(r, 'is_selfie') || '').toString().trim().toLowerCase();
    const selfie = selfieRaw === 'yes' ? 'yes' : (selfieRaw === 'no' ? 'no' : 'unknown');
    itemsByCategory[idx].push({ hex, selfie });
    selfieStatsByCategory[idx].total += 1;
    selfieStatsByCategory[idx][selfie] += 1;
  }
}

// Added: Animation and grid helpers
function startEnterAnimation(catIndex) {
  if (catIndex < 0 || catIndex >= CATEGORY_DISPLAY_ORDER.length) return;
  interactionState.selectedCategoryIndex = catIndex;
  prepareGridLayoutForCategory(catIndex);
  interactionState.animating = true;
  interactionState.animationStartMs = millis();
  interactionState.animationDirection = 1;
  interactionState.progress = 0;
  interactionState.mode = 'radial';
  loop();
}

function startExitAnimation() {
  if (interactionState.selectedCategoryIndex < 0) return;
  interactionState.animating = true;
  interactionState.animationStartMs = millis();
  interactionState.animationDirection = -1;
  interactionState.progress = 1;
  loop();
}

function updateAnimationState() {
  if (!interactionState.animating) return;
  const now = millis();
  const t = constrain((now - interactionState.animationStartMs) / Math.max(1, interactionState.animationDurationMs), 0, 1);
  const e = easeInOutCubic(t);
  if (interactionState.animationDirection === 1) {
    interactionState.progress = e;
    if (t >= 1) {
      interactionState.animating = false;
      interactionState.mode = 'grid';
      interactionState.progress = 1;
      noLoop();
    }
  } else if (interactionState.animationDirection === -1) {
    interactionState.progress = 1 - e;
    if (t >= 1) {
      interactionState.animating = false;
      interactionState.mode = 'radial';
      interactionState.selectedCategoryIndex = -1;
      tileMappingByIndex = {};
      gridLayout = null;
      interactionState.progress = 0;
      noLoop();
    }
  }
}

function prepareGridLayoutForCategory(catIndex, overrideScreenTop) {
  const items = (itemsByCategory[catIndex] || []).slice();
  // Sort by hue
  items.sort((a, b) => hueFromHex(a.hex) - hueFromHex(b.hex));

  // Compute grid area on the right side of the screen
  const screenLeft = computeGridScreenLeft();
  const screenTop = (overrideScreenTop != null ? overrideScreenTop : (overlayBottomY() + OVERLAY_AFTER_TEXT_PADDING));
  const screenRight = width - 24;
  const screenBottom = height - 36; // extra bottom padding
  const availW = Math.max(40, screenRight - screenLeft);
  const availH = Math.max(40, screenBottom - screenTop);

  const n = Math.max(1, items.length);
  // Choose columns to roughly square the grid
  let cols = Math.max(1, Math.floor(Math.sqrt(n * (availW / Math.max(1, availH)))));
  cols = Math.min(cols, Math.max(1, Math.floor(availW / Math.max(6, mosaicTileSize))));
  cols = Math.max(1, Math.min(cols, n));
  const rows = Math.ceil(n / cols);
  // Scale down a bit more to guarantee fit
  const tileSize = Math.floor(Math.min(availW / cols, availH / rows) * 0.7);

  const leftWorld = screenLeft - getRadialCenterX();
  const topWorld = screenTop - getRadialCenterY();

  gridLayout = {
    left: leftWorld,
    top: topWorld,
    cols,
    rows,
    tileSize,
    title: CATEGORY_DISPLAY_ORDER[catIndex],
    bounds: {
      x1: leftWorld + tileSize * 0.5,
      y1: topWorld + tileSize * 0.5,
      x2: leftWorld + cols * tileSize - tileSize * 0.5,
      y2: topWorld + rows * tileSize - tileSize * 0.5
    }
  };

  // Map from existing mosaic tiles of this category to target grid cells
  const tileIndices = [];
  for (let i = 0; i < mosaicTiles.length; i++) {
    if (mosaicTiles[i].catIndex === catIndex) tileIndices.push(i);
  }
  tileMappingByIndex = {};
  for (let i = 0; i < items.length && i < tileIndices.length; i++) {
    const idx = tileIndices[i];
    const col = i % cols;
    const row = Math.floor(i / cols);
    const toX = leftWorld + col * tileSize + tileSize * 0.5;
    const toY = topWorld + row * tileSize + tileSize * 0.5;
    tileMappingByIndex[idx] = {
      toX,
      toY,
      toSize: tileSize,
      targetHex: items[i].hex
    };
  }
}

function getTileRenderParams(i, t) {
  // Default: render as-is
  let x = t.x;
  let y = t.y;
  let size = t.size;
  let hex = t.hex;
  let visible = true;

  const sel = interactionState.selectedCategoryIndex;
  const p = interactionState.progress;
  const mapping = tileMappingByIndex[i];

  if (sel >= 0 && t.catIndex === sel && mapping) {
    // Animate from radial position to grid position
    const u = p; // already eased
    x = lerp(t.x, mapping.toX, u);
    y = lerp(t.y, mapping.toY, u);
    size = lerp(t.size, mapping.toSize, u);
    // Switch to target hex halfway-through for a clean hue-sorted grid
    hex = u < 0.5 ? t.hex : mapping.targetHex;
  }

  // When returning to radial, mapping still exists but progress decreases
  // When no category selected, ensure all tiles show
  return { x, y, size, hex, visible };
}

function drawGridOverlay() {
  const sel = interactionState.selectedCategoryIndex;
  if (sel < 0 || !gridLayout) return;

  // Determine alpha based on progress to fade in text
  const p = interactionState.progress;
  const alpha = Math.floor(255 * constrain(p, 0, 1));

  // Use screen coordinates
  push();
  resetMatrix();
  noStroke();
  fill(20, 20, 20, alpha);

  const title = CATEGORY_DISPLAY_ORDER[sel];
  const stats = selfieStatsByCategory[sel] || { total: 0, yes: 0, no: 0, unknown: 0 };

  const screenLeft = computeGridScreenLeft();
  const titleX = screenLeft;
  const titleY = OVERLAY_TITLE_Y;

  textFont('Futura');
  textAlign(LEFT, TOP);

  textSize(32);
  textStyle(BOLD);
  text(title, titleX, titleY);

  textStyle(NORMAL);
  textSize(18);
  const totalLine = (stats.total || 0) + ' images in ' + title.toLowerCase() + ' position';
  text(totalLine, titleX, titleY + OVERLAY_TITLE_OFFSET + OVERLAY_BLOCK_SPACING);

  const s1 = 'Selfies: ' + (stats.yes || 0);
  const s2 = 'Non-Selfies: ' + (stats.no || 0);
  const s3 = 'No Selfie Data: ' + (stats.unknown || 0);
  const firstLineY = titleY + OVERLAY_TITLE_OFFSET + OVERLAY_BLOCK_SPACING + OVERLAY_LINE_HEIGHT;
  text(s1, titleX, firstLineY);
  text(s2, titleX, firstLineY + OVERLAY_LINE_HEIGHT);
  text(s3, titleX, firstLineY + OVERLAY_LINE_HEIGHT * 2);

  pop();
}

function hueFromHex(hex) {
  const { r, g, b } = hexToRgb(hex);
  const { h } = rgbToHsl(r, g, b);
  return h;
}

function hexToRgb(hex) {
  const s = sanitizeHex(hex).slice(1);
  const r = parseInt(s.substring(0, 2), 16);
  const g = parseInt(s.substring(2, 4), 16);
  const b = parseInt(s.substring(4, 6), 16);
  return { r, g, b };
}

function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h, s, l = (max + min) / 2;
  if (max === min) {
    h = s = 0;
  } else {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      case b: h = (r - g) / d + 4; break;
    }
    h /= 6;
  }
  return { h: h * 360, s, l };
}

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

// Hit-testing and interaction
function mousePressed() {
  // Convert to world coordinates (after draw translates by width/2,height/2)
  const wx = mouseX - getRadialCenterX();
  const wy = mouseY - getRadialCenterY();

  if (interactionState.animating) return;

  if (interactionState.mode === 'radial') {
    // 1) Hit test labels
    const li = hitTestLabel(wx, wy);
    if (li >= 0) {
      startEnterAnimation(li);
      return;
    }
    // 2) Hit test tiles
    const ti = hitTestTile(wx, wy);
    if (ti >= 0) {
      startEnterAnimation(mosaicTiles[ti].catIndex);
      return;
    }
  } else if (interactionState.mode === 'grid') {
    // Click outside the grid bounds returns to radial
    if (!isPointInsideBounds(wx, wy, gridLayout && gridLayout.bounds)) {
      startExitAnimation();
    }
  }
}

function hitTestLabel(wx, wy) {
  for (let i = 0; i < labelHitboxes.length; i++) {
    const hb = labelHitboxes[i];
    if (!hb) continue;
    if (Math.abs(wx - hb.x) <= hb.w * 0.5 && Math.abs(wy - hb.y) <= hb.h * 0.5) return i;
  }
  return -1;
}

function hitTestTile(wx, wy) {
  // Prioritize nearest tile by simple scan (counts are modest)
  let bestIdx = -1;
  let bestDist2 = Infinity;
  for (let i = 0; i < mosaicTiles.length; i++) {
    const t = mosaicTiles[i];
    const half = t.size * 0.5;
    if (wx >= t.x - half && wx <= t.x + half && wy >= t.y - half && wy <= t.y + half) {
      const dx = wx - t.x;
      const dy = wy - t.y;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestDist2) { bestDist2 = d2; bestIdx = i; }
    }
  }
  return bestIdx;
}

function isPointInsideBounds(wx, wy, bounds) {
  if (!bounds) return false;
  return wx >= bounds.x1 && wx <= bounds.x2 && wy >= bounds.y1 && wy <= bounds.y2;
}
