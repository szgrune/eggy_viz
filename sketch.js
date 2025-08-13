let photoTable;

const CATEGORY_DISPLAY_ORDER = [
  'standing on all fours',
  'standing up on hind legs',
  'loaf',
  'curled up',
  'between legs',
  'belly up',
  'pretzel',
  'laying on side',
  'other'
];

let categoryCounts = {};
let countsArray = [];

let categoryColorsByIndex = [];
let categoryCenterAngles = [];

let mosaicTiles = [];
let mosaicTileSize = 10;
let mosaicComputedFor = { w: -1, h: -1 };


function preload() {
  photoTable = loadTable('cat_analysis_ref_no_comparison.csv', 'csv', 'header');
}

function setup() {
  createCanvas(windowWidth, windowHeight);
  pixelDensity(2);
  textFont('sans-serif');
  countCategories();
  noLoop();
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);

  mosaicComputedFor = { w: -1, h: -1 };

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
  if (!s) return 'other';
  if (s.includes('standing') && s.includes('hind')) return 'standing up on hind legs';
  if (s.includes('standing') && (s.includes('all fours') || s.includes('fours'))) return 'standing on all fours';
  if (s.includes('loaf')) return 'loaf';
  if (s.includes('curl')) return 'curled up';
  if (s.includes('between') && s.includes('legs')) return 'between legs';
  if (s.includes('belly')) return 'belly up';
  if (s.includes('pretzel')) return 'pretzel';
  if ((s.includes('laying') && s.includes('side')) || s.includes('on side')) return 'laying on side';
  if (s === 'other') return 'other';
  return 'other';
}

function draw() {
  background(253, 250, 246);

  translate(width / 2, height / 2);

  const faceRadius = Math.min(width, height) * 0.28;


  ensureMosaicComputed(faceRadius);

  drawCatFaceBase(faceRadius);

  const tickRadius = faceRadius * 0.95;
  const innerRadius = faceRadius * 0.28;
  const maxAdditional = faceRadius * 0.55;

  drawTicksAndLabels(tickRadius);
  drawRadialSpikes(innerRadius, maxAdditional);
  drawCatFeatures(faceRadius);
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
  push();
  noStroke();
  fill(0, 0, 0, 20);
  ellipse(8, 18, r * 2.06, r * 2.06);

  pop();

  drawMosaicTiles();

  // Overdraw ear outlines to keep silhouette readable above mosaic
  const earOffsetX = r * 0.62;
  const earOffsetY = -r * 0.65;
  const earSize = r * 0.9;
  drawEarOutline(-earOffsetX, earOffsetY, earSize, false);
  drawEarOutline(earOffsetX, earOffsetY, earSize, true);
}

function drawMosaicTiles() {
  if (!mosaicTiles || mosaicTiles.length === 0) return;
  push();
  noStroke();
  for (let i = 0; i < mosaicTiles.length; i++) {
    const t = mosaicTiles[i];
    fill(t.hex);
    rectMode(CENTER);
    rect(t.x, t.y, t.size, t.size, t.size * 0.15);
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
  noFill();
  stroke(40, 40, 40, 140);
  strokeWeight(Math.max(1, size * 0.04));
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
  fill(35, 35, 35);
  const eyeY = -r * 0.15;
  const eyeX = r * 0.35;
  const eyeW = r * 0.18;
  const eyeH = r * 0.22;
  ellipse(-eyeX, eyeY, eyeW, eyeH);
  ellipse(eyeX, eyeY, eyeW, eyeH);

  fill(255, 255, 255, 200);
  ellipse(-eyeX - eyeW * 0.15, eyeY - eyeH * 0.15, eyeW * 0.18, eyeH * 0.18);
  ellipse(eyeX - eyeW * 0.15, eyeY - eyeH * 0.15, eyeW * 0.18, eyeH * 0.18);

  fill(240, 120, 140);
  const noseW = r * 0.14;
  const noseY = r * 0.05;
  triangle(-noseW * 0.5, noseY, 0, noseY + noseW * 0.35, noseW * 0.5, noseY);

  noFill();
  stroke(80);
  strokeWeight(Math.max(1, r * 0.02));
  const mouthY = noseY + noseW * 0.4;
  const mouthR = r * 0.22;
  arc(-noseW * 0.2, mouthY, mouthR, mouthR * 0.8, 0.15, PI - 0.15);
  arc(noseW * 0.2, mouthY, mouthR, mouthR * 0.8, 0.15, PI - 0.15);

  stroke(80, 120);
  strokeWeight(Math.max(0.5, r * 0.015));
  const whiskerY = noseY + noseW * 0.2;
  for (let i = 0; i < 3; i++) {
    const dy = whiskerY + i * r * 0.08;
    line(-r * 0.15, dy, -r * 0.65, dy - r * (0.06 + i * 0.02));
    line(-r * 0.15, dy, -r * 0.65, dy + r * (0.01 + i * 0.03));
    line(r * 0.15, dy, r * 0.65, dy - r * (0.01 + i * 0.03));
    line(r * 0.15, dy, r * 0.65, dy + r * (0.06 + i * 0.02));
  }
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
  for (let i = 0; i < CATEGORY_DISPLAY_ORDER.length; i++) {
    const angle = -HALF_PI + (i * TWO_PI) / CATEGORY_DISPLAY_ORDER.length;
    const x1 = Math.cos(angle) * (radius - 6);
    const y1 = Math.sin(angle) * (radius - 6);
    const x2 = Math.cos(angle) * (radius + 6);
    const y2 = Math.sin(angle) * (radius + 6);
    line(x1, y1, x2, y2);

    const lx = Math.cos(angle) * labelRadius;
    const ly = Math.sin(angle) * labelRadius;
    push();
    translate(lx, ly);
    rotate(angle);
    const label = CATEGORY_DISPLAY_ORDER[i];
    noStroke();
    fill(50);
    text(shortenLabel(label), 0, 0);
    pop();
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
  fill(255, 200, 120, 180);
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

  noFill();
  stroke(255, 150, 60);
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
  textSize(Math.max(10, innerRadius * 0.15));
  for (let i = 0; i < n; i++) {
    const count = countsArray[i] || 0;
    const t = maxCount > 0 ? count / maxCount : 0;
    const radius = innerRadius + maxAdditional * t;
    const angle = -HALF_PI + (i * TWO_PI) / n;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    fill(35);
    ellipse(x, y, innerRadius * 0.08, innerRadius * 0.08);
    fill(20);
    text(count, x, y - innerRadius * 0.12);
  }
  pop();
}