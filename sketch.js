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
  redraw();
}

function countCategories() {
  categoryCounts = {};
  for (let label of CATEGORY_DISPLAY_ORDER) {
    categoryCounts[label] = 0;
  }
  if (photoTable) {
    for (let r = 0; r < photoTable.getRowCount(); r++) {
      const raw = photoTable.getString(r, 'body_position');
      const normalized = normalizeBodyPosition(raw);
      if (Object.prototype.hasOwnProperty.call(categoryCounts, normalized)) {
        categoryCounts[normalized] += 1;
      } else {
        categoryCounts['other'] += 1;
      }
    }
  }
  countsArray = CATEGORY_DISPLAY_ORDER.map(label => categoryCounts[label] || 0);
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
  drawCatFaceBase(faceRadius);

  const tickRadius = faceRadius * 0.95;
  const innerRadius = faceRadius * 0.28;
  const maxAdditional = faceRadius * 0.55;

  drawTicksAndLabels(tickRadius);
  drawRadialSpikes(innerRadius, maxAdditional);
  drawCatFeatures(faceRadius);
}

function drawCatFaceBase(r) {
  push();
  noStroke();
  fill(0, 0, 0, 20);
  ellipse(8, 18, r * 2.06, r * 2.06);

  fill(255, 229, 200);
  ellipse(0, 0, r * 2, r * 2);

  const earOffsetX = r * 0.62;
  const earOffsetY = -r * 0.65;
  const earSize = r * 0.9;
  drawEar(-earOffsetX, earOffsetY, earSize, false);
  drawEar(earOffsetX, earOffsetY, earSize, true);
  pop();
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