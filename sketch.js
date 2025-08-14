let photoTable;
let hexToImageMap = {}; // Store mapping from hex values to individual photo images
let bodyPositionImages = {}; // Store body position images for tick mark labels
let hoveredImage = null; // Currently displayed hover image
let hoveredImagePos = { x: 0, y: 0 }; // Position for hover image

const CATEGORY_DISPLAY_ORDER = [
  'Standing on all fours',
  'Standing up on hind legs',
  'Loaf',
  'Curled up',
  'Sitting on human',
  'Belly up',
  'Pretzel',
  'Laying on side',
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
function computeCanvasWidth() {
  return windowWidth;
}
function computeCanvasHeight() {
  return windowHeight;
}
function getRadialCenterX() { return Math.floor(windowWidth * 0.28); }
function getRadialCenterY() { return Math.floor(windowHeight / 2); }
function computeGridScreenLeft() { return getRadialCenterX() + currentFaceRadius * 1.4 + 40; }

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
  // Load the CSV table first, then load individual photos once it's ready
  photoTable = loadTable('cat_analysis_8cats.csv', 'csv', 'header', () => {
    console.log('CSV loaded successfully, now loading individual photos...');
    // Load individual photos after CSV is ready
    loadIndividualPhotos();
  });
  
  // Load body position images for tick mark labels
  loadBodyPositionImages();
}

// Load body position images for tick mark labels
function loadBodyPositionImages() {
  console.log('Loading body position images for tick marks...');
  
  for (let category of CATEGORY_DISPLAY_ORDER) {
    const imageName = category + '.png'; // Use PNG format as specified
    bodyPositionImages[category] = loadImage('body_position_drawings/' + imageName, 
      // Success callback
      () => console.log('Successfully loaded body position image for:', category),
      // Error callback
      () => {
        console.warn('Failed to load body position image for:', category);
        bodyPositionImages[category] = null;
      }
    );
  }
}

// Load individual photos and create hex-to-image mapping
function loadIndividualPhotos() {
  console.log('loadIndividualPhotos called');
  console.log('photoTable:', photoTable);
  
  if (!photoTable) {
    console.error('photoTable is null or undefined');
    return;
  }
  
  console.log('Photo table row count:', photoTable.getRowCount());
  console.log('Loading individual photos...');
  
  // Get the first row to see what columns are available
  if (photoTable.getRowCount() > 0) {
    const firstRow = photoTable.getRow(0);
    const columnNames = Object.keys(firstRow.obj);
    console.log('Available CSV columns:', columnNames);
    
    // Try different possible column names for the filename
    const possibleFilenameColumns = ['filename', 'image_filename', 'file', 'image', 'photo', 'img'];
    let filenameColumn = null;
    
    // Find the filename column
    for (const col of possibleFilenameColumns) {
      if (columnNames.includes(col)) {
        filenameColumn = col;
        console.log('Found filename column:', col);
        break;
      }
    }
    
    if (!filenameColumn) {
      console.warn('No filename column found. Available columns:', columnNames);
      return;
    }
    
    let loadedCount = 0;
    let totalCount = 0;
    
    console.log('Starting to process CSV rows...');
    
    // Load all images from the CSV rows
    for (let r = 0; r < photoTable.getRowCount(); r++) {
      const hex = sanitizeHex(photoTable.getString(r, 'average_hex_color'));
      const filename = photoTable.getString(r, filenameColumn);
      
      console.log(`Row ${r}: hex="${hex}", filename="${filename}"`);
      
      if (filename && hex) {
        totalCount++;
        console.log(`Loading image ${totalCount}: eggy_photos/${filename} for hex: ${hex}`);
        
        // Load the image from eggy_photos folder
        loadImage('eggy_photos/' + filename, 
          // Success callback
          (img) => {
            console.log(`✅ Successfully loaded image for hex ${hex}:`, filename);
            hexToImageMap[hex] = img;
            loadedCount++;
            console.log(`Loaded ${loadedCount}/${totalCount} images. Hashmap size: ${Object.keys(hexToImageMap).length}`);
            
            if (loadedCount === totalCount) {
              console.log(`🎉 All images loaded! Hashmap contains ${Object.keys(hexToImageMap).length} entries`);
              console.log('Hashmap keys (hex values):', Object.keys(hexToImageMap));
            }
          },
          // Error callback
          (err) => {
            console.error(`❌ Failed to load image: eggy_photos/${filename} for hex: ${hex}`, err);
            hexToImageMap[hex] = null;
          }
        );
      } else {
        console.warn(`Row ${r}: Missing hex or filename - hex: "${hex}", filename: "${filename}"`);
      }
    }
    
    console.log(`Total rows to process: ${totalCount}`);
  } else {
    console.error('Photo table has no rows!');
  }
}

function setup() {
  createCanvas(computeCanvasWidth(), computeCanvasHeight());
  pixelDensity(2);
  textFont('sans-serif');
  countCategories();
  noLoop();
  
  // Enable mouse tracking for hover effects
  loop(); // Enable continuous drawing for mouse tracking
  
  // Check if images loaded after a delay
  setTimeout(() => {
    console.log('🔍 Setup complete - checking image loading status:');
    console.log('  - Hashmap size:', Object.keys(hexToImageMap).length);
    console.log('  - Photo table rows:', photoTable ? photoTable.getRowCount() : 'null');
    if (Object.keys(hexToImageMap).length === 0) {
      console.warn('⚠️ No images loaded! This might indicate a loading issue.');
    } else {
      console.log('✅ Images loaded successfully!');
    }
  }, 3000); // Check after 3 seconds to allow for image loading
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
        //categoryCounts['other'] += 1;
        //const idx = CATEGORY_DISPLAY_ORDER.indexOf('other');
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
  //if (!s) return 'Other';
  if (s.includes('standing') && s.includes('hind')) return 'Standing up on hind legs';
  if (s.includes('standing') && (s.includes('all fours') || s.includes('fours'))) return 'Standing on all fours';
  if (s.includes('loaf')) return 'Loaf';
  if (s.includes('curl')) return 'Curled up';
  if (s.includes('sitting') && s.includes('human')) return 'Sitting on human';
  if (s.includes('belly')) return 'Belly up';
  if (s.includes('pretzel')) return 'Pretzel';
  if ((s.includes('laying') && s.includes('side')) || s.includes('on side')) return 'Laying on side';
  //if (s === 'other') return 'Other';
  //return 'Other';
}

function draw() {
  background(253, 250, 246);

  // Added: update animation progress if needed
  updateAnimationState();

  //translate(width / 2, height / 2);

  const radialCX = getRadialCenterX();
  const radialCY = getRadialCenterY();
  translate(radialCX, radialCY);

  const faceRadius = Math.min(windowWidth, windowHeight) * 0.28;
  currentFaceRadius = faceRadius;


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
  
  // Update hover state every frame for continuous detection
  updateHoverState();
  
  // Draw hover image if available
  drawHoverImage();
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
  // Make tiles bigger to fill more whitespace while keeping them within boundaries
  mosaicTileSize = Math.max(6, Math.min(Math.sqrt(circleArea / Math.max(1, totalCount)) * 1.0, r * 0.15));
  const step = mosaicTileSize; // grid step

  // Precompute ear triangles (world coordinates)
  const ears = computeEarWorldTriangles(r);

  // Build candidate grid inside face (circle + ears)
  const bounds = r * 1.85;
  const jitter = step * 0.25;
  const candidates = [];
  let idxCounter = 0;
  
  // Define section boundaries for each category
  const sectionAngles = [];
  for (let i = 0; i < CATEGORY_DISPLAY_ORDER.length; i++) {
    const startAngle = -HALF_PI + (i * TWO_PI) / CATEGORY_DISPLAY_ORDER.length;
    const endAngle = -HALF_PI + ((i + 1) * TWO_PI) / CATEGORY_DISPLAY_ORDER.length;
    const centerAngle = (startAngle + endAngle) / 2;
    const sectionWidth = TWO_PI / CATEGORY_DISPLAY_ORDER.length;
    sectionAngles.push({ start: startAngle, end: endAngle, center: centerAngle, width: sectionWidth });
  }
  
  for (let y = -bounds; y <= bounds; y += step) {
    for (let x = -bounds; x <= bounds; x += step) {
      const jx = x + (Math.random() * 2 - 1) * jitter;
      const jy = y + (Math.random() * 2 - 1) * jitter;
      if (!isRectInsideFace(jx, jy, step, r, ears)) continue;
      
      const phi = Math.atan2(jy, jx);
      // Normalize angle to [0, TWO_PI] for easier comparison
      let normalizedPhi = phi;
      if (normalizedPhi < 0) normalizedPhi += TWO_PI;
      
      // Find which section this point belongs to
      let sectionIndex = -1;
      for (let i = 0; i < sectionAngles.length; i++) {
        let start = sectionAngles[i].start;
        let end = sectionAngles[i].end;
        if (start < 0) start += TWO_PI;
        if (end < 0) end += TWO_PI;
        
        if (start <= end) {
          if (normalizedPhi >= start && normalizedPhi < end) {
            sectionIndex = i;
            break;
          }
        } else {
          // Handle case where section crosses 0/2π boundary
          if (normalizedPhi >= start || normalizedPhi < end) {
            sectionIndex = i;
            break;
          }
        }
      }
      
      if (sectionIndex === -1) continue;
      
      const dists = categoryCenterAngles.map(a => angularDistance(phi, a));
      let nearestCat = sectionIndex; // Force tiles to stay in their section
      let nearestDist = dists[sectionIndex];
      
      // Calculate distance from section center (tick mark) for better positioning
      const sectionCenter = sectionAngles[sectionIndex].center;
      const distFromCenter = angularDistance(phi, sectionCenter);
      
      candidates.push({ 
        idx: idxCounter++, 
        x: jx, 
        y: jy, 
        phi, 
        dists, 
        nearestDist, 
        nearestCat,
        distFromCenter,
        sectionCenter
      });
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
    // Sort by distance from section center (tick mark) for better centering
    byCat[c].sort((a, b) => a.distFromCenter - b.distFromCenter);
  }

  // Remaining needs per category
  const needs = countsArray.slice();
  const selectedByCat = CATEGORY_DISPLAY_ORDER.map(() => []);
  const taken = new Array(candidates.length).fill(false);

  // Step 1: Fill using cells whose nearest category is this one (strict section-based)
  // Prioritize tiles closer to the section center (tick mark) for better centering
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

  // Step 2: If still short, fill with remaining cells in the same section
  for (let c = 0; c < CATEGORY_DISPLAY_ORDER.length; c++) {
    const want = needs[c];
    if (selectedByCat[c].length >= want) continue;
    
    // Find remaining candidates in this category's section
    const remaining = [];
    for (let i = 0; i < candidates.length; i++) {
      const cand = candidates[i];
      if (!taken[cand.idx]) {
        // Check if this candidate is in the right section
        const phi = cand.phi;
        let normalizedPhi = phi;
        if (normalizedPhi < 0) normalizedPhi += TWO_PI;
        
        const startAngle = -HALF_PI + (c * TWO_PI) / CATEGORY_DISPLAY_ORDER.length;
        let endAngle = -HALF_PI + ((c + 1) * TWO_PI) / CATEGORY_DISPLAY_ORDER.length;
        let start = startAngle;
        let end = endAngle;
        if (start < 0) start += TWO_PI;
        if (end < 0) end += TWO_PI;
        
        let inSection = false;
        if (start <= end) {
          inSection = (normalizedPhi >= start && normalizedPhi < end);
        } else {
          // Handle case where section crosses 0/2π boundary
          inSection = (normalizedPhi >= start || normalizedPhi < end);
        }
        
        if (inSection) {
          remaining.push(cand);
        }
      }
    }
    
    // Sort by distance from section center for better clustering and spacing
    remaining.sort((a, b) => a.distFromCenter - b.distFromCenter);
    let p = 0;
    while (selectedByCat[c].length < want && p < remaining.length) {
      const cand = remaining[p++];
      if (!taken[cand.idx]) {
        taken[cand.idx] = true;
        selectedByCat[c].push(cand);
      }
    }
  }

  // Build tiles with category colors - ensure ALL tiles are created
  const colorIdx = CATEGORY_DISPLAY_ORDER.map(() => 0);
  for (let c = 0; c < CATEGORY_DISPLAY_ORDER.length; c++) {
    const targetCount = countsArray[c]; // Use the actual count from countsArray
    const arr = selectedByCat[c];
    
    // Create tiles for ALL items in this category, not just selected candidates
    for (let i = 0; i < targetCount; i++) {
      const idx = colorIdx[c] % Math.max(1, categoryColorsByIndex[c].length);
      const colorHex = categoryColorsByIndex[c][idx] || '#cccccc';
      colorIdx[c]++;
      
      // Generate random positions within each sector, with sector size proportional to tile count
      const sectionIndex = c;
      const totalSections = CATEGORY_DISPLAY_ORDER.length;
      
      // Calculate sector boundaries - each category gets a pie slice
      const sectionStartAngle = -HALF_PI + (sectionIndex * TWO_PI) / totalSections;
      const sectionEndAngle = -HALF_PI + ((sectionIndex + 1) * TWO_PI) / totalSections;
      
      // Use safe radius range that stays within circle boundaries
      const minRadius = r * 0.05; // Reduced to allow tiles in center
      const maxRadius = r * 0.85;
      
      // Generate completely random position within the sector
      const randomRadius = minRadius + Math.random() * (maxRadius - minRadius);
      const randomAngle = sectionStartAngle + Math.random() * (sectionEndAngle - sectionStartAngle);
      
      // Position tile
      tileX = Math.cos(randomAngle) * randomRadius;
      tileY = Math.sin(randomAngle) * randomRadius;
      
      // Apply rotation
      const rotationAngle = -22.5 * (PI / 180);
      const rotatedX = tileX * Math.cos(rotationAngle) - tileY * Math.sin(rotationAngle);
      const rotatedY = tileX * Math.sin(rotationAngle) + tileY * Math.cos(rotationAngle);
      tileX = rotatedX;
      tileY = rotatedY;
      
              mosaicTiles.push({ x: tileX, y: tileY, size: mosaicTileSize, hex: colorHex, catIndex: c });
    }
  }
}


function drawCatFaceBase(r) {
  // Draw background circle with same color as cat ears
  push();
  noStroke();
  fill(253, 250, 246); // Same color as cat ears
  ellipse(0, 0, r * 1.85, r * 1.85); // Full face circle
  pop();

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
  noStroke()
  fill(136, 134, 133);
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
  const eyeX = r * 0.3;
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
    
    const label = CATEGORY_DISPLAY_ORDER[i];
    const bodyImage = bodyPositionImages[label];
    
    if (bodyImage) {
      // Draw the body position image instead of text, preserving aspect ratio
      const baseSize = Math.max(40, radius * 0.5); // Base size for scaling
      const imgAspect = bodyImage.width / bodyImage.height;
      
      let displayWidth, displayHeight;
      if (imgAspect > 1) {
        // Landscape image - use baseSize for width
        displayWidth = baseSize;
        displayHeight = baseSize / imgAspect;
      } else {
        // Portrait or square image - use baseSize for height
        displayHeight = baseSize;
        displayWidth = baseSize * imgAspect;
      }
      
      imageMode(CENTER);
      image(bodyImage, 0, 0, displayWidth, displayHeight);
    } else {
      // Fallback to text if image not loaded
      noStroke();
      fill(0);
      textFont("Futura");
      text(shortenLabel(label), 0, 0);
    }
    
    pop();

    // Added: compute hitbox around rendered image (centered at lx*1.33, ly*1.33)
    const baseSize = Math.max(40, radius * 0.5);
    const currentBodyImage = bodyPositionImages[CATEGORY_DISPLAY_ORDER[i]];
    
    let hitboxWidth, hitboxHeight;
    if (currentBodyImage) {
      // Calculate hitbox based on actual image dimensions
      const imgAspect = currentBodyImage.width / currentBodyImage.height;
      if (imgAspect > 1) {
        hitboxWidth = baseSize + 10;
        hitboxHeight = (baseSize / imgAspect) + 10;
      } else {
        hitboxWidth = (baseSize * imgAspect) + 10;
        hitboxHeight = baseSize + 10;
      }
    } else {
      // Fallback to square hitbox if no image
      const hitboxSize = baseSize + 10;
      hitboxWidth = hitboxSize;
      hitboxHeight = hitboxSize;
    }
    
    labelHitboxes[i] = { x: lx*1.33, y: ly*1.33, w: hitboxWidth, h: hitboxHeight };
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
      // Keep loop running for hover detection in grid mode
      // noLoop(); // Removed to keep hover detection active
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
  const screenLeft = computeGridScreenLeft() + 80; // Shift right by 80 pixels total
  const screenTop = (overrideScreenTop != null ? overrideScreenTop : (overlayBottomY() + OVERLAY_AFTER_TEXT_PADDING + 85)); // Shift down by 45 pixels total (moved up 155px total)
  const screenRight = width - 24;
  const screenBottom = height - 36; // extra bottom padding
  const availW = Math.max(40, screenRight - screenLeft);
  const availH = Math.max(40, screenBottom - screenTop);

  const n = Math.max(1, items.length);
  // Choose columns to roughly square the grid
  const cols = 10; // Fixed width of 10 tiles for visual consistency
  const rows = Math.ceil(n / cols);
  // Use consistent tile size across all grids for visual consistency
  const tileSize = 35; // Fixed size that matches the 'Loaf' group tile size

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

  // Map existing mosaic tiles to grid positions for animation, sorted by hue
  tileMappingByIndex = {};
  
  // Get all existing mosaic tiles of this category
  const categoryTiles = [];
  for (let i = 0; i < mosaicTiles.length; i++) {
    if (mosaicTiles[i].catIndex === catIndex) {
      categoryTiles.push(i);
    }
  }
  
  // Sort tiles by hue for beautiful color gradient
  categoryTiles.sort((a, b) => {
    const hueA = hueFromHex(mosaicTiles[a].hex);
    const hueB = hueFromHex(mosaicTiles[b].hex);
    return hueA - hueB;
  });
  
  // Map each existing mosaic tile to a grid position
  for (let i = 0; i < categoryTiles.length; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const toX = leftWorld + col * tileSize + tileSize * 0.5;
    const toY = topWorld + row * tileSize + tileSize * 0.5;
    
    tileMappingByIndex[categoryTiles[i]] = {
      toX,
      toY,
      toSize: tileSize,
      targetHex: mosaicTiles[categoryTiles[i]].hex // Keep original hex color
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

function drawHoverImage() {
  if (!hoveredImage) {
    console.log('No hover image to draw');
    return;
  }
  
  console.log('Drawing hover image at:', hoveredImagePos);
  
  push();
  resetMatrix(); // Use screen coordinates
  
  // Calculate dimensions maintaining aspect ratio
  const maxSize = 180; // Maximum dimension (1.5x larger than before)
  const imgAspect = hoveredImage.width / hoveredImage.height;
  
  let displayWidth, displayHeight;
  if (imgAspect > 1) {
    // Landscape image
    displayWidth = maxSize;
    displayHeight = maxSize / imgAspect;
  } else {
    // Portrait image
    displayHeight = maxSize;
    displayWidth = maxSize * imgAspect;
  }
  
  // Draw image at cursor position with proper aspect ratio
  imageMode(CENTER);
  image(hoveredImage, hoveredImagePos.x, hoveredImagePos.y, displayWidth, displayHeight);
  
  pop();
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

  const screenLeft = computeGridScreenLeft() + 80; // Shift right by 80 pixels total to match grid
  const titleX = screenLeft;
  const titleY = OVERLAY_TITLE_Y + 45; // Shift down by 45 pixels total to match grid (moved up 115px total)

  textFont('Futura');
  textAlign(LEFT, TOP);

  textSize(32);
  textStyle(BOLD);
  text(title, titleX, titleY);

  textStyle(NORMAL);
  textSize(18);
  const totalLine = (stats.total || 0) + ' images in ' + title.toLowerCase() + ' position\n' + 'Hover over a tile to see the image';
  text(totalLine, titleX, titleY + OVERLAY_TITLE_OFFSET + OVERLAY_BLOCK_SPACING);

  const s1 = 'Selfies: ' + (stats.yes || 0);
  const s2 = 'Non-Selfies: ' + (stats.no || 0);
  const s3 = 'No Selfie Data: ' + (stats.unknown || 0);
  const firstLineY = titleY + OVERLAY_TITLE_OFFSET*2 + OVERLAY_BLOCK_SPACING + OVERLAY_LINE_HEIGHT;
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

// Mouse tracking for hover effects (called from draw loop for continuous checking)
function updateHoverState() {
  if (interactionState.mode === 'grid' && gridLayout) {
    // Use screen coordinates directly for simpler detection
    const gridScreenLeft = getRadialCenterX() + gridLayout.left;
    const gridScreenTop = getRadialCenterY() + gridLayout.top;
    const gridScreenRight = gridScreenLeft + gridLayout.cols * gridLayout.tileSize;
    const gridScreenBottom = gridScreenTop + gridLayout.rows * gridLayout.tileSize;
    
    // Check if mouse is over the grid area using screen coordinates
    if (mouseX >= gridScreenLeft && mouseX <= gridScreenRight && 
        mouseY >= gridScreenTop && mouseY <= gridScreenBottom) {
      
      // Find which specific tile is being hovered over
      const hoveredTile = findHoveredTile(mouseX, mouseY);
      
      if (hoveredTile && hoveredTile.hex) {
        // Log the filename for this hex value to console
        logFilenameForHex(hoveredTile.hex);
        
        // Look up the specific image for this tile's hex value
        const specificImage = hexToImageMap[hoveredTile.hex];
        
        if (specificImage) {
          hoveredImage = specificImage;
          hoveredImagePos = { x: mouseX + 30, y: mouseY - 30 }; // Offset from cursor
        } else {
          // No specific image found, don't show anything
          hoveredImage = null;
        }
      } else {
        hoveredImage = null;
      }
    } else {
      hoveredImage = null;
    }
  } else {
    hoveredImage = null;
  }
}

// Keep mouseMoved for p5.js compatibility, but it's not the main hover logic
function mouseMoved() {
  // This function is called by p5.js when mouse moves, but we handle hover in draw loop
}

// Log the filename for a specific hex value by looking it up in the CSV
function logFilenameForHex(hex) {
  if (!photoTable || !hex) return;
  
  // Look for the hex value in the CSV
  for (let r = 0; r < photoTable.getRowCount(); r++) {
    const csvHex = sanitizeHex(photoTable.getString(r, 'average_hex_color'));
    if (csvHex === hex) {
      // Found the matching hex, get the filename
      const filename = photoTable.getString(r, 'filename') || 
                      photoTable.getString(r, 'image_filename') || 
                      photoTable.getString(r, 'file') || 
                      photoTable.getString(r, 'image') || 
                      photoTable.getString(r, 'photo') || 
                      photoTable.getString(r, 'img');
      
      if (filename) {
        console.log('Hex:', hex, '-> Filename:', filename);
      } else {
        console.log('Hex:', hex, '-> No filename found');
      }
      break;
    }
  }
}

// Find which specific tile is being hovered over in grid mode
function findHoveredTile(mouseX, mouseY) {
  if (!gridLayout || !tileMappingByIndex) return null;
  
  // Calculate grid screen coordinates
  const gridScreenLeft = getRadialCenterX() + gridLayout.left;
  const gridScreenTop = getRadialCenterY() + gridLayout.top;
  
  // Convert mouse position to grid coordinates
  const gridX = Math.floor((mouseX - gridScreenLeft) / gridLayout.tileSize);
  const gridY = Math.floor((mouseY - gridScreenTop) / gridLayout.tileSize);
  
  // Check if we're within grid bounds
  if (gridX < 0 || gridX >= gridLayout.cols || gridY < 0 || gridY >= gridLayout.rows) {
    return null;
  }
  
  // Calculate the index in the items array for this grid position
  const itemIndex = gridY * gridLayout.cols + gridX;
  const category = CATEGORY_DISPLAY_ORDER[interactionState.selectedCategoryIndex];
  const items = itemsByCategory[interactionState.selectedCategoryIndex] || [];
  
  if (itemIndex >= 0 && itemIndex < items.length) {
    const item = items[itemIndex];
    return {
      hex: item.hex,
      gridX: gridX,
      gridY: gridY,
      itemIndex: itemIndex
    };
  }
  
  return null;
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
    if (!isPointInsideBounds(wx, wy, gridLayout.bounds)) {
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
