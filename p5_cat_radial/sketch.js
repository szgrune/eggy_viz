/* global p5 */

// Cat body positions in order for 9 ticks
const BODY_POSITIONS = [
	"standing on all fours",
	"standing up on hind legs",
	"loaf",
	"curled up",
	"between legs",
	"belly up",
	"pretzel",
	"laying on side",
	"other",
];

// CSV filenames to try in order (relative to this HTML file's folder)
const CANDIDATE_CSVS = [
	"../cat_analysis_ref_no_comparison.csv",
	"../cat_analysis_ref.csv",
	"../cat_analysis.csv",
	"../cat_analysis_clip.csv",
];

let table = null; // p5.Table
let counts = new Array(BODY_POSITIONS.length).fill(0);
let total = 0;
let dataLoaded = false;
let chosenCsv = null;

function setup() {
	const container = document.getElementById('sketch-container');
	const w = Math.min(window.innerWidth * 0.9, 1000);
	const h = Math.min(window.innerHeight * 0.9, 1000);
	const cnv = createCanvas(w, h);
	if (container) cnv.parent(container);

	textFont('system-ui');
	textAlign(CENTER, CENTER);
	angleMode(DEGREES);
	noLoop();

	// Try to load CSVs in order using async callbacks
	tryLoadCSVs(0);

	// Draw initial placeholder
	redraw();
}

function draw() {
	background(255);
	drawCatFaceRadial();
}

function windowResized() {
	const container = document.getElementById('sketch-container');
	const w = Math.min(window.innerWidth * 0.9, 1000);
	const h = Math.min(window.innerHeight * 0.9, 1000);
	resizeCanvas(w, h);
	redraw();
}

function tryLoadCSVs(idx) {
	if (idx >= CANDIDATE_CSVS.length) {
		// no data loaded
		dataLoaded = false;
		chosenCsv = null;
		redraw();
		return;
	}
	const path = CANDIDATE_CSVS[idx];
	loadTable(
		path,
		'csv',
		'header',
		(tbl) => {
			if (tbl && tbl.getRowCount() > 0 && tbl.columns.includes('body_position')) {
				table = tbl;
				chosenCsv = path;
				processData(table);
				dataLoaded = true;
				redraw();
			} else {
				// try next
				tryLoadCSVs(idx + 1);
			}
		},
		// error callback
		() => {
			tryLoadCSVs(idx + 1);
		}
	);
}

function processData(table) {
	// Expect a column named 'body_position'
	const bodyPosIdx = table.columns.indexOf('body_position');
	if (bodyPosIdx === -1) return;

	counts = new Array(BODY_POSITIONS.length).fill(0);
	for (let r = 0; r < table.getRowCount(); r++) {
		const raw = table.getString(r, bodyPosIdx) || '';
		const key = raw.trim().toLowerCase();
		const idx = BODY_POSITIONS.indexOf(key);
		if (idx >= 0) {
			counts[idx] += 1;
		} else {
			// anything else goes to 'other'
			counts[BODY_POSITIONS.length - 1] += 1;
		}
	}
	total = counts.reduce((a, b) => a + b, 0);
}

function drawCatFaceRadial() {
	push();
	translate(width / 2, height / 2);
	const minDim = Math.min(width, height);
	const faceRadius = minDim * 0.28; // base cat face radius
	const spikeMax = minDim * 0.22;   // maximum spike length beyond face
	const earSize = faceRadius * 0.9;
	const tickLength = minDim * 0.035;
	const tickRadius = faceRadius + spikeMax + tickLength * 1.2;

	// Draw cute cat face outline (no features) with thick stroke
	stroke(0);
	strokeWeight(Math.max(2, minDim * 0.01));
	noFill();

	// Head circle
	circle(0, 0, faceRadius * 2);

	// Ears: two simple triangles
	const earOffsetX = faceRadius * 0.75;
	const earTopY = -faceRadius * 1.25;
	const earBaseY = -faceRadius * 0.35;
	// Left ear
	triangle(
		-earOffsetX, earBaseY,
		-earOffsetX - earSize * 0.25, earBaseY,
		-earOffsetX + earSize * 0.08, earTopY
	);
	// Right ear
	triangle(
		earOffsetX, earBaseY,
		earOffsetX + earSize * 0.25, earBaseY,
		earOffsetX - earSize * 0.08, earTopY
	);

	// Compute scaling
	let maxCount = 1;
	for (const c of counts) maxCount = Math.max(maxCount, c);

	// Draw 9 ticks and spikes
	const num = BODY_POSITIONS.length;
	for (let i = 0; i < num; i++) {
		const angle = -90 + (360 / num) * i; // start at top, clockwise
		push();
		rotate(angle);

		// Tick mark at outer perimeter
		stroke(0);
		line(0, -tickRadius, 0, -tickRadius + tickLength);

		// Spike from face edge outward
		const c = counts[i];
		const t = maxCount > 0 ? c / maxCount : 0;
		const spikeLen = t * spikeMax;
		line(0, -faceRadius, 0, -faceRadius - spikeLen);

		pop();
	}

	// Labels around
	textSize(Math.max(10, minDim * 0.018));
	noStroke();
	fill(0);
	for (let i = 0; i < num; i++) {
		const angle = -90 + (360 / num) * i;
		const rad = (angle * Math.PI) / 180;
		const r = tickRadius + tickLength * 1.2;
		const x = Math.cos(rad) * r;
		const y = Math.sin(rad) * r;
		push();
		translate(x, y);
		rotate(angle + 90);
		const label = BODY_POSITIONS[i];
		text(label, 0, 0);
		pop();
	}

	// Title / meta
	textSize(Math.max(12, minDim * 0.022));
	textStyle(BOLD);
	textAlign(CENTER, TOP);
	noStroke();
	fill(0);
	text('Cat Body Positions – Radial Cat Face', 0, faceRadius + tickLength * 2.2);

	textSize(Math.max(10, minDim * 0.017));
	textStyle(NORMAL);
	const source = dataLoaded && chosenCsv ? `Data: ${chosenCsv}` : 'Data: none loaded';
	text(source, 0, faceRadius + tickLength * 3.4);

	pop();
}