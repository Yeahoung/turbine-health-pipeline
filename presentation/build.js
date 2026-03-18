#!/usr/bin/env node
/**
 * Build script: reads slides from slides/ and generates index.html
 * Outputs to public/ for Vercel deployment
 * Run: node build.js
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname);
const OUT_DIR = path.join(ROOT, 'public');
const SLIDES_DIR = path.join(ROOT, 'slides');
const MANIFEST_PATH = path.join(SLIDES_DIR, 'manifest.json');

function copyDir(src, dest) {
  if (!fs.existsSync(src)) return;
  fs.mkdirSync(dest, { recursive: true });
  for (const name of fs.readdirSync(src)) {
    const srcPath = path.join(src, name);
    const destPath = path.join(dest, name);
    if (fs.statSync(srcPath).isDirectory()) {
      copyDir(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

const DEMO_APP_URL = process.env.DEMO_APP_URL || 'https://yyeahoung-turbine-health-pipeline.hf.space';

function build() {
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'));
  const slidesHtml = manifest.map((slide) => {
    const contentPath = path.join(SLIDES_DIR, slide.file);
    let content = fs.readFileSync(contentPath, 'utf8').trim();
    content = content.replace(/\{\{DEMO_APP_URL\}\}/g, DEMO_APP_URL);
    const notesEscaped = escapeHtml(slide.notes || '');
    return `      <section class="slide" id="${slide.id}" data-title="${escapeHtml(slide.title)}" data-notes="${notesEscaped}">
        ${content}
      </section>`;
  }).join('\n');

  const indexHtml = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Wind Turbine AI Decision Support — Web PPT</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" crossorigin="anonymous">
  <link rel="stylesheet" href="css/style.css">
</head>
<body>
  <div class="progress-wrap"><div class="progress-bar" id="progressBar"></div></div>
  <div class="topbar">
    <div class="topbar-panel">
      <span class="topbar-title">Wind Turbine AI Decision Support</span>
      <span class="topbar-meta" id="slideMeta">Slide 1 / ${manifest.length}</span>
      <span class="topbar-meta" id="slideTitle">Title</span>
      <span class="mode-chip" id="modeChip">Presentation</span>
    </div>
    <div class="topbar-panel">
      <button class="control-btn" id="fullscreenBtn" type="button">Fullscreen</button>
      <button class="control-btn" id="presenterBtn" type="button">Presenter</button>
      <button class="control-btn" id="autoBtn" type="button">Auto Play</button>
      <button class="control-btn" id="startBtn" type="button">Go Title</button>
    </div>
  </div>

  <div class="nav-dots" id="navDots"></div>

  <div class="hint-bar">
    <span>Navigate slides</span>
    <span class="kbd">↑</span>
    <span class="kbd">↓</span>
    <span class="kbd">PgUp</span>
    <span class="kbd">PgDn</span>
  </div>

  <div class="app">
    <main class="deck" id="deck">
${slidesHtml}
    </main>

    <aside class="presenter-pane" id="presenterPane">
      <div class="presenter-shell">
        <div>
          <div class="presenter-title">Presenter View</div>
          <div class="presenter-slide-name" id="presenterSlideName">Title</div>
        </div>
        <div class="presenter-card">
          <h4>Status</h4>
          <div class="timer-row">
            <span class="timer-chip" id="elapsedChip">Elapsed 00:00</span>
            <span class="timer-chip" id="slideChip">Slide 1 / ${manifest.length}</span>
            <span class="timer-chip" id="autoChip">Manual mode</span>
          </div>
        </div>
        <div class="presenter-card">
          <h4>Speaking prompts</h4>
          <ul class="checklist" id="cueList">
            <li>State the problem clearly.</li>
            <li>Explain why the system is advisory-only.</li>
            <li>Close each section with a practical takeaway.</li>
          </ul>
        </div>
        <div class="presenter-card notes-box" id="presenterNotes"></div>
        <div class="presenter-card">
          <h4>Shortcuts</h4>
          <p style="font-size:0.92rem; color:var(--muted); line-height:1.7;">Press <strong>P</strong> for presenter mode, <strong>A</strong> for auto-play, <strong>F</strong> for fullscreen, and <strong>Esc</strong> to close zoomed plots.</p>
        </div>
      </div>
    </aside>
  </div>

  <div class="modal" id="imageModal" aria-hidden="true">
    <div class="modal-content">
      <button class="modal-close" id="modalClose" type="button">×</button>
      <img id="modalImage" alt="Expanded plot view">
      <div class="modal-caption" id="modalCaption"></div>
    </div>
  </div>

  <script src="js/script.js"></script>
</body>
</html>
`;

  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(path.join(OUT_DIR, 'index.html'), indexHtml);
  copyDir(path.join(ROOT, 'css'), path.join(OUT_DIR, 'css'));
  copyDir(path.join(ROOT, 'js'), path.join(OUT_DIR, 'js'));
  copyDir(path.join(ROOT, 'plots'), path.join(OUT_DIR, 'plots'));
  console.log(`Built index.html with ${manifest.length} slides to public/`);
}

build();
