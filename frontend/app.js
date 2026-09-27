/**
 * SatQuery AI — Frontend Logic
 * Manages interactive ROI drawing, natural-scale coordinate mapping,
 * live backend communication (/health, /analyze), and export generation.
 */

// Configuration
const API_BASE_URL = window.location.origin.includes(':8000') 
  ? window.location.origin 
  : 'http://127.0.0.1:8000';

// State
let loadedImage = null;       // HTMLImageElement
let currentFileBlob = null;   // File or Blob
let currentFilename = 'satellite_image.png';
let isDrawing = false;
let startX = 0, startY = 0;
let currentX = 0, currentY = 0;
let selectedRoi = null;       // { x, y, width, height } in natural image coordinates
let lastAnalysisResult = null;
let activeTab = 'geojson';

// DOM Elements
const fileInput = document.getElementById('fileInput');
const dropzoneOverlay = document.getElementById('dropzoneOverlay');
const canvasWrapper = document.getElementById('canvasWrapper');
const canvas = document.getElementById('imageCanvas');
const ctx = canvas.getContext('2d');
const roiBadge = document.getElementById('roiCoordinatesBadge');
const imgDimBadge = document.getElementById('imageDimensionsBadge');
const queryInput = document.getElementById('queryInput');
const btnRunAnalysis = document.getElementById('btnRunAnalysis');
const btnSelectAllRoi = document.getElementById('btnSelectAllRoi');
const btnClearRoi = document.getElementById('btnClearRoi');
const btnChangeImage = document.getElementById('btnChangeImage');

// Status & Result DOM
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const resultsStatusBadge = document.getElementById('resultsStatusBadge');
const resultsIdleState = document.getElementById('resultsIdleState');
const resultsLoadingState = document.getElementById('resultsLoadingState');
const resultsActiveState = document.getElementById('resultsActiveState');
const errorBanner = document.getElementById('errorBanner');
const errorMessageText = document.getElementById('errorMessageText');

// Output DOM
const roiThumbImg = document.getElementById('roiThumbImg');
const roiMetaCoords = document.getElementById('roiMetaCoords');
const roiMetaQuery = document.getElementById('roiMetaQuery');
const captionText = document.getElementById('captionText');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBarFill = document.getElementById('confidenceBarFill');
const humanReviewBanner = document.getElementById('humanReviewBanner');
const telemetryLatency = document.getElementById('telemetryLatency');
const telemetryModel = document.getElementById('telemetryModel');
const btnDownloadGeoJSON = document.getElementById('btnDownloadGeoJSON');
const btnDownloadAudit = document.getElementById('btnDownloadAudit');
const btnDownloadPDF = document.getElementById('btnDownloadPDF');
const tabBtnGeoJson = document.getElementById('tabBtnGeoJson');
const tabBtnAudit = document.getElementById('tabBtnAudit');
const tabBtnAgentic = document.getElementById('tabBtnAgentic');
const tabBtnMemory = document.getElementById('tabBtnMemory');
const inspectorContent = document.getElementById('inspectorContent');

// Continuous Learning & Memory DOM
const continuousLearningPanel = document.getElementById('continuousLearningPanel');
const memoryStatsBadge = document.getElementById('memoryStatsBadge');
const memoryCountPill = document.getElementById('memoryCountPill');
const btnTrainMemory = document.getElementById('btnTrainMemory');
const btnTrainMemoryText = document.getElementById('btnTrainMemoryText');
const btnClearMemory = document.getElementById('btnClearMemory');
const selectEpochs = document.getElementById('selectEpochs');
const autoTrainToggle = document.getElementById('autoTrainToggle');
const trainingStatusCard = document.getElementById('trainingStatusCard');
const trainingSpinner = document.getElementById('trainingSpinner');
const trainingStatusTitle = document.getElementById('trainingStatusTitle');
const trainingStatusDesc = document.getElementById('trainingStatusDesc');
const memoryCardsGrid = document.getElementById('memoryCardsGrid');
const memoryEmptyState = document.getElementById('memoryEmptyState');
const btnExportDataset = document.getElementById('btnExportDataset');
const toastContainer = document.getElementById('toastContainer');
let currentMemoryRecords = [];

// Innovation Controls & Displays
const languageSelect = document.getElementById('languageSelect');
const sarToggle = document.getElementById('sarToggle');
const sarActiveBadge = document.getElementById('sarActiveBadge');
const captionLangBadge = document.getElementById('captionLangBadge');
const captionIntentBadge = document.getElementById('captionIntentBadge');
const captionSubtext = document.getElementById('captionSubtext');
const sarFusionCard = document.getElementById('sarFusionCard');
const sarValVV = document.getElementById('sarValVV');
const sarValVH = document.getElementById('sarValVH');
const sarValRatio = document.getElementById('sarValRatio');
const sarValScattering = document.getElementById('sarValScattering');
const sarDielectricDesc = document.getElementById('sarDielectricDesc');

// SAR Toggle Event
if (sarToggle && sarActiveBadge) {
  sarToggle.addEventListener('change', () => {
    if (sarToggle.checked) {
      sarActiveBadge.textContent = 'ON';
      sarActiveBadge.classList.add('active');
    } else {
      sarActiveBadge.textContent = 'OFF';
      sarActiveBadge.classList.remove('active');
    }
  });
}

// ---------------------------------------------------------------------------
// 1. Health Polling
// ---------------------------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      const data = await res.json();
      statusDot.className = 'status-dot online';
      const variant = data.has_adapter ? 'BLIP + LoRA' : 'Base BLIP';
      statusText.textContent = `Pipeline Ready: ${variant} [${data.device.toUpperCase()}]`;
    } else {
      statusDot.className = 'status-dot offline';
      statusText.textContent = 'Backend Error: ' + res.status;
    }
  } catch (err) {
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'Backend Offline (Port 8000)';
  }
}

setInterval(checkHealth, 8000);
checkHealth();

// ---------------------------------------------------------------------------
// 2. Image Loading & Canvas Initialization
// ---------------------------------------------------------------------------
function handleFileSelect(file) {
  if (!file) return;
  currentFileBlob = file;
  currentFilename = file.name || 'satellite_image.png';

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      loadedImage = img;
      initCanvas();
      // Select central 80% as initial default ROI
      const padX = Math.round(img.naturalWidth * 0.1);
      const padY = Math.round(img.naturalHeight * 0.1);
      selectedRoi = {
        x: padX,
        y: padY,
        width: img.naturalWidth - padX * 2,
        height: img.naturalHeight - padY * 2,
      };
      drawCanvas();
      btnRunAnalysis.disabled = false;
      imgDimBadge.textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;
      hideError();
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function initCanvas() {
  dropzoneOverlay.classList.add('hidden');
  canvasWrapper.classList.remove('hidden');

  // Match canvas dimensions to container width while preserving aspect ratio
  const containerWidth = canvasWrapper.parentElement.clientWidth - 40;
  const aspect = loadedImage.naturalHeight / loadedImage.naturalWidth;
  const displayWidth = Math.min(containerWidth, 680);
  const displayHeight = Math.round(displayWidth * aspect);

  canvas.width = displayWidth;
  canvas.height = displayHeight;
}

// ---------------------------------------------------------------------------
// 3. Coordinate Conversion & Canvas Rendering
// ---------------------------------------------------------------------------
function canvasToNatural(cx, cy) {
  const scaleX = loadedImage.naturalWidth / canvas.width;
  const scaleY = loadedImage.naturalHeight / canvas.height;
  return {
    x: Math.round(cx * scaleX),
    y: Math.round(cy * scaleY),
  };
}

function naturalToCanvas(nx, ny) {
  const scaleX = canvas.width / loadedImage.naturalWidth;
  const scaleY = canvas.height / loadedImage.naturalHeight;
  return {
    x: nx * scaleX,
    y: ny * scaleY,
  };
}

function drawCanvas() {
  if (!loadedImage) return;

  // 1. Draw base image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(loadedImage, 0, 0, canvas.width, canvas.height);

  // 2. Draw active dragging box or committed ROI
  let roiBox = null;

  if (isDrawing) {
    const x = Math.min(startX, currentX);
    const y = Math.min(startY, currentY);
    const w = Math.abs(currentX - startX);
    const h = Math.abs(currentY - startY);
    roiBox = { x, y, w, h };
  } else if (selectedRoi) {
    const pt1 = naturalToCanvas(selectedRoi.x, selectedRoi.y);
    const pt2 = naturalToCanvas(selectedRoi.x + selectedRoi.width, selectedRoi.y + selectedRoi.height);
    roiBox = {
      x: pt1.x,
      y: pt1.y,
      w: pt2.x - pt1.x,
      h: pt2.y - pt1.y,
    };
  }

  if (roiBox && roiBox.w > 4 && roiBox.h > 4) {
    // Dim background outside ROI
    ctx.fillStyle = 'rgba(2, 6, 23, 0.45)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Clear ROI aperture
    ctx.clearRect(roiBox.x, roiBox.y, roiBox.w, roiBox.h);
    ctx.drawImage(
      loadedImage,
      (roiBox.x / canvas.width) * loadedImage.naturalWidth,
      (roiBox.y / canvas.height) * loadedImage.naturalHeight,
      (roiBox.w / canvas.width) * loadedImage.naturalWidth,
      (roiBox.h / canvas.height) * loadedImage.naturalHeight,
      roiBox.x, roiBox.y, roiBox.w, roiBox.h
    );

    // ROI Box Border & Neon Glow
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2;
    ctx.strokeRect(roiBox.x, roiBox.y, roiBox.w, roiBox.h);

    // Corner Anchors
    const cornerSize = 7;
    ctx.fillStyle = '#00f0ff';
    // Top-left
    ctx.fillRect(roiBox.x - 2, roiBox.y - 2, cornerSize, cornerSize);
    // Top-right
    ctx.fillRect(roiBox.x + roiBox.w - cornerSize + 2, roiBox.y - 2, cornerSize, cornerSize);
    // Bottom-left
    ctx.fillRect(roiBox.x - 2, roiBox.y + roiBox.h - cornerSize + 2, cornerSize, cornerSize);
    // Bottom-right
    ctx.fillRect(roiBox.x + roiBox.w - cornerSize + 2, roiBox.y + roiBox.h - cornerSize + 2, cornerSize, cornerSize);

    // Update Coordinate Badge
    const natStart = canvasToNatural(roiBox.x, roiBox.y);
    const natDim = canvasToNatural(roiBox.w, roiBox.h);
    roiBadge.style.display = 'block';
    roiBadge.textContent = `ROI: ${natDim.x} × ${natDim.y} px [X:${natStart.x}, Y:${natStart.y}]`;
  } else {
    roiBadge.style.display = 'none';
  }
}

// ---------------------------------------------------------------------------
// 4. Interactive Drawing Event Handlers
// ---------------------------------------------------------------------------
canvas.addEventListener('mousedown', (e) => {
  if (!loadedImage) return;
  const rect = canvas.getBoundingClientRect();
  startX = e.clientX - rect.left;
  startY = e.clientY - rect.top;
  currentX = startX;
  currentY = startY;
  isDrawing = true;
});

canvas.addEventListener('mousemove', (e) => {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  currentX = Math.max(0, Math.min(canvas.width, e.clientX - rect.left));
  currentY = Math.max(0, Math.min(canvas.height, e.clientY - rect.top));
  drawCanvas();
});

canvas.addEventListener('mouseup', () => {
  if (!isDrawing) return;
  isDrawing = false;

  const cx = Math.min(startX, currentX);
  const cy = Math.min(startY, currentY);
  const cw = Math.abs(currentX - startX);
  const ch = Math.abs(currentY - startY);

  if (cw > 8 && ch > 8) {
    const natP1 = canvasToNatural(cx, cy);
    const natP2 = canvasToNatural(cx + cw, cy + ch);
    selectedRoi = {
      x: natP1.x,
      y: natP1.y,
      width: natP2.x - natP1.x,
      height: natP2.y - natP1.y,
    };
    btnRunAnalysis.disabled = false;
  }
  drawCanvas();
});

// Canvas Tool Actions
btnSelectAllRoi.addEventListener('click', () => {
  if (!loadedImage) return;
  selectedRoi = {
    x: 0,
    y: 0,
    width: loadedImage.naturalWidth,
    height: loadedImage.naturalHeight,
  };
  drawCanvas();
  btnRunAnalysis.disabled = false;
});

btnClearRoi.addEventListener('click', () => {
  selectedRoi = null;
  drawCanvas();
  roiBadge.style.display = 'none';
  btnRunAnalysis.disabled = true;
});

btnChangeImage.addEventListener('click', () => {
  fileInput.click();
});

// Dropzone Events
dropzoneOverlay.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  if (e.target.files.length) handleFileSelect(e.target.files[0]);
});

['dragenter', 'dragover'].forEach((eventName) => {
  dropzoneOverlay.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzoneOverlay.classList.add('drag-over');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropzoneOverlay.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzoneOverlay.classList.remove('drag-over');
  });
});

dropzoneOverlay.addEventListener('drop', (e) => {
  if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
});

// Quick Demo Imagery Chips
document.querySelectorAll('.sample-chip').forEach((btn) => {
  btn.addEventListener('click', async () => {
    const sample = btn.getAttribute('data-sample');
    document.querySelectorAll('.sample-chip').forEach((c) => c.classList.remove('active'));
    btn.classList.add('active');

    const sampleUrl = `samples/${sample}.png`;
    try {
      const res = await fetch(sampleUrl);
      if (!res.ok) throw new Error('Sample image not found on server');
      const blob = await res.blob();
      const file = new File([blob], `${sample}_satellite.png`, { type: 'image/png' });
      handleFileSelect(file);
    } catch (err) {
      showError(`Could not load sample image (${sample}): ${err.message}`);
    }
  });
});

// Preset Query Chips
document.querySelectorAll('.query-chip').forEach((btn) => {
  btn.addEventListener('click', () => {
    queryInput.value = btn.getAttribute('data-query');
    queryInput.focus();
  });
});

queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !btnRunAnalysis.disabled) {
    runAnalysis();
  }
});

// ---------------------------------------------------------------------------
// 5. Backend Inference Execution
// ---------------------------------------------------------------------------
btnRunAnalysis.addEventListener('click', runAnalysis);

async function runAnalysis() {
  if (!loadedImage || !currentFileBlob) {
    showError('Please upload or load a satellite image first.');
    return;
  }

  // Fallback to full image if ROI not explicitly drawn
  const roi = selectedRoi || {
    x: 0,
    y: 0,
    width: loadedImage.naturalWidth,
    height: loadedImage.naturalHeight,
  };

  hideError();
  setLoadingState(true);

  // Prepare Multipart Form Data
  const formData = new FormData();
  formData.append('file', currentFileBlob, currentFilename);
  formData.append('x', roi.x);
  formData.append('y', roi.y);
  formData.append('width', roi.width);
  formData.append('height', roi.height);

  const query = queryInput.value.trim();
  if (query) {
    formData.append('query', query);
  }

  // Pass selected Indian Regional Language & Optical-SAR Toggle
  const selectedLang = languageSelect ? languageSelect.value : 'en';
  formData.append('language', selectedLang);

  const sarEnabled = sarToggle ? sarToggle.checked : false;
  formData.append('enable_sar', sarEnabled ? 'true' : 'false');

  try {
    const res = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: `HTTP error ${res.status}` }));
      throw new Error(errData.detail || `Server returned status ${res.status}`);
    }

    const data = await res.json();
    lastAnalysisResult = data;
    renderResults(data, roi);

    // Continuous Learning Q&A storage notification & refresh
    if (data.memory && data.memory.stored) {
      showToast(`💾 Question & Answer saved to training memory (Record #${data.memory.record_id})`, 'success');
      fetchMemory();
      if (autoTrainToggle && autoTrainToggle.checked) {
        trainMemory();
      }
    }
  } catch (err) {
    console.error('Inference error:', err);
    showError(`Analysis failed: ${err.message}. Please verify the backend is running.`);
  } finally {
    setLoadingState(false);
  }
}

// ---------------------------------------------------------------------------
// 6. Result Rendering & UI States
// ---------------------------------------------------------------------------
function renderResults(data, roi) {
  resultsIdleState.classList.add('hidden');
  resultsLoadingState.classList.add('hidden');
  resultsActiveState.classList.remove('hidden');
  resultsStatusBadge.textContent = 'Inference Complete';

  // Generate cropped thumbnail for UI
  generateThumbnail(roi);

  // ROI Meta
  roiMetaCoords.textContent = `Coordinates: [X:${roi.x}, Y:${roi.y}, W:${roi.width}, H:${roi.height}]`;
  roiMetaQuery.textContent = queryInput.value.trim() 
    ? `Prompt: "${queryInput.value.trim()}"` 
    : 'Mode: Autonomous Land-Cover Captioning';

  // Paragraph Answer
  captionText.textContent = data.caption || 'No caption generated.';

  // Language & Intent Meta Badges
  if (captionLangBadge && languageSelect) {
    const activeOpt = languageSelect.options[languageSelect.selectedIndex];
    captionLangBadge.textContent = activeOpt ? activeOpt.text : 'English (en)';
  }
  if (captionIntentBadge) {
    const rawIntent = (data.agentic_intel && data.agentic_intel.task_intent) || 'general_land_cover';
    const intentMap = {
      'water_hydrology': 'Hydrology & Water',
      'urban_infrastructure': 'Urban Infrastructure',
      'forest_ecology': 'Forest & Canopy',
      'agricultural_crop': 'Agricultural Crop',
      'disaster_assessment': 'Disaster & Hazard',
      'general_land_cover': 'General Land Cover',
    };
    captionIntentBadge.textContent = intentMap[rawIntent] || rawIntent;
  }
  if (captionSubtext) {
    if (data.short_caption && data.short_caption !== data.caption) {
      captionSubtext.textContent = `VLM Primary Target: "${data.short_caption}"`;
    } else {
      captionSubtext.textContent = '';
    }
  }

  // Optical-SAR Radar Telemetry Card
  if (sarFusionCard) {
    const sar = data.agentic_intel && data.agentic_intel.sar_fusion;
    if (sar && sar.sar_enabled) {
      sarValVV.textContent = `${sar.sigma0_vv_db} dB`;
      sarValVH.textContent = `${sar.sigma0_vh_db} dB`;
      sarValRatio.textContent = `${sar.cross_pol_ratio_db} dB`;
      sarValScattering.textContent = sar.dominant_scattering.split('(')[0].trim();
      sarDielectricDesc.textContent = `${sar.dielectric_interpretation} — ${sar.cloud_immunity}`;
      sarFusionCard.classList.remove('hidden');
    } else {
      sarFusionCard.classList.add('hidden');
    }
  }

  // Confidence Display & Color Coding
  const conf = data.confidence !== undefined ? data.confidence : 0;
  const pct = Math.round(conf * 100);
  confidenceValue.textContent = `${pct}%`;
  confidenceBarFill.style.width = `${pct}%`;

  // Color classes: high (>=70%), medium (55-69%), low (<55%)
  confidenceValue.className = 'confidence-value';
  confidenceBarFill.className = 'confidence-bar-fill';
  if (pct >= 70) {
    confidenceValue.classList.add('high');
    confidenceBarFill.classList.add('high');
  } else if (pct >= 55) {
    confidenceValue.classList.add('medium');
    confidenceBarFill.classList.add('medium');
  } else {
    confidenceValue.classList.add('low');
    confidenceBarFill.classList.add('low');
  }

  // Human Review Safety Banner
  if (data.low_confidence_flag) {
    humanReviewBanner.classList.remove('hidden');
  } else {
    humanReviewBanner.classList.add('hidden');
  }

  // Telemetry Badges
  if (data.audit && data.audit.latency_sec) {
    telemetryLatency.textContent = `Latency: ${(data.audit.latency_sec * 1000).toFixed(0)} ms`;
  }
  if (data.audit && data.audit.model) {
    telemetryModel.textContent = data.audit.model;
  }

  // Update Inspector
  updateInspector();
}

function generateThumbnail(roi) {
  const thumbCanvas = document.createElement('canvas');
  thumbCanvas.width = 120;
  thumbCanvas.height = 120;
  const tCtx = thumbCanvas.getContext('2d');

  tCtx.drawImage(
    loadedImage,
    roi.x, roi.y, roi.width, roi.height,
    0, 0, 120, 120
  );
  roiThumbImg.src = thumbCanvas.toDataURL('image/png');
}

function setLoadingState(isLoading) {
  if (isLoading) {
    btnRunAnalysis.disabled = true;
    btnRunAnalysis.innerHTML = `
      <div class="loading-spinner-ring" style="width:16px;height:16px;border-width:2px;"></div>
      <span>Analyzing Satellite Imagery...</span>
    `;
    resultsIdleState.classList.add('hidden');
    resultsActiveState.classList.add('hidden');
    resultsLoadingState.classList.remove('hidden');
    resultsStatusBadge.textContent = 'Processing...';

    // Step animation ticker
    const steps = [
      document.getElementById('step1'),
      document.getElementById('step2'),
      document.getElementById('step3'),
    ];
    let currentStep = 0;
    const interval = setInterval(() => {
      currentStep++;
      if (currentStep < steps.length) {
        steps.forEach((s) => s.classList.remove('active'));
        steps[currentStep].classList.add('active');
      } else {
        clearInterval(interval);
      }
    }, 400);
  } else {
    btnRunAnalysis.disabled = false;
    btnRunAnalysis.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <polygon points="5 3 19 12 5 21 5 3"></polygon>
      </svg>
      <span>Run VLM Analysis</span>
    `;
    resultsLoadingState.classList.add('hidden');
  }
}

function showError(msg) {
  errorBanner.classList.remove('hidden');
  errorMessageText.textContent = msg;
}

function hideError() {
  errorBanner.classList.add('hidden');
}

// ---------------------------------------------------------------------------
// 7. Data Inspector Tabs & Downloads
// ---------------------------------------------------------------------------
tabBtnGeoJson.addEventListener('click', () => {
  activeTab = 'geojson';
  tabBtnGeoJson.classList.add('active');
  tabBtnAudit.classList.remove('active');
  if (tabBtnAgentic) tabBtnAgentic.classList.remove('active');
  if (tabBtnMemory) tabBtnMemory.classList.remove('active');
  updateInspector();
});

tabBtnAudit.addEventListener('click', () => {
  activeTab = 'audit';
  tabBtnAudit.classList.add('active');
  tabBtnGeoJson.classList.remove('active');
  if (tabBtnAgentic) tabBtnAgentic.classList.remove('active');
  if (tabBtnMemory) tabBtnMemory.classList.remove('active');
  updateInspector();
});

if (tabBtnAgentic) {
  tabBtnAgentic.addEventListener('click', () => {
    activeTab = 'agentic';
    tabBtnAgentic.classList.add('active');
    tabBtnGeoJson.classList.remove('active');
    tabBtnAudit.classList.remove('active');
    if (tabBtnMemory) tabBtnMemory.classList.remove('active');
    updateInspector();
  });
}

if (tabBtnMemory) {
  tabBtnMemory.addEventListener('click', () => {
    activeTab = 'memory';
    tabBtnMemory.classList.add('active');
    tabBtnGeoJson.classList.remove('active');
    tabBtnAudit.classList.remove('active');
    if (tabBtnAgentic) tabBtnAgentic.classList.remove('active');
    updateInspector();
  });
}

function updateInspector() {
  if (!lastAnalysisResult && !currentMemoryRecords.length) return;
  if (activeTab === 'geojson') {
    inspectorContent.textContent = JSON.stringify((lastAnalysisResult && lastAnalysisResult.geojson) || {}, null, 2);
  } else if (activeTab === 'audit') {
    inspectorContent.textContent = JSON.stringify((lastAnalysisResult && lastAnalysisResult.audit) || {}, null, 2);
  } else if (activeTab === 'agentic') {
    inspectorContent.textContent = JSON.stringify((lastAnalysisResult && lastAnalysisResult.agentic_intel) || {}, null, 2);
  } else if (activeTab === 'memory') {
    const memPayload = (lastAnalysisResult && lastAnalysisResult.memory) || { records: currentMemoryRecords };
    inspectorContent.textContent = JSON.stringify(memPayload, null, 2);
  }
}

function triggerDownload(content, filename, contentType = 'application/json') {
  const blob = new Blob([content], { type: contentType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

btnDownloadGeoJSON.addEventListener('click', () => {
  if (lastAnalysisResult && lastAnalysisResult.geojson) {
    const geoStr = JSON.stringify(lastAnalysisResult.geojson, null, 2);
    triggerDownload(geoStr, 'satquery_roi.geojson', 'application/geo+json');
  }
});

btnDownloadAudit.addEventListener('click', () => {
  if (lastAnalysisResult && lastAnalysisResult.audit) {
    const auditStr = JSON.stringify(lastAnalysisResult.audit, null, 2);
    triggerDownload(auditStr, 'satquery_audit_report.json', 'application/json');
  }
});

// Official Auditable PDF Report Download Handler
if (btnDownloadPDF) {
  btnDownloadPDF.addEventListener('click', async () => {
    if (!lastAnalysisResult || !currentFileBlob) {
      showError('Please load an image and run analysis before exporting the audit PDF.');
      return;
    }

    const roi = selectedRoi || {
      x: 0,
      y: 0,
      width: loadedImage.naturalWidth,
      height: loadedImage.naturalHeight,
    };

    const pdfData = new FormData();
    pdfData.append('file', currentFileBlob, currentFilename);
    pdfData.append('x', roi.x);
    pdfData.append('y', roi.y);
    pdfData.append('width', roi.width);
    pdfData.append('height', roi.height);
    pdfData.append('query', queryInput.value.trim());
    pdfData.append('paragraph', lastAnalysisResult.caption || '');
    pdfData.append('confidence', lastAnalysisResult.confidence || 0.75);
    pdfData.append('latency_sec', (lastAnalysisResult.audit && lastAnalysisResult.audit.latency_sec) || 0.25);
    pdfData.append('language', languageSelect ? languageSelect.value : 'en');
    pdfData.append('enable_sar', (sarToggle && sarToggle.checked) ? 'true' : 'false');
    const engPara = (lastAnalysisResult.agentic_intel && lastAnalysisResult.agentic_intel.english_paragraph) || '';
    if (engPara) {
      pdfData.append('english_paragraph', engPara);
    }

    btnDownloadPDF.disabled = true;
    const origHTML = btnDownloadPDF.innerHTML;
    btnDownloadPDF.innerHTML = `<span>Generating PDF...</span>`;

    try {
      const res = await fetch(`${API_BASE_URL}/export/pdf`, {
        method: 'POST',
        body: pdfData,
      });

      if (!res.ok) throw new Error(`PDF export failed with status ${res.status}`);

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `satquery_audit_report_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('PDF export error:', err);
      showError(`Could not generate PDF: ${err.message}`);
    } finally {
      btnDownloadPDF.disabled = false;
      btnDownloadPDF.innerHTML = origHTML;
    }
  });
}

// ===========================================================================
// 8. Continuous Learning & Past Q&A Memory Management
// ===========================================================================

function showToast(message, type = 'info') {
  if (!toastContainer) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  const iconSvg = type === 'success' 
    ? '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'
    : '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';
  toast.innerHTML = `${iconSvg}<div class="toast-text">${message}</div>`;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('hiding');
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

async function fetchMemory() {
  try {
    const res = await fetch(`${API_BASE_URL}/memory`);
    if (!res.ok) return;
    const data = await res.json();
    currentMemoryRecords = data.records || [];
    const stats = data.stats || { total_records: 0, trained_records: 0, pending_training: 0 };
    
    if (memoryStatsBadge) {
      memoryStatsBadge.textContent = `${stats.total_records} Q&A Memorized (${stats.pending_training} Pending Train)`;
    }
    if (memoryCountPill) {
      memoryCountPill.textContent = `${stats.total_records} Samples`;
    }
    if (btnTrainMemory) {
      btnTrainMemory.disabled = (stats.total_records === 0);
      if (btnTrainMemoryText) {
        btnTrainMemoryText.textContent = stats.pending_training > 0 
          ? `Train Model on Previous Answers (${stats.pending_training} New)`
          : `Retrain Model on Previous Answers (${stats.total_records})`;
      }
    }

    renderMemoryCards(currentMemoryRecords);
  } catch (err) {
    console.warn('Could not fetch memory records:', err);
  }
}

function renderMemoryCards(records) {
  if (!memoryCardsGrid) return;
  if (!records || records.length === 0) {
    memoryCardsGrid.innerHTML = `
      <div class="memory-empty-state" id="memoryEmptyState">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
        <p>No questions memorized yet. Ask any question above to record your first Q&A training sample.</p>
      </div>`;
    return;
  }

  memoryCardsGrid.innerHTML = '';
  records.forEach((rec, idx) => {
    const card = document.createElement('div');
    card.className = 'memory-card' + (idx === 0 ? ' highlight' : '');
    
    const cropFilename = rec.crop_path ? rec.crop_path.split(/[\\/]/).pop() : '';
    const cropUrl = cropFilename ? `${API_BASE_URL}/memory/crops/${cropFilename}` : '';
    const imgEl = cropUrl 
      ? `<img class="memory-card-thumb" src="${cropUrl}" alt="ROI Crop" onerror="this.style.display='none'">`
      : `<div class="memory-card-thumb" style="display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:0.6rem;">ROI</div>`;

    const statusClass = rec.trained ? 'trained' : 'pending';
    const statusText = rec.trained ? 'Trained in LoRA' : 'Pending Adaptation';
    const confPct = Math.round(rec.confidence * 100);

    card.innerHTML = `
      <div class="memory-card-top">
        ${imgEl}
        <div class="memory-card-meta">
          <div class="memory-card-id">Memory #${rec.id} &bull; ${new Date(rec.timestamp).toLocaleTimeString()}</div>
          <div class="memory-card-question" title="${rec.question}">${rec.question}</div>
          <div class="memory-card-tags">
            <span class="status-pill ${statusClass}">
              <span class="status-pill-dot"></span>
              <span>${statusText}</span>
            </span>
            <span class="lang-pill" style="font-size:0.62rem;padding:0.1rem 0.35rem;">${confPct}% Conf</span>
          </div>
        </div>
      </div>
      <div class="memory-card-answer" title="${rec.answer}">
        <strong>Answer:</strong> ${rec.short_caption ? `<em>[${rec.short_caption}]</em> ` : ''}${rec.answer}
      </div>
      <div class="memory-card-footer" style="display:flex;align-items:center;justify-content:space-between;margin-top:0.35rem;padding-top:0.35rem;border-top:1px solid rgba(255,255,255,0.06);">
        <button type="button" class="btn-edit-memory" data-id="${rec.id}" style="background:none;border:none;color:var(--accent-cyan);font-size:0.7rem;cursor:pointer;display:inline-flex;align-items:center;gap:0.25rem;">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"></path><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path></svg>
          <span>Edit / Correct</span>
        </button>
        <span style="font-size:0.65rem;color:var(--text-muted);font-family:var(--font-mono);">${rec.intent || 'analysis'}</span>
      </div>
    `;
    memoryCardsGrid.appendChild(card);
  });

  document.querySelectorAll('.btn-edit-memory').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const id = btn.getAttribute('data-id');
      const rec = records.find((r) => String(r.id) === String(id));
      if (!rec) return;
      const newAnswer = prompt(`Edit / Correct Answer for Question:\n"${rec.question}"`, rec.answer);
      if (newAnswer !== null && newAnswer.trim() && newAnswer.trim() !== rec.answer) {
        try {
          const form = new FormData();
          form.append('record_id', id);
          form.append('answer', newAnswer.trim());
          const res = await fetch(`${API_BASE_URL}/memory/update`, { method: 'POST', body: form });
          if (res.ok) {
            showToast(`Memory #${id} updated with your correction! Ready for re-training.`, 'success');
            fetchMemory();
          }
        } catch (err) {
          showError(`Could not update record: ${err.message}`);
        }
      }
    });
  });
}

async function trainMemory() {
  if (!btnTrainMemory) return;
  btnTrainMemory.disabled = true;
  const origBtnText = btnTrainMemoryText ? btnTrainMemoryText.textContent : 'Train Model on Previous Answers';
  if (btnTrainMemoryText) btnTrainMemoryText.textContent = 'Training Model Weights...';

  if (trainingStatusCard) {
    trainingStatusCard.className = 'training-status-card';
    trainingStatusCard.classList.remove('hidden');
    if (trainingStatusTitle) trainingStatusTitle.textContent = 'Continuous LoRA Fine-Tuning in Progress...';
    const ep = selectEpochs ? selectEpochs.value : '2';
    if (trainingStatusDesc) trainingStatusDesc.textContent = `Adapting BLIP cross-attention matrices for ${ep} epoch(s) on stored previous answers...`;
    if (trainingSpinner) trainingSpinner.classList.remove('hidden');
  }

  const formData = new FormData();
  const epochsVal = selectEpochs ? selectEpochs.value : '2';
  formData.append('epochs', epochsVal);
  formData.append('train_all', 'true');

  try {
    const res = await fetch(`${API_BASE_URL}/train/memory`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || 'Training failed');
    }

    const result = await res.json();
    if (result.status === 'success') {
      if (trainingStatusCard) {
        trainingStatusCard.className = 'training-status-card success';
        if (trainingSpinner) trainingSpinner.classList.add('hidden');
        if (trainingStatusTitle) trainingStatusTitle.textContent = '✨ Model Successfully Retrained on Previous Answers!';
        const lossDrop = result.loss_history && result.loss_history.length > 1
          ? `Loss: ${result.loss_history[0]} &rarr; ${result.loss_history[result.loss_history.length - 1]}`
          : `Final Loss: ${result.final_loss}`;
        if (trainingStatusDesc) {
          trainingStatusDesc.innerHTML = `Trained on <strong>${result.samples_trained} memorized interactions</strong> in ${result.duration_sec}s (${lossDrop}). New LoRA weights hot-reloaded into inference pipeline!`;
        }
      }
      showToast(`✨ Model successfully fine-tuned on ${result.samples_trained} previous answers!`, 'success');
      fetchMemory();
      checkHealth();
    } else {
      if (trainingStatusCard) {
        trainingStatusCard.className = 'training-status-card error';
        if (trainingSpinner) trainingSpinner.classList.add('hidden');
        if (trainingStatusTitle) trainingStatusTitle.textContent = 'Training Info';
        if (trainingStatusDesc) trainingStatusDesc.textContent = result.message || 'No samples available to train.';
      }
    }
  } catch (err) {
    console.error('Training error:', err);
    if (trainingStatusCard) {
      trainingStatusCard.className = 'training-status-card error';
      if (trainingSpinner) trainingSpinner.classList.add('hidden');
      if (trainingStatusTitle) trainingStatusTitle.textContent = 'Continuous Adaptation Error';
      if (trainingStatusDesc) trainingStatusDesc.textContent = err.message;
    }
    showToast(`Training failed: ${err.message}`, 'error');
  } finally {
    btnTrainMemory.disabled = false;
    if (btnTrainMemoryText) btnTrainMemoryText.textContent = origBtnText;
  }
}

// Event Listeners for Training & Memory
if (btnTrainMemory) {
  btnTrainMemory.addEventListener('click', trainMemory);
}

if (btnClearMemory) {
  btnClearMemory.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to clear all stored question-answer memory and crops?')) return;
    try {
      const res = await fetch(`${API_BASE_URL}/memory/clear`, { method: 'POST' });
      if (res.ok) {
        showToast('Q&A memory database cleared.');
        if (trainingStatusCard) trainingStatusCard.classList.add('hidden');
        fetchMemory();
      }
    } catch (err) {
      showError(`Could not clear memory: ${err.message}`);
    }
  });
}

if (btnExportDataset) {
  btnExportDataset.addEventListener('click', () => {
    window.open(`${API_BASE_URL}/memory/export`, '_blank');
  });
}

// Auto-load Forest demo on startup for seamless first look & fetch memory
window.addEventListener('DOMContentLoaded', () => {
  const defaultChip = document.querySelector('.sample-chip[data-sample="forest"]');
  if (defaultChip) defaultChip.click();
  fetchMemory();
});
