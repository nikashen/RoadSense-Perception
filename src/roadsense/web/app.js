(() => {
  "use strict";

  const API_PATH = "/api/v1/demo";
  const PAGES_PAYLOAD_PATH = "demo.json";
  const WIDTH = 960;
  const HEIGHT = 540;
  const FIXTURE_CADENCE_MS = 100;
  const ROAD_BOTTOM = 540;
  const ROAD_HORIZON = 250;
  const SCENE_OBJECT_MIN_CONFIDENCE = 0.5;

  const COLORS = Object.freeze({
    car: "#b8ef67",
    bus: "#50c7ef",
    pedestrian: "#ffc46b",
    cyclist: "#ff7894",
    "traffic light": "#a99aff",
    default: "#f0f7f4",
    road: "#42e1c3",
    sidewalk: "#a99aff",
  });

  const state = {
    fixture: null,
    source: "loading",
    sourceReason: "",
    currentFrame: 0,
    playing: false,
    timer: null,
    selectedTrack: null,
    threshold: 0.5,
    layers: {
      detection: true,
      segmentation: true,
      tracking: true,
    },
  };

  const elements = {};

  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    cacheElements();
    bindEvents();
    configureCanvas();

    const requestedView = window.location.hash === "#evidence" ? "evidence" : "lab";
    setView(requestedView, false);

    try {
      const loaded = await loadFixture();
      state.fixture = loaded.fixture;
      state.source = loaded.source;
      state.sourceReason = loaded.reason ?? "";
      state.currentFrame = 0;
      state.selectedTrack = firstVisibleDetection()?.trackId ?? null;

      buildFrameStrip();
      updateSourceStatus();
      updateManifest();
      render();
      elements.copyManifest.disabled = false;
    } catch (error) {
      // Keep the page usable even if a future fixture builder or a browser API
      // fails before the fetch fallback is available. The visible status makes
      // this degraded path explicit instead of leaving a permanent spinner.
      state.fixture = buildBuiltinFixture();
      state.source = "fallback";
      state.sourceReason = error instanceof Error ? error.message : "initialization error";
      state.currentFrame = 0;
      state.selectedTrack = firstVisibleDetection()?.trackId ?? null;
      buildFrameStrip();
      updateSourceStatus();
      updateManifest();
      render();
      elements.copyManifest.disabled = false;
    } finally {
      elements.canvasLoading.classList.add("is-hidden");
      elements.canvasShell?.setAttribute("aria-busy", "false");
    }
  }

  function cacheElements() {
    elements.canvas = document.querySelector("#scene-canvas");
    elements.canvasShell = document.querySelector("#canvas-shell");
    elements.canvasLoading = document.querySelector("#canvas-loading");
    elements.sourceLabel = document.querySelector("#data-source-label");
    elements.headerStatus = document.querySelector(".header-status");
    elements.frameCounter = document.querySelector("#frame-counter");
    elements.timeCounter = document.querySelector("#time-counter");
    elements.timelineEnd = document.querySelector("#timeline-end");
    elements.timeline = document.querySelector("#timeline");
    elements.frameStrip = document.querySelector("#frame-strip");
    elements.playButton = document.querySelector("#play-button");
    elements.playLabel = document.querySelector("#play-label");
    elements.previousButton = document.querySelector("#previous-button");
    elements.nextButton = document.querySelector("#next-button");
    elements.resetButton = document.querySelector("#reset-button");
    elements.layersReset = document.querySelector("#layers-reset");
    elements.detectionToggle = document.querySelector("#detection-toggle");
    elements.segmentationToggle = document.querySelector("#segmentation-toggle");
    elements.trackingToggle = document.querySelector("#tracking-toggle");
    elements.confidenceRange = document.querySelector("#confidence-range");
    elements.confidenceOutput = document.querySelector("#confidence-output");
    elements.visibleCount = document.querySelector("#visible-count");
    elements.trackCount = document.querySelector("#track-count");
    elements.coverageCount = document.querySelector("#coverage-count");
    elements.iouCount = document.querySelector("#iou-count");
    elements.objectCount = document.querySelector("#object-count");
    elements.objectList = document.querySelector("#object-list");
    elements.objectDetail = document.querySelector("#object-detail");
    elements.copyManifest = document.querySelector("#copy-manifest");
    elements.manifestEvidence = document.querySelector("#manifest-evidence");
    elements.manifestAuthorization = document.querySelector("#manifest-authorization");
    elements.manifestBenchmark = document.querySelector("#manifest-benchmark");
  }

  function bindEvents() {
    document.querySelectorAll("[data-view]").forEach((button) => {
      button.addEventListener("click", () => setView(button.dataset.view));
    });

    document.querySelector(".brand").addEventListener("click", (event) => {
      event.preventDefault();
      setView("lab");
    });

    elements.playButton.addEventListener("click", togglePlayback);
    elements.previousButton.addEventListener("click", () => stepFrame(-1));
    elements.nextButton.addEventListener("click", () => stepFrame(1));
    elements.resetButton.addEventListener("click", resetSequence);

    elements.timeline.addEventListener("input", (event) => {
      pausePlayback();
      selectFrame(Number(event.target.value));
    });

    elements.detectionToggle.addEventListener("change", syncLayers);
    elements.segmentationToggle.addEventListener("change", syncLayers);
    elements.trackingToggle.addEventListener("change", syncLayers);

    elements.layersReset.addEventListener("click", () => {
      elements.detectionToggle.checked = true;
      elements.segmentationToggle.checked = true;
      elements.trackingToggle.checked = true;
      elements.confidenceRange.value = "0.50";
      syncLayers();
      updateThreshold();
    });

    elements.confidenceRange.addEventListener("input", updateThreshold);
    elements.canvas.addEventListener("click", handleCanvasSelection);
    elements.copyManifest.addEventListener("click", copyManifestSummary);
    window.addEventListener("resize", debounce(configureCanvas, 100));
    window.addEventListener("hashchange", () => {
      setView(window.location.hash === "#evidence" ? "evidence" : "lab", false);
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) pausePlayback();
    });
    document.addEventListener("keydown", handleKeyboard);
  }

  async function loadFixture() {
    const fallback = buildBuiltinFixture();
    if (!isLocalApiHost()) {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 1200);
      try {
        const response = await fetch(PAGES_PAYLOAD_PATH, {
          headers: { Accept: "application/json" },
          cache: "no-cache",
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Pages payload returned ${response.status}`);
        }
        return {
          fixture: normalizeFixture(await response.json()),
          source: "pages",
          reason: "pages_payload",
        };
      } catch (error) {
        return {
          fixture: fallback,
          source: "builtin",
          reason: error instanceof Error ? error.message : "pages_payload_unavailable",
        };
      } finally {
        window.clearTimeout(timeout);
      }
    }
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 1800);

    try {
      const response = await fetch(API_PATH, {
        headers: { Accept: "application/json" },
        cache: "no-store",
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`demo endpoint returned ${response.status}`);
      }

      const payload = await response.json();
      return {
        fixture: normalizeFixture(payload),
        source: "api",
        reason: "api_loaded",
      };
    } catch (error) {
      const apiReason = error instanceof Error ? error.message : "api_unavailable";
      // A Pages build is often previewed from localhost with a static server.
      // In that case the hostname looks like an API host, but there is no
      // /api/v1/demo route. Try the versioned replay before using the emergency
      // in-memory fixture so local static previews match GitHub Pages.
      const pagesController = new AbortController();
      const pagesTimeout = window.setTimeout(() => pagesController.abort(), 1200);
      try {
        const pagesResponse = await fetch(PAGES_PAYLOAD_PATH, {
          headers: { Accept: "application/json" },
          cache: "no-cache",
          signal: pagesController.signal,
        });
        if (!pagesResponse.ok) {
          throw new Error(`Pages payload returned ${pagesResponse.status}`);
        }
        return {
          fixture: normalizeFixture(await pagesResponse.json()),
          source: "pages",
          reason: `api_unavailable: ${apiReason}; pages_payload`,
        };
      } catch (pagesError) {
        const pagesReason =
          pagesError instanceof Error ? pagesError.message : "pages_payload_unavailable";
        return {
          fixture: fallback,
          source: "fallback",
          reason: `api_unavailable: ${apiReason}; ${pagesReason}`,
        };
      } finally {
        window.clearTimeout(pagesTimeout);
      }
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function isLocalApiHost() {
    return ["localhost", "127.0.0.1", "0.0.0.0", "[::1]", "::1"].includes(
      window.location.hostname,
    );
  }

  function normalizeFixture(payload) {
    const root = payload?.fixture ?? payload?.demo ?? payload;
    const rawFrames = root?.frames ?? root?.sequence?.frames;
    if (!root || !Array.isArray(rawFrames) || rawFrames.length !== 24) {
      throw new Error("invalid demo fixture: exactly 24 frames are required");
    }
    if (root.schema_version !== "roadsense.demo/v1") {
      throw new Error("invalid demo fixture: unsupported schema");
    }
    if (root.source !== "deterministic_geometric_fixture") {
      throw new Error("invalid demo fixture: unsupported source");
    }
    const evidence = root.evidence;
    if (
      !evidence ||
      evidence.level !== "fixture" ||
      evidence.evaluation_authorized !== false ||
      evidence.frozen !== false ||
      evidence.benchmark_claim_available !== false
    ) {
      throw new Error("invalid demo fixture: evidence boundary is not fixture-only");
    }
    const rawFixtureId = root.fixture_id ?? root.fixtureId;
    if (typeof rawFixtureId !== "string" || !rawFixtureId.trim()) {
      throw new Error("invalid demo fixture: fixture_id is required");
    }
    if (typeof evidence.claim_boundary !== "string" || !evidence.claim_boundary.trim()) {
      throw new Error("invalid demo fixture: claim boundary is required");
    }
    const canvas = root.canvas ?? root.metadata?.canvas ?? {};
    if (Number(canvas.width) !== WIDTH || Number(canvas.height) !== HEIGHT) {
      throw new Error("invalid demo fixture: canvas must be 960x540");
    }

    const frames = rawFrames.map((rawFrame, index) => {
      const declaredIndex = rawFrame?.frame_index ?? rawFrame?.frameIndex;
      if (declaredIndex !== undefined && Number(declaredIndex) !== index) {
        throw new Error("invalid demo fixture: frame indices must be ordered");
      }
      const rawDetections = rawFrame?.detections ?? rawFrame?.objects ?? [];
      const rawSegments = rawFrame?.segments ?? rawFrame?.segmentation ?? [];

      const timestampMs = strictFiniteNumber(
        rawFrame?.timestamp_ms ?? rawFrame?.timestampMs,
        index * FIXTURE_CADENCE_MS,
        "timestamp",
      );
      if (timestampMs < 0) {
        throw new Error("invalid demo fixture: timestamps must be non-negative");
      }
      return {
        index,
        timestampMs,
        egoSpeedKph: Math.max(
          0,
          strictFiniteNumber(
            rawFrame?.ego_speed_kph ?? rawFrame?.egoSpeedKph,
            28 + index * 0.3,
            "ego speed",
          ),
        ),
        detections: rawDetections.map((item, itemIndex) => normalizeDetection(item, index, itemIndex)),
        segments: rawSegments.map(normalizeSegment).filter(Boolean),
      };
    });

    if (frames.some((frame, index) => index > 0 && frame.timestampMs < frames[index - 1].timestampMs)) {
      throw new Error("invalid demo fixture: timestamps must be monotonic");
    }

    if (!frames.some((frame) => frame.detections.length)) {
      throw new Error("invalid demo fixture: detections are required");
    }

    const rawCadence =
      root.cadence_ms ??
      root.cadenceMs ??
      (root.fps === undefined || root.fps === null ? undefined : 1000 / Number(root.fps));
    const cadenceMs = strictFiniteNumber(rawCadence, FIXTURE_CADENCE_MS, "cadence");
    if (cadenceMs !== FIXTURE_CADENCE_MS) {
      throw new Error(`invalid demo fixture: cadence must be ${FIXTURE_CADENCE_MS} ms`);
    }
    if (
      root.fps !== undefined &&
      root.fps !== null &&
      Math.abs(strictFiniteNumber(root.fps, 0, "fps") * cadenceMs - 1000) > 1e-6
    ) {
      throw new Error("invalid demo fixture: fps and cadence must describe the same cadence");
    }
    if (frames.some((frame, index) => frame.timestampMs !== index * cadenceMs)) {
      throw new Error("invalid demo fixture: timestamps must follow cadence_ms from frame zero");
    }

    return {
      fixtureId: rawFixtureId,
      schemaVersion: root.schema_version,
      source: root.source,
      evidence: {
        level: evidence.level,
        evaluationAuthorized: evidence.evaluation_authorized,
        frozen: evidence.frozen,
        benchmarkClaimAvailable: evidence.benchmark_claim_available,
        claimBoundary: evidence.claim_boundary,
      },
      kind: "deterministic_fixture",
      canvas: {
        width: WIDTH,
        height: HEIGHT,
      },
      cadenceMs,
      frames,
    };
  }

  function normalizeDetection(item, frameIndex, itemIndex) {
    const rawBox = item?.bbox ?? item?.box ?? [0, 0, 1, 1];
    const bbox = Array.isArray(rawBox)
      ? rawBox.slice(0, 4)
      : [rawBox.x, rawBox.y, rawBox.width ?? rawBox.w, rawBox.height ?? rawBox.h];
    const safeBox = [0, 1, 2, 3].map((index) => {
      return requiredFiniteNumber(bbox[index], `bbox coordinate ${index}`);
    });
    const className = String(item?.class_name ?? item?.className ?? item?.label ?? "object").toLowerCase();
    const trackId = String(item?.track_id ?? item?.trackId ?? item?.id ?? `D-${frameIndex}-${itemIndex}`);
    const x = clamp(safeBox[0], 0, WIDTH - 1);
    const y = clamp(safeBox[1], 0, HEIGHT - 1);

    return {
      id: String(item?.id ?? `${trackId}-f${frameIndex}`),
      trackId,
      className,
      confidence: clamp(
        strictFiniteNumber(item?.confidence ?? item?.score, 0.5, "confidence"),
        0,
        1,
      ),
      bbox: [
        x,
        y,
        clamp(safeBox[2], 1, WIDTH - x),
        clamp(safeBox[3], 1, HEIGHT - y),
      ],
      occlusion: String(item?.occlusion ?? "none"),
    };
  }

  function normalizeSegment(item, index) {
    const polygon = item?.polygon ?? item?.points;
    if (!Array.isArray(polygon) || polygon.length < 3) return null;

    const normalizedPolygon = polygon
      .map((point) => {
        if (Array.isArray(point)) {
          return [
            requiredFiniteNumber(point[0], "polygon coordinate"),
            requiredFiniteNumber(point[1], "polygon coordinate"),
          ];
        }
        return [
          requiredFiniteNumber(point?.x, "polygon coordinate"),
          requiredFiniteNumber(point?.y, "polygon coordinate"),
        ];
      })
      .map(([x, y]) => [clamp(x, 0, WIDTH), clamp(y, 0, HEIGHT)]);

    const className = String(item?.class_name ?? item?.className ?? item?.label ?? `region-${index}`).toLowerCase();
    return {
      id: String(item?.id ?? `${className}-${index}`),
      className,
      polygon: normalizedPolygon,
      color: String(item?.color ?? COLORS[className] ?? COLORS.default),
    };
  }

  function buildBuiltinFixture() {
    // Emergency fallback only: it preserves the normalized UI contract when a
    // Pages payload or local API cannot be loaded, and carries a distinct ID so
    // it cannot be mistaken for the hashed city-loop payload.
    const frames = Array.from({ length: 24 }, (_, index) => {
      const detections = [
        detection(
          "T-01",
          "car",
          0.95 - index * 0.004,
          laneVehicleBox(index, "right", 330, 420, 72, 112, 0.28),
          index,
        ),
        detection(
          "T-02",
          "car",
          0.87 + index * 0.003,
          laneVehicleBox(index, "left", 305, 360, 55, 78, 0.28),
          index,
        ),
        detection("T-07", "pedestrian", 0.91 - Math.abs(index - 5) * 0.009, [815 - index * 1.4, 300 + index * 0.8, 27, 67], index),
        detection("T-20", "traffic light", 0.73 + (index % 3) * 0.02, [668, 174, 23, 48], index),
      ];

      if (index <= 8) {
        detections.push(
          detection(
            "T-04",
            "bus",
            0.82 - index * 0.012,
            laneVehicleBox(index, "left", 340, 410, 78, 112, 0.55),
            index,
            index >= 7 ? "partial" : "none",
          ),
        );
      }

      if (index >= 3) {
        detections.push(
          detection("T-12", "cyclist", 0.68 + (index - 3) * 0.019, [125 + index * 8, 350 - index * 0.5, 48, 59], index),
        );
      }

      return {
        index,
        timestampMs: index * FIXTURE_CADENCE_MS,
        egoSpeedKph: Number((27.8 + index * 0.34).toFixed(1)),
        detections,
        segments: [
          {
            id: "road-region",
            className: "road",
            color: COLORS.road,
            polygon: [
              [236, 540],
              [779, 540],
              [543, 250],
              [419, 250],
            ],
          },
          {
            id: "left-sidewalk",
            className: "sidewalk",
            color: COLORS.sidewalk,
            polygon: [
              [0, 540],
              [236, 540],
              [419, 250],
              [345, 250],
            ],
          },
          {
            id: "right-sidewalk",
            className: "sidewalk",
            color: COLORS.sidewalk,
            polygon: [
              [779, 540],
              [960, 540],
              [615, 250],
              [543, 250],
            ],
          },
        ],
      };
    });

    return {
      fixtureId: "roadsense-emergency-fallback-v4",
      schemaVersion: "roadsense.demo/v1",
      source: "deterministic_geometric_fixture",
      evidence: {
        level: "fixture",
        evaluationAuthorized: false,
        frozen: false,
        benchmarkClaimAvailable: false,
        claimBoundary: "Emergency synthetic fallback only; no benchmark claim is authorized.",
      },
      kind: "deterministic_fixture",
      canvas: { width: WIDTH, height: HEIGHT },
      cadenceMs: FIXTURE_CADENCE_MS,
      frames,
    };
  }

  function laneVehicleBox(frameIndex, lane, startBottom, endBottom, startWidth, endWidth, laneFraction) {
    const progress = frameIndex / 23;
    const bottom = startBottom + (endBottom - startBottom) * progress;
    const width = startWidth + (endWidth - startWidth) * progress;
    const height = width * 0.58;
    const roadProgress = (bottom - ROAD_HORIZON) / (ROAD_BOTTOM - ROAD_HORIZON);
    const roadCenter = 481 + 22 * roadProgress;
    const leftEdge = 419 - 183 * roadProgress;
    const rightEdge = 543 + 236 * roadProgress;
    const centerX =
      lane === "left"
        ? roadCenter - (roadCenter - leftEdge) * laneFraction
        : roadCenter + (rightEdge - roadCenter) * laneFraction;
    return [centerX - width / 2, bottom - height, width, height];
  }

  function detection(trackId, className, confidence, bbox, frameIndex, occlusion = "none") {
    return {
      id: `${trackId}-f${String(frameIndex).padStart(2, "0")}`,
      trackId,
      className,
      confidence: Number(clamp(confidence, 0, 1).toFixed(3)),
      bbox: bbox.map((value) => Number(value.toFixed(2))),
      occlusion,
    };
  }

  function configureCanvas() {
    if (!elements.canvas) return;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    elements.canvas.width = Math.round(WIDTH * pixelRatio);
    elements.canvas.height = Math.round(HEIGHT * pixelRatio);
    elements.canvas.dataset.pixelRatio = String(pixelRatio);
    if (state.fixture) drawCurrentFrame();
  }

  function render() {
    if (!state.fixture) return;
    const frame = currentFrame();
    const visible = visibleDetections(frame);

    ensureSelectedTrack(visible);
    drawCurrentFrame();
    updateTransport(frame);
    updateFrameStrip();
    updateDiagnostics(frame, visible);
    renderObjectList(visible);
    renderObjectDetail(visible);
  }

  function drawCurrentFrame() {
    const canvas = elements.canvas;
    const context = canvas.getContext?.("2d");
    if (!context) return;
    const ratio = Number(canvas.dataset.pixelRatio || 1);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, WIDTH, HEIGHT);

    const frame = currentFrame();
    drawBaseScene(context, frame);

    if (state.layers.segmentation) {
      drawSegments(context, frame.segments);
    }

    // Low-confidence injected false positives remain overlay candidates, but
    // they must not materialize as physical actors in the synthetic scene.
    [...frame.detections]
      .filter((item) => item.confidence >= SCENE_OBJECT_MIN_CONFIDENCE)
      .sort((a, b) => a.bbox[1] + a.bbox[3] - (b.bbox[1] + b.bbox[3]))
      .forEach((item) => drawSceneObject(context, item));

    const visible = visibleDetections(frame);
    if (state.layers.tracking) {
      drawTrackTrails(context, visible);
    }
    if (state.layers.detection) {
      visible.forEach((item) => drawDetection(context, item, item.trackId === state.selectedTrack));
    }

    drawCanvasFooter(context, frame, visible);
  }

  function drawBaseScene(context, frame) {
    const sky = context.createLinearGradient(0, 0, 0, 300);
    sky.addColorStop(0, "#6f9fab");
    sky.addColorStop(0.62, "#a9c1bc");
    sky.addColorStop(1, "#d2d2b8");
    context.fillStyle = sky;
    context.fillRect(0, 0, WIDTH, HEIGHT);

    context.save();
    context.globalAlpha = 0.46;
    context.fillStyle = "#edf2d0";
    context.beginPath();
    context.arc(785, 84, 37, 0, Math.PI * 2);
    context.fill();
    context.restore();

    drawCloud(context, 170, 88, 0.82);
    drawCloud(context, 548, 112, 0.56);

    context.fillStyle = "#657f78";
    context.beginPath();
    context.moveTo(0, 245);
    context.lineTo(130, 174);
    context.lineTo(263, 238);
    context.lineTo(385, 182);
    context.lineTo(525, 244);
    context.lineTo(690, 165);
    context.lineTo(960, 240);
    context.lineTo(960, 290);
    context.lineTo(0, 290);
    context.closePath();
    context.fill();

    drawBuildings(context, frame.index);

    context.fillStyle = "#596566";
    context.beginPath();
    context.moveTo(236, HEIGHT);
    context.lineTo(419, 250);
    context.lineTo(543, 250);
    context.lineTo(779, HEIGHT);
    context.closePath();
    context.fill();

    const roadShade = context.createLinearGradient(0, 250, 0, HEIGHT);
    roadShade.addColorStop(0, "rgba(255,255,255,0.02)");
    roadShade.addColorStop(1, "rgba(0,0,0,0.18)");
    context.fillStyle = roadShade;
    context.fill();

    context.fillStyle = "#8b8f87";
    context.beginPath();
    context.moveTo(0, HEIGHT);
    context.lineTo(236, HEIGHT);
    context.lineTo(419, 250);
    context.lineTo(345, 250);
    context.lineTo(0, 377);
    context.closePath();
    context.fill();

    context.beginPath();
    context.moveTo(779, HEIGHT);
    context.lineTo(WIDTH, HEIGHT);
    context.lineTo(WIDTH, 349);
    context.lineTo(615, 250);
    context.lineTo(543, 250);
    context.closePath();
    context.fill();

    context.strokeStyle = "rgba(242,241,211,0.75)";
    context.lineWidth = 3;
    context.beginPath();
    context.moveTo(236, HEIGHT);
    context.lineTo(419, 250);
    context.moveTo(779, HEIGHT);
    context.lineTo(543, 250);
    context.stroke();

    drawLaneMarkings(context, frame.index);
    drawCrosswalk(context, frame.index);
    drawRoadFurniture(context);
  }

  function drawCloud(context, x, y, scale) {
    context.save();
    context.translate(x, y);
    context.scale(scale, scale);
    context.fillStyle = "rgba(236,244,229,0.36)";
    context.beginPath();
    context.arc(0, 4, 20, 0, Math.PI * 2);
    context.arc(25, -4, 27, 0, Math.PI * 2);
    context.arc(53, 7, 18, 0, Math.PI * 2);
    context.rect(-2, 4, 58, 23);
    context.fill();
    context.restore();
  }

  function drawBuildings(context, frameIndex) {
    const buildings = [
      [0, 197, 128, 131, "#526d69"],
      [80, 169, 120, 164, "#668079"],
      [190, 216, 91, 105, "#4e6966"],
      [717, 191, 112, 142, "#536c68"],
      [810, 154, 150, 192, "#647c75"],
    ];

    buildings.forEach(([x, y, width, height, color], buildingIndex) => {
      context.fillStyle = color;
      context.fillRect(x, y, width, height);
      context.fillStyle = "rgba(216,226,194,0.28)";
      const columnCount = Math.max(2, Math.floor(width / 28));
      const rowCount = Math.max(2, Math.floor(height / 32));
      for (let column = 0; column < columnCount; column += 1) {
        for (let row = 0; row < rowCount; row += 1) {
          const flicker = (column + row + buildingIndex + frameIndex) % 4 === 0;
          context.globalAlpha = flicker ? 0.35 : 0.16;
          context.fillRect(x + 11 + column * 25, y + 13 + row * 28, 8, 11);
        }
      }
      context.globalAlpha = 1;
    });
  }

  function drawLaneMarkings(context, frameIndex) {
    const offset = (frameIndex * 17) % 68;
    context.fillStyle = "rgba(242,241,211,0.84)";
    for (let index = -1; index < 5; index += 1) {
      const y = 276 + index * 71 + offset;
      if (y < 260 || y > HEIGHT + 20) continue;
      const perspective = (y - 245) / (HEIGHT - 245);
      const x = 481 + perspective * 22;
      const width = 2 + perspective * 7;
      const length = 13 + perspective * 31;
      context.beginPath();
      context.moveTo(x - width, y);
      context.lineTo(x + width, y);
      context.lineTo(x + width * 1.25, y + length);
      context.lineTo(x - width * 1.25, y + length);
      context.closePath();
      context.fill();
    }
  }

  function drawCrosswalk(context, frameIndex) {
    const y = 420 + Math.min(frameIndex * 1.2, 11);
    context.save();
    context.fillStyle = "rgba(235,234,210,0.56)";
    for (let index = 0; index < 7; index += 1) {
      const x = 352 + index * 52;
      context.beginPath();
      context.moveTo(x, y);
      context.lineTo(x + 28, y);
      context.lineTo(x + 38, y + 16);
      context.lineTo(x + 4, y + 16);
      context.closePath();
      context.fill();
    }
    context.restore();
  }

  function drawRoadFurniture(context) {
    context.strokeStyle = "#344b48";
    context.lineWidth = 5;
    context.beginPath();
    context.moveTo(680, 250);
    context.lineTo(680, 160);
    context.lineTo(716, 160);
    context.stroke();

    context.fillStyle = "#263d39";
    context.fillRect(666, 169, 27, 55);
    ["#ff6d69", "#e8bd57", "#83d774"].forEach((color, index) => {
      context.beginPath();
      context.fillStyle = color;
      context.globalAlpha = index === 0 ? 0.92 : 0.22;
      context.arc(679.5, 181 + index * 15, 4.2, 0, Math.PI * 2);
      context.fill();
    });
    context.globalAlpha = 1;

    context.fillStyle = "#49615a";
    context.fillRect(89, 293, 7, 109);
    context.fillStyle = "#d7d4b5";
    context.fillRect(69, 289, 46, 29);
    context.fillStyle = "#55706a";
    context.font = "700 10px ui-sans-serif, system-ui";
    context.fillText("CITY", 79, 307);
  }

  function drawSegments(context, segments) {
    context.save();
    segments.forEach((segment) => {
      context.beginPath();
      segment.polygon.forEach(([x, y], index) => {
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.closePath();
      context.fillStyle = withAlpha(segment.color, segment.className === "road" ? 0.18 : 0.11);
      context.fill();
      context.strokeStyle = withAlpha(segment.color, 0.58);
      context.lineWidth = 1.5;
      context.setLineDash([7, 5]);
      context.stroke();
    });
    context.setLineDash([]);

    const road = segments.find((segment) => segment.className === "road");
    if (road) {
      const center = polygonCentroid(road.polygon);
      drawPill(context, center.x - 43, center.y + 84, "ROAD REGION", COLORS.road, true);
    }
    context.restore();
  }

  function drawSceneObject(context, item) {
    const [x, y, width, height] = item.bbox;
    context.save();

    if (item.className === "car" || item.className === "bus") {
      const isBus = item.className === "bus";
      const primaryTrack = item.trackId === "1" || item.trackId === "T-01";
      const bodyColor = primaryTrack ? "#314d5a" : isBus ? "#416c72" : "#8e7759";
      drawRoadVehicle(context, item, bodyColor, isBus);
    } else if (item.className === "pedestrian") {
      context.fillStyle = "#c98d54";
      context.beginPath();
      context.arc(x + width * 0.5, y + height * 0.12, width * 0.19, 0, Math.PI * 2);
      context.fill();
      context.strokeStyle = "#314d49";
      context.lineWidth = Math.max(3, width * 0.18);
      context.lineCap = "round";
      context.beginPath();
      context.moveTo(x + width * 0.5, y + height * 0.25);
      context.lineTo(x + width * 0.48, y + height * 0.64);
      context.moveTo(x + width * 0.48, y + height * 0.39);
      context.lineTo(x + width * 0.2, y + height * 0.54);
      context.moveTo(x + width * 0.48, y + height * 0.4);
      context.lineTo(x + width * 0.78, y + height * 0.52);
      context.moveTo(x + width * 0.48, y + height * 0.63);
      context.lineTo(x + width * 0.23, y + height * 0.94);
      context.moveTo(x + width * 0.48, y + height * 0.63);
      context.lineTo(x + width * 0.74, y + height * 0.94);
      context.stroke();
    } else if (item.className === "cyclist") {
      context.strokeStyle = "#263e3c";
      context.lineWidth = 3;
      context.beginPath();
      context.arc(x + width * 0.24, y + height * 0.78, width * 0.19, 0, Math.PI * 2);
      context.arc(x + width * 0.77, y + height * 0.78, width * 0.19, 0, Math.PI * 2);
      context.moveTo(x + width * 0.24, y + height * 0.78);
      context.lineTo(x + width * 0.5, y + height * 0.5);
      context.lineTo(x + width * 0.68, y + height * 0.78);
      context.lineTo(x + width * 0.24, y + height * 0.78);
      context.stroke();
      context.fillStyle = "#895064";
      context.beginPath();
      context.arc(x + width * 0.5, y + height * 0.18, width * 0.11, 0, Math.PI * 2);
      context.fill();
      context.strokeStyle = "#895064";
      context.lineWidth = 5;
      context.beginPath();
      context.moveTo(x + width * 0.5, y + height * 0.28);
      context.lineTo(x + width * 0.52, y + height * 0.53);
      context.stroke();
    }

    context.restore();
  }

  function drawRoadVehicle(context, item, bodyColor, isBus) {
    const [x, y, width, height] = item.bbox;
    const rearCenter = { x: x + width / 2, y: y + height * 0.9 };
    const forward = { x: 0, y: -1 };
    const right = { x: 1, y: 0 };
    const bodyLength = height * (isBus ? 0.88 : 0.84);
    const rearHalfWidth = Math.min(
      width * (isBus ? 0.38 : 0.31),
      height * (isBus ? 0.48 : 0.38),
    );
    const frontHalfWidth = rearHalfWidth * (isBus ? 0.82 : 0.64);
    const frontCenter = offsetPoint(rearCenter, forward, bodyLength);
    const body = [
      offsetPoint(rearCenter, right, -rearHalfWidth),
      offsetPoint(rearCenter, right, rearHalfWidth),
      offsetPoint(frontCenter, right, frontHalfWidth),
      offsetPoint(frontCenter, right, -frontHalfWidth),
    ];
    const crossAngle = Math.atan2(forward.y, forward.x) + Math.PI / 2;

    context.fillStyle = "rgba(0,0,0,0.24)";
    context.beginPath();
    context.ellipse(
      rearCenter.x - forward.x * height * 0.08,
      rearCenter.y - forward.y * height * 0.08,
      rearHalfWidth * 1.12,
      Math.max(2, height * 0.1),
      crossAngle,
      0,
      Math.PI * 2,
    );
    context.fill();

    drawPolygonPath(context, body);
    context.fillStyle = bodyColor;
    context.fill();
    context.strokeStyle = "rgba(12,28,27,0.72)";
    context.lineWidth = Math.max(1.2, height * 0.035);
    context.stroke();

    const windowRear = offsetPoint(rearCenter, forward, bodyLength * 0.43);
    const windowFront = offsetPoint(rearCenter, forward, bodyLength * 0.76);
    const windowRearHalf = rearHalfWidth * 0.62;
    const windowFrontHalf = frontHalfWidth * 0.76;
    drawPolygonPath(context, [
      offsetPoint(windowRear, right, -windowRearHalf),
      offsetPoint(windowRear, right, windowRearHalf),
      offsetPoint(windowFront, right, windowFrontHalf),
      offsetPoint(windowFront, right, -windowFrontHalf),
    ]);
    context.fillStyle = "#8eaaa8";
    context.fill();
    context.strokeStyle = "rgba(224,239,232,0.34)";
    context.lineWidth = 1;
    context.stroke();

    const wheelLongRadius = Math.max(2.5, height * 0.1);
    const wheelShortRadius = Math.max(1.3, height * 0.035);
    const headingAngle = Math.atan2(forward.y, forward.x);
    context.fillStyle = "#17201f";
    for (const direction of [-1, 1]) {
      const wheel = offsetPoint(
        offsetPoint(rearCenter, forward, bodyLength * 0.12),
        right,
        direction * rearHalfWidth * 0.92,
      );
      context.beginPath();
      context.ellipse(
        wheel.x,
        wheel.y,
        wheelLongRadius,
        wheelShortRadius,
        headingAngle,
        0,
        Math.PI * 2,
      );
      context.fill();
    }

    const rearPanel = offsetPoint(rearCenter, forward, bodyLength * 0.1);
    context.fillStyle = "#ef7a68";
    for (const direction of [-1, 1]) {
      const light = offsetPoint(rearPanel, right, direction * rearHalfWidth * 0.66);
      context.beginPath();
      context.arc(light.x, light.y, Math.max(1.5, height * 0.045), 0, Math.PI * 2);
      context.fill();
    }

    const headingCenter = offsetPoint(rearCenter, forward, bodyLength * 0.56);
    const headingTip = offsetPoint(headingCenter, forward, height * 0.1);
    const headingLeft = offsetPoint(headingCenter, right, -height * 0.07);
    const headingRight = offsetPoint(headingCenter, right, height * 0.07);
    context.strokeStyle = "rgba(231,244,235,0.72)";
    context.lineWidth = Math.max(1.2, height * 0.03);
    context.lineCap = "round";
    context.beginPath();
    context.moveTo(headingLeft.x, headingLeft.y);
    context.lineTo(headingTip.x, headingTip.y);
    context.lineTo(headingRight.x, headingRight.y);
    context.stroke();
  }

  function offsetPoint(point, direction, distance) {
    return {
      x: point.x + direction.x * distance,
      y: point.y + direction.y * distance,
    };
  }

  function drawPolygonPath(context, points) {
    context.beginPath();
    points.forEach((point, index) => {
      if (index === 0) context.moveTo(point.x, point.y);
      else context.lineTo(point.x, point.y);
    });
    context.closePath();
  }

  function drawTrackTrails(context, visible) {
    visible.forEach((item) => {
      const history = getTrackHistory(item.trackId, state.currentFrame);
      if (history.length < 2) return;
      const color = objectColor(item.className);
      context.save();
      context.strokeStyle = withAlpha(color, 0.82);
      context.fillStyle = color;
      context.lineWidth = 2.2;
      context.setLineDash([5, 5]);
      context.beginPath();
      history.forEach((detectionItem, index) => {
        const [x, y, width, height] = detectionItem.bbox;
        const centerX = x + width / 2;
        const centerY = y + height;
        if (index === 0) context.moveTo(centerX, centerY);
        else context.lineTo(centerX, centerY);
      });
      context.stroke();
      context.setLineDash([]);
      history.slice(0, -1).forEach((detectionItem, index) => {
        const [x, y, width, height] = detectionItem.bbox;
        context.globalAlpha = 0.28 + (index / history.length) * 0.46;
        context.beginPath();
        context.arc(x + width / 2, y + height, 2.6, 0, Math.PI * 2);
        context.fill();
      });
      context.restore();
    });
  }

  function drawDetection(context, item, selected) {
    const [x, y, width, height] = item.bbox;
    const color = objectColor(item.className);
    context.save();
    context.strokeStyle = color;
    context.lineWidth = selected ? 3 : 2;
    context.shadowColor = selected ? withAlpha(color, 0.58) : "transparent";
    context.shadowBlur = selected ? 12 : 0;
    context.strokeRect(x, y, width, height);
    context.shadowBlur = 0;

    const corner = Math.min(12, width * 0.18, height * 0.22);
    context.lineWidth = selected ? 4 : 3;
    [
      [x, y, 1, 1],
      [x + width, y, -1, 1],
      [x, y + height, 1, -1],
      [x + width, y + height, -1, -1],
    ].forEach(([cornerX, cornerY, directionX, directionY]) => {
      context.beginPath();
      context.moveTo(cornerX, cornerY + directionY * corner);
      context.lineTo(cornerX, cornerY);
      context.lineTo(cornerX + directionX * corner, cornerY);
      context.stroke();
    });

    const label = `${item.trackId}  ${item.className.toUpperCase()}  ${item.confidence.toFixed(2)}`;
    context.font = "700 11px ui-monospace, SFMono-Regular, Menlo, monospace";
    const textWidth = context.measureText(label).width;
    const labelY = y > 25 ? y - 23 : y + 4;
    context.fillStyle = "rgba(4,13,11,0.9)";
    roundedRect(context, x, labelY, textWidth + 14, 20, 4);
    context.fill();
    context.fillStyle = color;
    context.fillRect(x, labelY, 3, 20);
    context.fillStyle = "#f4faf7";
    context.fillText(label, x + 8, labelY + 14);
    context.restore();
  }

  function drawCanvasFooter(context, frame, visible) {
    context.save();
    const label = `FRAME ${String(frame.index + 1).padStart(2, "0")}  ·  ${visible.length} VISIBLE  ·  FIXTURE ONLY`;
    context.font = "700 10px ui-monospace, SFMono-Regular, Menlo, monospace";
    const width = context.measureText(label).width + 18;
    context.fillStyle = "rgba(3,11,10,0.72)";
    roundedRect(context, WIDTH - width - 12, HEIGHT - 31, width, 20, 5);
    context.fill();
    context.fillStyle = "rgba(238,247,243,0.82)";
    context.fillText(label, WIDTH - width - 3, HEIGHT - 17);
    context.restore();
  }

  function drawPill(context, x, y, label, color, translucent = false) {
    context.save();
    context.font = "700 10px ui-monospace, SFMono-Regular, Menlo, monospace";
    const width = context.measureText(label).width + 14;
    context.fillStyle = translucent ? "rgba(3,13,11,0.56)" : "rgba(3,13,11,0.85)";
    roundedRect(context, x, y, width, 20, 5);
    context.fill();
    context.fillStyle = color;
    context.fillText(label, x + 7, y + 14);
    context.restore();
  }

  function roundedRect(context, x, y, width, height, radius) {
    const safeRadius = Math.min(radius, width / 2, height / 2);
    context.beginPath();
    context.moveTo(x + safeRadius, y);
    context.arcTo(x + width, y, x + width, y + height, safeRadius);
    context.arcTo(x + width, y + height, x, y + height, safeRadius);
    context.arcTo(x, y + height, x, y, safeRadius);
    context.arcTo(x, y, x + width, y, safeRadius);
    context.closePath();
  }

  function updateTransport(frame) {
    const frames = state.fixture.frames;
    elements.frameCounter.textContent = `FRAME ${String(frame.index + 1).padStart(2, "0")} / ${String(frames.length).padStart(2, "0")}`;
    elements.timeCounter.textContent = formatTimestamp(frame.timestampMs);
    if (elements.timelineEnd) {
      const finalFrame = frames[frames.length - 1];
      elements.timelineEnd.textContent = `${(finalFrame.timestampMs / 1000).toFixed(1)} SEC`;
    }
    elements.timeline.max = String(frames.length - 1);
    elements.timeline.value = String(state.currentFrame);
    elements.timeline.setAttribute(
      "aria-valuetext",
      `Frame ${frame.index + 1} of ${frames.length}, ${formatTimestamp(frame.timestampMs)}`,
    );
    setRangeProgress(elements.timeline, state.currentFrame / Math.max(frames.length - 1, 1));
    elements.playButton.classList.toggle("is-playing", state.playing);
    elements.playButton.setAttribute("aria-pressed", String(state.playing));
    elements.playLabel.textContent = state.playing ? "Pause" : "Play";
  }

  function buildFrameStrip() {
    const fragment = document.createDocumentFragment();
    state.fixture.frames.forEach((frame) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "frame-chip";
      button.dataset.frame = String(frame.index);
      button.dataset.label = String(frame.index + 1).padStart(2, "0");
      button.textContent = String(frame.index + 1).padStart(2, "0");
      button.title = `Go to frame ${frame.index + 1}`;
      button.setAttribute("aria-label", `Go to frame ${frame.index + 1}`);
      button.style.setProperty("--activity", String(clamp(frame.detections.length / 7, 0.25, 1)));
      button.addEventListener("click", () => {
        pausePlayback();
        selectFrame(frame.index);
      });
      fragment.append(button);
    });
    elements.frameStrip.replaceChildren(fragment);
  }

  function updateFrameStrip() {
    elements.frameStrip.querySelectorAll(".frame-chip").forEach((button) => {
      const active = Number(button.dataset.frame) === state.currentFrame;
      button.classList.toggle("is-active", active);
      if (active) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
  }

  function updateDiagnostics(frame, visible) {
    const trackIds = new Set(visible.map((item) => item.trackId));
    const roadSegment = frame.segments.find((segment) => segment.className === "road");
    const roadCoverage = roadSegment ? (polygonArea(roadSegment.polygon) / (WIDTH * HEIGHT)) * 100 : 0;
    const maxOverlap = maximumPairwiseIou(visible);

    elements.visibleCount.textContent = String(visible.length);
    elements.trackCount.textContent = String(trackIds.size);
    elements.coverageCount.textContent = `${roadCoverage.toFixed(1)}%`;
    elements.iouCount.textContent = maxOverlap.toFixed(2);
  }

  function renderObjectList(visible) {
    elements.objectCount.textContent = `${visible.length} shown`;
    if (visible.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "No fixture boxes meet the current threshold.";
      elements.objectList.replaceChildren(empty);
      return;
    }

    const fragment = document.createDocumentFragment();
    visible.forEach((item) => {
      const button = document.createElement("button");
      const color = objectColor(item.className);
      button.type = "button";
      button.className = "object-row";
      button.style.setProperty("--object-color", color);
      button.classList.toggle("is-active", item.trackId === state.selectedTrack);
      button.setAttribute("aria-pressed", String(item.trackId === state.selectedTrack));
      button.setAttribute("aria-label", `${item.trackId}, ${item.className}, confidence ${item.confidence.toFixed(2)}`);

      const icon = document.createElement("span");
      icon.className = "object-icon";
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = classAbbreviation(item.className);

      const copy = document.createElement("span");
      copy.className = "object-copy";
      const name = document.createElement("strong");
      name.textContent = item.className;
      const track = document.createElement("small");
      track.textContent = `TRACK ${item.trackId}`;
      copy.append(name, track);

      const confidence = document.createElement("span");
      confidence.className = "confidence-value";
      confidence.textContent = item.confidence.toFixed(2);

      button.append(icon, copy, confidence);
      button.addEventListener("click", () => {
        state.selectedTrack = item.trackId;
        render();
      });
      fragment.append(button);
    });
    elements.objectList.replaceChildren(fragment);
  }

  function renderObjectDetail(visible) {
    const item = visible.find((candidate) => candidate.trackId === state.selectedTrack);
    if (!item) {
      const empty = document.createElement("div");
      empty.className = "detail-empty";
      empty.textContent = "Select an object to inspect its fixture track state.";
      elements.objectDetail.replaceChildren(empty);
      return;
    }

    const history = getTrackHistory(item.trackId, state.fixture.frames.length - 1);
    const seenThroughCurrent = getTrackHistory(item.trackId, state.currentFrame).length;
    const confidences = history.map((entry) => entry.confidence);
    const [x, y, width, height] = item.bbox;

    const heading = document.createElement("div");
    heading.className = "detail-heading";
    const title = document.createElement("strong");
    title.textContent = `${item.trackId} · ${capitalize(item.className)}`;
    const status = document.createElement("span");
    status.textContent = "FIXTURE TRACK";
    heading.append(title, status);

    const grid = document.createElement("div");
    grid.className = "detail-grid";
    [
      ["Seen", `${seenThroughCurrent}/${history.length} frames`],
      ["Conf. range", `${Math.min(...confidences).toFixed(2)}–${Math.max(...confidences).toFixed(2)}`],
      ["Occlusion", item.occlusion],
      ["Position", `${Math.round(x)}, ${Math.round(y)}`],
      ["Box", `${Math.round(width)} × ${Math.round(height)}`],
      ["Ego speed", `${currentFrame().egoSpeedKph.toFixed(1)} km/h`],
    ].forEach(([label, value]) => {
      const cell = document.createElement("div");
      const labelNode = document.createElement("span");
      const valueNode = document.createElement("strong");
      labelNode.textContent = label;
      valueNode.textContent = value;
      cell.append(labelNode, valueNode);
      grid.append(cell);
    });

    elements.objectDetail.replaceChildren(heading, grid);
  }

  function updateManifest() {
    const fixture = state.fixture;
    const sourceLabel =
      state.source === "api"
        ? "API deterministic fixture"
        : state.source === "fallback"
          ? "Built-in fallback (API unavailable)"
          : state.source === "pages"
            ? "Pages fixture · versioned payload"
            : "Built-in static fixture (Pages fallback)";
    document.querySelector("#manifest-id").textContent = fixture.fixtureId;
    document.querySelector("#manifest-schema").textContent = fixture.schemaVersion;
    document.querySelector("#manifest-canvas").textContent = `${fixture.canvas.width} × ${fixture.canvas.height}`;
    document.querySelector("#manifest-frames").textContent = `${fixture.frames.length} frames @ ${fixture.cadenceMs} ms`;
    document.querySelector("#manifest-source").textContent = sourceLabel;
    if (fixture.evidence) {
      elements.manifestEvidence.textContent = fixture.evidence.level;
      elements.manifestAuthorization.textContent = fixture.evidence.evaluationAuthorized
        ? "authorized"
        : "not authorized";
      elements.manifestBenchmark.textContent = fixture.evidence.benchmarkClaimAvailable
        ? "available"
        : "unavailable";
    }
  }

  function updateSourceStatus() {
    elements.headerStatus.classList.add("is-ready");
    const labels = {
      api: "API fixture · no inference",
      pages: "Pages fixture · versioned payload",
      builtin: "Static fixture · Pages fallback",
      fallback: "Fallback fixture · API unavailable",
    };
    elements.sourceLabel.textContent = labels[state.source] ?? "Fixture loaded";
    if (state.sourceReason && ["builtin", "fallback"].includes(state.source)) {
      const reasonLabel =
        state.source === "builtin"
          ? `Versioned Pages payload unavailable; using built-in fixture (${state.sourceReason})`
          : `The local demo endpoint was not accepted (${state.sourceReason})`;
      elements.headerStatus.title = reasonLabel;
      elements.headerStatus.setAttribute(
        "aria-label",
        `${elements.sourceLabel.textContent}. ${reasonLabel}`,
      );
    } else {
      elements.headerStatus.removeAttribute("title");
      elements.headerStatus.removeAttribute("aria-label");
    }
  }

  function syncLayers() {
    state.layers.detection = elements.detectionToggle.checked;
    state.layers.segmentation = elements.segmentationToggle.checked;
    state.layers.tracking = elements.trackingToggle.checked;
    render();
  }

  function updateThreshold() {
    state.threshold = Number(elements.confidenceRange.value);
    elements.confidenceOutput.value = state.threshold.toFixed(2);
    elements.confidenceOutput.textContent = state.threshold.toFixed(2);
    const min = Number(elements.confidenceRange.min);
    const max = Number(elements.confidenceRange.max);
    setRangeProgress(elements.confidenceRange, (state.threshold - min) / (max - min));
    render();
  }

  function setView(view, updateHash = true) {
    const safeView = view === "evidence" ? "evidence" : "lab";
    document.querySelectorAll("[data-view-panel]").forEach((panel) => {
      const active = panel.dataset.viewPanel === safeView;
      panel.hidden = !active;
      panel.classList.toggle("is-active", active);
    });
    document.querySelectorAll("[data-view]").forEach((button) => {
      const active = button.dataset.view === safeView;
      button.classList.toggle("is-active", active);
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    if (safeView !== "lab") pausePlayback();
    if (updateHash) {
      window.history.replaceState(null, "", safeView === "evidence" ? "#evidence" : "#lab");
    }
    if (safeView === "lab" && state.fixture) window.requestAnimationFrame(configureCanvas);
  }

  function togglePlayback() {
    if (state.playing) pausePlayback();
    else startPlayback();
  }

  function startPlayback() {
    if (!state.fixture || state.playing) return;
    state.playing = true;
    elements.playButton.classList.add("is-playing");
    elements.playButton.setAttribute("aria-pressed", "true");
    elements.playLabel.textContent = "Pause";
    state.timer = window.setInterval(() => {
      const next = (state.currentFrame + 1) % state.fixture.frames.length;
      selectFrame(next, false);
    }, Math.max(50, state.fixture.cadenceMs));
  }

  function pausePlayback() {
    if (state.timer !== null) window.clearInterval(state.timer);
    state.timer = null;
    state.playing = false;
    if (!elements.playButton) return;
    elements.playButton.classList.remove("is-playing");
    elements.playButton.setAttribute("aria-pressed", "false");
    elements.playLabel.textContent = "Play";
  }

  function stepFrame(direction) {
    if (!state.fixture) return;
    pausePlayback();
    const length = state.fixture.frames.length;
    selectFrame((state.currentFrame + direction + length) % length);
  }

  function resetSequence() {
    pausePlayback();
    state.selectedTrack = null;
    selectFrame(0);
  }

  function selectFrame(index, pause = true) {
    if (!state.fixture) return;
    if (pause) pausePlayback();
    state.currentFrame = clamp(Math.round(index), 0, state.fixture.frames.length - 1);
    render();
  }

  function handleCanvasSelection(event) {
    if (!state.fixture) return;
    const rect = elements.canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * WIDTH;
    const y = ((event.clientY - rect.top) / rect.height) * HEIGHT;
    const candidates = visibleDetections(currentFrame())
      .filter((item) => pointInsideBox(x, y, item.bbox))
      .sort((a, b) => boxArea(a.bbox) - boxArea(b.bbox));
    if (candidates.length) {
      state.selectedTrack = candidates[0].trackId;
      render();
    }
  }

  function handleKeyboard(event) {
    const tagName = event.target?.tagName?.toLowerCase();
    if (["input", "button", "textarea", "select"].includes(tagName)) return;
    if (window.location.hash === "#evidence") return;

    if (event.code === "Space") {
      event.preventDefault();
      togglePlayback();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      stepFrame(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      stepFrame(1);
    } else if (event.key === "Home") {
      event.preventDefault();
      resetSequence();
    }
  }

  async function copyManifestSummary() {
    if (!state.fixture) return;
    const fixture = state.fixture;
    const sourceLabel =
      state.source === "api"
        ? "api_fixture"
        : state.source === "fallback"
          ? "builtin_fallback_fixture"
          : state.source === "pages"
            ? "pages_static_payload"
            : "builtin_emergency_fallback";
    const summary = [
      `fixture_id=${fixture.fixtureId}`,
      `schema_version=${fixture.schemaVersion}`,
      `canvas=${fixture.canvas.width}x${fixture.canvas.height}`,
      `frames=${fixture.frames.length}`,
      `cadence_ms=${fixture.cadenceMs}`,
      `source=${sourceLabel}`,
      `evidence_level=${fixture.evidence?.level ?? "fixture"}`,
      `evaluation_authorized=${fixture.evidence?.evaluationAuthorized ?? false}`,
      `frozen=${fixture.evidence?.frozen ?? false}`,
      `benchmark_claim_available=${fixture.evidence?.benchmarkClaimAvailable ?? false}`,
      `claim_boundary=${fixture.evidence?.claimBoundary ?? "FIXTURE ONLY / NO BENCHMARK CLAIM"}`,
    ].join("\n");

    const label = elements.copyManifest.querySelector("span");
    if (!label) return;
    let copied = false;
    try {
      await navigator.clipboard.writeText(summary);
      copied = true;
    } catch (_error) {
      try {
        const textarea = document.createElement("textarea");
        textarea.value = summary;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.append(textarea);
        try {
          textarea.select();
          copied = document.execCommand("copy");
        } finally {
          textarea.remove();
        }
      } catch (_fallbackError) {
        copied = false;
      }
    }
    label.textContent = copied ? "Copied" : "Copy unavailable";
    window.setTimeout(() => {
      label.textContent = "Copy manifest summary";
    }, 1600);
  }

  function currentFrame() {
    return state.fixture.frames[state.currentFrame];
  }

  function visibleDetections(frame) {
    return frame.detections.filter((item) => item.confidence >= state.threshold);
  }

  function firstVisibleDetection() {
    if (!state.fixture) return null;
    return visibleDetections(currentFrame())[0] ?? null;
  }

  function ensureSelectedTrack(visible) {
    if (!visible.some((item) => item.trackId === state.selectedTrack)) {
      state.selectedTrack = visible[0]?.trackId ?? null;
    }
  }

  function getTrackHistory(trackId, throughFrame) {
    return state.fixture.frames
      .slice(0, throughFrame + 1)
      .flatMap((frame) => frame.detections.filter((item) => item.trackId === trackId));
  }

  function maximumPairwiseIou(detections) {
    let maximum = 0;
    for (let first = 0; first < detections.length; first += 1) {
      for (let second = first + 1; second < detections.length; second += 1) {
        maximum = Math.max(maximum, intersectionOverUnion(detections[first].bbox, detections[second].bbox));
      }
    }
    return maximum;
  }

  function intersectionOverUnion(first, second) {
    const left = Math.max(first[0], second[0]);
    const top = Math.max(first[1], second[1]);
    const right = Math.min(first[0] + first[2], second[0] + second[2]);
    const bottom = Math.min(first[1] + first[3], second[1] + second[3]);
    const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
    const union = boxArea(first) + boxArea(second) - intersection;
    return union > 0 ? intersection / union : 0;
  }

  function polygonArea(points) {
    let sum = 0;
    points.forEach(([x, y], index) => {
      const [nextX, nextY] = points[(index + 1) % points.length];
      sum += x * nextY - nextX * y;
    });
    return Math.abs(sum) / 2;
  }

  function polygonCentroid(points) {
    const sum = points.reduce(
      (accumulator, [x, y]) => ({ x: accumulator.x + x, y: accumulator.y + y }),
      { x: 0, y: 0 },
    );
    return { x: sum.x / points.length, y: sum.y / points.length };
  }

  function boxArea([, , width, height]) {
    return Math.max(0, width) * Math.max(0, height);
  }

  function pointInsideBox(x, y, [boxX, boxY, width, height]) {
    return x >= boxX && x <= boxX + width && y >= boxY && y <= boxY + height;
  }

  function objectColor(className) {
    return COLORS[className] ?? COLORS.default;
  }

  function classAbbreviation(className) {
    const abbreviations = {
      car: "CAR",
      bus: "BUS",
      pedestrian: "PED",
      cyclist: "CYC",
      "traffic light": "SIG",
    };
    return abbreviations[className] ?? className.slice(0, 3).toUpperCase();
  }

  function formatTimestamp(milliseconds) {
    const safeMilliseconds = Math.max(0, Math.round(milliseconds));
    const seconds = Math.floor(safeMilliseconds / 1000);
    const remainder = safeMilliseconds % 1000;
    const minutes = Math.floor(seconds / 60);
    return `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}.${String(remainder).padStart(3, "0")}`;
  }

  function setRangeProgress(element, ratio) {
    element.style.setProperty("--range-progress", `${clamp(ratio, 0, 1) * 100}%`);
  }

  function withAlpha(hexColor, alpha) {
    const hex = hexColor.replace("#", "");
    if (!/^[0-9a-fA-F]{6}$/.test(hex)) return `rgba(240,247,244,${alpha})`;
    const red = Number.parseInt(hex.slice(0, 2), 16);
    const green = Number.parseInt(hex.slice(2, 4), 16);
    const blue = Number.parseInt(hex.slice(4, 6), 16);
    return `rgba(${red},${green},${blue},${alpha})`;
  }

  function strictFiniteNumber(value, fallback, fieldName) {
    if (value === undefined || value === null || value === "") return fallback;
    const number = Number(value);
    if (!Number.isFinite(number)) {
      throw new Error(`invalid demo fixture: ${fieldName} must be finite`);
    }
    return number;
  }

  function requiredFiniteNumber(value, fieldName) {
    if (value === undefined || value === null || value === "") {
      throw new Error(`invalid demo fixture: ${fieldName} is required`);
    }
    return strictFiniteNumber(value, 0, fieldName);
  }

  function clamp(value, minimum, maximum) {
    return Math.min(Math.max(value, minimum), maximum);
  }

  function capitalize(value) {
    return value ? `${value.charAt(0).toUpperCase()}${value.slice(1)}` : "";
  }

  function debounce(callback, wait) {
    let timeout;
    return (...args) => {
      window.clearTimeout(timeout);
      timeout = window.setTimeout(() => callback(...args), wait);
    };
  }
})();
