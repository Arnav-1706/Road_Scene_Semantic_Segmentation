(() => {
    "use strict";

    const CITYSCAPES_COLORS = {
        "Road": "rgb(128, 64, 128)",
        "Sidewalk": "rgb(244, 35, 232)",
        "Building": "rgb(70, 70, 70)",
        "Wall": "rgb(102, 102, 156)",
        "Fence": "rgb(190, 153, 153)",
        "Pole": "rgb(153, 153, 153)",
        "Traffic Light": "rgb(250, 170, 30)",
        "Traffic Sign": "rgb(220, 220, 0)",
        "Vegetation": "rgb(107, 142, 35)",
        "Terrain": "rgb(152, 251, 152)",
        "Sky": "rgb(70, 130, 180)",
        "Person": "rgb(220, 20, 60)",
        "Rider": "rgb(255, 0, 0)",
        "Car": "rgb(0, 0, 142)",
        "Truck": "rgb(0, 0, 70)",
        "Bus": "rgb(0, 60, 100)",
        "Train": "rgb(0, 80, 100)",
        "Motorcycle": "rgb(0, 0, 230)",
        "Bicycle": "rgb(119, 11, 32)"
    };

    const MODEL_NAMES = {
        unet: "U-Net",
        deeplab: "DeepLabV3+"
    };

    const MODE_HINTS = {
        high: "High Quality analyzes every frame for maximum temporal precision.",
        balanced: "Balanced analyzes every 3rd frame and reuses recent masks on intermediate frames without inflating stats.",
        fast: "Fast analyzes every 5th frame for maximum speed while preserving original video duration."
    };

    const IMAGE_EXTS = new Set(["jpg", "jpeg", "png"]);
    const VIDEO_EXTS = new Set(["mp4", "avi", "mov"]);

    let selectedFile = null;
    let selectedKind = null;
    let previewUrl = null;
    let classChart = null;
    let isRunning = false;
    let activeJobId = null;
    let pollTimer = null;

    const el = {
        deviceBadge: document.getElementById("deviceBadge"),
        noticeBar: document.getElementById("noticeBar"),
        noticeText: document.getElementById("noticeText"),
        noticeClose: document.getElementById("noticeClose"),

        modelSelect: document.getElementById("modelSelect"),
        modelRecommendation: document.getElementById("modelRecommendation"),
        yoloToggle: document.getElementById("yoloToggle"),
        videoModeSelect: document.getElementById("videoModeSelect"),
        videoModeHint: document.getElementById("videoModeHint"),

        uploadArea: document.getElementById("uploadArea"),
        fileInput: document.getElementById("fileInput"),
        uploadButton: document.getElementById("uploadButton"),
        changeFileButton: document.getElementById("changeFileButton"),
        clearFileButton: document.getElementById("clearFileButton"),
        dropPrompt: document.getElementById("dropPrompt"),
        fileChosen: document.getElementById("fileChosen"),
        fileName: document.getElementById("fileName"),
        fileMeta: document.getElementById("fileMeta"),

        analyzeButton: document.getElementById("analyzeButton"),
        cancelButton: document.getElementById("cancelButton"),

        videoProgressPanel: document.getElementById("videoProgressPanel"),
        progressStatusLabel: document.getElementById("progressStatusLabel"),
        progressPercent: document.getElementById("progressPercent"),
        progressBar: document.getElementById("progressBar"),
        progressFrames: document.getElementById("progressFrames"),
        progressFps: document.getElementById("progressFps"),
        progressElapsed: document.getElementById("progressElapsed"),
        progressEta: document.getElementById("progressEta"),

        resultMeta: document.getElementById("resultMeta"),
        imageResultsGrid: document.getElementById("imageResultsGrid"),
        videoResultsGrid: document.getElementById("videoResultsGrid"),
        originalPreview: document.getElementById("originalPreview"),
        segmentationPreview: document.getElementById("segmentationPreview"),
        detectionPreview: document.getElementById("detectionPreview"),
        videoOriginalPreview: document.getElementById("videoOriginalPreview"),
        videoOutputPreview: document.getElementById("videoOutputPreview"),
        downloadVideoLink: document.getElementById("downloadVideoLink"),
        videoCodecNotice: document.getElementById("videoCodecNotice"),

        summaryScopeNote: document.getElementById("summaryScopeNote"),
        totalLabel: document.getElementById("totalLabel"),
        totalObjects: document.getElementById("totalObjects"),
        totalSub: document.getElementById("totalSub"),
        carCount: document.getElementById("carCount"),
        carSub: document.getElementById("carSub"),
        personCount: document.getElementById("personCount"),
        personSub: document.getElementById("personSub"),
        processingTime: document.getElementById("processingTime"),
        processingSub: document.getElementById("processingSub"),

        videoFrameBreakdown: document.getElementById("videoFrameBreakdown"),
        breakdownTotal: document.getElementById("breakdownTotal"),
        breakdownAnalyzed: document.getElementById("breakdownAnalyzed"),
        breakdownReused: document.getElementById("breakdownReused"),
        breakdownSpeed: document.getElementById("breakdownSpeed"),

        detectedTags: document.getElementById("detectedTags"),
        chartTitle: document.getElementById("chartTitle"),
        classChart: document.getElementById("classChart"),
        chartEmpty: document.getElementById("chartEmpty")
    };

    // --------------------------------------------------
    // Load Hardware Info & Recommended Model
    // --------------------------------------------------

    async function loadSystemInfo() {
        try {
            const resp = await fetch("/api/system-info");
            if (!resp.ok) throw new Error("Unable to fetch system info");
            const info = await resp.json();

            if (info.device === "GPU") {
                const shortName = (info.device_name || "CUDA GPU").replace("NVIDIA GeForce ", "");
                el.deviceBadge.textContent = `GPU · ${shortName}`;
            } else {
                el.deviceBadge.textContent = "CPU Fallback";
            }

            const rec = info.recommended_video_model || "unet";
            const unetOpt = el.modelSelect.querySelector('option[value="unet"]');
            const dlOpt = el.modelSelect.querySelector('option[value="deeplab"]');
            if (unetOpt) {
                unetOpt.textContent = rec === "unet" ? "U-Net (Recommended)" : "U-Net";
            }
            if (dlOpt) {
                dlOpt.textContent = rec === "deeplab" ? "DeepLabV3+ (Recommended)" : "DeepLabV3+";
            }
            el.modelSelect.value = rec;
            if (info.recommended_reason) {
                el.modelRecommendation.textContent = `Recommended (${MODEL_NAMES[rec]}): ${info.recommended_reason}`;
            }
        } catch (err) {
            el.deviceBadge.textContent = "Local Runtime";
        }
    }

    // --------------------------------------------------
    // Helpers
    // --------------------------------------------------

    function showNotice(text, isError = false) {
        el.noticeText.textContent = text;
        el.noticeBar.className = isError ? "notice is-error" : "notice";
    }

    function hideNotice() {
        el.noticeBar.classList.add("hidden");
    }

    function getExt(filename) {
        const dot = filename.lastIndexOf(".");
        return dot >= 0 ? filename.slice(dot + 1).toLowerCase() : "";
    }

    function detectKind(file) {
        const ext = getExt(file.name);
        if (IMAGE_EXTS.has(ext)) return "image";
        if (VIDEO_EXTS.has(ext)) return "video";
        if (file.type === "image/jpeg" || file.type === "image/png") return "image";
        if (["video/mp4", "video/avi", "video/x-msvideo", "video/quicktime"].includes(file.type)) {
            return "video";
        }
        return null;
    }

    function formatSize(bytes) {
        if (!bytes) return "0 B";
        const units = ["B", "KB", "MB", "GB"];
        const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
        return `${(bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
    }

    function formatSeconds(sec) {
        if (sec === null || sec === undefined || !Number.isFinite(sec)) return "--";
        if (sec < 60) return `${sec.toFixed(1)}s`;
        const mins = Math.floor(sec / 60);
        const rem = (sec % 60).toFixed(0).padStart(2, "0");
        return `${mins}m ${rem}s`;
    }

    function getCount(counts, target) {
        if (!counts) return 0;
        const lower = target.toLowerCase();
        for (const [k, v] of Object.entries(counts)) {
            if (k.toLowerCase() === lower) return Number(v) || 0;
        }
        return 0;
    }

    function makePlaceholder(text) {
        const span = document.createElement("span");
        span.className = "placeholder-text";
        span.textContent = text;
        return span;
    }

    // --------------------------------------------------
    // Controls & File Handling
    // --------------------------------------------------

    el.videoModeSelect.addEventListener("change", () => {
        el.videoModeHint.textContent = MODE_HINTS[el.videoModeSelect.value] || "";
    });

    function clearSelectedFile() {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }
        selectedFile = null;
        selectedKind = null;
        el.fileInput.value = "";
        el.fileChosen.classList.add("hidden");
        el.dropPrompt.classList.remove("hidden");
        el.analyzeButton.disabled = true;
    }

    function handleFile(file) {
        hideNotice();
        const kind = detectKind(file);
        if (!kind) {
            clearSelectedFile();
            showNotice("Please upload a JPG, PNG, MP4, AVI, or MOV file.", true);
            return;
        }

        if (previewUrl) URL.revokeObjectURL(previewUrl);
        selectedFile = file;
        selectedKind = kind;
        previewUrl = URL.createObjectURL(file);

        el.dropPrompt.classList.add("hidden");
        el.fileChosen.classList.remove("hidden");
        el.fileName.textContent = file.name;
        el.fileMeta.textContent = `${kind === "video" ? "Video" : "Image"} · ${formatSize(file.size)}`;
        el.analyzeButton.disabled = isRunning;
        el.resultMeta.textContent = "";

        if (kind === "image") {
            el.videoResultsGrid.classList.add("hidden");
            el.imageResultsGrid.classList.remove("hidden");

            el.originalPreview.innerHTML = "";
            const img = document.createElement("img");
            img.src = previewUrl;
            img.alt = "Uploaded road scene";
            el.originalPreview.appendChild(img);

            el.segmentationPreview.innerHTML = "";
            el.segmentationPreview.appendChild(makePlaceholder("Ready to analyze"));

            el.detectionPreview.innerHTML = "";
            el.detectionPreview.appendChild(makePlaceholder("Ready to analyze"));
        } else {
            el.imageResultsGrid.classList.add("hidden");
            el.videoResultsGrid.classList.remove("hidden");
            el.videoCodecNotice.classList.add("hidden");
            el.downloadVideoLink.classList.add("hidden");

            el.videoOriginalPreview.innerHTML = "";
            const vid = document.createElement("video");
            vid.src = previewUrl;
            vid.controls = true;
            vid.playsInline = true;
            el.videoOriginalPreview.appendChild(vid);

            el.videoOutputPreview.innerHTML = "";
            el.videoOutputPreview.appendChild(makePlaceholder("Ready to analyze"));
        }
    }

    el.uploadButton.addEventListener("click", (e) => {
        e.stopPropagation();
        el.fileInput.click();
    });

    el.changeFileButton.addEventListener("click", (e) => {
        e.stopPropagation();
        if (!isRunning) el.fileInput.click();
    });

    el.clearFileButton.addEventListener("click", (e) => {
        e.stopPropagation();
        if (!isRunning) clearSelectedFile();
    });

    el.uploadArea.addEventListener("click", (e) => {
        if (!selectedFile && !isRunning && !e.target.closest("button")) {
            el.fileInput.click();
        }
    });

    el.uploadArea.addEventListener("keydown", (e) => {
        if ((e.key === "Enter" || e.key === " ") && !selectedFile && !isRunning) {
            e.preventDefault();
            el.fileInput.click();
        }
    });

    el.fileInput.addEventListener("change", (e) => {
        if (e.target.files[0]) handleFile(e.target.files[0]);
    });

    el.uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        if (!isRunning) el.uploadArea.classList.add("dragging");
    });

    el.uploadArea.addEventListener("dragleave", () => {
        el.uploadArea.classList.remove("dragging");
    });

    el.uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        el.uploadArea.classList.remove("dragging");
        if (isRunning) return;
        const file = e.dataTransfer.files[0];
        if (file) {
            try {
                el.fileInput.files = e.dataTransfer.files;
            } catch (err) {}
            handleFile(file);
        }
    });

    el.noticeClose.addEventListener("click", hideNotice);

    // --------------------------------------------------
    // Charts & Summary Rendering
    // --------------------------------------------------

    function renderDetectedTags(counts, useYolo) {
        el.detectedTags.innerHTML = "";
        if (!useYolo) {
            el.detectedTags.classList.add("hidden");
            return;
        }

        const items = Object.entries(counts || {})
            .filter(([, count]) => Number(count) > 0)
            .sort((a, b) => b[1] - a[1]);

        if (items.length === 0) {
            el.detectedTags.classList.add("hidden");
            return;
        }

        el.detectedTags.classList.remove("hidden");
        for (const [label, count] of items) {
            const tag = document.createElement("span");
            tag.className = "det-tag";
            tag.textContent = label;
            const strong = document.createElement("strong");
            strong.textContent = String(count);
            tag.appendChild(strong);
            el.detectedTags.appendChild(tag);
        }
    }

    function renderClassChart(distribution, isVideo = false) {
        if (classChart) {
            classChart.destroy();
            classChart = null;
        }

        const chartBox = el.classChart.parentElement;

        if (!Array.isArray(distribution) || distribution.length === 0) {
            if (chartBox) chartBox.style.height = "260px";
            el.chartEmpty.classList.remove("hidden");
            el.chartEmpty.textContent = "Run an analysis to view semantic class percentages.";
            return;
        }

        el.chartEmpty.classList.add("hidden");
        el.chartTitle.textContent = isVideo
            ? "Class Coverage (Averaged Across Analyzed Frames Only)"
            : "Class Coverage";

        if (typeof Chart === "undefined") return;

        const sorted = [...distribution].sort((a, b) => b.percentage - a.percentage);
        if (chartBox) {
            chartBox.style.height = `${Math.max(220, sorted.length * 26 + 44)}px`;
        }

        classChart = new Chart(el.classChart, {
            type: "bar",
            data: {
                labels: sorted.map((d) => d.class),
                datasets: [
                    {
                        label: "Coverage (%)",
                        data: sorted.map((d) => d.percentage),
                        backgroundColor: sorted.map((d) => CITYSCAPES_COLORS[d.class] || "#5c3a21"),
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: "y",
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const item = sorted[ctx.dataIndex];
                                return ` ${item.percentage}% (${item.pixels.toLocaleString()} px)`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: "rgba(111, 86, 70, 0.14)" },
                        ticks: {
                            color: "#6f5646",
                            callback: (val) => `${val}%`
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: {
                            autoSkip: false,
                            color: "#231710",
                            font: { size: 12 }
                        }
                    }
                }
            }
        });
    }

    function renderVideoCompleted(data, useYolo) {
        const modelLabel = MODEL_NAMES[data.model] || data.model;
        el.imageResultsGrid.classList.add("hidden");
        el.videoResultsGrid.classList.remove("hidden");
        el.videoCodecNotice.classList.add("hidden");

        el.resultMeta.textContent =
            `${modelLabel} · ${data.mode_label} · ${data.device} · ${data.analyzed_frames}/${data.frames_processed} analyzed`;

        el.videoOutputPreview.innerHTML = "";
        const outVideo = document.createElement("video");
        outVideo.src = data.video;
        outVideo.controls = true;
        outVideo.playsInline = true;
        outVideo.addEventListener("error", () => {
            el.videoCodecNotice.classList.remove("hidden");
        });
        el.videoOutputPreview.appendChild(outVideo);

        el.downloadVideoLink.href = data.video;
        el.downloadVideoLink.classList.remove("hidden");

        const counts = data.detection_counts || {};
        el.summaryScopeNote.textContent =
            data.reused_frames > 0
                ? `Metrics computed strictly from ${data.analyzed_frames} analyzed frames (not inflated by ${data.reused_frames} reused frames).`
                : `Computed across all ${data.analyzed_frames} frames.`;

        el.totalLabel.textContent = "Analyzed Detections";
        el.totalObjects.textContent = useYolo ? String(data.total_detections ?? 0) : "Off";
        el.totalSub.textContent = useYolo
            ? `Avg ${data.avg_detections_per_analyzed_frame ?? 0} / analyzed frame`
            : "YOLOv8 disabled";

        el.carCount.textContent = useYolo ? String(getCount(counts, "car")) : "--";
        el.carSub.textContent = useYolo ? "Across analyzed frames" : "";

        el.personCount.textContent = useYolo ? String(getCount(counts, "person")) : "--";
        el.personSub.textContent = useYolo ? "Across analyzed frames" : "";

        el.processingTime.textContent = `${data.processing_time} s`;
        el.processingSub.textContent = `${data.device} · ${data.processing_fps} FPS`;

        el.videoFrameBreakdown.classList.remove("hidden");
        el.breakdownTotal.textContent = `Total Frames: ${data.frames_processed} (${data.duration_seconds}s @ ${data.original_fps} fps)`;
        el.breakdownAnalyzed.textContent = `Analyzed Frames: ${data.analyzed_frames}`;
        el.breakdownReused.textContent = `Reused Predictions: ${data.reused_frames}`;
        el.breakdownSpeed.textContent = `Throughput: ${data.processing_fps} FPS`;

        renderDetectedTags(counts, useYolo);
        renderClassChart(data.class_distribution || [], true);
        showNotice(
            `Finished ${data.mode_label} video analysis with ${modelLabel} on ${data.device} in ${data.processing_time}s (${data.processing_fps} FPS).`
        );
    }

    // --------------------------------------------------
    // Execution & Real-Time Polling / Cancellation
    // --------------------------------------------------

    function setRunningUI(running, isVideo) {
        isRunning = running;
        el.analyzeButton.disabled = running || !selectedFile;
        el.modelSelect.disabled = running;
        el.yoloToggle.disabled = running;
        el.videoModeSelect.disabled = running;

        if (running && isVideo) {
            el.analyzeButton.textContent = "Processing...";
            el.cancelButton.classList.remove("hidden");
            el.cancelButton.disabled = false;
            el.cancelButton.textContent = "Cancel Processing";
            el.videoProgressPanel.classList.remove("hidden");
        } else if (running) {
            el.analyzeButton.textContent = "Analyzing...";
            el.cancelButton.classList.add("hidden");
            el.videoProgressPanel.classList.add("hidden");
        } else {
            el.analyzeButton.textContent = "Analyze Scene";
            el.cancelButton.classList.add("hidden");
        }
    }

    function updateProgressDisplay(status) {
        const pct = Math.max(0, Math.min(100, Number(status.percent) || 0));
        el.progressBar.style.width = `${pct}%`;
        el.progressPercent.textContent = `${pct.toFixed(1)}%`;

        const modeName = status.mode_label || "Video";
        const dev = status.device || "GPU";
        el.progressStatusLabel.textContent =
            status.status === "cancelling"
                ? "Cancelling and cleaning up..."
                : `${modeName} · Running on ${dev}`;

        const total = status.total_frames || 0;
        const done = status.processed_frames || 0;
        const analyzed = status.analyzed_frames || 0;
        el.progressFrames.textContent =
            total > 0
                ? `${done} / ${total} frames (${analyzed} analyzed)`
                : `${done} frames (${analyzed} analyzed)`;

        el.progressFps.textContent = `${(Number(status.current_fps) || 0).toFixed(1)} FPS`;
        el.progressElapsed.textContent = `Elapsed: ${formatSeconds(status.elapsed_seconds || 0)}`;
        el.progressEta.textContent =
            status.eta_seconds !== null && status.eta_seconds !== undefined
                ? `ETA: ${formatSeconds(status.eta_seconds)}`
                : "ETA: --";
    }

    function stopPolling() {
        if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
    }

    el.cancelButton.addEventListener("click", async () => {
        if (!activeJobId) return;
        el.cancelButton.disabled = true;
        el.cancelButton.textContent = "Cancelling...";
        try {
            await fetch(`/api/video-cancel/${activeJobId}`, { method: "POST" });
        } catch (err) {}
    });

    async function runVideoAsync(formData, useYolo) {
        updateProgressDisplay({
            status: "running",
            mode_label: el.videoModeSelect.options[el.videoModeSelect.selectedIndex].text.split(" (")[0],
            device: el.deviceBadge.textContent.startsWith("GPU") ? "GPU" : "CPU",
            percent: 0,
            total_frames: 0,
            processed_frames: 0,
            analyzed_frames: 0,
            current_fps: 0,
            elapsed_seconds: 0,
            eta_seconds: null
        });

        const startResp = await fetch("/api/analyze-video/start", {
            method: "POST",
            body: formData
        });
        const startData = await startResp.json();
        if (!startResp.ok || startData.error) {
            throw new Error(startData.error || "Failed to start video processing.");
        }

        activeJobId = startData.job_id;

        return new Promise((resolve, reject) => {
            const pollOnce = async () => {
                if (!activeJobId) return;
                try {
                    const stResp = await fetch(`/api/video-status/${activeJobId}`);
                    const st = await stResp.json();
                    if (!stResp.ok) {
                        stopPolling();
                        reject(new Error(st.error || "Lost job status."));
                        return;
                    }

                    updateProgressDisplay(st);

                    if (st.status === "completed") {
                        stopPolling();
                        activeJobId = null;
                        renderVideoCompleted(st.result, useYolo);
                        resolve();
                    } else if (st.status === "cancelled") {
                        stopPolling();
                        activeJobId = null;
                        el.videoOutputPreview.innerHTML = "";
                        el.videoOutputPreview.appendChild(makePlaceholder("Processing cancelled"));
                        showNotice("Video processing was cancelled and temporary files were cleaned up.");
                        resolve();
                    } else if (st.status === "error") {
                        stopPolling();
                        activeJobId = null;
                        reject(new Error(st.error || "Video processing failed."));
                    }
                } catch (err) {
                    stopPolling();
                    activeJobId = null;
                    reject(err);
                }
            };

            pollTimer = setInterval(pollOnce, 250);
            pollOnce();
        });
    }

    el.analyzeButton.addEventListener("click", async () => {
        if (isRunning) return;
        if (!selectedFile || !selectedKind) {
            showNotice("Please choose an image or video first.", true);
            return;
        }

        hideNotice();
        const isVideo = selectedKind === "video";
        const useYolo = el.yoloToggle.checked;

        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("model", el.modelSelect.value);
        formData.append("use_yolo", String(useYolo));
        if (isVideo) {
            formData.append("mode", el.videoModeSelect.value);
        }

        setRunningUI(true, isVideo);

        try {
            if (isVideo) {
                await runVideoAsync(formData, useYolo);
            } else {
                const response = await fetch("/api/analyze", {
                    method: "POST",
                    body: formData
                });
                const raw = await response.text();
                let data;
                try {
                    data = JSON.parse(raw);
                } catch (err) {
                    throw new Error("Server returned an unexpected response.");
                }
                if (!response.ok || data.error) {
                    throw new Error(data.error || "Image analysis failed.");
                }

                const modelLabel = MODEL_NAMES[data.model] || data.model;
                el.videoResultsGrid.classList.add("hidden");
                el.imageResultsGrid.classList.remove("hidden");
                el.videoFrameBreakdown.classList.add("hidden");
                el.summaryScopeNote.textContent = "";

                el.resultMeta.textContent = `${modelLabel} · ${data.device} · ${data.image_width}×${data.image_height}`;

                el.originalPreview.innerHTML = "";
                const origImg = document.createElement("img");
                origImg.src = data.original;
                origImg.alt = "Original road scene";
                el.originalPreview.appendChild(origImg);

                el.segmentationPreview.innerHTML = "";
                const segImg = document.createElement("img");
                segImg.src = data.segmentation;
                segImg.alt = "Semantic segmentation result";
                el.segmentationPreview.appendChild(segImg);

                el.detectionPreview.innerHTML = "";
                if (data.detection) {
                    const detImg = document.createElement("img");
                    detImg.src = data.detection;
                    detImg.alt = "Object detection result";
                    el.detectionPreview.appendChild(detImg);
                } else {
                    el.detectionPreview.appendChild(makePlaceholder("Object detection disabled"));
                }

                const counts = data.object_counts || {};
                el.totalLabel.textContent = "Total Objects";
                el.totalObjects.textContent = useYolo ? String(data.total_objects ?? 0) : "Off";
                el.totalSub.textContent = "";
                el.carCount.textContent = useYolo ? String(getCount(counts, "car")) : "--";
                el.carSub.textContent = "";
                el.personCount.textContent = useYolo ? String(getCount(counts, "person")) : "--";
                el.personSub.textContent = "";
                el.processingTime.textContent = `${data.processing_time} s`;
                el.processingSub.textContent = data.device || "";

                renderDetectedTags(counts, useYolo);
                renderClassChart(data.class_distribution || [], false);
                showNotice(`Finished analyzing with ${modelLabel} on ${data.device} in ${data.processing_time}s.`);
            }
        } catch (err) {
            showNotice(err.message || "Analysis failed.", true);
        } finally {
            setRunningUI(false, isVideo);
        }
    });

    loadSystemInfo();
})();
