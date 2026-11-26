(() => {
    "use strict";

    // Cityscapes class RGB colors matching app.py
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

    const IMAGE_EXTS = new Set(["jpg", "jpeg", "png"]);
    const VIDEO_EXTS = new Set(["mp4", "avi", "mov"]);

    let selectedFile = null;
    let selectedKind = null;
    let previewUrl = null;
    let classChart = null;
    let isRunning = false;

    const el = {
        noticeBar: document.getElementById("noticeBar"),
        noticeText: document.getElementById("noticeText"),
        noticeClose: document.getElementById("noticeClose"),

        modelSelect: document.getElementById("modelSelect"),
        yoloToggle: document.getElementById("yoloToggle"),
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

        totalLabel: document.getElementById("totalLabel"),
        totalObjects: document.getElementById("totalObjects"),
        carCount: document.getElementById("carCount"),
        personCount: document.getElementById("personCount"),
        processingTime: document.getElementById("processingTime"),
        detectedTags: document.getElementById("detectedTags"),
        classChart: document.getElementById("classChart"),
        chartEmpty: document.getElementById("chartEmpty")
    };

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
        el.analyzeButton.disabled = false;
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
        el.fileInput.click();
    });

    el.clearFileButton.addEventListener("click", (e) => {
        e.stopPropagation();
        clearSelectedFile();
    });

    el.uploadArea.addEventListener("click", (e) => {
        if (!selectedFile && !e.target.closest("button")) {
            el.fileInput.click();
        }
    });

    el.uploadArea.addEventListener("keydown", (e) => {
        if ((e.key === "Enter" || e.key === " ") && !selectedFile) {
            e.preventDefault();
            el.fileInput.click();
        }
    });

    el.fileInput.addEventListener("change", (e) => {
        if (e.target.files[0]) handleFile(e.target.files[0]);
    });

    el.uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        el.uploadArea.classList.add("dragging");
    });

    el.uploadArea.addEventListener("dragleave", () => {
        el.uploadArea.classList.remove("dragging");
    });

    el.uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        el.uploadArea.classList.remove("dragging");
        const file = e.dataTransfer.files[0];
        if (file) {
            try {
                el.fileInput.files = e.dataTransfer.files;
            } catch (err) {}
            handleFile(file);
        }
    });

    el.noticeClose.addEventListener("click", hideNotice);

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

    function renderClassChart(distribution) {
        if (classChart) {
            classChart.destroy();
            classChart = null;
        }

        const chartBox = el.classChart.parentElement;

        if (!Array.isArray(distribution) || distribution.length === 0) {
            if (chartBox) {
                chartBox.style.height = "260px";
            }
            el.chartEmpty.classList.remove("hidden");
            el.chartEmpty.textContent =
                selectedKind === "video"
                    ? "Class percentages are generated for single-image analysis."
                    : "Run an image analysis to view semantic class percentages.";
            return;
        }

        el.chartEmpty.classList.add("hidden");

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

    el.analyzeButton.addEventListener("click", async () => {
        if (isRunning) return;
        if (!selectedFile || !selectedKind) {
            showNotice("Please choose an image or video first.", true);
            return;
        }

        hideNotice();
        isRunning = true;
        el.analyzeButton.disabled = true;
        el.analyzeButton.textContent = selectedKind === "video" ? "Processing video..." : "Analyzing...";

        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("model", el.modelSelect.value);
        formData.append("use_yolo", String(el.yoloToggle.checked));

        const endpoint = selectedKind === "video" ? "/api/analyze-video" : "/api/analyze";
        const useYolo = el.yoloToggle.checked;

        try {
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData
            });

            const raw = await response.text();
            let data;
            try {
                data = JSON.parse(raw);
            } catch (err) {
                throw new Error("Server returned an unexpected response. Check the Flask terminal.");
            }

            if (!response.ok || data.error) {
                throw new Error(data.error || "Analysis failed.");
            }

            const modelLabel = MODEL_NAMES[data.model] || data.model;

            if (selectedKind === "video") {
                el.imageResultsGrid.classList.add("hidden");
                el.videoResultsGrid.classList.remove("hidden");
                el.videoCodecNotice.classList.add("hidden");

                el.resultMeta.textContent = `${modelLabel} · ${data.frames_processed} frames · ${data.original_fps} fps`;

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
                el.totalLabel.textContent = "Frame Detections";
                el.totalObjects.textContent = useYolo ? String(data.total_detections ?? 0) : "Off";
                el.carCount.textContent = useYolo ? String(getCount(counts, "car")) : "--";
                el.personCount.textContent = useYolo ? String(getCount(counts, "person")) : "--";
                el.processingTime.textContent = `${data.processing_time} s`;

                renderDetectedTags(counts, useYolo);
                renderClassChart([]);
            } else {
                el.videoResultsGrid.classList.add("hidden");
                el.imageResultsGrid.classList.remove("hidden");

                el.resultMeta.textContent = `${modelLabel} · ${data.image_width}×${data.image_height}`;

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
                el.carCount.textContent = useYolo ? String(getCount(counts, "car")) : "--";
                el.personCount.textContent = useYolo ? String(getCount(counts, "person")) : "--";
                el.processingTime.textContent = `${data.processing_time} s`;

                renderDetectedTags(counts, useYolo);
                renderClassChart(data.class_distribution || []);
            }

            showNotice(`Finished analyzing with ${modelLabel} in ${data.processing_time} seconds.`);
        } catch (err) {
            showNotice(err.message || "Analysis failed.", true);
        } finally {
            isRunning = false;
            el.analyzeButton.disabled = !selectedFile;
            el.analyzeButton.textContent = "Analyze Scene";
        }
    });
})();
