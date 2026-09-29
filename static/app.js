/** Dog Breed Recognition: local uploads, previews and predictions. No dependencies. */
document.addEventListener("DOMContentLoaded", () => {
    const ids = [
        "imageInput", "fileLabel", "predictBtn", "resetBtn", "previewContainer",
        "imagePreview", "previewMessage", "fileNameDisplay", "loadingIndicator",
        "errorContainer", "errorMessage", "statusMessage", "resultContainer",
        "top1Container", "top1Breed", "top1Confidence", "resultNote",
        "top5Container", "top5TableBody"
    ];
    const elements = ids.map(id => document.getElementById(id));
    if (elements.some(element => !element)) {
        console.error("The prediction page is missing a required UI element.");
        return;
    }
    const [imageInput, fileLabel, predictBtn, resetBtn, previewContainer,
        imagePreview, previewMessage, fileNameDisplay, loadingIndicator,
        errorContainer, errorMessage, statusMessage, resultContainer,
        top1Container, top1Breed, top1Confidence, resultNote,
        top5Container, top5TableBody] = elements;

    // Flask limits the entire multipart request, including headers and boundary.
    const MAX_FILE_SIZE = 10 * 1024 * 1024 - 64 * 1024;
    const REQUEST_TIMEOUT_MS = 60_000;
    const DEFAULT_RESULT_NOTE = "The model compares 120 dog breeds and applies a separate OOD check before showing a breed.";
    let selectedFile = null;
    let previewObjectUrl = null;
    let previewLoader = null;
    let activeRequest = null;

    function formatBreedName(rawName) {
        return rawName.replace(/_/g, " ").toLowerCase()
            .replace(/(^|[ -])([a-z])/g, (_, separator, letter) => separator + letter.toUpperCase());
    }

    function clearError() {
        errorContainer.classList.add("hidden");
        errorMessage.textContent = "";
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorContainer.classList.remove("hidden");
        statusMessage.textContent = "";
    }

    function clearResults() {
        resultContainer.classList.add("hidden");
        top1Container.classList.remove("hidden");
        top5Container.classList.remove("hidden");
        top1Breed.textContent = "";
        top1Confidence.textContent = "";
        resultNote.textContent = DEFAULT_RESULT_NOTE;
        resultNote.classList.remove("decision-note-not-dog");
        resultNote.classList.remove("decision-note-uncertain");
        top5TableBody.replaceChildren();
    }

    function setLoading(loading) {
        loadingIndicator.classList.toggle("hidden", !loading);
        predictBtn.textContent = loading ? "Predicting..." : "Predict Breed";
        predictBtn.disabled = loading || !selectedFile;
        predictBtn.setAttribute("aria-busy", String(loading));
        resetBtn.disabled = !selectedFile;
    }

    function cancelPrediction() {
        if (activeRequest) {
            const request = activeRequest;
            activeRequest = null; // Invalidates late responses even if abort arrives too late.
            clearTimeout(request.timeout);
            request.controller.abort();
        }
        setLoading(false);
    }

    function releasePreview() {
        if (previewLoader) {
            previewLoader.onload = null;
            previewLoader.onerror = null;
            previewLoader.removeAttribute("src");
            previewLoader = null;
        }
        imagePreview.removeAttribute("src");
        if (previewObjectUrl) {
            URL.revokeObjectURL(previewObjectUrl);
            previewObjectUrl = null;
        }
    }

    function resetSelection() {
        cancelPrediction();
        releasePreview();
        selectedFile = null;
        imageInput.value = "";
        imagePreview.alt = "Dog image preview";
        previewContainer.classList.add("hidden");
        previewMessage.textContent = "";
        fileNameDisplay.textContent = "";
        fileLabel.classList.remove("drag-over");
        clearResults();
        clearError();
        statusMessage.textContent = "";
        setLoading(false);
    }

    function handleFileSelection(file) {
        // Cancelling the native picker should preserve the current image and result.
        if (!file) return;
        resetSelection();
        if (file.size === 0) {
            showError("The selected file is empty. Please choose a JPEG, PNG, or WEBP image.");
            return;
        }
        if (file.size > MAX_FILE_SIZE) {
            showError("The selected image is too large. Choose a file up to 9.94 MiB (10 MiB total upload limit).");
            return;
        }

        // File.type and the extension are advisory. Pillow validates bytes on /predict.
        selectedFile = file;
        fileNameDisplay.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MiB)`;
        previewContainer.classList.remove("hidden");
        imagePreview.classList.add("hidden");
        previewMessage.textContent = "Loading preview...";
        statusMessage.textContent = `${file.name} selected. Ready to predict.`;
        setLoading(false);

        // Each selection owns its loader, so old decode events cannot change a new preview.
        try {
            const objectUrl = URL.createObjectURL(file);
            previewObjectUrl = objectUrl;
            const loader = new Image();
            previewLoader = loader;
            loader.onload = () => {
                if (previewObjectUrl !== objectUrl) return;
                imagePreview.src = objectUrl;
                imagePreview.alt = `Preview of selected image: ${file.name}`;
                imagePreview.classList.remove("hidden");
                previewMessage.textContent = "";
                previewLoader = null;
            };
            loader.onerror = () => {
                if (previewObjectUrl !== objectUrl) return;
                previewMessage.textContent = "Preview unavailable. Predict Breed will validate the image on the server.";
                previewLoader = null;
            };
            loader.src = objectUrl;
        } catch {
            previewMessage.textContent = "Preview unavailable. You can still request a prediction.";
        }
    }

    imageInput.addEventListener("change", event => {
        handleFileSelection(event.target.files && event.target.files[0]);
    });
    resetBtn.addEventListener("click", () => {
        resetSelection();
        statusMessage.textContent = "Image and results cleared.";
        imageInput.focus();
    });

    // The input remains a native keyboard control, with a visible focus ring on its label.
    imageInput.addEventListener("focus", () => fileLabel.classList.add("keyboard-focus"));
    imageInput.addEventListener("blur", () => fileLabel.classList.remove("keyboard-focus"));
    fileLabel.addEventListener("dragover", event => {
        event.preventDefault();
        fileLabel.classList.add("drag-over");
    });
    fileLabel.addEventListener("dragleave", () => fileLabel.classList.remove("drag-over"));
    fileLabel.addEventListener("drop", event => {
        event.preventDefault();
        fileLabel.classList.remove("drag-over");
        const files = event.dataTransfer && event.dataTransfer.files;
        if (!files || files.length === 0) return;
        if (files.length !== 1) {
            showError("Please drop one image at a time.");
            return;
        }
        handleFileSelection(files[0]);
    });

    function validPrediction(item) {
        return item && Number.isInteger(item.index) && item.index >= 0 && item.index < 120
            && typeof item.breed === "string" && item.breed.trim().length > 0
            && Number.isFinite(item.confidence) && item.confidence >= 0 && item.confidence <= 1;
    }

    function getStatus(data) {
        if (!data) return "";
        let s = (data.final_status || data.status || "").toUpperCase();
        if ((s === "SUCCESS" || !["DOG", "UNCERTAIN", "NOT_DOG"].includes(s)) && data.decision) {
            s = String(data.decision).toUpperCase();
        }
        return s;
    }

    function validResponse(data) {
        if (!data) return false;
        const status = getStatus(data);
        if (["DOG", "UNCERTAIN", "NOT_DOG"].includes(status)) {
            const breed = data.breed || data.top1_breed || (data.prediction && data.prediction.breed);
            const confidence = data.confidence !== undefined ? data.confidence : (data.prediction && data.prediction.confidence);
            if (typeof breed === "string" && breed.trim().length > 0
                && Number.isFinite(confidence) && confidence >= 0 && confidence <= 1
                && Array.isArray(data.top5)) {
                if (data.top5.length > 0) {
                    if (!data.top5.every(validPrediction)) return false;
                    if (new Set(data.top5.map(item => item.index)).size !== data.top5.length) return false;
                    if (!data.top5.every((item, idx) => idx === 0 || item.confidence <= data.top5[idx - 1].confidence)) return false;
                }
                return true;
            }
            return (data.breed === null || data.breed === undefined || typeof data.prediction === "string") && Array.isArray(data.top5);
        }
        if (data.status === "success" && ["DOG", "UNCERTAIN", "NOT_DOG"].includes(data.decision)) {
            return true;
        }
        if (data.prediction && validPrediction(data.prediction) && Array.isArray(data.top5)
            && data.top5.length === 5 && data.top5.every(validPrediction)) {
            const top5 = data.top5;
            return new Set(top5.map(item => item.index)).size === 5
                && top5.every((item, index) => index === 0 || item.confidence <= top5[index - 1].confidence)
                && top5[0].index === data.prediction.index
                && top5[0].breed === data.prediction.breed
                && top5[0].confidence === data.prediction.confidence;
        }
        return false;
    }

    function renderPredictions(data) {
        if (!validResponse(data)) {
            showError("The server returned an invalid prediction response. Please try again.");
            return;
        }
        resultContainer.classList.remove("hidden");
        const status = getStatus(data);

        if (status === "NOT_DOG") {
            top1Container.classList.add("hidden");
            top5Container.classList.add("hidden");
            resultNote.textContent = data.message || "No dog was confidently detected.";
            resultNote.classList.remove("decision-note-uncertain");
            resultNote.classList.add("decision-note-not-dog");
            statusMessage.textContent = resultNote.textContent;
            return;
        }

        const breed = data.breed || data.top1_breed || (data.prediction && data.prediction.breed) || "";
        const confidence = data.confidence !== undefined ? data.confidence : (data.prediction && data.prediction.confidence) || 0;
        const oodScore = (data.ood_score !== undefined && data.ood_score !== null) ? Number(data.ood_score) : null;

        top1Container.classList.remove("hidden");
        top5Container.classList.remove("hidden");

        top1Breed.textContent = formatBreedName(breed);
        top1Confidence.textContent = `${(confidence * 100).toFixed(2)}%`;

        const rows = (data.top5 || []).map((item, index) => {
            const row = document.createElement("tr");
            for (const [className, value] of [
                ["col-rank", String(index + 1)],
                ["col-breed", formatBreedName(item.breed)],
                ["col-confidence", `${(item.confidence * 100).toFixed(2)}%`]
            ]) {
                const cell = document.createElement("td");
                cell.className = className;
                cell.textContent = value;
                row.appendChild(cell);
            }
            const barCell = document.createElement("td");
            barCell.className = "col-bar";
            barCell.setAttribute("aria-hidden", "true");
            const track = document.createElement("div");
            track.className = "progress-track";
            const fill = document.createElement("div");
            fill.className = index === 0 ? "progress-fill top1-fill" : "progress-fill";
            fill.style.width = `${item.confidence * 100}%`;
            track.appendChild(fill);
            barCell.appendChild(track);
            row.appendChild(barCell);
            return row;
        });
        top5TableBody.replaceChildren(...rows);

        if (status === "UNCERTAIN") {
            const oodDetails = (oodScore !== null) ? ` (OOD score: ${oodScore.toFixed(4)})` : "";
            resultNote.textContent = "⚠️ Outlier Warning: " + (data.message || "The image is ambiguous or outside the reliable dog-breed region.") + oodDetails;
            resultNote.classList.remove("decision-note-not-dog");
            resultNote.classList.add("decision-note-uncertain");
            statusMessage.textContent = `Prediction uncertain: ${top1Breed.textContent}, ${top1Confidence.textContent} confidence. Warning: ambiguous or outlier image.`;
            return;
        }

        // DOG
        resultNote.textContent = "Confidence is a model score for one of the 120 supported dog breeds.";
        resultNote.classList.remove("decision-note-not-dog");
        resultNote.classList.remove("decision-note-uncertain");
        statusMessage.textContent = `Prediction complete: ${top1Breed.textContent}, ${top1Confidence.textContent} confidence. Top five results are available below.`;
    }

    predictBtn.addEventListener("click", async () => {
        if (activeRequest) return;
        if (!selectedFile) {
            showError("Please select an image first.");
            return;
        }
        clearError();
        clearResults();
        const request = { controller: new AbortController(), timeout: null };
        activeRequest = request;
        setLoading(true);
        statusMessage.textContent = "Predicting breed. Please wait.";
        request.timeout = setTimeout(() => {
            if (activeRequest !== request) return;
            cancelPrediction();
            showError("Prediction timed out. Check that the local app is running, then try again.");
        }, REQUEST_TIMEOUT_MS);

        try {
            const formData = new FormData();
            // A short transport filename keeps multipart overhead bounded; original name stays local.
            formData.append("image", selectedFile, "upload");
            const response = await fetch("/predict", {
                method: "POST",
                body: formData,
                signal: request.controller.signal
            }); // The browser supplies multipart Content-Type and boundary.
            if (activeRequest !== request) return;
            if (!response.ok) {
                if (response.status === 400) {
                    showError("Please choose a valid JPEG, PNG, or WEBP image up to 20 megapixels.");
                } else if (response.status === 413) {
                    showError("The upload exceeds the 10 MiB total request limit. Choose a smaller image.");
                } else if (response.status >= 500) {
                    showError("Prediction failed. Please try again.");
                } else {
                    showError(`Prediction request failed (HTTP ${response.status}). Please try again.`);
                }
                return;
            }
            let data;
            try {
                data = await response.json();
            } catch {
                if (activeRequest === request) {
                    showError("The server returned an unreadable prediction response. Please try again.");
                }
                return;
            }
            if (activeRequest === request) renderPredictions(data);
        } catch (error) {
            if (activeRequest === request && error.name !== "AbortError") {
                showError("Cannot reach the local app. Check that it is running and try again.");
            }
        } finally {
            clearTimeout(request.timeout);
            if (activeRequest === request) {
                activeRequest = null;
                setLoading(false);
            }
        }
    });

    window.addEventListener("pagehide", resetSelection);
});
