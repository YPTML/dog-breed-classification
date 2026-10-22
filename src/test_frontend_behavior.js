/** Browser-independent regression tests. Run: node --test src/test_frontend_behavior.js */
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const script = readFileSync(path.join(__dirname, "../static/app.js"), "utf8");
const html = readFileSync(path.join(__dirname, "../templates/index.html"), "utf8");
const FILE_LIMIT = 10 * 1024 * 1024 - 64 * 1024;
const file = (overrides = {}) => ({ name: "synthetic.png", type: "image/png", size: 100, ...overrides });
const prediction = () => {
    const top5 = [0.6, 0.2, 0.1, 0.05, 0.03].map((confidence, index) => ({
        index, breed: index === 0 ? "soft-coated_wheaten_terrier" : `breed_${index}`, confidence
    }));
    return { prediction: { ...top5[0] }, top5 };
};
const threeStatePrediction = (status = "DOG") => {
    const data = prediction();
    return { ...data, status, breed: data.prediction.breed, confidence: data.prediction.confidence };
};
function deferred() {
    let resolve, reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    return { promise, resolve, reject };
}

function harness(missingId = null) {
    const requests = [], createdUrls = [], revokedUrls = [], loaders = [], timers = new Map();
    const windowEvents = {}, errors = [], predictionEvents = [];
    let focused = null, pickerFiles = [], timerId = 0;
    class Element {
        constructor(tag = "div") {
            this.tagName = tag.toUpperCase();
            this.textContent = "";
            this.children = [];
            this.listeners = {};
            this.attributes = {};
            this.style = {};
            this.value = "";
            this.disabled = false;
            const classes = new Set();
            this.classList = {
                add: value => classes.add(value),
                remove: value => classes.delete(value),
                contains: value => classes.has(value),
                toggle: (value, force = !classes.has(value)) => {
                    if (force) classes.add(value); else classes.delete(value);
                }
            };
        }
        set innerHTML(_) { throw new Error("Unsafe HTML assignment"); }
        addEventListener(name, listener) { this.listeners[name] = listener; }
        emit(name, event = {}) {
            return this.listeners[name]?.({ target: this, preventDefault() {}, ...event });
        }
        appendChild(child) { this.children.push(child); }
        replaceChildren(...children) { this.children = children; }
        setAttribute(name, value) { this.attributes[name] = value; }
        removeAttribute(name) { delete this.attributes[name]; if (name === "src") this.src = undefined; }
        focus() { focused = this; this.emit("focus"); }
    }
    const elements = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(([, id]) => [id, new Element()]));
    if (missingId) delete elements[missingId];
    for (const id of ["previewContainer", "loadingIndicator", "errorContainer", "resultContainer"]) {
        elements[id].classList.add("hidden");
    }
    elements.predictBtn.disabled = true;
    elements.resetBtn.disabled = true;
    Object.defineProperty(elements.imageInput, "files", {
        get: () => pickerFiles,
        set: () => { throw new Error("Drop must not assign a browser FileList"); }
    });
    class PreviewImage extends Element {
        constructor() { super("img"); loaders.push(this); }
    }
    class UploadForm {
        constructor() { this.fields = []; }
        append(...field) { this.fields.push(field); }
    }
    vm.runInNewContext(script, {
        document: {
            getElementById: id => elements[id],
            createElement: tag => new Element(tag),
            addEventListener: (name, callback) => { if (name === "DOMContentLoaded") callback(); },
            dispatchEvent: event => { if (event.type === "prediction:result") predictionEvents.push(event.detail); }
        },
        window: { addEventListener: (name, callback) => { windowEvents[name] = callback; } },
        Image: PreviewImage, FormData: UploadForm, AbortController,
        CustomEvent: class { constructor(type, options) { this.type = type; this.detail = options.detail; } },
        URL: {
            createObjectURL: () => { const url = `blob:test/${createdUrls.length}`; createdUrls.push(url); return url; },
            revokeObjectURL: url => revokedUrls.push(url)
        },
        fetch: (url, options) => {
            const response = deferred();
            requests.push({ url, options, ...response });
            // Deliberately do not settle on abort: test late responses from an uncooperative transport.
            return response.promise;
        },
        setTimeout: callback => { timers.set(++timerId, callback); return timerId; },
        clearTimeout: id => timers.delete(id),
        console: { error: message => errors.push(message) }
    });
    return {
        elements, requests, createdUrls, revokedUrls, loaders, timers, errors, predictionEvents,
        focused: () => focused,
        select: selected => { pickerFiles = selected ? [selected] : []; elements.imageInput.emit("change"); },
        drop: files => elements.fileLabel.emit("drop", { dataTransfer: { files } }),
        click: () => elements.predictBtn.emit("click"),
        clear: () => elements.resetBtn.emit("click"),
        pagehide: () => windowEvents.pagehide(),
        expire: () => [...timers.values()].forEach(callback => callback()),
        respond: (index, data = prediction(), status = 200) => requests[index].resolve({
            ok: status >= 200 && status < 300, status, json: async () => data
        })
    };
}

const hidden = element => element.classList.contains("hidden");

test("uploads FormData without manual content type and renders five safe results", async () => {
    const h = harness();
    const chosen = file({ name: '<img src=x onerror=alert(1)>.png' });
    h.select(chosen);
    h.loaders[0].onload();
    assert.equal(h.elements.imagePreview.src, h.createdUrls[0]);
    assert.equal(hidden(h.elements.imagePreview), false);
    assert.ok(h.elements.fileNameDisplay.textContent.startsWith(chosen.name));
    const pending = h.click();
    assert.equal(h.elements.predictBtn.disabled, true);
    assert.equal(hidden(h.elements.loadingIndicator), false);
    assert.equal(h.requests[0].url, "/predict");
    assert.equal(h.requests[0].options.method, "POST");
    assert.equal(h.requests[0].options.headers, undefined);
    assert.deepEqual(h.requests[0].options.body.fields[0], ["image", chosen, "upload"]);
    h.respond(0);
    await pending;
    assert.equal(h.elements.top1Breed.textContent, "Soft-Coated Wheaten Terrier");
    assert.equal(h.elements.top1Confidence.textContent, "60.00%");
    assert.equal(h.elements.top5TableBody.children.length, 5);
    assert.equal(hidden(h.elements.resultContainer), false);
    assert.equal(h.elements.predictBtn.disabled, false);
    assert.equal(h.timers.size, 0);
    assert.match(h.elements.statusMessage.textContent, /Prediction complete/);
});

test("rejects empty and oversized files while accepting exact multipart-aware file boundary", () => {
    for (const size of [0, FILE_LIMIT + 1, 10 * 1024 * 1024]) {
        const h = harness();
        h.select(file({ size }));
        assert.equal(h.elements.predictBtn.disabled, true);
        assert.equal(hidden(h.elements.errorContainer), false);
        assert.equal(h.createdUrls.length, 0);
    }
    const h = harness();
    h.select(file({ size: FILE_LIMIT }));
    assert.equal(h.elements.predictBtn.disabled, false);
});

test("missing or misleading client MIME and extension remain advisory", () => {
    for (const type of ["", "application/octet-stream", "text/plain", "image/jpeg", "image/webp"]) {
        const h = harness();
        h.select(file({ name: "unknown.bin", type }));
        assert.equal(h.elements.predictBtn.disabled, false);
        assert.equal(hidden(h.elements.errorContainer), true);
    }
});

test("preview failure stays usable and cancelling picker preserves selection", () => {
    const h = harness();
    h.select(file());
    h.loaders[0].onerror();
    assert.match(h.elements.previewMessage.textContent, /Preview unavailable/);
    assert.equal(h.elements.predictBtn.disabled, false);
    h.select(null);
    assert.equal(h.elements.predictBtn.disabled, false);
    assert.equal(h.revokedUrls.length, 0);
});

test("replacement releases old URLs and ignores stale preview decoder callbacks", () => {
    const h = harness();
    h.select(file({ name: "first.png" }));
    const lateLoad = h.loaders[0].onload, lateError = h.loaders[0].onerror;
    h.select(file({ name: "second.png" }));
    lateLoad();
    lateError();
    assert.equal(h.elements.imagePreview.src, undefined);
    assert.equal(h.elements.previewMessage.textContent, "Loading preview...");
    h.loaders[1].onload();
    assert.equal(h.elements.imagePreview.src, h.createdUrls[1]);
    assert.deepEqual(h.revokedUrls, [h.createdUrls[0]]);
    h.pagehide();
    assert.deepEqual(h.revokedUrls, h.createdUrls);
    assert.equal(h.elements.imagePreview.src, undefined);
});

test("clear cancels pending request, revokes preview, resets state and restores keyboard focus", async () => {
    const h = harness();
    h.select(file());
    const pending = h.click();
    h.clear();
    assert.equal(h.requests[0].options.signal.aborted, true);
    assert.equal(h.elements.predictBtn.disabled, true);
    assert.equal(h.elements.resetBtn.disabled, true);
    assert.equal(hidden(h.elements.previewContainer), true);
    assert.equal(h.focused(), h.elements.imageInput);
    assert.equal(h.elements.fileLabel.classList.contains("keyboard-focus"), true);
    h.respond(0);
    await pending;
    assert.equal(hidden(h.elements.resultContainer), true);
    assert.equal(h.elements.predictBtn.disabled, true);
    assert.equal(h.timers.size, 0);
});

test("reselection prevents stale response and stale finally from affecting current request", async () => {
    const h = harness();
    h.select(file({ name: "first.png" }));
    const first = h.click();
    h.select(file({ name: "second.png" }));
    const second = h.click();
    assert.equal(h.requests[0].options.signal.aborted, true);
    h.respond(0);
    await first;
    assert.equal(hidden(h.elements.resultContainer), true);
    assert.equal(h.elements.predictBtn.disabled, true);
    assert.equal(hidden(h.elements.loadingIndicator), false);
    h.respond(1);
    await second;
    assert.equal(hidden(h.elements.resultContainer), false);
    assert.equal(h.elements.predictBtn.disabled, false);
});

test("late network rejection cannot overwrite a newer result", async () => {
    const h = harness();
    h.select(file());
    const first = h.click();
    h.select(file({ name: "second.png" }));
    const second = h.click();
    h.respond(1);
    await second;
    h.requests[0].reject(new Error("old connection failed"));
    await first;
    assert.equal(hidden(h.elements.errorContainer), true);
    assert.equal(hidden(h.elements.resultContainer), false);
});

test("reselection during JSON parsing ignores the old response", async () => {
    const h = harness(), body = deferred();
    h.select(file());
    const pending = h.click();
    h.requests[0].resolve({ ok: true, status: 200, json: () => body.promise });
    await Promise.resolve();
    h.select(file({ name: "second.png" }));
    body.resolve(prediction());
    await pending;
    assert.equal(hidden(h.elements.resultContainer), true);
    assert.equal(h.elements.predictBtn.disabled, false);
});

test("double click sends only one request", async () => {
    const h = harness();
    h.select(file());
    const first = h.click();
    await h.click();
    assert.equal(h.requests.length, 1);
    h.respond(0);
    await first;
});

test("timeout recovers controls and blocks late completion", async () => {
    const h = harness();
    h.select(file());
    const pending = h.click();
    h.expire();
    assert.equal(h.requests[0].options.signal.aborted, true);
    assert.equal(h.elements.predictBtn.disabled, false);
    assert.match(h.elements.errorMessage.textContent, /timed out/);
    h.respond(0);
    await pending;
    assert.equal(hidden(h.elements.resultContainer), true);
    assert.match(h.elements.errorMessage.textContent, /timed out/);
});

test("drop selects one file without writing FileList and reports multi-file drop", () => {
    const h = harness();
    h.elements.fileLabel.emit("dragover");
    assert.equal(h.elements.fileLabel.classList.contains("drag-over"), true);
    h.drop([file()]);
    assert.equal(h.elements.predictBtn.disabled, false);
    assert.equal(h.elements.fileLabel.classList.contains("drag-over"), false);
    h.drop([file(), file()]);
    assert.match(h.elements.errorMessage.textContent, /one image at a time/);
    assert.equal(h.createdUrls.length, 1);
});

for (const status of [400, 413, 500, 503, 404]) {
    test(`HTTP ${status} produces friendly error and restores controls`, async () => {
        const h = harness();
        h.select(file());
        const pending = h.click();
        h.respond(0, { error: "Traceback: C:/private/file.py" }, status);
        await pending;
        assert.equal(hidden(h.elements.errorContainer), false);
        assert.doesNotMatch(h.elements.errorMessage.textContent, /Traceback|private/);
        assert.equal(h.elements.predictBtn.disabled, false);
        assert.equal(hidden(h.elements.loadingIndicator), true);
    });
}

test("non-JSON success has a distinct response error, while network failure has a local-app error", async () => {
    const h = harness();
    h.select(file());
    let pending = h.click();
    h.requests[0].resolve({ ok: true, status: 200, json: async () => { throw new SyntaxError("HTML"); } });
    await pending;
    assert.match(h.elements.errorMessage.textContent, /unreadable prediction response/);
    pending = h.click();
    h.requests[1].reject(new TypeError("Failed to fetch"));
    await pending;
    assert.match(h.elements.errorMessage.textContent, /Cannot reach the local app/);
});

const malformed = [
    ["null body", () => null],
    ["missing prediction", () => ({ top5: [] })],
    ["empty top five", () => ({ ...prediction(), top5: [] })],
    ["null row", () => { const data = prediction(); data.top5[2] = null; return data; }],
    ["numeric string confidence", () => { const data = prediction(); data.top5[2].confidence = "0.1"; return data; }],
    ["nonfinite confidence", () => { const data = prediction(); data.top5[2].confidence = NaN; return data; }],
    ["out-of-range confidence", () => { const data = prediction(); data.top5[2].confidence = 1.1; return data; }],
    ["duplicate index", () => { const data = prediction(); data.top5[2].index = 1; return data; }],
    ["unsorted top five", () => { const data = prediction(); data.top5[2].confidence = 0.4; return data; }],
    ["empty breed", () => { const data = prediction(); data.top5[2].breed = " "; return data; }],
    ["out-of-range index", () => { const data = prediction(); data.top5[2].index = 120; return data; }],
    ["mismatched top one", () => { const data = prediction(); data.prediction.index = 10; return data; }],
    ["numeric status", () => ({ ...prediction(), status: 123 })],
    ["object final status", () => ({ ...prediction(), final_status: {} })],
    ["unknown status", () => ({ ...prediction(), status: "error" })],
    ["DOG without breed or confidence", () => ({ status: "DOG", top5: [] })],
    ["UNCERTAIN without breed or confidence", () => ({ status: "UNCERTAIN", top5: [] })],
    ["DOG with a string prediction", () => ({ status: "DOG", prediction: "not a prediction", top5: [] })],
    ["DOG with empty top five", () => ({ ...threeStatePrediction(), top5: [] })],
    ["DOG with mismatched top breed", () => ({ ...threeStatePrediction(), breed: "wrong_breed" })],
    ["UNCERTAIN with mismatched top confidence", () => ({ ...threeStatePrediction("UNCERTAIN"), confidence: 0.9 })],
    ["DOG with more than five rows", () => {
        const data = threeStatePrediction();
        data.top5.push({ index: 5, breed: "breed_5", confidence: 0.01 });
        return data;
    }],
    ["UNCERTAIN with invalid OOD score", () => ({ ...threeStatePrediction("UNCERTAIN"), ood_score: "broken" })],
    ["NOT_DOG with malformed hidden rows", () => ({ status: "NOT_DOG", breed: null, top5: [null] })]
];
for (const [name, makeData] of malformed) {
    test(`malformed API: ${name} never renders misleading results`, async () => {
        const h = harness();
        h.select(file());
        const pending = h.click();
        h.respond(0, makeData());
        await pending;
        assert.match(h.elements.errorMessage.textContent, /invalid prediction response/);
        assert.equal(hidden(h.elements.resultContainer), true);
        assert.equal(h.elements.top5TableBody.children.length, 0);
        assert.equal(h.elements.predictBtn.disabled, false);
    });
}

test("HTML-like breed is only text and cannot create injected nodes", async () => {
    const h = harness(), data = prediction();
    data.prediction.breed = data.top5[0].breed = '<img src=x onerror=alert(1)>';
    h.select(file());
    const pending = h.click();
    h.respond(0, data);
    await pending;
    assert.equal(h.elements.top1Breed.children.length, 0);
    assert.match(h.elements.top1Breed.textContent, /<img/i);
    assert.equal(h.elements.top5TableBody.children[0].children[1].children.length, 0);
});

test("missing DOM element fails initialization safely", () => {
    const h = harness("previewMessage");
    assert.equal(h.errors.length, 1);
    assert.equal(h.elements.predictBtn.listeners.click, undefined);
});

test("OE V2 DOG response contract renders breed, confidence, and Top-5", async () => {
    const h = harness();
    h.select(file());
    const pending = h.click();
    h.respond(0, {
        status: "DOG",
        final_status: "DOG",
        breed: "golden_retriever",
        confidence: 0.95,
        top5: [
            { rank: 1, breed: "golden_retriever", confidence: 0.95, index: 45 },
            { rank: 2, breed: "labrador_retriever", confidence: 0.03, index: 56 }
        ],
        probability_margin: 0.92,
        ood_score: 0.20,
        T_uncertain: 0.343358,
        T_not_dog: 0.457513
    });
    await pending;
    assert.equal(hidden(h.elements.resultContainer), false);
    assert.equal(hidden(h.elements.top1Container), false);
    assert.equal(hidden(h.elements.top5Container), false);
    assert.equal(h.elements.top1Breed.textContent, "Golden Retriever");
    assert.equal(h.elements.top1Confidence.textContent, "95.00%");
    assert.equal(h.elements.top5TableBody.children.length, 2);
    assert.match(h.elements.resultNote.textContent, /Confidence is a model score/);
});

test("OE V2 UNCERTAIN response contract renders likely breed, confidence, Top-5, and OOD warning", async () => {
    const h = harness();
    h.select(file());
    const pending = h.click();
    h.respond(0, {
        status: "UNCERTAIN",
        final_status: "UNCERTAIN",
        breed: "eskimo_dog",
        confidence: 0.398673,
        top5: [
            { rank: 1, breed: "eskimo_dog", confidence: 0.398673, index: 43 },
            { rank: 2, breed: "siberian_husky", confidence: 0.210076, index: 101 }
        ],
        probability_margin: 0.188597,
        ood_score: 0.416950,
        T_uncertain: 0.343358,
        T_not_dog: 0.457513,
        message: "The image is ambiguous or outside the reliable dog-breed region."
    });
    await pending;
    assert.equal(hidden(h.elements.resultContainer), false);
    assert.equal(hidden(h.elements.top1Container), false);
    assert.equal(hidden(h.elements.top5Container), false);
    assert.equal(h.elements.top1Breed.textContent, "Eskimo Dog");
    assert.equal(h.elements.top1Confidence.textContent, "39.87%");
    assert.equal(h.elements.top5TableBody.children.length, 2);
    assert.match(h.elements.resultNote.textContent, /⚠️ Outlier Warning/);
    assert.match(h.elements.resultNote.textContent, /0\.4169|0\.4170/);
    assert.equal(h.elements.resultNote.classList.contains("decision-note-uncertain"), true);
});

test("OE V2 NOT_DOG response contract suppresses breed and Top-5 while retaining message", async () => {
    const h = harness();
    h.select(file());
    const pending = h.click();
    h.respond(0, {
        status: "NOT_DOG",
        final_status: "NOT_DOG",
        breed: "tabby_cat",
        confidence: 0.15,
        top5: [
            { rank: 1, breed: "tabby_cat", confidence: 0.15, index: 10 }
        ],
        probability_margin: 0.05,
        ood_score: 0.58,
        T_uncertain: 0.343358,
        T_not_dog: 0.457513,
        message: "No dog was confidently detected."
    });
    await pending;
    assert.equal(hidden(h.elements.resultContainer), false);
    assert.equal(hidden(h.elements.top1Container), true);
    assert.equal(hidden(h.elements.top5Container), true);
    assert.equal(h.elements.resultNote.textContent, "No dog was confidently detected.");
    assert.equal(h.elements.resultNote.classList.contains("decision-note-not-dog"), true);
});

test("NOT_DOG may omit breed predictions without displaying an empty result", async () => {
    const h = harness();
    h.select(file());
    const pending = h.click();
    h.respond(0, { status: "success", decision: "NOT_DOG", breed: null, top5: [] });
    await pending;
    assert.equal(hidden(h.elements.errorContainer), true);
    assert.equal(hidden(h.elements.resultContainer), false);
    assert.equal(hidden(h.elements.top1Container), true);
    assert.equal(hidden(h.elements.top5Container), true);
    assert.equal(h.elements.resultNote.textContent, "No dog was confidently detected.");
});


test("RAG receives exact inference score and clears context when image is reset", async () => {
    const h = harness();
    h.select(file());
    const pending = h.click();
    const data = {...threeStatePrediction("UNCERTAIN"), ood_score: 0.387654321};
    h.respond(0, data);
    await pending;
    assert.equal(h.predictionEvents.at(-1).ood_score, data.ood_score);
    assert.equal(h.predictionEvents.at(-1).status, "UNCERTAIN");
    h.clear();
    assert.equal(h.predictionEvents.at(-1), null);
});
