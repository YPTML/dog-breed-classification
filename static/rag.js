/**
 * Dog Breed Recognition: RAG Question Answering Assistant Interface.
 * Handles user chat, suggested chips, source citation inspection, and active prediction context.
 */
document.addEventListener("DOMContentLoaded", () => {
    const ragSection = document.getElementById("ragSection");
    if (!ragSection) return;

    const ragInput = document.getElementById("ragInput");
    const ragSendBtn = document.getElementById("ragSendBtn");
    const ragClearChatBtn = document.getElementById("ragClearChatBtn");
    const ragChatHistory = document.getElementById("ragChatHistory");
    const ragLoading = document.getElementById("ragLoading");
    const ragErrorBox = document.getElementById("ragErrorBox");
    const ragErrorText = document.getElementById("ragErrorText");
    const ragRetryBtn = document.getElementById("ragRetryBtn");
    const ragPredictionContext = document.getElementById("ragPredictionContext");
    const ragContextBreed = document.getElementById("ragContextBreed");
    const ragContextConf = document.getElementById("ragContextConf");
    const ragContextStatus = document.getElementById("ragContextStatus");
    const ragClearContextBtn = document.getElementById("ragClearContextBtn");
    const ragChips = document.querySelectorAll(".rag-chip");
    const statusBadge = document.getElementById("ragStatusBadge");
    const coverageLabel = document.getElementById("ragCoverage");
    async function fetchWithTimeout(url, options = {}, timeout = 10000) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeout);
        try {
            return await fetch(url, { ...options, signal: controller.signal });
        } catch (error) {
            if (error.name === 'AbortError') throw new Error('Hết thời gian chờ. Bạn có thể thử lại; lần nạp model đầu tiên có thể lâu hơn.');
            throw error;
        } finally {
            clearTimeout(timer);
        }
    }
    async function refreshKnowledgeStatus() {
        try {
            const response = await fetchWithTimeout('/api/rag/status');
            if (!response.ok) throw new Error('Không đọc được trạng thái');
            const data = await response.json();
            statusBadge.textContent = data.status === 'index_missing' ? 'Chưa có chỉ mục' :
                data.llm_status === 'live' ? (data.llm_provider === 'ollama' ? 'Ollama cục bộ sẵn sàng' : 'Đã cấu hình LLM') : 'Chế độ trích đoạn · chưa có LLM';
            statusBadge.title = data.missing_config_message || data.knowledge_version;
            coverageLabel.textContent = `${data.files_count} tệp · ${data.chunks_count} đoạn · ${data.paper_files || 0} PDF · ${data.index_issues || 0} vấn đề cần kiểm tra.`;
        } catch (error) {
            statusBadge.textContent = 'Chưa kiểm tra được kho tài liệu';
            coverageLabel.textContent = error.message;
        }
    }
    refreshKnowledgeStatus();

    let conversationHistory = [];
    let lastQueryText = "";
    let activePredictionContext = null;
    let isQuerying = false;

    // Keep the exact API values; DOM labels round confidence and omit OOD scores.
    document.addEventListener("prediction:result", event => {
        const data = event.detail;
        if (!data) {
            activePredictionContext = null;
            ragPredictionContext.classList.add("hidden");
            return;
        }
        activePredictionContext = {
            status: data.status,
            breed: data.breed || data.top1_breed || data.prediction?.breed,
            confidence: data.confidence ?? data.prediction?.confidence,
            ood_score: data.ood_score,
            top5: data.top5,
        };
        ragContextBreed.textContent = activePredictionContext.breed || "(Không có giống dự đoán)";
        ragContextConf.textContent = Number.isFinite(activePredictionContext.confidence)
            ? `${(activePredictionContext.confidence * 100).toFixed(2)}%` : "OOD";
        ragContextStatus.textContent = data.status;
        ragPredictionContext.classList.remove("hidden");
    });

    if (ragClearContextBtn) {
        ragClearContextBtn.addEventListener("click", () => {
            activePredictionContext = null;
            ragPredictionContext.classList.add("hidden");
        });
    }

    // 2. Safe text escaping & lightweight markdown formatting
    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    function formatMarkdown(text) {
        let escaped = escapeHtml(text);

        // Bold: **text**
        escaped = escaped.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

        // Code: `code`
        escaped = escaped.replace(/`([^`]+)`/g, "<code>$1</code>");

        // Source citations: [path#lines]
        escaped = escaped.replace(/\[([^\]]+)\]/g, (match, p1) => {
            return `<span class="source-tag" title="Nguồn đối chiếu">${p1}</span>`;
        });

        // Lists: - item
        const lines = escaped.split("\n");
        const formattedLines = [];
        let inList = false;

        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith("- ")) {
                if (!inList) {
                    formattedLines.push("<ul>");
                    inList = true;
                }
                formattedLines.push(`<li>${trimmed.substring(2)}</li>`);
            } else {
                if (inList) {
                    formattedLines.push("</ul>");
                    inList = false;
                }
                if (trimmed) {
                    formattedLines.push(`<p>${line}</p>`);
                }
            }
        }
        if (inList) formattedLines.push("</ul>");

        return formattedLines.join("");
    }

    // 3. Render messages in chat history
    function appendMessage(role, text, sources = [], llmStatus = null, missingConfig = null) {
        const msgDiv = document.createElement("div");
        msgDiv.className = role === "user" ? "rag-message rag-message-user" : "rag-message rag-message-assistant";

        const headerDiv = document.createElement("div");
        headerDiv.className = "rag-message-header";

        const senderSpan = document.createElement("span");
        senderSpan.className = "sender-name";
        senderSpan.textContent = role === "user" ? "Bạn" : "Trợ lý mô hình";
        headerDiv.appendChild(senderSpan);

        if (role === "assistant" && llmStatus) {
            const badge = document.createElement("span");
            badge.className = llmStatus === "live" ? "rag-badge badge-live" : "rag-badge badge-offline";
            badge.textContent = llmStatus === "live" ? "LLM tổng hợp" :
                llmStatus === 'error' ? 'Lỗi LLM' : llmStatus === 'no_evidence' ? 'Thiếu bằng chứng' : 'Trích đoạn / dữ liệu xác định';
            badge.title = missingConfig || "Phản hồi bám sát tài liệu";
            headerDiv.appendChild(badge);
        }

        const bodyDiv = document.createElement("div");
        bodyDiv.className = "rag-message-body";
        bodyDiv.innerHTML = formatMarkdown(text);

        msgDiv.appendChild(headerDiv);
        msgDiv.appendChild(bodyDiv);

        // Sources Drawer
        if (sources && sources.length > 0) {
            const sourcesContainer = document.createElement("div");
            sourcesContainer.className = "rag-sources-container";

            const sourcesTitle = document.createElement("div");
            sourcesTitle.className = "sources-title";
            sourcesTitle.textContent = "Nguồn đối chiếu:";
            sourcesContainer.appendChild(sourcesTitle);

            const sourcesList = document.createElement("div");
            sourcesList.className = "sources-list";

            sources.forEach((src) => {
                const pill = document.createElement("button");
                pill.type = "button";
                pill.className = "source-pill";
                const lineInfo = src.page ? ` · trang ${src.page}` : src.row ? ` · dòng dữ liệu ${src.row}` :
                    src.json_pointer ? ` · ${src.json_pointer}` : src.locator ? `#${src.locator}` :
                    src.start_line ? `#L${src.start_line}${src.end_line && src.end_line > src.start_line ? `-L` + src.end_line : ""}` : "";
                pill.textContent = `${src.source_path}${lineInfo}`;
                pill.title = `${src.title || "Tài liệu đối chiếu"} (Độ khớp: ${src.relevance_score || "1.0"})`;

                const excerpt = document.createElement('pre');
                excerpt.className = 'rag-source-excerpt';
                excerpt.hidden = true;
                pill.addEventListener("click", async () => {
                    excerpt.hidden = !excerpt.hidden;
                    if (excerpt.hidden || excerpt.dataset.loaded) return;
                    if (!src.chunk_id) {
                        excerpt.textContent = `${src.title || 'Nguồn dữ liệu'}\n${src.source_path}\nNguồn tính toán có cấu trúc; xem tệp gốc trong dự án.`;
                        return;
                    }
                    excerpt.textContent = 'Đang đọc trích đoạn...';
                    try {
                        const response = await fetchWithTimeout(`/api/rag/source?chunk_id=${encodeURIComponent(src.chunk_id)}`);
                        if (!response.ok) throw new Error('Trích đoạn không còn trong chỉ mục.');
                        const data = await response.json();
                        excerpt.textContent = data.content;
                        excerpt.dataset.loaded = 'true';
                    } catch (error) { excerpt.textContent = error.message; }
                });

                sourcesList.appendChild(pill);
                sourcesList.appendChild(excerpt);
            });

            sourcesContainer.appendChild(sourcesList);
            msgDiv.appendChild(sourcesContainer);
        }

        ragChatHistory.appendChild(msgDiv);
        ragChatHistory.scrollTop = ragChatHistory.scrollHeight;
    }

    function setLoading(loading) {
        isQuerying = loading;
        ragLoading.classList.toggle("hidden", !loading);
        ragSendBtn.disabled = loading;
        ragInput.disabled = loading;
        ragClearChatBtn.disabled = loading;
        if (loading) {
            ragErrorBox.classList.add("hidden");
        }
    }

    // 4. Send Query to /api/rag/query
    async function sendQuery(queryText) {
        if (!queryText || !queryText.trim() || isQuerying) return;
        const cleanQuery = queryText.trim();
        lastQueryText = cleanQuery;

        // Append user question
        appendMessage("user", cleanQuery);
        conversationHistory.push({ role: "user", content: cleanQuery });

        ragInput.value = "";
        setLoading(true);

        const payload = {
            query: cleanQuery,
            history: conversationHistory.slice(-6),
            last_prediction: activePredictionContext,
        };

        try {
            const response = await fetchWithTimeout("/api/rag/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            }, 240000);

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Lỗi máy chủ (HTTP ${response.status})`);
            }

            const data = await response.json();
            const answer = data.answer || "Không nhận được phản hồi.";
            const sources = data.sources || [];
            const llmStatus = data.llm_status;
            const missingConfig = data.missing_config_message;

            appendMessage("assistant", answer, sources, llmStatus, missingConfig);
            if (llmStatus === 'error') {
                ragErrorText.textContent = answer;
                ragErrorBox.classList.remove('hidden');
            } else {
                conversationHistory.push({ role: "assistant", content: answer });
            }
        } catch (error) {
            ragErrorText.textContent = error.message || "Có lỗi xảy ra khi truy vấn kho kiến thức.";
            ragErrorBox.classList.remove("hidden");
        } finally {
            setLoading(false);
        }
    }

    // 5. Event Listeners
    ragSendBtn.addEventListener("click", () => {
        sendQuery(ragInput.value);
    });

    ragInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendQuery(ragInput.value);
        }
    });

    ragRetryBtn.addEventListener("click", () => {
        ragErrorBox.classList.add("hidden");
        sendQuery(lastQueryText);
    });

    ragChips.forEach((chip) => {
        chip.addEventListener("click", () => {
            const q = chip.getAttribute("data-question");
            if (q) {
                ragInput.value = q;
                sendQuery(q);
            }
        });
    });

    ragClearChatBtn.addEventListener("click", () => {
        conversationHistory = [];
        ragChatHistory.replaceChildren();
        appendMessage("assistant", "Đã xóa lịch sử trò chuyện. Bạn có thể đặt câu hỏi mới!");
    });
});
