/**
 * DocQuery AI — Frontend Logic
 * Handles PDF upload, chat messaging, and source citation display.
 */

// ─── State ──────────────────────────────────────────────────────────────────
let currentSessionId = null;
let isLoading = false;

// ─── DOM References ─────────────────────────────────────────────────────────
const uploadSection = document.getElementById("upload-section");
const chatSection = document.getElementById("chat-section");
const apiKeySection = document.getElementById("api-key-section");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const processingSection = document.getElementById("processing-section");
const processingText = document.getElementById("processing-text");
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const btnSend = document.getElementById("btn-send");
const btnNewSession = document.getElementById("btn-new-session");
const sessionBadge = document.getElementById("session-badge");
const sessionFilename = document.getElementById("session-filename");
const docName = document.getElementById("doc-name");
const docPages = document.getElementById("doc-pages");
const docChunks = document.getElementById("doc-chunks");
const toast = document.getElementById("toast");

// ─── Startup: check if API key is configured ─────────────────────────────────
(async function checkApiKey() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        if (!data.groq_api_configured) {
            // Hide upload, show API key setup screen
            uploadSection.classList.add("hidden");
            apiKeySection.classList.remove("hidden");
        }
    } catch (e) {
        console.warn("Health check failed:", e);
    }
})();

// ─── Save API Key ─────────────────────────────────────────────────────────────
async function saveApiKey() {
    const input = document.getElementById("api-key-input");
    const btn = document.getElementById("btn-save-key");
    const key = input.value.trim();

    if (!key) { showToast("Please paste your Groq API key.", "error"); return; }
    if (!key.startsWith("gsk_")) { showToast("Invalid key — Groq keys start with 'gsk_'.", "error"); return; }

    btn.textContent = "Saving…";
    btn.disabled = true;

    try {
        const res = await fetch("/api/set-key", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: key }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to save key.");

        showToast("API key saved! You can now upload documents.", "success");
        apiKeySection.classList.add("hidden");
        uploadSection.classList.remove("hidden");
    } catch (err) {
        showToast(err.message, "error");
        btn.textContent = "Save API Key & Continue";
        btn.disabled = false;
    }
}



// ─── Event Listeners ────────────────────────────────────────────────────────

// Drop zone — click
dropZone.addEventListener("click", () => fileInput.click());

// Drop zone — drag events
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileUpload(files[0]);
});

// File input change
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
});

// Chat input — auto-resize + enter to send
chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
    btnSend.disabled = !chatInput.value.trim() || isLoading;
});

chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (chatInput.value.trim() && !isLoading) sendMessage();
    }
});

// Send button
btnSend.addEventListener("click", () => {
    if (chatInput.value.trim() && !isLoading) sendMessage();
});

// New session button
btnNewSession.addEventListener("click", resetToUpload);


// ─── File Upload ────────────────────────────────────────────────────────────

let _pollInterval = null;   // setInterval handle for status polling
let _currentJobId = null;   // active background job ID

async function handleFileUpload(file) {
    // Validate
    if (!file.name.toLowerCase().endsWith(".pdf")) {
        showToast("Please upload a PDF file.", "error");
        return;
    }
    if (file.size === 0) {
        showToast("File is empty. Please upload a valid PDF.", "error");
        return;
    }
    if (file.size > 20 * 1024 * 1024) {
        showToast("File too large. Maximum size is 20MB.", "error");
        return;
    }

    // Show processing UI
    dropZone.classList.add("hidden");
    processingSection.classList.remove("hidden");

    // Animate processing steps while waiting
    const steps = ["step-extract", "step-chunk", "step-embed", "step-index"];
    const stepMessages = [
        "Extracting text from PDF...",
        "Chunking into semantic segments...",
        "Generating vector embeddings...",
        "Building FAISS search index...",
    ];
    let currentStep = 0;
    const stepInterval = setInterval(() => {
        if (currentStep < steps.length) {
            if (currentStep > 0) {
                document.getElementById(steps[currentStep - 1]).classList.remove("active");
                document.getElementById(steps[currentStep - 1]).classList.add("done");
                document.getElementById(steps[currentStep - 1]).querySelector(".step-icon").textContent = "✓";
            }
            document.getElementById(steps[currentStep]).classList.add("active");
            document.getElementById(steps[currentStep]).querySelector(".step-icon").textContent = "●";
            processingText.textContent = stepMessages[currentStep];
            currentStep++;
        }
    }, 800);

    try {
        const formData = new FormData();
        formData.append("file", file);

        // POST upload — returns immediately with job_id
        const response = await fetch("/api/upload", { method: "POST", body: formData });
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: "Upload failed" }));
            throw new Error(error.detail || `Upload failed with status ${response.status}`);
        }
        const { job_id, filename } = await response.json();
        _currentJobId = job_id;

        // Enable cancel button
        const btnCancel = document.getElementById("btn-cancel-upload");
        if (btnCancel) btnCancel.disabled = false;

        // Poll /api/status/{job_id} every 2 seconds
        _pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/status/${job_id}`);
                if (!res.ok) return;
                const data = await res.json();

                if (data.status === "complete") {
                    clearInterval(_pollInterval);
                    clearInterval(stepInterval);
                    _pollInterval = null;

                    // Mark all steps done
                    steps.forEach((id) => {
                        const el = document.getElementById(id);
                        el.classList.remove("active");
                        el.classList.add("done");
                        el.querySelector(".step-icon").textContent = "✓";
                    });
                    processingText.textContent = "Document ready!";

                    setTimeout(() => {
                        showChatInterface({
                            session_id: data.job_id,
                            filename: data.filename,
                            total_pages: data.total_pages,
                            total_chunks: data.total_chunks,
                        });
                        currentSessionId = data.job_id;
                        showToast(`"${data.filename}" indexed successfully!`, "success");
                    }, 600);

                } else if (data.status === "error") {
                    clearInterval(_pollInterval);
                    clearInterval(stepInterval);
                    _pollInterval = null;
                    throw new Error(data.error || "Processing failed on server.");

                } else if (data.status === "cancelled") {
                    clearInterval(_pollInterval);
                    clearInterval(stepInterval);
                    _pollInterval = null;
                    _resetUploadUI();
                    showToast("Upload cancelled.", "");
                }
            } catch (pollErr) {
                if (pollErr.message !== "polling") {
                    clearInterval(_pollInterval);
                    clearInterval(stepInterval);
                    _pollInterval = null;
                    showToast(pollErr.message || "Error while checking status.", "error");
                    _resetUploadUI();
                }
            }
        }, 2000);

    } catch (error) {
        clearInterval(stepInterval);
        if (_pollInterval) { clearInterval(_pollInterval); _pollInterval = null; }
        console.error("Upload error:", error);
        showToast(error.message || "Failed to process document.", "error");
        _resetUploadUI();
    }
}

async function cancelUpload() {
    const btnCancel = document.getElementById("btn-cancel-upload");
    if (btnCancel) btnCancel.disabled = true;

    if (_pollInterval) { clearInterval(_pollInterval); _pollInterval = null; }

    if (_currentJobId) {
        try {
            await fetch(`/api/cancel/${_currentJobId}`, { method: "DELETE" });
        } catch (e) {
            console.error("Cancel request failed:", e);
        }
        _currentJobId = null;
    }

    _resetUploadUI();
    showToast("Upload cancelled. You can upload a different PDF.", "");
}

function _resetUploadUI() {
    processingSection.classList.add("hidden");
    dropZone.classList.remove("hidden");
    resetProcessingSteps();
    const btnCancel = document.getElementById("btn-cancel-upload");
    if (btnCancel) btnCancel.disabled = false;
    fileInput.value = "";
}


// ─── Chat Interface ─────────────────────────────────────────────────────────

function showChatInterface(data) {
    // Update header
    sessionFilename.textContent = data.filename;
    sessionBadge.classList.remove("hidden");
    btnNewSession.classList.remove("hidden");

    // Update doc info bar
    docName.textContent = data.filename;
    docPages.textContent = `${data.total_pages} pages`;
    docChunks.textContent = `${data.total_chunks} chunks`;

    // Switch views
    uploadSection.classList.add("hidden");
    chatSection.classList.remove("hidden");

    // Focus input
    chatInput.focus();
}

async function sendMessage() {
    const question = chatInput.value.trim();
    if (!question || isLoading) return;

    // Basic validation
    if (question.length > 2000) {
        showToast("Question is too long. Maximum 2000 characters.", "error");
        return;
    }

    isLoading = true;
    btnSend.disabled = true;

    // Clear welcome message on first question
    const welcome = chatMessages.querySelector(".welcome-message");
    if (welcome) welcome.remove();

    // Add user message
    addMessage("user", question);

    // Clear input
    chatInput.value = "";
    chatInput.style.height = "auto";

    // Add typing indicator
    const typingEl = addTypingIndicator();

    try {
        const response = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: currentSessionId,
                question: question,
            }),
        });

        // Remove typing indicator
        typingEl.remove();

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: "Query failed" }));
            throw new Error(error.detail || `Query failed with status ${response.status}`);
        }

        const data = await response.json();
        addMessage("assistant", data.answer, data.sources);

    } catch (error) {
        typingEl.remove();
        console.error("Query error:", error);
        addMessage("assistant", "Sorry, I encountered an error processing your question. Please try again.");
        showToast(error.message || "Failed to get answer.", "error");
    } finally {
        isLoading = false;
        btnSend.disabled = !chatInput.value.trim();
    }
}


// ─── Message Rendering ──────────────────────────────────────────────────────

function addMessage(role, content, sources = []) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "assistant" ? "AI" : "You";

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = content;

    contentDiv.appendChild(bubble);

    // Add sources for assistant messages
    if (role === "assistant" && sources.length > 0) {
        const sourcesSection = document.createElement("div");
        sourcesSection.className = "sources-section";

        const toggleBtn = document.createElement("button");
        toggleBtn.className = "sources-toggle";
        toggleBtn.innerHTML = `📄 ${sources.length} source${sources.length > 1 ? "s" : ""} found`;

        const sourcesList = document.createElement("div");
        sourcesList.className = "sources-list hidden";

        sources.forEach((src) => {
            const item = document.createElement("div");
            item.className = "source-item";
            item.innerHTML = `
                <div class="source-page">📑 Page ${src.page_number}</div>
                <div>${escapeHtml(src.content_preview)}</div>
            `;
            sourcesList.appendChild(item);
        });

        toggleBtn.addEventListener("click", () => {
            sourcesList.classList.toggle("hidden");
            toggleBtn.innerHTML = sourcesList.classList.contains("hidden")
                ? `📄 ${sources.length} source${sources.length > 1 ? "s" : ""} found`
                : `📄 Hide sources`;
        });

        sourcesSection.appendChild(toggleBtn);
        sourcesSection.appendChild(sourcesList);
        contentDiv.appendChild(sourcesSection);
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(contentDiv);
    chatMessages.appendChild(msgDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addTypingIndicator() {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant";
    msgDiv.id = "typing-indicator";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "AI";

    const typing = document.createElement("div");
    typing.className = "typing-indicator";
    typing.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(typing);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return msgDiv;
}


// ─── Utilities ──────────────────────────────────────────────────────────────

function askSuggestion(question) {
    chatInput.value = question;
    chatInput.dispatchEvent(new Event("input"));
    sendMessage();
}

function resetToUpload() {
    // Clear session on server
    if (currentSessionId) {
        fetch(`/api/session/${currentSessionId}`, { method: "DELETE" })
            .catch((err) => console.error("Failed to clear session:", err));
    }

    currentSessionId = null;
    isLoading = false;

    // Reset UI
    chatSection.classList.add("hidden");
    uploadSection.classList.remove("hidden");
    sessionBadge.classList.add("hidden");
    btnNewSession.classList.add("hidden");

    processingSection.classList.add("hidden");
    dropZone.classList.remove("hidden");
    resetProcessingSteps();

    // Clear chat messages
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            </div>
            <h3>Document Ready!</h3>
            <p>Your document has been indexed. Ask any question about its content and I'll provide answers with source citations.</p>
            <div class="suggestion-chips">
                <button class="chip" onclick="askSuggestion('What is the main topic of this document?')">📄 Main topic?</button>
                <button class="chip" onclick="askSuggestion('Summarize the key points of this document')">📝 Key points</button>
                <button class="chip" onclick="askSuggestion('What are the conclusions or findings?')">🎯 Conclusions</button>
            </div>
        </div>
    `;

    fileInput.value = "";
}

function resetProcessingSteps() {
    ["step-extract", "step-chunk", "step-embed", "step-index"].forEach((id) => {
        const el = document.getElementById(id);
        el.classList.remove("active", "done");
        el.querySelector(".step-icon").textContent = "◯";
    });
    processingText.textContent = "Processing document...";
}

function showToast(message, type = "") {
    toast.textContent = message;
    toast.className = `toast ${type}`;

    // Trigger reflow for animation
    void toast.offsetWidth;
    toast.classList.add("show");

    setTimeout(() => {
        toast.classList.remove("show");
    }, 3500);
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
