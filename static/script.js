
document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("verifyForm");
    const submitBtn = document.getElementById("submitBtn");
    const spinner = document.querySelector(".spinner");
    const btnText = document.querySelector(".btn-text");

    // Input Groups
    const questionGroup = document.getElementById("questionGroup");
    const answerGroup = document.getElementById("answerGroup");
    const modeInput = document.getElementById("modeInput");
    const questionInput = document.getElementById("question");
    const answerInput = document.getElementById("ai_answer");

    // Progress Elements
    const loadingStatus = document.getElementById("loadingStatus");
    const loadingText = document.getElementById("loadingText");
    // const progressBar = document.getElementById("progressBar"); // Not used currently
    // const percentText = document.getElementById("percentText"); // Not used currently

    const resultSection = document.getElementById("resultSection");
    const verdictBanner = document.getElementById("verdictBanner");

    // Stats Elements
    const totalClaimsEl = document.getElementById("totalClaims");
    const verifiedCountEl = document.getElementById("verifiedCount");
    const hallucinatedCountEl = document.getElementById("hallucinatedCount");
    const missingCountEl = document.getElementById("missingCount");

    const correctedContainer = document.getElementById("correctedContainer");
    const finalAnswerText = document.getElementById("finalAnswerText");
    const claimsList = document.getElementById("claimsList");

    // Analysis Elements
    const analysisSection = document.getElementById("analysisSection");
    const beforeAcc = document.getElementById("beforeAcc");
    const afterAcc = document.getElementById("afterAcc");
    const impAcc = document.getElementById("impAcc");
    const beforeHall = document.getElementById("beforeHall");
    const afterHall = document.getElementById("afterHall");
    const impHall = document.getElementById("impHall");
    const analysisReportText = document.getElementById("analysisReportText");

    // Expose switchTab globally
    window.switchTab = function (mode) {
        modeInput.value = mode;

        // Clear inputs on mode switch
        questionInput.value = "";
        answerInput.value = "";

        // Update Tabs UI
        document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
        const activeBtn = document.querySelector(`.tab-btn[onclick="switchTab('${mode}')"]`);
        if (activeBtn) activeBtn.classList.add("active");

        const answerLabel = document.getElementById("answerLabel");

        // Update Form Logic
        if (mode === 'generate') {
            // Mode 1: Ask & Verify (First Tab)
            questionGroup.classList.remove("hidden"); // Show Question

            // HIDE answerGroup (User requested removal of context input)
            answerGroup.classList.add("hidden");

            // Required? No, context is optional
            answerInput.removeAttribute("required");

            questionInput.setAttribute("required", "true");
            questionInput.placeholder = "e.g., Is metformin effective for weight loss in non-diabetics?";

            btnText.textContent = "Generate & Verify";
        } else {
            // Mode 2: Verify Text (Second Tab)
            questionGroup.classList.add("hidden"); // Hide Question

            answerGroup.classList.remove("hidden");

            answerInput.setAttribute("required", "true");

            questionInput.removeAttribute("required");
            // questionInput.placeholder = "Optional..."; // Hidden

            // Change Label & Placeholder for Verification Text
            answerLabel.innerHTML = '<i class="fa-solid fa-robot"></i> Text to Verify';
            answerInput.placeholder = "Paste the text or LLM response here for verification...";

            btnText.textContent = "Verify Provided Text";
        }

        // Reset UI when switching tabs
        resetUI();
    }

    // State for multi-step
    let currentRefinedText = "";
    let currentRefinedQuestion = "";

    // Refinement Elements
    const refinementSection = document.getElementById("refinementSection");
    const refinementTitle = document.getElementById("refinementTitle");
    const refinementSubtitle = document.getElementById("refinementSubtitle");
    const refinedInput = document.getElementById("refinedInput");
    const finalVerifyBtn = document.getElementById("finalVerifyBtn");

    form.addEventListener("submit", async function (e) {
        e.preventDefault();

        // STEP 1: PREVIEW / REFINE
        setLoading(true);
        resetUI(); // Hide previous results

        const formData = new FormData(form);
        formData.append("action", "preview"); // Tell backend to just refine/generate

        try {
            const response = await fetch("/verify", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || "Refinement failed. Service may be busy.");
            }

            const data = await response.json();

            if (data.status === "PREVIEW") {
                renderRefinementPreview(data);
            } else {
                renderResults(data);
            }

        } catch (err) {
            console.error(err);
            alert(`Error: ${err.message}`);
        } finally {
            setLoading(false);
        }
    });

    finalVerifyBtn.addEventListener("click", async function () {
        // STEP 2: FINAL EXECUTION
        setLoading(true);

        // Prepare data for final analysis
        // We reuse the form data but update the 'ai_answer' (and 'question' if needed) with REFINED content
        const formData = new FormData(form);
        formData.append("action", "analyze");

        const mode = modeInput.value;

        if (mode === 'generate') {
            // Mode 1: Refined Question -> Generate -> Verify
            // But wait, the PREVIEW step already Generated the answer. 
            // So we just need to verify the *Generated Answer* (which user might have edited in refinedInput)
            // And we should pass the *Refined Question* too.
            formData.set("question", currentRefinedQuestion);
            formData.set("ai_answer", refinedInput.value); // The generated answer is here
        } else {
            // Mode 2: Refined Text -> Verify
            formData.set("ai_answer", refinedInput.value);
        }

        try {
            const response = await fetch("/verify", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || "Verification failed. Check your API limits.");
            }

            const data = await response.json();
            renderResults(data);

        } catch (err) {
            console.error(err);
            alert(`Error: ${err.message}`);
        } finally {
            setLoading(false);
        }
    });

    function renderRefinementPreview(data) {
        refinementSection.classList.remove("hidden");
        // Scroll to it
        refinementSection.scrollIntoView({ behavior: 'smooth' });

        const mode = modeInput.value;
        const refinedLabel = document.getElementById("refinedLabel");

        if (mode === 'generate') {
            refinementTitle.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Review Refined Question';
            refinementSubtitle.textContent = `Refined Question: "${data.refined_question}"`;

            // Update Label for Mode 1
            refinedLabel.innerHTML = '<i class="fa-solid fa-robot"></i> LLM Answer Is...';

            refinedInput.value = data.generated_answer;
            currentRefinedQuestion = data.refined_question;

        } else {
            refinementTitle.innerHTML = '<i class="fa-solid fa-spell-check"></i> Review Refined Text';
            refinementSubtitle.textContent = `Refined Text: "${data.refined_text.substring(0, 80)}..."`;

            // Update Label for Mode 2
            refinedLabel.innerHTML = '<i class="fa-solid fa-robot"></i> LLM Corrected Text';

            refinedInput.value = data.refined_text;
            currentRefinedText = data.refined_text;
        }
    }

    function resetUI() {
        resultSection.classList.add("hidden");
        refinementSection.classList.add("hidden");
        if (analysisSection) analysisSection.classList.add("hidden");

        claimsList.innerHTML = "";
        correctedContainer.classList.add("hidden");
    }

    function setLoading(isLoading) {
        submitBtn.disabled = isLoading;
        // Disable tabs too
        document.querySelectorAll(".tab-btn").forEach(btn => btn.disabled = isLoading);

        if (isLoading) {
            spinner.classList.remove("hidden");
            btnText.textContent = "Processing...";
            loadingStatus.classList.remove("hidden");
        } else {
            spinner.classList.add("hidden");
            // Check mode to set correct text
            const mode = document.getElementById("modeInput").value;
            // Mode 1 (value='generate') is now labeled "Verify Text" per user swap? 
            // Logic: Mode 1 button says "Generate & Verify", Mode 2 says "Verify Provided Text".
            // The swap was only for TAB labels.

            if (mode === 'generate') {
                btnText.textContent = "Generate & Verify";
            } else {
                btnText.textContent = "Verify Provided Text";
            }

            // Keep loading status visible for a moment if needed, or hide it
            setTimeout(() => {
                loadingStatus.classList.add("hidden");
            }, 1000);
        }
    }

    function renderResults(data) {
        resultSection.classList.remove("hidden");

        // 1. Verdict Banner
        if (data.status === "PASSED") {
            verdictBanner.className = "verdict-banner verdict-safe";
            verdictBanner.innerHTML = '<i class="fa-solid fa-shield-halved"></i> VERDICT: SAFE (Verified Content)';
        } else if (data.status === "FAILED") {
            verdictBanner.className = "verdict-banner verdict-unsafe";
            verdictBanner.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> VERDICT: HALLUCINATIONS DETECTED';
        } else {
            verdictBanner.className = "verdict-banner";
            verdictBanner.style.backgroundColor = "#f3f4f6";
            verdictBanner.innerHTML = '<i class="fa-solid fa-bug"></i> VERDICT: ERROR / UNKNOWN';
        }

        // 2. Stats
        const total = data.claims ? data.claims.length : 0;
        const verified = data.claims ? data.claims.filter(c => c.verification_status === "Verified").length : 0;
        const hallucinated = data.claims ? data.claims.filter(c => c.verification_status === "Hallucinated").length : 0;
        const missing = data.claims ? data.claims.filter(c => c.verification_status === "Evidence Not Found").length : 0;

        animateValue(totalClaimsEl, 0, total, 1000);
        animateValue(verifiedCountEl, 0, verified, 1000);
        animateValue(hallucinatedCountEl, 0, hallucinated, 1000);
        animateValue(missingCountEl, 0, missing, 1000);

        if (data.analysis && analysisSection) {
            analysisSection.classList.remove("hidden");
            if (beforeAcc) beforeAcc.textContent = `${data.analysis.before.accuracy}%`;
            if (afterAcc) afterAcc.textContent = `${data.analysis.after.accuracy}%`;
            if (impAcc) impAcc.innerHTML = formatImprovement(data.analysis.improvement.accuracy, true);

            if (beforeHall) beforeHall.textContent = `${data.analysis.before.hallucination_rate}%`;
            if (afterHall) afterHall.textContent = `${data.analysis.after.hallucination_rate}%`;
            if (impHall) impHall.innerHTML = formatImprovement(data.analysis.improvement.hallucination, false);

            if (analysisReportText) analysisReportText.textContent = data.analysis.report;
        }

        // 3. Final Answer Display
        const mode = document.getElementById("modeInput").value;
        const answerTitle = mode === 'generate' ? "Generated & Verified Answer" : "Corrected Answer";

        if (data.final_answer) {
            correctedContainer.classList.remove("hidden");
            correctedContainer.querySelector("h3").innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> ${answerTitle}`;
            finalAnswerText.textContent = data.final_answer;
        }

        // 4. Detailed Claims List
        if (data.claims && data.claims.length > 0) {
            data.claims.forEach(claim => {
                const card = createClaimCard(claim, false);
                claimsList.appendChild(card);
            });
        }

        // Scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    function createClaimCard(claim, isPass2) {
        const div = document.createElement("div");

        let statusClass = "claim-card";
        let badgeClass = "claim-badge";
        let badgeText = "UNKNOWN";
        let icon = "";

        if (claim.verification_status === "Verified") {
            statusClass += " verified";
            badgeClass += " badge-verified";
            badgeText = "VERIFIED";
            icon = '<i class="fa-solid fa-check"></i>';
        } else if (claim.verification_status === "Hallucinated") {
            statusClass += " hallucinated";
            badgeClass += " badge-hallucinated";
            badgeText = "HALLUCINATED";
            icon = '<i class="fa-solid fa-xmark"></i>';
        } else {
            statusClass += " evidence-not-found";
            badgeClass += " badge-missing";
            badgeText = "NO EVIDENCE";
            icon = '<i class="fa-solid fa-magnifying-glass-minus"></i>';
        }

        div.className = statusClass;

        // Optional: differentiate pass 2 style slightly?
        if (isPass2) {
            div.style.borderLeft = "4px solid #059669"; // Green border for pass 2
        }

        let html = `
            <div class="claim-header">
                <span class="${badgeClass}">${icon} ${badgeText}</span>
                ${isPass2 ? '<span style="font-size:0.8rem; color:#6b7280; margin-left:auto;">Double Check</span>' : ''}
            </div>
            <p class="claim-text">"${claim.claim}"</p>
        `;

        if (claim.correction && claim.verification_status === "Hallucinated") {
            html += `
            <div class="correction-area">
                <strong><i class="fa-solid fa-pen-nib"></i> Correction:</strong> ${claim.correction}
            </div>`;
        }

        if (claim.evidence && claim.evidence.length > 0) {
            html += `
            <div class="evidence-area">
                <button type="button" class="evidence-toggle" onclick="toggleEvidence(this)">
                    <i class="fa-solid fa-book-medical"></i> View Supporting Evidence (${claim.evidence.length})
                </button>
                <div class="evidence-content hidden">
                    <ul>
                        ${claim.evidence.slice(0, 3).map(e => `<li>${e}</li>`).join('')}
                    </ul>
                </div>
            </div>`;
        }

        div.innerHTML = html;
        return div;
    }

    function animateValue(obj, start, end, duration) {
        if (start === end) {
            obj.innerHTML = end;
            return;
        }
        let range = end - start;
        let current = start;
        let increment = end > start ? 1 : -1;
        let stepTime = Math.abs(Math.floor(duration / range));
        let timer = setInterval(function () {
            current += increment;
            obj.innerHTML = current;
            if (current == end) {
                clearInterval(timer);
            }
        }, stepTime);
    }

    function formatImprovement(value, isPositiveGood) {
        const val = parseFloat(value);
        const absVal = Math.abs(val);
        let arrow = val >= 0 ? '<i class="fa-solid fa-arrow-up"></i>' : '<i class="fa-solid fa-arrow-down"></i>';

        let color = "#1e293b"; // Standard slate
        if (val !== 0) {
            const isGood = isPositiveGood ? val > 0 : val < 0;
            // color = isGood ? "#10b981" : "#ef4444"; // Optional: color codes
        }

        return `<span style="color: ${color}">${arrow} ${absVal}%</span>`;
    }

    // Initialize UI
    window.switchTab('generate');
});

function toggleEvidence(btn) {
    const content = btn.nextElementSibling;
    content.classList.toggle("hidden");
}
