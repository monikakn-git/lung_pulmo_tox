// Configuration: Point this to your FastAPI backend URL.
// When testing locally with FastAPI, use localhost:8000
const API_URL = "https://lung-pulmo-tox.onrender.com";

document.addEventListener("DOMContentLoaded", () => {
    // Navigation Logic
    const navLinks = document.querySelectorAll('.nav-links a');
    const pages = document.querySelectorAll('.page');
    
    function switchPage(targetId) {
        // Update Nav Links
        navLinks.forEach(link => link.classList.remove('active'));
        document.querySelector(`[data-target="${targetId}"]`)?.classList.add('active');
        
        // Update Pages
        pages.forEach(page => page.classList.remove('active'));
        document.getElementById(targetId)?.classList.add('active');
    }

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            switchPage(link.getAttribute('data-target'));
        });
    });

    document.getElementById('get-started-btn')?.addEventListener('click', () => {
        switchPage('dashboard-page');
    });

    // Predictor Logic
    const form = document.getElementById("predict-form");
    const input = document.getElementById("drug-input");
    const predictBtn = document.getElementById("predict-btn");
    const btnText = predictBtn.querySelector(".btn-text");
    const spinner = predictBtn.querySelector(".spinner");
    const errorMessage = document.getElementById("error-message");
    
    const resultSection = document.getElementById("result-section");
    const displayDrugName = document.getElementById("display-drug-name");
    const riskValue = document.getElementById("risk-value");
    const probValue = document.getElementById("prob-value");
    const confValue = document.getElementById("conf-value");
    const explanationText = document.getElementById("explanation-text");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const drug_name = input.value.trim();
        if (!drug_name) return;

        // UI Loading State
        predictBtn.disabled = true;
        btnText.classList.add("hidden");
        spinner.classList.remove("hidden");
        errorMessage.classList.add("hidden");
        resultSection.classList.add("hidden");
        
        // Reset old classes
        riskValue.className = "metric-value";

        try {
            // Fix 404 by hitting the correct endpoint
            const endpoint = API_URL.endsWith('/') ? API_URL + 'api/predict' : API_URL + '/api/predict';
            
            const response = await fetch(endpoint, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ drug_name })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "An error occurred while analyzing the structure.");
            }

            // Populate Data
            displayDrugName.textContent = data.drug_name;
            riskValue.textContent = data.risk_category;
            probValue.textContent = `${(data.probability * 100).toFixed(1)}%`;
            confValue.textContent = `${data.confidence_score.toFixed(1)}%`;
            explanationText.textContent = data.explanation;

            // Apply risk color coding
            if (data.risk_category === "HIGH RISK") {
                riskValue.classList.add("risk-high");
            } else if (data.risk_category === "MEDIUM RISK") {
                riskValue.classList.add("risk-medium");
            } else {
                riskValue.classList.add("risk-low");
            }

            // Reveal Results
            resultSection.classList.remove("hidden");

        } catch (error) {
            errorMessage.textContent = error.message;
            errorMessage.classList.remove("hidden");
        } finally {
            // Restore UI
            predictBtn.disabled = false;
            btnText.classList.remove("hidden");
            spinner.classList.add("hidden");
        }
    });
});
