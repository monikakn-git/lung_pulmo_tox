// Configuration: Point this to your FastAPI backend URL.
// Use window.location.origin for relative deployments (e.g. Vercel)
const API_URL = window.location.origin;

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
    
    const resPage = document.getElementById("results-page");
    const resDrugName = document.getElementById("res-drug-name");
    const resSmiles = document.getElementById("res-smiles");
    const resMoleculeImage = document.getElementById("res-molecule-image");
    const resRiskValue = document.getElementById("res-risk-value");
    const resProbValue = document.getElementById("res-prob-value");
    const resConfValue = document.getElementById("res-conf-value");
    const resExplanationText = document.getElementById("res-explanation-text");
    const modelsList = document.getElementById("models-list");
    const shapChart = document.getElementById("shap-chart");
    const neighborName = document.getElementById("neighbor-name");
    const neighborSim = document.getElementById("neighbor-sim");
    const neighborLabel = document.getElementById("neighbor-label");
    const similarityAlert = document.getElementById("similarity-alert");
    const similarityText = document.getElementById("similarity-text");

    document.getElementById("back-to-dash-btn")?.addEventListener("click", () => {
        switchPage('dashboard-page');
    });

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const drug_name = input.value.trim();
        if (!drug_name) return;

        // UI Loading State
        predictBtn.disabled = true;
        btnText.classList.add("hidden");
        spinner.classList.remove("hidden");
        errorMessage.classList.add("hidden");
        
        // Reset old classes
        resRiskValue.className = "huge-value";

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

            // Populate Large Results Page
            resDrugName.textContent = data.drug_name;
            resSmiles.textContent = `SMILES: ${data.smiles}`;
            resRiskValue.textContent = data.risk_category;
            resProbValue.textContent = `${(data.probability * 100).toFixed(1)}%`;
            resConfValue.textContent = `${data.confidence_score.toFixed(1)}%`;
            resExplanationText.textContent = data.explanation;

            if (data.image_base64) {
                resMoleculeImage.src = `data:image/png;base64,${data.image_base64}`;
            }

            // Populate Models List
            modelsList.innerHTML = '';
            if (data.all_models) {
                // Sort by probability descending
                Object.entries(data.all_models)
                    .sort(([,a], [,b]) => b - a)
                    .forEach(([name, prob]) => {
                        const probPct = (prob * 100).toFixed(1);
                        const item = document.createElement('div');
                        item.className = 'model-item';
                        item.innerHTML = `
                            <span class="model-name">${name}</span>
                            <div class="model-prob-bar">
                                <div class="prob-fill" style="width: ${probPct}%"></div>
                            </div>
                            <span class="model-val">${probPct}%</span>
                        `;
                        modelsList.appendChild(item);
                    });
            }

            // Populate SHAP Chart
            shapChart.innerHTML = '';
            if (data.shap_values) {
                // Find max abs value for normalization
                const maxAbs = Math.max(...data.shap_values.map(v => Math.abs(v.value)));
                
                data.shap_values.forEach(item => {
                    const width = (Math.abs(item.value) / maxAbs * 100).toFixed(0);
                    const colorClass = item.value > 0 ? 'pos-bar' : 'neg-bar';
                    const row = document.createElement('div');
                    row.className = 'shap-row';
                    row.innerHTML = `
                        <div class="shap-label">${item.feature}</div>
                        <div class="shap-bar-bg">
                            <div class="shap-bar-fill ${colorClass}" style="width: ${width}%"></div>
                        </div>
                        <div class="shap-val">${item.value.toFixed(3)}</div>
                    `;
                    shapChart.appendChild(row);
                });
            }

            // Populate Nearest Neighbor
            if (data.nearest_neighbor) {
                neighborName.textContent = data.nearest_neighbor.name;
                neighborSim.textContent = `Structural Similarity: ${(data.nearest_neighbor.similarity * 100).toFixed(1)}%`;
                neighborLabel.textContent = data.nearest_neighbor.label;
                neighborLabel.style.color = data.nearest_neighbor.label === 'TOXIC' ? 'var(--risk-high)' : 'var(--accent-primary)';
                
                // Show similarity warning if below 30%
                if (data.nearest_neighbor.similarity < 0.3) {
                    similarityAlert.classList.remove('hidden');
                    similarityText.textContent = "Out-of-distribution: This molecule is structurally unique vs our training set.";
                } else {
                    similarityAlert.classList.add('hidden');
                }
            }

            // Apply risk color coding
            if (data.risk_category === "HIGH RISK") {
                resRiskValue.classList.add("risk-high");
            } else if (data.risk_category === "MEDIUM RISK") {
                resRiskValue.classList.add("risk-medium");
            } else {
                resRiskValue.classList.add("risk-low");
            }

            // Transition to results page
            switchPage('results-page');

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
