const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const resultCard = document.getElementById('result-card');
const resultContent = document.getElementById('result-content');
const errorBox = document.getElementById('error-message');
const resetBtn = document.getElementById('resetBtn');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError();

    // 1. Gather Inputs
    const income = parseFloat(document.getElementById('monthly_income').value);
    const debt = parseFloat(document.getElementById('monthly_debt_payments').value);
    const loan = parseFloat(document.getElementById('loan_amount').value);
    const missed = parseInt(document.getElementById('missed_installments').value);
    const balance = parseFloat(document.getElementById('credit_card_balance').value);
    const accounts = parseInt(document.getElementById('total_open_accounts').value);
    const home = document.getElementById('home_ownership').value;

    // 2. Client-Side Validation (Step 3 & 4)
    if (debt > income) {
        showError("Your monthly debt payments cannot exceed your income.");
        return;
    }
    if (loan > income * 60) {
        // Just a sanity check warning, though maybe valid for mortgages, but risky.
        // We won't block, but it's high risk.
    }

    // 3. Prepare Payload
    const payload = {
        monthly_income: income,
        monthly_debt_payments: debt,
        loan_amount: loan,
        missed_installments: missed,
        credit_card_balance: balance,
        total_open_accounts: accounts,
        home_ownership: home
    };

    // 4. Send to Backend
    setLoading(true);
    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.statusText}`);
        }

        const data = await response.json();
        displayResult(data);

    } catch (err) {
        showError("Failed to get prediction. Ensure backend is running. " + err.message);
    } finally {
        setLoading(false);
    }
});

resetBtn.addEventListener('click', () => {
    resultCard.classList.add('hidden');
    form.classList.remove('hidden');
    form.reset();
});

function displayResult(data) {
    form.classList.add('hidden');
    resultCard.classList.remove('hidden');

    const scoreEl = document.getElementById('risk-score');
    const catEl = document.getElementById('risk-category');
    const msgEl = document.getElementById('risk-message');

    // Update visuals
    scoreEl.textContent = data.default_probability + "%";
    scoreEl.style.color = data.color;

    catEl.textContent = data.risk_category;
    catEl.style.color = data.color;

    msgEl.textContent = data.message;
}

function showError(msg) {
    errorBox.textContent = `⚠️ ${msg}`;
    errorBox.classList.remove('hidden');
}

function hideError() {
    errorBox.classList.add('hidden');
}

function setLoading(isLoading) {
    if (isLoading) {
        predictBtn.textContent = "Analyzing...";
        predictBtn.disabled = true;
    } else {
        predictBtn.textContent = "Predict Risk";
        predictBtn.disabled = false;
    }
}
