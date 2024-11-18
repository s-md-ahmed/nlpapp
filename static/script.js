async function makePrediction() {
    const message = document.getElementById('message').value;
    if(message===''){
        alert('Please enter a message');
        return;
    }
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message })
    });

    const data = await response.json();
    
    // Update the result with the prediction
    document.getElementById('result').textContent = `Prediction: ${data.result}`;

    // Dynamically generate and insert the report content
    const reportsHTML = `
        <div id="train-report" class="report">
            <h2>Train Set Classification Report</h2>
            <pre id="train-report-content">${data.train_report}</pre>
        </div>
        <div id="test-report" class="report">
            <h2>Test Set Classification Report</h2>
            <pre id="test-report-content">${data.test_report}</pre>
        </div>
    `;

    // Inject the reports HTML into the page
    document.getElementById('result').insertAdjacentHTML('beforeend', reportsHTML);
}
