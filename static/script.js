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

    
}
