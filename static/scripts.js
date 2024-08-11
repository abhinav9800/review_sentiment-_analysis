function getSentiment() {
    const inputText = document.getElementById('inputText').value;
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ texts: [inputText] })
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `Sentiment: ${data.sentiments[0]}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
