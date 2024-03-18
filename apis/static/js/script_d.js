// Handle form submission to summarize text
document.getElementById('summarizationForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission
    
    // Get the model link from the input field
    var modelLink = document.getElementById('modelLink').value;
    
    // Make a POST request to the SummarizationAPIView
    fetch('/apis/summarize/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'model_link': modelLink }),
    })
    .then(response => response.json())
    .then(data => {
        // Display summary and metric scores on index.html
        document.getElementById('generatedSummary').textContent = data.generated_summary;
        document.getElementById('metricScores').textContent = `Rouge-1 Score: ${data.metric_scores.rouge_1_score}, Rouge-2 Score: ${data.metric_scores.rouge_2_score}, BLEU Score: ${data.metric_scores.bleu_score}`;
    })
    .catch(error => {
        console.error('Error:', error);
        // Handle errors
    });
});

// Handle click event for detailed report button
document.getElementById('detailedReportButton').addEventListener('click', function() {
    // Make a POST request to the VisualizationAPIView
    fetch('/apis/visualization/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'dataset_path': 'path/to/your/dataset.csv' }), // Replace with actual dataset path
    })
    .then(response => response.json())
    .then(data => {
        // Redirect to report.html and pass metric scores and visualization data as URL parameters
        window.location.href = `/report.html?compression_ratio=${data.compression_ratio}&unique_words=${data.unique_words}&visualization=${data.visualization}`;
    })
    .catch(error => {
        console.error('Error:', error);
        // Handle errors
    });
});
