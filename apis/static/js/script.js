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
        console.log('Received data:', data);
        // Display summary and metric scores on index.html
        document.getElementById('originalSummary').textContent = data.original_summary;
        document.getElementById('generatedSummary').textContent = data.generated_summary;
        document.getElementById('metricScores').textContent = `Rouge-1 {Precision}: ${data.metric_scores.rouge_1_score.p},
        Rouge-1 {Recall}: ${data.metric_scores.rouge_1_score.r}, 
        Rouge-1 {f-score}: ${data.metric_scores.rouge_1_score.f},
        Rouge-2 Score {Precision}: ${data.metric_scores.rouge_2_score.p}, 
        Rouge-2 Score {Recall}: ${data.metric_scores.rouge_2_score.r},
        Rouge-2 Score {f-score}: ${data.metric_scores.rouge_2_score.f},
        BLEU Score: ${data.metric_scores.bleu_score},
        BERT Score: ${data.metric_scores.bert_score},
        Mean Perplexity Score: ${data.metric_scores.perplexity_scores.mean_perplexity},
        METEOR Score: ${data.metric_scores.meteor_score_value},
        chrF Score: ${data.metric_scores.chrf_score},
        BLANC Score: ${data.metric_scores.blanc_score}`;
        document.getElementById('compressionRatio').textContent = data.histogram_data.compression_ratio;
        document.getElementById('totalWords').textContent = data.histogram_data.total_words;
        document.getElementById('uniqueWords').textContent = data.histogram_data.unique_words;
        //'perplexity_scores': {'perplexities': [41.79629135131836, 51.028106689453125], 'mean_perplexity': 46.41219902038574}

        // Render histograms
        renderHistograms(data.histogram_data);
        
        // Generate word clouds for unigrams, bigrams, and trigrams
        generateWordClouds(data.histogram_data.unigrams, 'unigrams');
        generateWordClouds(data.histogram_data.bigrams, 'bigrams');
        generateWordClouds(data.histogram_data.trigrams, 'trigrams');
    })
    .catch(error => {
        console.error('Error:', error);
        // Handle errors
    });
});

// Function to generate word clouds for unigrams, bigrams, or trigrams
function generateWordClouds(data, type) {
    // Clear previous word cloud
    document.getElementById(`${type}WordCloudContainer`).innerHTML = '';

    // Convert data object into an array of objects
    var wordsArray = Object.entries(data).map(([word, size]) => ({ text: word, size: size }));

    // Set the dimensions and margins of the graph
    var margin = { top: 10, right: 10, bottom: 10, left: 10 },
        width = 450 - margin.left - margin.right,
        height = 450 - margin.top - margin.bottom;

    // Append the SVG object to the container
    var svg = d3.select(`#${type}WordCloudContainer`).append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    // Constructs a new cloud layout instance
    var layout = d3.layout.cloud()
        .size([width, height])
        .words(wordsArray)
        .padding(5) // Space between words
        .rotate(function () { return ~~(Math.random() * 2) * 90; })
        .fontSize(function (d) { return d.size * 10; }) // Font size of words
        .on("end", draw);

    layout.start();

    // Draw the words
    function draw(words) {
        svg.append("g")
            .attr("transform", "translate(" + layout.size()[0] / 2 + "," + layout.size()[1] / 2 + ")")
            .selectAll("text")
            .data(words)
            .enter().append("text")
            .style("font-size", function (d) { return d.size; })
            .style("fill", "#69b3a2")
            .attr("text-anchor", "middle")
            .style("font-family", "Impact")
            .attr("transform", function (d) {
                return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
            })
            .text(function (d) { return d.text; });
    }
}

// Function to render histograms
function renderHistograms(histogram_data) {
    // Clear the histogram container before rendering new histograms
    document.getElementById('histogramContainer').innerHTML = '';

    // Create a grid container for the histograms
    const gridContainer = document.createElement('div');
    gridContainer.style.display = 'grid';
    gridContainer.style.gridTemplateColumns = 'repeat(2, 1fr)'; // Two columns
    gridContainer.style.gridGap = '1px'; // Gap between histograms
    document.getElementById('histogramContainer').appendChild(gridContainer);

    // Render histogram for compression ratio
    renderHistogram('Compression Ratio', { [histogram_data.compression_ratio]: 1 }, 'Compression Ratio', 'Frequency', 'salmon', 200, 150, gridContainer);

    // Render histogram for total words
    const totalWordsHistogramData = {};
    totalWordsHistogramData[histogram_data.total_words] = 1;
    renderHistogram('Total Words', totalWordsHistogramData, 'Total Words', 'Frequency', 'skyblue', 200, 150, gridContainer);

    // Render histogram for type-token ratio
    renderHistogram('Type-Token Ratio', { [histogram_data.type_token_ratio]: 1 }, 'Type-Token Ratio', 'Frequency', 'green', 200, 150, gridContainer);

    // Render histogram for unique words
    const uniqueWordsHistogramData = {};
    uniqueWordsHistogramData[histogram_data.unique_words] = 1;
    renderHistogram('Unique Words', uniqueWordsHistogramData, 'Unique Words', 'Frequency', 'purple', 200, 150, gridContainer);
}

function renderHistogram(title, data, xAxisLabel, yAxisLabel, color, width, height, container) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    container.appendChild(canvas);

    const chart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: Object.keys(data),
            datasets: [{
                label: title,
                data: Object.values(data),
                backgroundColor: color,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 12,
                    },
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: xAxisLabel,
                        font: {
                            size: 10,
                        },
                    },
                    ticks: {
                        beginAtZero: true,
                        stepSize: 1,
                        font: {
                            size: 8,
                        },
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: yAxisLabel,
                        font: {
                            size: 10,
                        },
                    },
                    ticks: {
                        beginAtZero: true,
                        font: {
                            size: 8,
                        },
                    }
                }
            }
        }
    });
}