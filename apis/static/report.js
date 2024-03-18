// report.js
document.addEventListener('DOMContentLoaded', function() {
    // Function to parse URL parameters
    function getUrlParameter(name) {
        name = name.replace(/[\[]/, '\\[').replace(/[\]]/, '\\]');
        var regex = new RegExp('[\\?&]' + name + '=([^&#]*)');
        var results = regex.exec(location.search);
        return results === null ? '' : decodeURIComponent(results[1].replace(/\+/g, ' '));
    };

    // Get metric scores and visualization data from URL parameters
    var compressionRatio = getUrlParameter('compression_ratio');
    var uniqueWords = getUrlParameter('unique_words');
    var visualization = getUrlParameter('visualization');

    // Render metric scores
    document.getElementById('compressionRatio').textContent = compressionRatio;
    document.getElementById('uniqueWords').textContent = uniqueWords;

    // Render visualization charts (you'll need to implement this)

    // Handle download button click event
    document.getElementById('downloadPdfButton').addEventListener('click', function() {
        // Perform logic to download report as PDF (you'll need to implement this)
    });
});
