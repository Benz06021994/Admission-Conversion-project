new Chart(document.getElementById("barChart"), {
    type: 'bar',
    data: {
        labels: Object.keys(sourceData),
        datasets: [{ label: "Conversion Rate", data: Object.values(sourceData) }]
    }
});

new Chart(document.getElementById("lineChart"), {
    type: 'line',
    data: {
        labels: Object.keys(monthlyTrend),
        datasets: [{ label: "Monthly Trend", data: Object.values(monthlyTrend) }]
    }
});

new Chart(document.getElementById("pieChart"), {
    type: 'pie',
    data: {
        labels: Object.keys(funnelData),
        datasets: [{ data: Object.values(funnelData) }]
    }
});

new Chart(document.getElementById("featureChart"), {
    type: 'bar',
    data: {
        labels: Object.keys(featureImportance),
        datasets: [{ label: "Importance", data: Object.values(featureImportance) }]
    }
});