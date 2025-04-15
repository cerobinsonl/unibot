<?php
// list_traces.php - Lists all JSON trace files in the traces directory

// Directory where trace files are stored
$directory = './traces/';

// Check if directory exists
if (!file_exists($directory) || !is_dir($directory)) {
    header('HTTP/1.1 500 Internal Server Error');
    echo json_encode(['error' => 'Traces directory not found']);
    exit;
}

// Scan the directory for files
$files = scandir($directory);

// Filter to only include JSON files
$traces = array_filter($files, function($file) {
    return pathinfo($file, PATHINFO_EXTENSION) === 'json';
});

// Remove '.' and '..' from results
$traces = array_values(array_diff($traces, array('.', '..')));

// Sort by modification time (newest first)
usort($traces, function($a, $b) use ($directory) {
    return filemtime($directory . $b) - filemtime($directory . $a);
});

// Send JSON response
header('Content-Type: application/json');
echo json_encode($traces);
?>