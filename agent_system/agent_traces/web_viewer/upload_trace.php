<?php
// upload_trace.php - Handles trace file uploads

// Directory where trace files are stored
$directory = './traces/';

// Check if directory exists, create if not
if (!file_exists($directory)) {
    mkdir($directory, 0755, true);
}

// Check if a file was uploaded
if (isset($_FILES['trace_file']) && $_FILES['trace_file']['error'] === UPLOAD_ERR_OK) {
    $tmp_name = $_FILES['trace_file']['tmp_name'];
    $name = $_FILES['trace_file']['name'];
    
    // Ensure the file has a .json extension
    if (pathinfo($name, PATHINFO_EXTENSION) !== 'json') {
        header('HTTP/1.1 400 Bad Request');
        echo json_encode(['error' => 'File must be a JSON file']);
        exit;
    }
    
    // Generate a unique filename to avoid overwriting
    $timestamp = date('YmdHis');
    $filename = $timestamp . '_' . $name;
    $destination = $directory . $filename;
    
    // Move the uploaded file to the destination
    if (move_uploaded_file($tmp_name, $destination)) {
        echo json_encode([
            'status' => 'success',
            'message' => 'Trace file uploaded successfully',
            'filename' => $filename
        ]);
    } else {
        header('HTTP/1.1 500 Internal Server Error');
        echo json_encode(['error' => 'Failed to save file']);
    }
} else {
    // No file uploaded or upload error
    header('HTTP/1.1 400 Bad Request');
    echo json_encode([
        'error' => 'No file uploaded or upload error',
        'details' => isset($_FILES['trace_file']) ? $_FILES['trace_file']['error'] : 'No file specified'
    ]);
}
?>