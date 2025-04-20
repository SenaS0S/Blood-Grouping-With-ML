// Main JavaScript for Blood Grouping System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips and popovers
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // File upload handling
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.querySelector('.upload-area');
    const previewContainer = document.getElementById('image-preview');
    const loadingSpinner = document.getElementById('loading-spinner');

    if (uploadArea) {
        // Drag and drop functionality
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('border-primary');
        });

        uploadArea.addEventListener('dragleave', function() {
            uploadArea.classList.remove('border-primary');
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('border-primary');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFilePreview(e.dataTransfer.files[0]);
            }
        });

        uploadArea.addEventListener('click', function() {
            fileInput.click();
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length) {
                handleFilePreview(fileInput.files[0]);
            }
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            if (fileInput.files.length === 0) {
                e.preventDefault();
                showAlert('Please select an image to analyze.', 'warning');
                return;
            }
            
            // Show loading state
            if (loadingSpinner) {
                loadingSpinner.classList.remove('d-none');
            }
        });
    }

    function handleFilePreview(file) {
        // Validate file type
        if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
            showAlert('Please select a valid JPEG or PNG image.', 'warning');
            return;
        }
        
        // Validate file size (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
            showAlert('File size exceeds 5MB. Please select a smaller image.', 'warning');
            return;
        }
        
        // Display preview
        if (previewContainer) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewContainer.innerHTML = `
                    <div class="card">
                        <div class="card-header">Image Preview</div>
                        <div class="card-body text-center">
                            <img src="${e.target.result}" class="img-fluid mb-3" style="max-height: 300px; border-radius: 0.25rem;">
                            <p class="mb-0 text-muted">${file.name} (${formatFileSize(file.size)})</p>
                        </div>
                    </div>
                `;
            };
            reader.readAsDataURL(file);
        }
    }

    // Batch processing functionality
    const batchUploadForm = document.getElementById('batch-upload-form');
    const batchFileInput = document.getElementById('batch-file-input');
    const batchResultsContainer = document.getElementById('batch-results');
    const batchLoadingSpinner = document.getElementById('batch-loading-spinner');

    if (batchUploadForm) {
        batchUploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            if (batchFileInput.files.length === 0) {
                showAlert('Please select at least one image to analyze.', 'warning');
                return;
            }
            
            // Show loading state
            if (batchLoadingSpinner) {
                batchLoadingSpinner.classList.remove('d-none');
            }
            
            // Create FormData and append files
            const formData = new FormData();
            for (let i = 0; i < batchFileInput.files.length; i++) {
                formData.append('files[]', batchFileInput.files[i]);
            }
            
            // Send AJAX request
            fetch('/batch/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading spinner
                if (batchLoadingSpinner) {
                    batchLoadingSpinner.classList.add('d-none');
                }
                
                // Display results
                if (batchResultsContainer) {
                    displayBatchResults(data.results);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showAlert('An error occurred during batch processing. Please try again.', 'danger');
                
                if (batchLoadingSpinner) {
                    batchLoadingSpinner.classList.add('d-none');
                }
            });
        });
    }

    function displayBatchResults(results) {
        if (!batchResultsContainer) return;
        
        if (results.length === 0) {
            batchResultsContainer.innerHTML = '<div class="alert alert-info">No results to display.</div>';
            return;
        }
        
        let html = `
            <h3 class="mt-4">Batch Processing Results</h3>
            <div class="table-responsive">
                <table class="table table-striped batch-results-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Filename</th>
                            <th>Status</th>
                            <th>Blood Type</th>
                            <th>Confidence</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        results.forEach((result, index) => {
            const statusClass = result.status === 'success' ? 'text-success' : 'text-danger';
            const statusIcon = result.status === 'success' ? 'check-circle' : 'x-circle';
            
            html += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${result.filename}</td>
                    <td><i class="bi bi-${statusIcon} ${statusClass}"></i> ${result.status}</td>
                    <td>${result.status === 'success' ? result.blood_type : '-'}</td>
                    <td>
                        ${result.status === 'success' ? 
                            `<div class="confidence-meter">
                                <div class="confidence-bar ${getConfidenceClass(result.confidence)}" 
                                     style="width: ${result.confidence}%"></div>
                            </div>
                            <small>${result.confidence.toFixed(1)}%</small>` 
                            : '-'}
                    </td>
                    <td>
                        ${result.status === 'success' && result.report_path ? 
                            `<a href="/report/${result.report_path}" target="_blank" class="btn btn-sm btn-primary">
                                <i class="bi bi-file-earmark-image"></i> View Report
                            </a>` 
                            : result.message ? `<span class="text-danger">${result.message}</span>` : '-'}
                    </td>
                </tr>
            `;
        });
        
        html += `
                    </tbody>
                </table>
            </div>
        `;
        
        batchResultsContainer.innerHTML = html;
    }

    // Utility functions
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' bytes';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / 1048576).toFixed(1) + ' MB';
    }

    function getConfidenceClass(confidence) {
        if (confidence >= 80) return 'high-confidence';
        else if (confidence >= 60) return 'medium-confidence';
        else return 'low-confidence';
    }

    function showAlert(message, type = 'info') {
        const alertContainer = document.getElementById('alert-container');
        if (!alertContainer) return;
        
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        alertContainer.appendChild(alert);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            alert.classList.remove('show');
            setTimeout(() => alert.remove(), 150);
        }, 5000);
    }
});
