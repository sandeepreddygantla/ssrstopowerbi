// Main app class - handles file uploads and analysis
class MigrationManager {
    constructor() {
        this.files = [];  // uploaded files
        this.selected = [];  // which ones are checked
        this.results = null;
        this.jobId = null;
        this.setup();
    }
    
    setup() {
        // setup drag and drop etc
        this.initUpload();
        this.bindEvents();
    }
    
    initUpload() {
        const zone = document.getElementById('drop-zone');
        const input = document.getElementById('file-input');
        
        // basic click handler
        zone.onclick = () => input.click();
        
        // handle file selection
        input.onchange = (e) => this.processFiles(e.target.files);
        
        // drag and drop stuff
        zone.ondragover = (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        };
        
        zone.ondragleave = (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
        };
        
        zone.ondrop = (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            this.processFiles(e.dataTransfer.files);
        };
    }
    
    bindEvents() {
        // button handlers
        document.getElementById('select-all-btn').onclick = () => this.selectAll();
        document.getElementById('clear-files-btn').onclick = () => this.clearFiles();
        document.getElementById('start-analysis-btn').onclick = () => this.runAnalysis();
        document.getElementById('start-migration-btn').onclick = () => this.runMigration();
    }
    
    processFiles(fileList) {
        // filter for rdl files only
        let rdlFiles = [];
        for (let i = 0; i < fileList.length; i++) {
            if (fileList[i].name.toLowerCase().endsWith('.rdl')) {
                rdlFiles.push(fileList[i]);
            }
        }
        
        if (rdlFiles.length == 0) {
            this.showToast('error', 'No RDL files found', 'Only .rdl files are supported');
            return;
        }
        
        this.uploadToServer(rdlFiles);
    }
    
    async uploadToServer(files) {
        const progress = document.getElementById('upload-progress');
        const bar = document.getElementById('upload-progress-bar');
        const text = document.getElementById('upload-percentage');
        
        try {
            progress.classList.remove('hidden');
            
            // prep the form data
            const data = new FormData();
            for (let file of files) {
                data.append('files', file);
            }
            
            // show some progress
            bar.style.width = '50%';
            text.textContent = '50%';
            
            console.log(`Uploading ${files.length} files...`);
            
            // actually upload
            const resp = await fetch('/rdlmigration/api/upload', {
                method: 'POST',
                body: data
            });
            
            if (!resp.ok) {
                throw new Error(`Upload failed: ${resp.status}`);
            }
            
            const result = await resp.json();
            console.log('Upload done:', result);
            
            // finish progress
            bar.style.width = '100%';
            text.textContent = '100%';
            
            // save the files
            this.files = [];
            if (result.files) {
                result.files.forEach((f, i) => {
                    this.files.push({
                        id: Date.now() + i,
                        name: f.name,
                        size: f.size,
                        path: f.path,
                        file: files[i],
                        selected: true
                    });
                });
            } else {
                // fallback if server doesn't return file info
                files.forEach((f, i) => {
                    this.files.push({
                        id: Date.now() + i,
                        name: f.name,
                        size: f.size,
                        file: f,
                        selected: true
                    });
                });
            }
            
            setTimeout(() => {
                progress.classList.add('hidden');
                this.updateFileList();
                this.showFileListSection();
                
                // Show success notification with server response
                const message = result.message || `Successfully uploaded ${files.length} RDL files!`;
                this.showSuccessNotification(message);
            }, 500);
            
        } catch (error) {
            console.error('[ERROR] File upload failed:', error);
            
            // Hide progress and show error
            progress.classList.add('hidden');
            
            this.showToast('error', 'Upload Failed', 
                error.message || 'Failed to upload files to server. Please check your connection and try again.');
        }
    }
    
    updateFileList() {
        const container = document.getElementById('file-table-container');
        
        if (this.files.length === 0) {
            container.innerHTML = '<div class="text-center text-gray">No files uploaded</div>';
            return;
        }
        
        const table = document.createElement('table');
        table.className = 'modern-table';
        
        table.innerHTML = `
            <thead>
                <tr>
                    <th><input type="checkbox" id="select-all-checkbox" ${this.allFilesSelected() ? 'checked' : ''}></th>
                    <th>File Name</th>
                    <th>Size</th>
                    <th>Status</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                ${this.files.map(file => `
                    <tr>
                        <td>
                            <input type="checkbox" class="file-checkbox" data-file-id="${file.id}" 
                                   ${file.selected ? 'checked' : ''}>
                        </td>
                        <td>
                            <div class="flex items-center gap-sm">
                                <i class="fas fa-file-code" style="color: var(--primary-orange);"></i>
                                ${file.name}
                            </div>
                        </td>
                        <td>${this.formatFileSize(file.size)}</td>
                        <td>
                            <span class="status-badge status-success">
                                <i class="fas fa-check"></i>
                                Ready
                            </span>
                        </td>
                        <td>
                            <button onclick="migrationManager.removeFile(${file.id})" 
                                    class="btn btn-secondary btn-sm">
                                <i class="fas fa-trash"></i>
                            </button>
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        `;
        
        container.innerHTML = '';
        container.appendChild(table);
        
        // Setup checkbox listeners
        document.getElementById('select-all-checkbox').addEventListener('change', (e) => {
            this.selectAll(e.target.checked);
        });
        
        document.querySelectorAll('.file-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateFileSelection();
            });
        });
        
        this.updateFileSelection();
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    updateFileSelection() {
        const checkboxes = document.querySelectorAll('.file-checkbox');
        const selectedCount = Array.from(checkboxes).filter(cb => cb.checked).length;
        const totalSize = Array.from(checkboxes)
            .filter(cb => cb.checked)
            .reduce((sum, cb) => {
                const fileId = parseInt(cb.dataset.fileId);
                const file = this.files.find(f => f.id === fileId);
                return sum + (file ? file.size : 0);
            }, 0);
        
        document.getElementById('selected-count').textContent = selectedCount;
        document.getElementById('selected-size').textContent = this.formatFileSize(totalSize);
        
        const summary = document.getElementById('selection-summary');
        if (selectedCount > 0) {
            summary.classList.remove('hidden');
            document.getElementById('analysis-section').classList.remove('hidden');
        } else {
            summary.classList.add('hidden');
            document.getElementById('analysis-section').classList.add('hidden');
        }
    }
    
    selectAll(checked = true) {
        const checkboxes = document.querySelectorAll('.file-checkbox');
        checkboxes.forEach(cb => {
            cb.checked = checked;
        });
        this.updateFileSelection();
    }
    
    allFilesSelected() {
        const checkboxes = document.querySelectorAll('.file-checkbox');
        return Array.from(checkboxes).every(cb => cb.checked);
    }
    
    clearFiles() {
        if (confirm('Are you sure you want to clear all uploaded files?')) {
            this.files = [];
            this.updateFileList();
            document.getElementById('file-list-section').classList.add('hidden');
            document.getElementById('analysis-section').classList.add('hidden');
            document.getElementById('results-section').classList.add('hidden');
            document.getElementById('migration-section').classList.add('hidden');
        }
    }
    
    removeFile(fileId) {
        this.files = this.files.filter(f => f.id !== fileId);
        this.updateFileList();
        
        if (this.files.length === 0) {
            document.getElementById('file-list-section').classList.add('hidden');
            document.getElementById('analysis-section').classList.add('hidden');
        }
    }
    
    showFileListSection() {
        document.getElementById('file-list-section').classList.remove('hidden');
        document.getElementById('analysis-section').classList.remove('hidden');
    }
    
    async runAnalysis() {
        const selectedFiles = this.getSelectedFiles();
        if (selectedFiles.length === 0) {
            this.showToast('warning', 'No files selected', 'Please select files to analyze');
            return;
        }
        
        if (selectedFiles.length < 2) {
            this.showToast('warning', 'Minimum 2 files required', 'Please select at least 2 files for similarity analysis');
            return;
        }
        
        const progressDiv = document.getElementById('analysis-progress');
        const progressBar = document.getElementById('analysis-progress-bar');
        const progressText = document.getElementById('analysis-percentage');
        const statusText = document.getElementById('analysis-status');
        
        progressDiv.classList.remove('hidden');
        
        try {
            // Prepare files data for API - Create temporary files for analysis
            const formData = new FormData();
            selectedFiles.forEach(fileData => {
                if (fileData.file) {
                    formData.append('files', fileData.file);
                }
            });
            
            // First upload files to get proper paths, then analyze
            statusText.textContent = 'Uploading files for analysis...';
            progressBar.style.width = '20%';
            progressText.textContent = '20%';
            
            const uploadResponse = await fetch('/rdlmigration/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const uploadResult = await uploadResponse.json();
            
            console.log('[DEBUG] Upload result:', uploadResult);
            
            // Check for files in the response (server returns 'files', not 'uploaded_files')
            const serverFiles = uploadResult.files || [];
            if (serverFiles.length === 0) {
                throw new Error('File upload failed for analysis - no files returned from server');
            }
            
            // Now prepare the files data with proper paths for analysis
            const filesData = serverFiles.map(file => ({
                name: file.name,
                path: file.path,
                size: file.size
            }));
            
            // Call the real analysis API
            statusText.textContent = 'Starting AI-powered business logic analysis...';
            progressBar.style.width = '40%';
            progressText.textContent = '40%';
            
            const response = await fetch('/rdlmigration/api/ai-analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    job_id: uploadResult.job_id,
                    enable_ai_analysis: true,
                    cache_enabled: true
                })
            });
            
            statusText.textContent = 'Processing analysis results...';
            progressBar.style.width = '80%';
            progressText.textContent = '80%';
            
            const result = await response.json();
            
            if (result.success && result.analysis_results) {
                this.analysisResults = result.analysis_results;
                
                statusText.textContent = 'Analysis complete!';
                progressBar.style.width = '100%';
                progressText.textContent = '100%';
                
                setTimeout(() => {
                    this.showDetailedAnalysisResults();
                }, 500);
            } else {
                throw new Error(result.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            statusText.textContent = 'Analysis failed - using demo data';
            
            // Fallback to demo data for testing
            this.analysisResults = this.generateDemoAnalysisResults(selectedFiles);
            
            progressBar.style.width = '100%';
            progressText.textContent = '100%';
            
            setTimeout(() => {
                this.showDetailedAnalysisResults();
            }, 500);
        }
    }
    
    showDetailedAnalysisResults() {
        document.getElementById('analysis-progress').classList.add('hidden');
        document.getElementById('results-section').classList.remove('hidden');
        document.getElementById('migration-section').classList.remove('hidden');
        
        const resultsContent = document.getElementById('results-content');
        
        if (!this.analysisResults || !this.analysisResults.similarity_pairs) {
            resultsContent.innerHTML = '<div class="text-center" style="color: var(--dark-gray);">No analysis results available</div>';
            return;
        }
        
        const pairs = this.analysisResults.similarity_pairs;
        const totalFiles = this.analysisResults.total_files || pairs.length;
        
        // Filter pairs while preserving original indices
        const highSimilarityPairs = pairs.map((p, i) => ({...p, originalIndex: i})).filter(p => p.similarity >= 70);
        const mediumSimilarityPairs = pairs.map((p, i) => ({...p, originalIndex: i})).filter(p => p.similarity >= 40 && p.similarity < 70);
        const lowSimilarityPairs = pairs.map((p, i) => ({...p, originalIndex: i})).filter(p => p.similarity < 40);
        
        resultsContent.innerHTML = `
            <!-- Analysis Report Header -->
            <div style="background: #002677; padding: 2rem; border-radius: 12px; color: #FFFFFF; margin-bottom: 2rem;">
                <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 600;">
                    <i class="fas fa-chart-line" style="margin-right: 0.5rem;"></i>
                    Analysis Report 154 - Smart Consolidation Results
                </h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; color: #FF612B;">${totalFiles}</div>
                        <div style="opacity: 0.9;">Files Analyzed</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; color: #FF612B;">${pairs.length}</div>
                        <div style="opacity: 0.9;">Comparisons Made</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; color: #FF612B;">${highSimilarityPairs.length}</div>
                        <div style="opacity: 0.9;">High Similarity Pairs</div>
                    </div>
                </div>
            </div>
            
            <!-- Analysis Summary Cards -->
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-bottom: 2rem;">
                <div style="background: #FFFFFF; padding: 1.5rem; border-radius: 8px; text-align: center; border: 2px solid #D9F6FA; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="font-size: 2rem; font-weight: 700; color: #002677; margin-bottom: 0.5rem;">${highSimilarityPairs.length}</div>
                    <div style="color: #4B4D4F; margin-bottom: 0.5rem; font-weight: 600;">High Similarity (≥70%)</div>
                    <div style="font-size: 0.875rem; color: #4B4D4F;">Strong consolidation candidates</div>
                </div>
                <div style="background: #FFFFFF; padding: 1.5rem; border-radius: 8px; text-align: center; border: 2px solid #FAF8F2; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="font-size: 2rem; font-weight: 700; color: #FF612B; margin-bottom: 0.5rem;">${mediumSimilarityPairs.length}</div>
                    <div style="color: #4B4D4F; margin-bottom: 0.5rem; font-weight: 600;">Medium Similarity (40-69%)</div>
                    <div style="font-size: 0.875rem; color: #4B4D4F;">Review for potential consolidation</div>
                </div>
                <div style="background: #FFFFFF; padding: 1.5rem; border-radius: 8px; text-align: center; border: 2px solid #FAF8F2; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="font-size: 2rem; font-weight: 700; color: #4B4D4F; margin-bottom: 0.5rem;">${lowSimilarityPairs.length}</div>
                    <div style="color: #4B4D4F; margin-bottom: 0.5rem; font-weight: 600;">Low Similarity (<40%)</div>
                    <div style="font-size: 0.875rem; color: #4B4D4F;">Maintain as separate reports</div>
                </div>
            </div>
            
            <!-- Detailed Analysis Results -->
            <div style="background: #FFFFFF; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h4 style="margin: 0 0 2rem 0; color: #002677; font-size: 1.25rem; font-weight: 600;">
                    <i class="fas fa-microscope" style="margin-right: 0.5rem;"></i>
                    Detailed Similarity Analysis
                </h4>
                
                <!-- Simple Tab Navigation -->
                <div style="margin-bottom: 2rem;">
                    <div style="display: flex; gap: 0; border-bottom: 3px solid #FAF8F2;">
                        <button class="tab-button active" data-tab="high-similarity" style="padding: 1rem 2rem; border: none; background: #002677; color: #FFFFFF; font-weight: 600; cursor: pointer; border-radius: 8px 8px 0 0;">
                            High (${highSimilarityPairs.length})
                        </button>
                        <button class="tab-button" data-tab="medium-similarity" style="padding: 1rem 2rem; border: none; background: #FAF8F2; color: #4B4D4F; font-weight: 600; cursor: pointer; border-radius: 8px 8px 0 0;">
                            Medium (${mediumSimilarityPairs.length})
                        </button>
                        <button class="tab-button" data-tab="low-similarity" style="padding: 1rem 2rem; border: none; background: #FAF8F2; color: #4B4D4F; font-weight: 600; cursor: pointer; border-radius: 8px 8px 0 0;">
                            Low (${lowSimilarityPairs.length})
                        </button>
                    </div>
                </div>
                
                <div id="high-similarity" class="tab-content">
                    ${this.renderSimplifiedSimilarityPairs(highSimilarityPairs, 'high')}
                </div>
                
                <div id="medium-similarity" class="tab-content hidden">
                    ${this.renderSimplifiedSimilarityPairs(mediumSimilarityPairs, 'medium')}
                </div>
                
                <div id="low-similarity" class="tab-content hidden">
                    ${this.renderSimplifiedSimilarityPairs(lowSimilarityPairs, 'low')}
                </div>
            </div>
            
            <!-- MODAL MOVED TO BODY LEVEL FOR COMPLETE SEPARATION -->
        `;
        
        // Setup tab functionality
        this.setupAnalysisTabsLogic();
        this.updateMigrationPreview();
    }
    
    renderSimplifiedSimilarityPairs(pairs, category) {
        if (pairs.length === 0) {
            return `
                <div style="text-align: center; padding: 3rem; color: #4B4D4F;">
                    <i class="fas fa-search" style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5; color: #4B4D4F;"></i>
                    <h4 style="margin: 0.5rem 0; color: #002677;">No ${category} similarity pairs found</h4>
                    <p style="margin: 0; color: #4B4D4F;">No file pairs match the ${category} similarity criteria.</p>
                </div>
            `;
        }
        
        return pairs.map((pair, index) => `
            <div style="background: #FAF8F2; border: 2px solid #D9F6FA; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <!-- File Comparison Header -->
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 2px solid #D9F6FA;">
                    <div>
                        <h5 style="margin: 0 0 0.5rem 0; color: #002677; font-size: 1.1rem; font-weight: 600;">
                            <i class="fas fa-files" style="margin-right: 0.5rem; color: #FF612B;"></i>
                            ${pair.file1} ↔ ${pair.file2}
                        </h5>
                        <div style="font-size: 0.875rem; color: #4B4D4F;">
                            Comparison ${index + 1} of ${pairs.length}
                        </div>
                    </div>
                    <div style="text-align: center; background: #FFFFFF; padding: 1rem; border-radius: 8px; border: 2px solid #D9F6FA;">
                        <div style="font-size: 2rem; font-weight: 700; color: #002677;">
                            ${pair.similarity}%
                        </div>
                        <div style="font-size: 0.875rem; color: #4B4D4F; font-weight: 600;">Overall Match</div>
                    </div>
                </div>
                
                <!-- Business Logic Metrics -->
                <div style="margin-bottom: 1.5rem;">
                    <h6 style="margin: 0 0 1rem 0; color: #002677; font-size: 1rem; font-weight: 600;">
                        <i class="fas fa-chart-bar" style="margin-right: 0.5rem; color: #FF612B;"></i>
                        Business Logic Analysis
                    </h6>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                        ${this.renderSimplifiedMetrics(pair.details)}
                    </div>
                </div>
                
                <!-- Action Buttons -->
                <div style="display: flex; gap: 1rem; justify-content: center;">
                    <button onclick="migrationManager.openSQLModal('${pair.file1}', '${pair.file2}', ${pair.originalIndex})" style="background: #FF612B; color: #FFFFFF; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.875rem;">
                        <i class="fas fa-code" style="margin-right: 0.5rem;"></i>
                        View SQL Queries
                    </button>
                    <button onclick="migrationManager.exportPairAnalysis(${pair.originalIndex})" style="background: #FFFFFF; color: #002677; border: 2px solid #002677; padding: 0.75rem 1.5rem; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.875rem;">
                        <i class="fas fa-download" style="margin-right: 0.5rem;"></i>
                        Export Analysis
                    </button>
                </div>
                
                <!-- Recommendation Badge -->
                <div style="margin-top: 1rem; text-align: center;">
                    ${this.renderRecommendationBadge(pair.similarity)}
                </div>
            </div>
        `).join('');
    }
    
    renderSimplifiedMetrics(details) {
        const metrics = [
            { label: 'Data Source', value: details.data_source_similarity || 0, icon: 'fas fa-database' },
            { label: 'Filter Logic', value: details.filter_logic_similarity || 0, icon: 'fas fa-filter' },
            { label: 'Business Purpose', value: details.business_purpose_similarity || 0, icon: 'fas fa-bullseye' },
            { label: 'Calculations', value: details.calculation_similarity || 0, icon: 'fas fa-calculator' },
            { label: 'Parameters', value: details.parameter_similarity || 0, icon: 'fas fa-sliders-h' }
        ];
        
        return metrics.map(metric => `
            <div style="background: #FFFFFF; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #D9F6FA;">
                <div style="color: #FF612B; font-size: 1.2rem; margin-bottom: 0.5rem;">
                    <i class="${metric.icon}"></i>
                </div>
                <div style="font-size: 0.875rem; color: #4B4D4F; margin-bottom: 0.5rem; font-weight: 600;">
                    ${metric.label}
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #002677;">
                    ${Math.round(metric.value)}%
                </div>
            </div>
        `).join('');
    }
    
    renderRecommendationBadge(similarity) {
        let recommendation, bgColor, textColor;
        
        if (similarity >= 70) {
            recommendation = "✅ Recommend Consolidation";
            bgColor = "#D9F6FA";
            textColor = "#002677";
        } else if (similarity >= 40) {
            recommendation = "⚠️ Review for Consolidation";
            bgColor = "#FAF8F2";
            textColor = "#FF612B";
        } else {
            recommendation = "❌ Keep Separate";
            bgColor = "#FFFFFF";
            textColor = "#4B4D4F";
        }
        
        return `
            <div style="background: ${bgColor}; color: ${textColor}; padding: 0.75rem 1.5rem; border-radius: 20px; display: inline-block; font-weight: 600; font-size: 0.875rem;">
                ${recommendation}
            </div>
        `;
    }
    
    renderSimilarityPairs(pairs, category) {
        if (pairs.length === 0) {
            return `
                <div class="text-center" style="padding: 3rem; color: var(--dark-gray);">
                    <i class="fas fa-search" style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;"></i>
                    <h4>No ${category} similarity pairs found</h4>
                    <p>No file pairs match the ${category} similarity criteria.</p>
                </div>
            `;
        }
        
        return pairs.map((pair, index) => `
            <div class="similarity-pair-card modern-card" style="margin-bottom: 2rem; border-left: 4px solid ${this.getSimilarityColor(pair.similarity)};">
                <!-- Pair Header -->
                <div class="flex items-center justify-between mb-lg" style="border-bottom: 1px solid var(--light-blue); padding-bottom: 1rem;">
                    <div>
                        <h5 style="margin: 0 0 0.5rem 0; color: var(--navy-blue);">
                            <i class="fas fa-copy" style="margin-right: 0.5rem;"></i>
                            ${pair.file1} ↔ ${pair.file2}
                        </h5>
                        <div style="font-size: 0.875rem; color: var(--dark-gray);">
                            Comparison ${index + 1} of ${pairs.length}
                        </div>
                    </div>
                    <div class="similarity-score" style="text-align: center;">
                        <div style="font-size: 2rem; font-weight: 700; color: ${this.getSimilarityColor(pair.similarity)};">
                            ${pair.similarity}%
                        </div>
                        <div style="font-size: 0.875rem; color: var(--dark-gray);">Overall Similarity</div>
                    </div>
                </div>
                
                <!-- Detailed Breakdown -->
                <div class="similarity-breakdown grid grid-2" style="margin-bottom: 2rem; gap: 1rem;">
                    <div>
                        <h6 style="margin-bottom: 1rem; color: var(--navy-blue);">
                            <i class="fas fa-chart-bar" style="margin-right: 0.5rem;"></i>
                            Business Logic Breakdown
                        </h6>
                        ${this.renderSimilarityMetrics(pair.details)}
                    </div>
                    <div>
                        <h6 style="margin-bottom: 1rem; color: var(--navy-blue);">
                            <i class="fas fa-lightbulb" style="margin-right: 0.5rem;"></i>
                            Consolidation Recommendation
                        </h6>
                        ${this.renderConsolidationRecommendation(pair.similarity)}
                    </div>
                </div>
                
                <!-- SQL Query Comparison -->
                ${this.renderSQLComparison(pair, index)}
                
                <div class="flex justify-end gap-md" style="margin-top: 1.5rem;">
                    <button onclick="migrationManager.exportPairAnalysis(${index})" class="btn btn-outline btn-sm">
                        <i class="fas fa-download"></i>
                        Export Analysis
                    </button>
                    <button onclick="migrationManager.openSQLModal('${pair.file1}', '${pair.file2}', ${index})" class="btn btn-primary btn-sm">
                        <i class="fas fa-code"></i>
                        View SQL Queries
                    </button>
                </div>
            </div>
        `).join('');
    }
    
    renderSimilarityMetrics(details) {
        const metrics = [
            { label: 'Data Source', value: details.data_source_similarity || 0, icon: 'fas fa-database' },
            { label: 'Filter Logic', value: details.filter_logic_similarity || 0, icon: 'fas fa-filter' },
            { label: 'Business Purpose', value: details.business_purpose_similarity || 0, icon: 'fas fa-bullseye' },
            { label: 'Calculations', value: details.calculation_similarity || 0, icon: 'fas fa-calculator' },
            { label: 'Parameters', value: details.parameter_similarity || 0, icon: 'fas fa-sliders-h' }
        ];
        
        return metrics.map(metric => `
            <div class="flex items-center justify-between mb-sm" style="padding: 0.5rem; background: var(--light-cream); border-radius: 8px; margin-bottom: 0.5rem;">
                <div class="flex items-center gap-sm">
                    <i class="${metric.icon}" style="color: var(--primary-orange); width: 16px;"></i>
                    <span style="font-size: 0.875rem;">${metric.label}</span>
                </div>
                <div class="flex items-center gap-sm">
                    <div class="progress-mini" style="width: 60px; height: 6px; background: rgba(0,0,0,0.1); border-radius: 3px; overflow: hidden;">
                        <div style="width: ${metric.value}%; height: 100%; background: ${this.getSimilarityColor(metric.value)}; transition: width 0.3s;"></div>
                    </div>
                    <span style="font-weight: 600; color: ${this.getSimilarityColor(metric.value)}; font-size: 0.875rem; min-width: 35px;">
                        ${Math.round(metric.value)}%
                    </span>
                </div>
            </div>
        `).join('');
    }
    
    renderConsolidationRecommendation(similarity) {
        let recommendation, icon, color, action;
        
        if (similarity >= 70) {
            recommendation = "Strong consolidation candidate";
            icon = "fas fa-check-circle";
            color = "#10B981";
            action = "Merge these reports into a single consolidated dashboard";
        } else if (similarity >= 40) {
            recommendation = "Review for potential consolidation";
            icon = "fas fa-exclamation-triangle";
            color = "#F59E0B";
            action = "Manual review recommended to determine consolidation feasibility";
        } else {
            recommendation = "Maintain as separate reports";
            icon = "fas fa-times-circle";
            color = "#6B7280";
            action = "Keep these reports separate due to distinct business logic";
        }
        
        return `
            <div style="background: var(--light-cream); padding: 1rem; border-radius: 8px; border-left: 4px solid ${color};">
                <div class="flex items-center gap-sm mb-sm">
                    <i class="${icon}" style="color: ${color};"></i>
                    <span class="font-semibold" style="color: ${color};">${recommendation}</span>
                </div>
                <div style="font-size: 0.875rem; color: var(--dark-gray);">
                    ${action}
                </div>
            </div>
        `;
    }
    
    renderSQLComparison(pair, pairIndex) {
        const file1Queries = pair.sql_queries?.file1_queries || [];
        const file2Queries = pair.sql_queries?.file2_queries || [];
        
        if (file1Queries.length === 0 && file2Queries.length === 0) {
            return `
                <div class="sql-comparison" style="background: var(--light-cream); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--light-blue);">
                    <h6 style="margin-bottom: 1rem; color: var(--navy-blue);">
                        <i class="fas fa-code" style="margin-right: 0.5rem;"></i>
                        SQL Query Analysis
                    </h6>
                    <div class="text-center" style="color: var(--dark-gray); padding: 2rem;">
                        <i class="fas fa-info-circle" style="font-size: 2rem; margin-bottom: 1rem; opacity: 0.5;"></i>
                        <p>No SQL queries found in the analyzed RDL files</p>
                    </div>
                </div>
            `;
        }
        
        return `
            <div class="sql-comparison" id="sql-comparison-${pairIndex}" style="background: var(--light-cream); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--light-blue);">
                <!-- SQL Query Summary Header -->
                <div class="sql-summary-header" style="margin-bottom: 1.5rem;">
                    <h6 style="margin-bottom: 1rem; color: var(--navy-blue);">
                        <i class="fas fa-code" style="margin-right: 0.5rem;"></i>
                        SQL Query Comparison - Manual Review
                    </h6>
                    
                    <div class="grid grid-2" style="gap: 1.5rem; margin-bottom: 1rem;">
                        <div style="background: var(--white); padding: 1rem; border-radius: 8px; border-left: 4px solid var(--primary-orange);">
                            <div class="flex items-center justify-between mb-sm">
                                <div class="flex items-center gap-sm">
                                    <i class="fas fa-file-alt" style="color: var(--primary-orange);"></i>
                                    <span class="font-semibold" style="color: var(--navy-blue);">${pair.file1}</span>
                                </div>
                                <span style="background: var(--primary-orange); color: var(--white); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600;">
                                    ${file1Queries.length} ${file1Queries.length === 1 ? 'Query' : 'Queries'}
                                </span>
                            </div>
                            <div style="font-size: 0.875rem; color: var(--dark-gray);">
                                Left side comparison queries
                            </div>
                        </div>
                        <div style="background: var(--white); padding: 1rem; border-radius: 8px; border-left: 4px solid var(--navy-blue);">
                            <div class="flex items-center justify-between mb-sm">
                                <div class="flex items-center gap-sm">
                                    <i class="fas fa-file-alt" style="color: var(--navy-blue);"></i>
                                    <span class="font-semibold" style="color: var(--navy-blue);">${pair.file2}</span>
                                </div>
                                <span style="background: var(--navy-blue); color: var(--white); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600;">
                                    ${file2Queries.length} ${file2Queries.length === 1 ? 'Query' : 'Queries'}
                                </span>
                            </div>
                            <div style="font-size: 0.875rem; color: var(--dark-gray);">
                                Right side comparison queries
                            </div>
                        </div>
                    </div>
                    
                    <!-- Toggle Button for SQL Details -->
                    <div class="text-center">
                        <button onclick="migrationManager.openSQLModal('${pairs[pairIndex].file1}', '${pairs[pairIndex].file2}', ${pairIndex})" 
                                id="sql-toggle-${pairIndex}" 
                                class="btn btn-outline btn-sm" 
                                style="background: var(--white); border: 2px solid var(--primary-orange); color: var(--primary-orange);">
                            <i class="fas fa-code" style="margin-right: 0.5rem;"></i>
                            View SQL Queries
                        </button>
                    </div>
                </div>
                
                <!-- OLD inline SQL details completely removed - now using overlay modal only -->
            </div>
        `;
    }
    
    // OLD renderDetailedSQLQueries method REMOVED - now using overlay modal only
    
    // OLD inline SQL rendering methods REMOVED - now using overlay modal only
    
    setupAnalysisTabsLogic() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.dataset.tab;
                
                // Update button states with new color scheme
                tabButtons.forEach(btn => {
                    btn.classList.remove('active');
                    btn.style.background = '#FAF8F2';
                    btn.style.color = '#4B4D4F';
                });
                
                button.classList.add('active');
                button.style.background = '#002677';
                button.style.color = '#FFFFFF';
                
                // Update content visibility
                tabContents.forEach(content => {
                    content.classList.add('hidden');
                });
                
                document.getElementById(targetTab).classList.remove('hidden');
            });
        });
    }
    
    openSQLModal(file1, file2, pairIndex) {
        console.log('Opening SQL Modal - NEW IMPLEMENTATION');
        console.log('DEBUG: Analysis results structure:', this.analysisResults);
        console.log('DEBUG: Pair index:', pairIndex);
        
        // PREVENT any inline content from showing
        this.hideInlineAnalysisContent();
        
        const modal = document.getElementById('sql-query-modal');
        const modalContent = document.getElementById('sql-modal-content');
        
        if (!modal || !modalContent) {
            console.error('Modal elements not found');
            return;
        }
        
        // Get the pair data
        const pair = this.analysisResults.similarity_pairs[pairIndex];
        console.log('DEBUG: Pair data:', pair);
        const file1Queries = pair.sql_queries?.file1_queries || [];
        const file2Queries = pair.sql_queries?.file2_queries || [];
        console.log('DEBUG: File1 queries:', file1Queries);
        console.log('DEBUG: File2 queries:', file2Queries);
        
        // Generate modal content with improved structure and better sizing
        modalContent.innerHTML = `
            <!-- Comparison Summary -->
            <div style="background: linear-gradient(135deg, var(--light-cream) 0%, var(--light-blue) 100%); padding: 1rem 1.5rem; border-bottom: 3px solid var(--primary-orange); flex-shrink: 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <h4 style="margin: 0; color: var(--navy-blue); font-size: 1.1rem; font-weight: 700;">
                        <i class="fas fa-exchange-alt" style="color: var(--primary-orange); margin-right: 0.5rem;"></i>
                        ${file1} ↔ ${file2}
                    </h4>
                    <div style="background: var(--white); padding: 0.5rem 1rem; border-radius: 8px; border: 2px solid var(--primary-orange);">
                        <span style="color: var(--navy-blue); font-weight: 700; font-size: 0.9rem;">${pair.similarity}% Overall Match</span>
                    </div>
                </div>
                <div style="display: flex; gap: 2rem; font-size: 0.85rem; color: var(--dark-gray);">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <i class="fas fa-file-code" style="color: var(--primary-orange);"></i>
                        <strong>${file1Queries.length}</strong> queries in ${file1}
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <i class="fas fa-file-code" style="color: var(--navy-blue);"></i>
                        <strong>${file2Queries.length}</strong> queries in ${file2}
                    </div>
                </div>
            </div>
            
            <!-- SQL Queries Side by Side with Improved Layout -->
            <div style="flex: 1; display: flex; overflow: hidden; min-height: 0; gap: clamp(4px, 1vw, 8px); background: var(--light-cream); padding: clamp(0.25rem, 1vw, 0.5rem); border-radius: 8px; margin: clamp(0.5rem, 1vw, 1rem);">
                <!-- Left Side -->
                <div style="flex: 1; display: flex; flex-direction: column; min-height: 0; background: var(--white); border-radius: 8px; overflow: hidden; border: 1px solid var(--light-blue); box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <div style="background: linear-gradient(135deg, var(--primary-orange) 0%, #FF7A47 100%); color: var(--white); padding: clamp(0.75rem, 2vw, 1.25rem); font-weight: 600; text-align: center; flex-shrink: 0; font-size: clamp(0.85rem, 1.8vw, 1.1rem); display: flex; align-items: center; justify-content: center; gap: 0.5rem; min-height: clamp(50px, 8vh, 60px);">
                        <i class="fas fa-file-alt" style="flex-shrink: 0;"></i>
                        <span style="flex: 1; min-width: 0; word-break: break-word; text-align: center; line-height: 1.2;" title="${file1}">${file1.length > 25 ? file1.substring(0, 22) + '...' : file1}</span>
                        <span style="background: rgba(255,255,255,0.25); padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.8em; flex-shrink: 0; font-weight: 700;">${file1Queries.length}</span>
                    </div>
                    <div style="flex: 1; overflow: auto; min-height: 0; background: var(--white);">
                        ${this.renderSQLQueriesInModal(file1Queries, 'left')}
                    </div>
                </div>
                
                <!-- Right Side -->
                <div style="flex: 1; display: flex; flex-direction: column; min-height: 0; background: var(--white); border-radius: 8px; overflow: hidden; border: 1px solid var(--light-blue); box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <div style="background: linear-gradient(135deg, var(--navy-blue) 0%, #1a3a8a 100%); color: var(--white); padding: clamp(0.75rem, 2vw, 1.25rem); font-weight: 600; text-align: center; flex-shrink: 0; font-size: clamp(0.85rem, 1.8vw, 1.1rem); display: flex; align-items: center; justify-content: center; gap: 0.5rem; min-height: clamp(50px, 8vh, 60px);">
                        <i class="fas fa-file-alt" style="flex-shrink: 0;"></i>
                        <span style="flex: 1; min-width: 0; word-break: break-word; text-align: center; line-height: 1.2;" title="${file2}">${file2.length > 25 ? file2.substring(0, 22) + '...' : file2}</span>
                        <span style="background: rgba(255,255,255,0.25); padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.8em; flex-shrink: 0; font-weight: 700;">${file2Queries.length}</span>
                    </div>
                    <div style="flex: 1; overflow: auto; min-height: 0; background: var(--white);">
                        ${this.renderSQLQueriesInModal(file2Queries, 'right')}
                    </div>
                </div>
            </div>
        `;
        
        // Lock body scroll to prevent conflicts
        document.body.style.overflow = 'hidden';
        
        // Show modal with proper display
        modal.style.display = 'flex';
        
        // Clean event listeners
        this.cleanupModalEvents();
        
        // Simple click outside to close
        this.modalClickHandler = (e) => {
            if (e.target === modal) {
                this.closeSQLModal();
            }
        };
        modal.addEventListener('click', this.modalClickHandler);
        
        // Escape key to close
        this.modalKeyHandler = (e) => {
            if (e.key === 'Escape') {
                this.closeSQLModal();
            }
        };
        document.addEventListener('keydown', this.modalKeyHandler);
    }
    
    hideInlineAnalysisContent() {
        // Hide any existing inline analysis content that might interfere
        const existingAnalysisSection = document.querySelector('[class*="sql"], [id*="sql"]:not(#sql-query-modal)');
        if (existingAnalysisSection && existingAnalysisSection.id !== 'sql-query-modal') {
            existingAnalysisSection.style.display = 'none';
        }
    }
    
    cleanupModalEvents() {
        const modal = document.getElementById('sql-query-modal');
        if (this.modalClickHandler) {
            modal.removeEventListener('click', this.modalClickHandler);
        }
        if (this.modalKeyHandler) {
            document.removeEventListener('keydown', this.modalKeyHandler);
        }
    }
    
    renderSQLQueriesInModal(queries, side) {
        if (queries.length === 0) {
            return `
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: var(--dark-gray); text-align: center; padding: clamp(1rem, 3vw, 2rem);">
                    <i class="fas fa-info-circle" style="font-size: clamp(2rem, 4vw, 3rem); margin-bottom: 1rem; opacity: 0.5; color: var(--primary-orange);"></i>
                    <h4 style="margin: 0.5rem 0; color: var(--navy-blue); font-size: clamp(1rem, 2vw, 1.25rem); font-weight: 600;">No SQL Queries Found</h4>
                    <p style="margin: 0; font-size: clamp(0.8rem, 1.5vw, 1rem); line-height: 1.5;">This RDL file doesn't contain extractable SQL queries.</p>
                </div>
            `;
        }
        
        return `
            <div style="padding: clamp(0.75rem, 2.5vw, 1.25rem); height: 100%; box-sizing: border-box; overflow-y: auto;">
                ${queries.map((query, index) => `
                    <div style="background: var(--light-cream); border: 1px solid var(--light-blue); border-radius: clamp(6px, 1vw, 8px); padding: clamp(1rem, 2.5vw, 1.25rem); margin-bottom: clamp(1rem, 2vw, 1.25rem); box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08); overflow: hidden;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem; gap: clamp(0.5rem, 1.5vw, 1rem); flex-wrap: wrap;">
                            <h5 style="margin: 0; color: var(--navy-blue); font-size: clamp(0.95rem, 2vw, 1.1rem); font-weight: 700; flex: 1; min-width: 0; line-height: 1.3;">
                                <i class="fas fa-database" style="color: var(--primary-orange); margin-right: 0.5rem; flex-shrink: 0;"></i>
                                <span style="word-break: break-word; display: inline-block;">Query ${index + 1}: ${(query.dataset_name || 'Unknown Dataset').length > 20 ? (query.dataset_name || 'Unknown Dataset').substring(0, 17) + '...' : (query.dataset_name || 'Unknown Dataset')}</span>
                            </h5>
                            <div style="background: var(--primary-orange); color: var(--white); padding: clamp(0.3rem, 1vw, 0.5rem) clamp(0.5rem, 1.5vw, 0.75rem); border-radius: 4px; font-size: clamp(0.7rem, 1.3vw, 0.8rem); font-weight: 600; white-space: nowrap; flex-shrink: 0;">
                                ${query.query_type || 'SQL'}
                            </div>
                        </div>
                        
                        <div style="background: var(--white); border: 1px solid var(--navy-blue); border-radius: 6px; overflow: hidden; margin-bottom: 0.75rem;">
                            <pre style="margin: 0; padding: clamp(0.75rem, 2vw, 1rem); font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: clamp(0.75rem, 1.4vw, 0.85rem); line-height: 1.5; overflow-x: auto; color: var(--navy-blue); max-height: clamp(250px, 35vh, 350px); overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; background: var(--white); border: none;">${this.formatSQLForDisplay(query.sql_query)}</pre>
                        </div>
                        
                        ${query.parameters && query.parameters.length > 0 ? `
                            <div style="margin-top: 0.5rem;">
                                <div style="font-size: clamp(0.7rem, 1.3vw, 0.8rem); color: var(--dark-gray); font-weight: 600; margin-bottom: 0.5rem;">
                                    <i class="fas fa-sliders-h" style="color: var(--primary-orange); margin-right: 0.25rem;"></i>
                                    Parameters:
                                </div>
                                <div style="display: flex; gap: 0.25rem; flex-wrap: wrap;">
                                    ${query.parameters.map(param => `
                                        <span style="background: var(--navy-blue); color: var(--white); padding: 0.25rem 0.5rem; border-radius: 4px; font-size: clamp(0.6rem, 1.1vw, 0.7rem); font-weight: 600; word-break: break-word;">
                                            ${param.length > 12 ? param.substring(0, 9) + '...' : param}
                                        </span>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    formatSQLForDisplay(sqlQuery) {
        if (!sqlQuery || typeof sqlQuery !== 'string') {
            return 'No query available';
        }
        
        // Basic SQL formatting
        return sqlQuery
            .replace(/\bSELECT\b/gi, 'SELECT')
            .replace(/\bFROM\b/gi, 'FROM')
            .replace(/\bWHERE\b/gi, 'WHERE')
            .replace(/\bJOIN\b/gi, 'JOIN')
            .replace(/\bINNER JOIN\b/gi, 'INNER JOIN')
            .replace(/\bLEFT JOIN\b/gi, 'LEFT JOIN')
            .replace(/\bRIGHT JOIN\b/gi, 'RIGHT JOIN')
            .replace(/\bORDER BY\b/gi, 'ORDER BY')
            .replace(/\bGROUP BY\b/gi, 'GROUP BY')
            .replace(/\bHAVING\b/gi, 'HAVING')
            .trim();
    }
    
    closeSQLModal() {
        console.log('Closing SQL Modal - NEW IMPLEMENTATION');
        
        const modal = document.getElementById('sql-query-modal');
        if (!modal) return;
        
        // Clean up all event listeners first
        this.cleanupModalEvents();
        
        // Hide modal
        modal.style.display = 'none';
        
        // Restore body scroll
        document.body.style.overflow = '';
        
        // Show any hidden inline content again (cleanup)
        this.showInlineAnalysisContent();
        
        // Clear modal content to free memory
        const modalContent = document.getElementById('sql-modal-content');
        if (modalContent) {
            modalContent.innerHTML = '';
        }
    }
    
    showInlineAnalysisContent() {
        // Re-show any inline analysis content that was hidden
        const hiddenAnalysisElements = document.querySelectorAll('[class*="sql"], [id*="sql"]:not(#sql-query-modal)');
        hiddenAnalysisElements.forEach(element => {
            if (element.id !== 'sql-query-modal' && element.style.display === 'none') {
                element.style.display = '';
            }
        });
    }
    
    startProgressPolling() {
        if (!this.currentJobId) return;
        
        console.log('[DEBUG] Starting progress polling for job:', this.currentJobId);
        
        // Initial progress setup
        this.updateProgressStatus({
            status: 'running',
            progress: 0,
            processed: 0,
            total: this.getSelectedFiles().length,
            message: 'Starting migration...'
        });
        
        // Start polling every 2 seconds
        this.progressInterval = setInterval(async () => {
            try {
                const response = await fetch(`/rdlmigration/api/progress/${this.currentJobId}`);
                const progressData = await response.json();
                
                console.log('[DEBUG] Progress update:', progressData);
                console.log('[DEBUG] Progress data breakdown - processed:', progressData.processed, 'total:', progressData.total, 'progress:', progressData.progress);
                
                if (progressData.error) {
                    console.error('Progress error:', progressData.error);
                    this.stopProgressPolling();
                    return;
                }
                
                this.updateProgressStatus(progressData);
                
                // Check if migration is complete
                if (progressData.status === 'completed' || progressData.status === 'error') {
                    this.stopProgressPolling();
                    
                    if (progressData.status === 'completed') {
                        await this.fetchMigrationResults();
                    }
                }
                
            } catch (error) {
                console.error('Progress polling error:', error);
            }
        }, 2000);
    }
    
    stopProgressPolling() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
            console.log('[DEBUG] Progress polling stopped');
        }
    }
    
    updateProgressStatus(progressData) {
        // Handle completion state values
        const isCompleted = progressData.status === 'completed';
        const totalFiles = this.getSelectedFiles().length;
        
        // For completed jobs, use final values
        let displayProcessed, displayTotal, displayProgress;
        if (isCompleted) {
            displayProcessed = totalFiles;
            displayTotal = totalFiles;
            displayProgress = 100;
        } else {
            displayProcessed = progressData.processed || 0;
            displayTotal = progressData.total || totalFiles;
            displayProgress = progressData.progress || 0;
        }
        
        // Update progress cards
        const progressStatus = document.getElementById('progress-status');
        progressStatus.innerHTML = `
            <div style="background: #FAF8F2; padding: 1rem; border-radius: 8px; text-align: center; border: 2px solid #D9F6FA;">
                <div style="color: #FF612B; font-size: 1.2rem; margin-bottom: 0.5rem;">
                    <i class="fas fa-tasks"></i>
                </div>
                <div style="font-size: 0.875rem; color: #4B4D4F; font-weight: 600; margin-bottom: 0.5rem;">Status</div>
                <div style="font-size: 1rem; font-weight: 700; color: #002677; text-transform: uppercase;">
                    ${progressData.status || 'Running'}
                </div>
            </div>
            
            <div style="background: #FAF8F2; padding: 1rem; border-radius: 8px; text-align: center; border: 2px solid #D9F6FA;">
                <div style="color: #FF612B; font-size: 1.2rem; margin-bottom: 0.5rem;">
                    <i class="fas fa-file-alt"></i>
                </div>
                <div style="font-size: 0.875rem; color: #4B4D4F; font-weight: 600; margin-bottom: 0.5rem;">Files Processed</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #002677;">
                    ${displayProcessed} / ${displayTotal}
                </div>
            </div>
            
            <div style="background: #FAF8F2; padding: 1rem; border-radius: 8px; text-align: center; border: 2px solid #D9F6FA;">
                <div style="color: #FF612B; font-size: 1.2rem; margin-bottom: 0.5rem;">
                    <i class="fas fa-clock"></i>
                </div>
                <div style="font-size: 0.875rem; color: #4B4D4F; font-weight: 600; margin-bottom: 0.5rem;">Time Remaining</div>
                <div style="font-size: 1rem; font-weight: 700; color: #002677;">
                    ${isCompleted ? 'Completed' : this.formatETA(progressData.eta_seconds)}
                </div>
            </div>
            
            <div style="background: #FAF8F2; padding: 1rem; border-radius: 8px; text-align: center; border: 2px solid #D9F6FA;">
                <div style="color: #FF612B; font-size: 1.2rem; margin-bottom: 0.5rem;">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <div style="font-size: 0.875rem; color: #4B4D4F; font-weight: 600; margin-bottom: 0.5rem;">Errors</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: ${(progressData.errors || 0) > 0 ? '#FF612B' : '#002677'};">
                    ${progressData.errors || 0}
                </div>
            </div>
        `;
        
        // Update progress bar
        document.getElementById('progress-percentage').textContent = `${displayProgress}%`;
        document.getElementById('progress-bar').style.width = `${displayProgress}%`;
        
        // Update progress details
        document.getElementById('files-processed').textContent = 
            `${displayProcessed} of ${displayTotal} files processed`;
        document.getElementById('eta-display').textContent = isCompleted ? 'Completed' : this.formatETA(progressData.eta_seconds);
        
        // Update current activity
        const currentFile = document.getElementById('current-file');
        if (progressData.status === 'completed') {
            currentFile.innerHTML = `
                <i class="fas fa-check-circle" style="color: #10B981; margin-right: 0.5rem;"></i>
                Migration completed successfully!
            `;
        } else if (progressData.status === 'error') {
            currentFile.innerHTML = `
                <i class="fas fa-exclamation-circle" style="color: #EF4444; margin-right: 0.5rem;"></i>
                Migration failed with errors
            `;
        } else if (progressData.message) {
            currentFile.innerHTML = `
                <i class="fas fa-cog fa-spin" style="color: #FF612B; margin-right: 0.5rem;"></i>
                ${progressData.message}
            `;
        }
    }
    
    formatETA(etaSeconds) {
        if (!etaSeconds || etaSeconds <= 0) return 'Calculating...';
        
        const minutes = Math.floor(etaSeconds / 60);
        const seconds = Math.floor(etaSeconds % 60);
        
        if (minutes > 0) {
            return `${minutes}m ${seconds}s`;
        } else {
            return `${seconds}s`;
        }
    }
    
    async fetchMigrationResults() {
        try {
            console.log('[DEBUG] Fetching migration results for job:', this.currentJobId);
            
            const response = await fetch(`/rdlmigration/api/results/${this.currentJobId}`);
            const resultsData = await response.json();
            
            console.log('[DEBUG] Migration results:', resultsData);
            
            if (resultsData.error) {
                console.error('Results error:', resultsData.error);
                return;
            }
            
            await this.displayMigrationResults(resultsData);
            
        } catch (error) {
            console.error('Error fetching migration results:', error);
        }
    }
    
    async displayMigrationResults(resultsData) {
        console.log('[DEBUG] Displaying migration results, showing section...');
        const resultsSection = document.getElementById('migration-results-section');
        const resultsContent = document.getElementById('migration-results-content');
        
        console.log('[DEBUG] Results section element:', resultsSection);
        console.log('[DEBUG] Results content element:', resultsContent);
        
        // Fetch detailed file list
        try {
            // Remove _migration suffix for results directory lookup
            const resultJobId = this.currentJobId.replace('_migration', '');
            const filesResponse = await fetch(`/rdlmigration/api/results/${resultJobId}/files`);
            const filesData = await filesResponse.json();
            
            console.log('[DEBUG] Generated files:', filesData);
            console.log('[DEBUG] Files array:', filesData.files);
            console.log('[DEBUG] Files count:', filesData.files ? filesData.files.length : 'undefined');
            
            if (filesData.files && filesData.files.length > 0) {
                console.log('[DEBUG] Rendering files with renderGeneratedFiles');
                
                // Store available files for context switching
                this.availableFiles = filesData.files;
                
                const renderedHTML = this.renderGeneratedFiles(filesData.files);
                console.log('[DEBUG] Rendered HTML length:', renderedHTML.length);
                resultsContent.innerHTML = renderedHTML;
            } else {
                console.log('[DEBUG] No files found, showing no files message');
                resultsContent.innerHTML = `
                    <div style="text-align: center; padding: 2rem; color: #4B4D4F;">
                        <i class="fas fa-info-circle" style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;"></i>
                        <h4 style="margin: 0.5rem 0; color: #002677;">No files generated</h4>
                        <p style="margin: 0;">Migration completed but no output files were created.</p>
                    </div>
                `;
            }
            
            console.log('[DEBUG] Showing migration results section...');
            resultsSection.classList.remove('hidden');
            
            // Scroll to results section
            resultsSection.scrollIntoView({ 
                behavior: 'smooth',
                block: 'start'
            });
            
        } catch (error) {
            console.error('Error fetching file list:', error);
            resultsContent.innerHTML = `
                <div style="text-align: center; padding: 2rem; color: #4B4D4F;">
                    <i class="fas fa-exclamation-triangle" style="font-size: 3rem; margin-bottom: 1rem; color: #FF612B;"></i>
                    <h4 style="margin: 0.5rem 0; color: #002677;">Error Loading Results</h4>
                    <p style="margin: 0;">Unable to fetch generated files. Please check the migration log.</p>
                </div>
            `;
            resultsSection.classList.remove('hidden');
        }
    }
    
    renderGeneratedFiles(files) {
        // Group files by type
        const filesByType = files.reduce((acc, file) => {
            const ext = file.type || '.unknown';
            if (!acc[ext]) acc[ext] = [];
            acc[ext].push(file);
            return acc;
        }, {});
        
        let html = '';
        
        // Render each file type group
        Object.keys(filesByType).forEach(fileType => {
            const typeFiles = filesByType[fileType];
            const typeIcon = this.getFileTypeIcon(fileType);
            const typeLabel = this.getFileTypeLabel(fileType);
            
            html += `
                <div style="margin-bottom: 2rem;">
                    <h5 style="margin: 0 0 1rem 0; color: #002677; font-size: 1.1rem; font-weight: 600; display: flex; align-items: center;">
                        <i class="${typeIcon}" style="margin-right: 0.5rem; color: #FF612B;"></i>
                        ${typeLabel} (${typeFiles.length} files)
                    </h5>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem;">
                        ${typeFiles.map(file => this.renderFileCard(file)).join('')}
                    </div>
                </div>
            `;
        });
        
        // Add download all button
        html += `
            <div style="text-align: center; margin-top: 2rem; padding-top: 2rem; border-top: 2px solid #D9F6FA;">
                <button onclick="migrationManager.downloadAllResults()" style="background: #002677; color: #FFFFFF; border: none; padding: 1rem 2rem; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 1rem;">
                    <i class="fas fa-download" style="margin-right: 0.5rem;"></i>
                    Download All Results (ZIP)
                </button>
            </div>
        `;
        
        return html;
    }
    
    renderFileCard(file) {
        return `
            <div style="background: #FAF8F2; border: 2px solid #D9F6FA; border-radius: 8px; padding: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
                    <h6 style="margin: 0; color: #002677; font-weight: 600; font-size: 0.9rem; line-height: 1.2;">
                        ${file.name}
                    </h6>
                    <span style="background: #D9F6FA; color: #002677; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">
                        ${this.formatFileSize(file.size)}
                    </span>
                </div>
                
                <div style="margin-bottom: 1rem; font-size: 0.875rem; color: #4B4D4F;">
                    Path: ${file.path}
                </div>
                
                <div style="display: flex; gap: 0.5rem;">
                    <button onclick="migrationManager.previewFile('${file.path}')" style="background: #FF612B; color: #FFFFFF; border: none; padding: 0.5rem 1rem; border-radius: 6px; font-size: 0.875rem; font-weight: 600; cursor: pointer; flex: 1;">
                        <i class="fas fa-eye" style="margin-right: 0.25rem;"></i>
                        Preview
                    </button>
                    <button onclick="migrationManager.downloadFile('${file.path}', '${file.name}')" style="background: #FFFFFF; color: #002677; border: 2px solid #002677; padding: 0.5rem 1rem; border-radius: 6px; font-size: 0.875rem; font-weight: 600; cursor: pointer; flex: 1;">
                        <i class="fas fa-download" style="margin-right: 0.25rem;"></i>
                        Download
                    </button>
                </div>
            </div>
        `;
    }
    
    getFileTypeIcon(fileType) {
        const icons = {
            '.m': 'fas fa-code',        // Power Query
            '.dax': 'fas fa-function',  // DAX
            '.md': 'fas fa-book',       // Migration Guide
            '.json': 'fas fa-file-code',
            '.txt': 'fas fa-file-alt',
            '.py': 'fas fa-code'
        };
        return icons[fileType] || 'fas fa-file';
    }
    
    getFileTypeLabel(fileType) {
        const labels = {
            '.m': 'Power Query Files',
            '.dax': 'DAX Files',
            '.md': 'Migration Guides',
            '.json': 'JSON Files',
            '.txt': 'Text Files',
            '.py': 'Python Files'
        };
        return labels[fileType] || 'Other Files';
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }
    
    async downloadAllResults() {
        try {
            const jobId = this.currentJobId.replace('_migration', '');
            
            // create temp download link
            const downloadLink = document.createElement('a');
            downloadLink.href = `/rdlmigration/api/download/${jobId}`;
            downloadLink.download = `migration_results_${jobId}.zip`;
            
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            
            this.showToast('success', 'Download Started', 'Downloading all migration results as ZIP file');
        } catch (error) {
            console.error('Download error:', error);
            this.showToast('error', 'Download Failed', 'Unable to download results');
        }
    }
    
    async downloadFile(filePath, fileName) {
        try {
            const jobId = this.currentJobId.replace('_migration', '');
            
            // trigger file download
            const fileLink = document.createElement('a');
            fileLink.href = `/rdlmigration/api/results/${jobId}/download/${filePath}`;
            fileLink.download = fileName;
            
            document.body.appendChild(fileLink);
            fileLink.click();
            document.body.removeChild(fileLink);
            
            this.showToast('success', 'Download Started', `Downloading ${fileName}`);
        } catch (error) {
            console.error('Download error:', error);
            this.showToast('error', 'Download Failed', `Unable to download ${fileName}`);
        }
    }
    
    async previewFile(filePath, contextFile = null) {
        try {
            const jobId = this.currentJobId.replace('_migration', '');
            
            // build URL with context param if needed
            let previewUrl = `/rdlmigration/api/results/${jobId}/preview/${filePath}`;
            if (contextFile) {
                previewUrl += `?context_file=${encodeURIComponent(contextFile)}`;
            }
            
            const response = await fetch(previewUrl);
            const previewData = await response.json();
            
            if (previewData.error) {
                this.showToast('error', 'Preview Failed', previewData.error);
                return;
            }
            
            // Check if this is a batch migration guide
            const isBatchGuide = filePath.includes('batch_migration_guide.md');
            
            this.showFilePreviewModal(previewData, filePath, isBatchGuide);
            
        } catch (error) {
            console.error('Preview error:', error);
            this.showToast('error', 'Preview Failed', 'Unable to preview file');
        }
    }
    
    showFilePreviewModal(previewData, filePath, isBatchGuide = false) {
        // Create preview modal 
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
            background: rgba(0,0,0,0.5); z-index: 1000; display: flex; 
            justify-content: center; align-items: center;
        `;
        
        // Show info banner for batch guides
        let infoBanner = '';
        if (isBatchGuide) {
            infoBanner = `
                <div style="background: #EFF6FF; padding: 1rem; border-bottom: 1px solid #DBEAFE; border-left: 4px solid #3B82F6;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <div style="color: #3B82F6; font-size: 1.25rem;">🚀</div>
                        <div>
                            <div style="font-weight: 600; color: #1E40AF;">Token-Optimized Batch Migration Guide</div>
                            <div style="color: #1E40AF; font-size: 0.875rem;">This guide contains all file-specific details and migration steps for your entire batch</div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        modal.innerHTML = `
            <div style="background: #FFFFFF; width: 90%; max-width: 900px; height: 85%; border-radius: 12px; overflow: hidden; display: flex; flex-direction: column;">
                <div style="background: #002677; color: #FFFFFF; padding: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin: 0; font-size: 1.25rem; font-weight: 600;">
                        <i class="fas fa-file-alt" style="margin-right: 0.5rem;"></i>
                        ${isBatchGuide ? '🚀 Smart Migration Guide' : `File Preview: ${filePath}`}
                    </h3>
                    <button onclick="this.closest('.modal-overlay').remove()" style="background: none; border: none; color: #FFFFFF; font-size: 1.5rem; cursor: pointer;">×</button>
                </div>
                ${infoBanner}
                <div id="previewContent" style="flex: 1; overflow: auto; padding: 1.5rem;">
                    ${this.renderFilePreview(previewData)}
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Close on click outside
        modal.onclick = (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        };
    }
    
    async switchGuideContext(contextFile, filePath) {
        try {
            // Show loading indicator
            const previewContent = document.getElementById('previewContent');
            if (previewContent) {
                previewContent.innerHTML = `
                    <div style="text-align: center; padding: 2rem;">
                        <div class="spinner" style="display: inline-block; width: 32px; height: 32px; border: 3px solid #E5E7EB; border-top: 3px solid #002677; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                        <p style="margin-top: 1rem; color: #6B7280;">Loading context-specific guide...</p>
                    </div>
                `;
            }
            
            // Fetch with new context
            const previewData = await this.getContextualPreview(filePath, contextFile);
            
            // Update content
            if (previewContent) {
                previewContent.innerHTML = this.renderFilePreview(previewData);
            }
            
        } catch (error) {
            console.error('Context switch error:', error);
            const previewContent = document.getElementById('previewContent');
            if (previewContent) {
                previewContent.innerHTML = `
                    <div style="text-align: center; padding: 2rem; color: #EF4444;">
                        <i class="fas fa-exclamation-triangle" style="font-size: 2rem; margin-bottom: 1rem;"></i>
                        <p>Failed to load contextual guide. Please try again.</p>
                    </div>
                `;
            }
        }
    }
    
    async getContextualPreview(filePath, contextFile) {
        const resultJobId = this.currentJobId.replace('_migration', '');
        let previewUrl = `/rdlmigration/api/results/${resultJobId}/preview/${filePath}`;
        if (contextFile) {
            previewUrl += `?context_file=${encodeURIComponent(contextFile)}`;
        }
        
        const response = await fetch(previewUrl);
        return await response.json();
    }
    
    renderFilePreview(previewData) {
        if (previewData.type === 'text') {
            return `
                <pre style="background: #FAF8F2; padding: 1rem; border-radius: 8px; overflow-x: auto; font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 0.875rem; line-height: 1.5; color: #002677; margin: 0;">${previewData.content}</pre>
            `;
        } else if (previewData.type === 'html') {
            return previewData.content;
        } else if (previewData.type === 'image') {
            return `<img src="${previewData.content}" style="max-width: 100%; height: auto;" alt="File preview">`;
        } else {
            return `
                <div style="text-align: center; padding: 2rem; color: #4B4D4F;">
                    <i class="fas fa-file" style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;"></i>
                    <h4 style="margin: 0.5rem 0; color: #002677;">Preview Not Available</h4>
                    <p style="margin: 0;">This file type cannot be previewed. Please download to view.</p>
                </div>
            `;
        }
    }
    
    generateDemoAnalysisResults(selectedFiles) {
        // Generate demo data for testing when API is not available
        const pairs = [];
        
        for (let i = 0; i < selectedFiles.length; i++) {
            for (let j = i + 1; j < selectedFiles.length; j++) {
                const similarity = Math.floor(Math.random() * 100);
                
                pairs.push({
                    file1: selectedFiles[i].name.replace(/^[a-f0-9-]+_/, ''),
                    file2: selectedFiles[j].name.replace(/^[a-f0-9-]+_/, ''),
                    similarity: similarity,
                    details: {
                        data_source_similarity: Math.floor(Math.random() * 100),
                        filter_logic_similarity: Math.floor(Math.random() * 100),
                        business_purpose_similarity: Math.floor(Math.random() * 100),
                        calculation_similarity: Math.floor(Math.random() * 100),
                        parameter_similarity: Math.floor(Math.random() * 100)
                    },
                    sql_queries: {
                        file1_queries: [
                            `SELECT CustomerID, OrderDate, TotalAmount 
FROM Orders o 
INNER JOIN Customers c ON o.CustomerID = c.CustomerID 
WHERE OrderDate >= '2024-01-01' 
AND Status = 'Completed'
ORDER BY OrderDate DESC`
                        ],
                        file2_queries: [
                            `SELECT o.CustomerID, o.OrderDate, o.TotalAmount, c.CustomerName
FROM Orders o 
JOIN Customers c ON o.CustomerID = c.CustomerID 
WHERE o.OrderDate >= '2024-01-01' 
AND o.Status = 'Completed'
ORDER BY o.OrderDate DESC`
                        ]
                    }
                });
            }
        }
        
        return { pairs, total_files: selectedFiles.length };
    }
    
    // OLD toggleSQLDetails method REMOVED to prevent conflicts
    // All SQL viewing now uses openSQLModal() overlay implementation
    
    // OLD toggleSQLComparison method REMOVED to prevent conflicts
    
    exportPairAnalysis(pairIndex) {
        // Placeholder for export functionality
        alert(`Export analysis for pair ${pairIndex + 1} - Feature coming soon!`);
    }
    
    getSelectedFiles() {
        const checkboxes = document.querySelectorAll('.file-checkbox:checked');
        return Array.from(checkboxes).map(cb => {
            const fileId = parseInt(cb.dataset.fileId);
            return this.files.find(f => f.id === fileId);
        }).filter(Boolean);
    }
    
    updateMigrationPreview() {
        const selectedMode = document.querySelector('input[name="migration-mode"]:checked')?.value;
        const selectedFiles = this.getSelectedFiles();
        const summaryElement = document.getElementById('migration-summary');
        
        if (selectedFiles.length === 0) {
            summaryElement.innerHTML = `
                <div class="text-center" style="color: var(--dark-gray);">
                    Select files to see migration preview
                </div>
            `;
            return;
        }
        
        if (selectedMode === 'individual') {
            summaryElement.innerHTML = `
                <div class="grid grid-2" style="gap: 1.5rem;">
                    <div>
                        <div class="font-semibold mb-sm">Mode: Individual</div>
                        <div style="font-size: 0.875rem; color: var(--dark-gray);">
                            ${selectedFiles.length} separate Power BI reports
                        </div>
                    </div>
                    <div>
                        <div class="font-semibold mb-sm">Output Files</div>
                        <div style="font-size: 0.875rem; color: var(--dark-gray);">
                            Power Query (.m) and DAX (.dax) files
                        </div>
                    </div>
                </div>
            `;
        } else {
            summaryElement.innerHTML = `
                <div class="grid grid-2" style="gap: 1.5rem;">
                    <div>
                        <div class="font-semibold mb-sm">Mode: Consolidated</div>
                        <div style="font-size: 0.875rem; color: var(--dark-gray);">
                            Smart consolidation based on similarity
                        </div>
                    </div>
                    <div>
                        <div class="font-semibold mb-sm">Estimated Output</div>
                        <div style="font-size: 0.875rem; color: var(--dark-gray);">
                            ~${Math.ceil(selectedFiles.length * 0.7)} consolidated reports
                        </div>
                    </div>
                </div>
            `;
        }
    }
    
    async runMigration() {
        const selectedFiles = this.getSelectedFiles();
        if (selectedFiles.length === 0) {
            this.showToast('warning', 'No files selected', 'Please select files to migrate');
            return;
        }
        
        const migrationMode = document.querySelector('input[name="migration-mode"]:checked')?.value;
        const enableConsolidation = document.getElementById('enable-consolidation').checked;
        const deepAnalysis = document.getElementById('deep-analysis').checked;
        
        try {
            // Prepare JSON data for migration
            const migrationData = {
                files: selectedFiles.map(fileData => ({
                    name: fileData.name,
                    path: fileData.path,
                    size: fileData.size
                })),
                mode: migrationMode || 'individual',
                consolidate: enableConsolidation,
                deep_analysis: deepAnalysis
            };
            
            console.log('[DEBUG] Starting migration with data:', migrationData);
            
            const response = await fetch('/rdlmigration/api/migrate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(migrationData)
            });
            
            const result = await response.json();
            console.log('[DEBUG] Migration response:', result);
            
            if (result.success) {
                this.currentJobId = result.job_id;
                this.showProgressDashboard();
                this.startProgressPolling();
                this.showToast('success', 'Migration Started', `Job ID: ${result.job_id}`);
            } else {
                this.showToast('error', 'Migration Failed', result.error || result.message || 'Unknown error');
            }
        } catch (error) {
            console.error('Migration error:', error);
            this.showToast('error', 'Migration Failed', 'Please try again');
        }
    }
    
    showProgressDashboard() {
        document.getElementById('progress-dashboard').classList.remove('hidden');
        document.getElementById('job-id-display').innerHTML = `
            <i class="fas fa-clock"></i>
            <span>Job: ${this.currentJobId}</span>
        `;
        
        // Scroll to progress dashboard
        document.getElementById('progress-dashboard').scrollIntoView({ 
            behavior: 'smooth' 
        });
    }
    
    showToast(type, title, message) {
        // Simple alert for now - in a real implementation, this would show a proper toast
        alert(`${title}: ${message}`);
    }
    
    showSuccessNotification(message) {
        // Create and show a temporary success notification
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: linear-gradient(135deg, #10B981, #34D399);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            transform: translateX(400px);
            transition: transform 0.3s ease;
        `;
        
        notification.innerHTML = `
            <i class="fas fa-check-circle"></i>
            ${message}
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        // Animate out and remove
        setTimeout(() => {
            notification.style.transform = 'translateX(400px)';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
    
    getSimilarityColor(score) {
        // Return color based on similarity score
        if (score >= 80) {
            return '#10B981'; // Green for high similarity
        } else if (score >= 60) {
            return '#F59E0B'; // Orange for medium-high similarity
        } else if (score >= 40) {
            return '#EF4444'; // Red for medium similarity
        } else {
            return '#6B7280'; // Gray for low similarity
        }
    }
    
    renderAdvancedSimilarityMatrix(analysisData) {
        // Advanced similarity matrix for 50+ files
        const files = [...new Set([
            ...analysisData.pairs.map(p => p.file1),
            ...analysisData.pairs.map(p => p.file2)
        ])];
        
        if (files.length <= 10) {
            return this.renderSimilarityPairs(analysisData.pairs, 'all');
        }
        
        return `
            <div class="similarity-matrix-container">
                <div class="matrix-header">
                    <h3><i class="fas fa-th"></i> Similarity Matrix (${files.length} files)</h3>
                    <div class="matrix-controls">
                        <button onclick="migrationManager.toggleMatrixView()" class="btn btn-outline btn-sm">
                            <i class="fas fa-eye"></i> Toggle View
                        </button>
                        <button onclick="migrationManager.exportMatrix()" class="btn btn-outline btn-sm">
                            <i class="fas fa-download"></i> Export Matrix
                        </button>
                    </div>
                </div>
                
                <div class="matrix-legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #10B981;"></div>
                        <span>High (80%+)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #F59E0B;"></div>
                        <span>Medium (40-79%)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #6B7280;"></div>
                        <span>Low (<40%)</span>
                    </div>
                </div>
                
                <div class="matrix-scroll-container">
                    ${this.renderMatrixGrid(files, analysisData.pairs)}
                </div>
                
                <div class="matrix-summary">
                    <div class="summary-stat">
                        <span class="stat-label">High Similarity Pairs:</span>
                        <span class="stat-value">${analysisData.pairs.filter(p => p.similarity >= 80).length}</span>
                    </div>
                    <div class="summary-stat">
                        <span class="stat-label">Medium Similarity Pairs:</span>
                        <span class="stat-value">${analysisData.pairs.filter(p => p.similarity >= 40 && p.similarity < 80).length}</span>
                    </div>
                    <div class="summary-stat">
                        <span class="stat-label">Consolidation Opportunities:</span>
                        <span class="stat-value">${this.countConsolidationOpportunities(analysisData.pairs)}</span>
                    </div>
                </div>
            </div>
            
            <style>
                .similarity-matrix-container {
                    background: white;
                    border-radius: 12px;
                    padding: 1.5rem;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    margin-bottom: 2rem;
                }
                
                .matrix-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1rem;
                    padding-bottom: 1rem;
                    border-bottom: 1px solid #E5E7EB;
                }
                
                .matrix-controls {
                    display: flex;
                    gap: 0.5rem;
                }
                
                .matrix-legend {
                    display: flex;
                    gap: 1rem;
                    margin-bottom: 1rem;
                    padding: 0.75rem;
                    background: #F9FAFB;
                    border-radius: 8px;
                }
                
                .legend-item {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-size: 0.875rem;
                }
                
                .legend-color {
                    width: 16px;
                    height: 16px;
                    border-radius: 4px;
                }
                
                .matrix-scroll-container {
                    overflow: auto;
                    max-height: 500px;
                    border: 1px solid #E5E7EB;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                }
                
                .matrix-grid {
                    display: grid;
                    gap: 1px;
                    background: #E5E7EB;
                    min-width: max-content;
                }
                
                .matrix-cell {
                    background: white;
                    padding: 0.5rem;
                    text-align: center;
                    font-size: 0.75rem;
                    min-width: 80px;
                    min-height: 40px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                
                .matrix-cell:hover {
                    transform: scale(1.05);
                    z-index: 10;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                }
                
                .matrix-cell.header {
                    background: #F3F4F6;
                    font-weight: 600;
                    writing-mode: vertical-rl;
                    text-orientation: mixed;
                }
                
                .matrix-cell.diagonal {
                    background: #F9FAFB;
                    color: #9CA3AF;
                }
                
                .matrix-summary {
                    display: flex;
                    gap: 2rem;
                    padding: 1rem;
                    background: #F9FAFB;
                    border-radius: 8px;
                }
                
                .summary-stat {
                    display: flex;
                    flex-direction: column;
                    gap: 0.25rem;
                }
                
                .stat-label {
                    font-size: 0.875rem;
                    color: #6B7280;
                }
                
                .stat-value {
                    font-size: 1.25rem;
                    font-weight: 700;
                    color: #1F2937;
                }
            </style>
        `;
    }
    
    renderMatrixGrid(files, pairs) {
        // Create similarity lookup
        const similarityLookup = {};
        pairs.forEach(pair => {
            const key1 = `${pair.file1}:${pair.file2}`;
            const key2 = `${pair.file2}:${pair.file1}`;
            similarityLookup[key1] = pair.similarity;
            similarityLookup[key2] = pair.similarity;
        });
        
        const gridSize = files.length + 1;
        
        let gridHTML = `<div class="matrix-grid" style="grid-template-columns: repeat(${gridSize}, 1fr);">`;
        
        // Header row
        gridHTML += '<div class="matrix-cell header"></div>'; // Top-left corner
        files.forEach(file => {
            const shortName = file.length > 12 ? file.substring(0, 12) + '...' : file;
            gridHTML += `<div class="matrix-cell header" title="${file}">${shortName}</div>`;
        });
        
        // Data rows
        files.forEach((rowFile, i) => {
            // Row header
            const shortName = rowFile.length > 12 ? rowFile.substring(0, 12) + '...' : rowFile;
            gridHTML += `<div class="matrix-cell header" title="${rowFile}">${shortName}</div>`;
            
            // Data cells
            files.forEach((colFile, j) => {
                if (i === j) {
                    // Diagonal cell
                    gridHTML += '<div class="matrix-cell diagonal">—</div>';
                } else {
                    const similarity = similarityLookup[`${rowFile}:${colFile}`] || 0;
                    const color = this.getSimilarityColor(similarity);
                    const textColor = similarity > 60 ? 'white' : 'black';
                    
                    gridHTML += `
                        <div class="matrix-cell" 
                             style="background-color: ${color}; color: ${textColor};"
                             onclick="migrationManager.showPairDetails('${rowFile}', '${colFile}', ${similarity})"
                             title="Similarity: ${similarity.toFixed(1)}%">
                            ${similarity > 0 ? similarity.toFixed(0) + '%' : '—'}
                        </div>
                    `;
                }
            });
        });
        
        gridHTML += '</div>';
        return gridHTML;
    }
    
    countConsolidationOpportunities(pairs) {
        // Count potential consolidation clusters
        const highSimilarityPairs = pairs.filter(p => p.similarity >= 75);
        const fileGroups = new Set();
        
        highSimilarityPairs.forEach(pair => {
            const key = [pair.file1, pair.file2].sort().join(':');
            fileGroups.add(key);
        });
        
        return fileGroups.size;
    }
    
    toggleMatrixView() {
        const container = document.querySelector('.matrix-scroll-container');
        if (container.style.maxHeight === '500px') {
            container.style.maxHeight = 'none';
            container.style.height = 'auto';
        } else {
            container.style.maxHeight = '500px';
        }
    }
    
    exportMatrix() {
        // Export similarity matrix data
        if (!this.results || !this.results.pairs) {
            this.showNotification('No analysis data to export', 'error');
            return;
        }
        
        const csvData = this.convertMatrixToCSV(this.results.pairs);
        const blob = new Blob([csvData], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `similarity_matrix_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showNotification('Matrix exported successfully');
    }
    
    convertMatrixToCSV(pairs) {
        // Get all unique files
        const files = [...new Set([
            ...pairs.map(p => p.file1),
            ...pairs.map(p => p.file2)
        ])];
        
        // Create similarity lookup
        const similarityLookup = {};
        pairs.forEach(pair => {
            similarityLookup[`${pair.file1}:${pair.file2}`] = pair.similarity;
            similarityLookup[`${pair.file2}:${pair.file1}`] = pair.similarity;
        });
        
        // Build CSV
        let csv = 'File,' + files.join(',') + '\n';
        
        files.forEach(rowFile => {
            let row = `"${rowFile}",`;
            const values = files.map(colFile => {
                if (rowFile === colFile) return '100';
                return (similarityLookup[`${rowFile}:${colFile}`] || 0).toFixed(1);
            });
            row += values.join(',');
            csv += row + '\n';
        });
        
        return csv;
    }
    
    showPairDetails(file1, file2, similarity) {
        // Show detailed comparison for a specific pair
        if (!this.results || !this.results.pairs) return;
        
        const pair = this.results.pairs.find(p => 
            (p.file1 === file1 && p.file2 === file2) || 
            (p.file1 === file2 && p.file2 === file1)
        );
        
        if (pair) {
            this.openSQLModal(file1, file2, 0); // Reuse existing modal
        } else {
            this.showNotification(`No detailed analysis available for ${file1} ↔ ${file2}`, 'info');
        }
    }
    
    renderAIInsights(analysisData) {
        // Render AI-powered insights and recommendations
        if (!analysisData.ai_insights) {
            return '';
        }
        
        const insights = analysisData.ai_insights;
        
        return `
            <div class="ai-insights-container modern-card">
                <div class="insights-header">
                    <h3><i class="fas fa-brain"></i> AI-Powered Insights</h3>
                    <div class="ai-badge">
                        <i class="fas fa-robot"></i>
                        AI Analysis
                    </div>
                </div>
                
                <div class="insights-grid">
                    <div class="insight-card">
                        <div class="insight-icon">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <div class="insight-content">
                            <h4>Consolidation Potential</h4>
                            <div class="insight-value">${insights.consolidation_score || 'N/A'}%</div>
                            <p class="insight-description">
                                Based on query similarity and business logic analysis
                            </p>
                        </div>
                    </div>
                    
                    <div class="insight-card">
                        <div class="insight-icon">
                            <i class="fas fa-coins"></i>
                        </div>
                        <div class="insight-content">
                            <h4>Token Efficiency</h4>
                            <div class="insight-value">${insights.token_savings || 'N/A'}%</div>
                            <p class="insight-description">
                                Cost optimization through smart pre-filtering
                            </p>
                        </div>
                    </div>
                    
                    <div class="insight-card">
                        <div class="insight-icon">
                            <i class="fas fa-clock"></i>
                        </div>
                        <div class="insight-content">
                            <h4>Time Savings</h4>
                            <div class="insight-value">${insights.time_savings || 'N/A'} hrs</div>
                            <p class="insight-description">
                                Estimated migration time reduction
                            </p>
                        </div>
                    </div>
                </div>
                
                <div class="ai-recommendations">
                    <h4><i class="fas fa-lightbulb"></i> Smart Recommendations</h4>
                    <div class="recommendations-list">
                        ${(insights.recommendations || []).map(rec => `
                            <div class="recommendation-item">
                                <div class="rec-priority ${rec.priority.toLowerCase()}">
                                    ${rec.priority}
                                </div>
                                <div class="rec-content">
                                    <strong>${rec.title}</strong>
                                    <p>${rec.description}</p>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
            
            <style>
                .ai-insights-container {
                    margin: 2rem 0;
                    padding: 1.5rem;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 16px;
                }
                
                .insights-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1.5rem;
                }
                
                .ai-badge {
                    background: rgba(255,255,255,0.2);
                    padding: 0.5rem 1rem;
                    border-radius: 20px;
                    font-size: 0.875rem;
                    backdrop-filter: blur(10px);
                }
                
                .insights-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2rem;
                }
                
                .insight-card {
                    background: rgba(255,255,255,0.1);
                    padding: 1.5rem;
                    border-radius: 12px;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255,255,255,0.2);
                    display: flex;
                    gap: 1rem;
                }
                
                .insight-icon {
                    font-size: 2rem;
                    color: #FCD34D;
                }
                
                .insight-content h4 {
                    margin: 0 0 0.5rem 0;
                    font-size: 1rem;
                    font-weight: 600;
                }
                
                .insight-value {
                    font-size: 2rem;
                    font-weight: 700;
                    color: #FCD34D;
                    margin-bottom: 0.5rem;
                }
                
                .insight-description {
                    font-size: 0.875rem;
                    opacity: 0.8;
                    margin: 0;
                }
                
                .ai-recommendations {
                    background: rgba(255,255,255,0.1);
                    padding: 1.5rem;
                    border-radius: 12px;
                }
                
                .ai-recommendations h4 {
                    margin: 0 0 1rem 0;
                    color: #FCD34D;
                }
                
                .recommendation-item {
                    display: flex;
                    gap: 1rem;
                    margin-bottom: 1rem;
                    align-items: flex-start;
                }
                
                .rec-priority {
                    padding: 0.25rem 0.75rem;
                    border-radius: 12px;
                    font-size: 0.75rem;
                    font-weight: 600;
                    min-width: 60px;
                    text-align: center;
                }
                
                .rec-priority.high {
                    background: #EF4444;
                    color: white;
                }
                
                .rec-priority.medium {
                    background: #F59E0B;
                    color: white;
                }
                
                .rec-priority.low {
                    background: #10B981;
                    color: white;
                }
                
                .rec-content {
                    flex: 1;
                }
                
                .rec-content strong {
                    color: #FCD34D;
                }
                
                .rec-content p {
                    margin: 0.25rem 0 0 0;
                    opacity: 0.9;
                    font-size: 0.875rem;
                }
            </style>
        `;
    }
}

// Migration mode selection
function selectMigrationMode(mode) {
    document.querySelectorAll('.migration-option').forEach(option => {
        option.classList.remove('selected');
    });
    
    event.currentTarget.classList.add('selected');
    document.querySelector(`input[value="${mode}"]`).checked = true;
    
    if (window.migrationManager) {
        migrationManager.updateMigrationPreview();
    }
}

// Initialize migration manager when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Migration Manager (External Access Compatible)');
    window.migrationManager = new MigrationManager();
});
