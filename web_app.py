#!/usr/bin/env python3
"""
Enterprise RDL to Power BI Migration Web Application
Flask + JavaScript UI for bulk processing 10,000+ RDL files
"""

from flask import Flask, render_template, request, jsonify, send_file, session, send_from_directory
from flask_socketio import SocketIO, emit
import os
import json
import uuid
import threading
import time
import re
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
import zipfile
import shutil
from dotenv import load_dotenv
# Import unified LLM configuration
from llm_config import llm, embedding_model, refresh_clients
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Import existing RDL processing modules  
from app import RDLParser, PowerBIConverter, ReportElement

# Import new business logic analyzer
from rdl_business_analyzer import RDLBusinessAnalyzer

app = Flask(__name__)  # Enable built-in static handling
app.config['SECRET_KEY'] = 'rdl-migration-secret-key-2025'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16GB max upload
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['RESULTS_FOLDER'] = './results'

# Configure static files for external access
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development
app.config['PREFERRED_URL_SCHEME'] = 'http'  # For external access

# Add CORS support manually
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response

# Configure LLM with OpenAI client
if llm:
    print("✅ LLM client initialized successfully. AI-powered analysis enabled.")
    USE_AI = True
else:
    print("⚠️  Warning: LLM client not initialized. Using basic similarity analysis.")
    print("   - Set OPENAI_API_KEY environment variable to enable AI features")
    USE_AI = False

# Initialize SocketIO for real-time updates with proper CORS
socketio = SocketIO(app, cors_allowed_origins="*", 
                   allow_headers=['Content-Type'], 
                   allow_methods=['GET', 'POST'], 
                   logger=False, engineio_logger=False,
                   ping_timeout=120, ping_interval=25)

# Global job tracking
active_jobs = {}
job_results = {}

class MigrationJob:
    """Track migration job progress and status"""
    def __init__(self, job_id, files, job_type='migration'):
        self.job_id = job_id
        self.files = files
        self.job_type = job_type
        self.status = 'pending'
        self.progress = 0
        self.total_files = len(files)
        self.processed_files = 0
        self.errors = []
        self.results = {}
        self.start_time = datetime.now()
        self.estimated_completion = None

    def update_progress(self, processed_count, status_message=""):
        self.processed_files = processed_count
        self.progress = int((processed_count / self.total_files) * 100)
        
        # Estimate completion time
        if processed_count > 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = processed_count / elapsed
            remaining = self.total_files - processed_count
            eta_seconds = remaining / rate if rate > 0 else 0
            self.estimated_completion = eta_seconds
        
        # Emit real-time update
        socketio.emit('job_progress', {
            'job_id': self.job_id,
            'progress': self.progress,
            'processed': self.processed_files,
            'total': self.total_files,
            'status': self.status,
            'eta_seconds': self.estimated_completion,
            'message': status_message
        })

# Ensure required directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
    Path(folder).mkdir(parents=True, exist_ok=True)

@app.route('/')
def index():
    """Main dashboard page - using embedded version"""
    return render_template('index-embedded.html')

@app.route('/migration')
def migration():
    """Migration interface page - using embedded version"""
    return render_template('migration-embedded.html')

@app.route('/results')
def results():
    """Results dashboard page - using embedded version"""
    return render_template('results.html')

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle bulk file uploads"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Create job ID and upload directory
        job_id = str(uuid.uuid4())
        upload_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        uploaded_files = []
        invalid_files = []
        
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                if filename.lower().endswith('.rdl'):
                    file_path = upload_dir / filename
                    file.save(str(file_path))
                    uploaded_files.append({
                        'name': filename,
                        'path': str(file_path),
                        'size': file_path.stat().st_size
                    })
                else:
                    invalid_files.append(filename)
        
        return jsonify({
            'job_id': job_id,
            'uploaded_files': len(uploaded_files),
            'invalid_files': len(invalid_files),
            'total_size': sum(f['size'] for f in uploaded_files),
            'files': uploaded_files[:10],  # Return first 10 for preview
            'message': f'Successfully uploaded {len(uploaded_files)} RDL files'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_single_file():
    """Handle single file upload for modern UI"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        if not filename.lower().endswith('.rdl'):
            return jsonify({'success': False, 'error': 'Only .rdl files are supported'}), 400
        
        # Create uploads directory if it doesn't exist
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique file ID and save file
        file_id = str(uuid.uuid4())
        file_path = upload_dir / f"{file_id}_{filename}"
        file.save(str(file_path))
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'file_path': str(file_path),
            'filename': filename,
            'size': file_path.stat().st_size,
            'message': 'File uploaded successfully'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_files():
    """Start similarity analysis"""
    try:
        print(f"[DEBUG] Analysis request received")
        data = request.get_json()
        print(f"[DEBUG] Request data: {data}")
        
        # Handle both old format (job_id) and new format (files array)
        if 'job_id' in data:
            # Old format - batch processing
            job_id = data.get('job_id')
            upload_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
            if not job_id or not upload_dir.exists():
                return jsonify({'error': 'Invalid job ID'}), 400
            rdl_files = list(upload_dir.glob('*.rdl'))
        elif 'files' in data:
            # New format - individual files
            files = data.get('files', [])
            if len(files) < 2:
                return jsonify({'error': 'At least 2 files required for similarity analysis'}), 400
                
            rdl_files = []
            for file_info in files:
                file_path = file_info.get('path')
                if file_path and Path(file_path).exists():
                    rdl_files.append(Path(file_path))
                    
            if len(rdl_files) < 2:
                return jsonify({'error': 'At least 2 valid files required'}), 400
        else:
            return jsonify({'error': 'Either job_id or files array required'}), 400
        
        # Create analysis job
        analysis_job_id = str(uuid.uuid4())
        
        # Start analysis directly using the business logic analyzer
        try:
            print(f"[DEBUG] Starting analysis with {len(rdl_files)} files")
            for i, file in enumerate(rdl_files):
                print(f"[DEBUG] File {i+1}: {file} (exists: {file.exists()})")
            
            analyzer = RDLBusinessAnalyzer()
            pairs = []
            
            num_comparisons = len(rdl_files) * (len(rdl_files) - 1) // 2
            print(f"[DEBUG] Will perform {num_comparisons} pairwise comparisons")
            
            comparison_count = 0
            # Compare all pairs of files
            for i in range(len(rdl_files)):
                for j in range(i + 1, len(rdl_files)):
                    comparison_count += 1
                    file1_path = str(rdl_files[i])
                    file2_path = str(rdl_files[j])
                    
                    print(f"[DEBUG] Comparison {comparison_count}/{num_comparisons}: {rdl_files[i].name} vs {rdl_files[j].name}")
                    
                    try:
                        similarity_result = analyzer.calculate_business_similarity(file1_path, file2_path)
                        print(f"[DEBUG] Comparison {comparison_count} completed: {similarity_result.get('overall_similarity', 0):.1f}%")
                        
                        # Extract SQL queries for manual validation
                        file1_context = analyzer.analyze_rdl_file(file1_path)
                        file2_context = analyzer.analyze_rdl_file(file2_path)
                        
                        sql_queries = {
                            'file1_queries': file1_context.sql_queries if hasattr(file1_context, 'sql_queries') else [],
                            'file2_queries': file2_context.sql_queries if hasattr(file2_context, 'sql_queries') else []
                        }
                        
                        # Clean file names (remove UUID prefix)
                        file1_clean = rdl_files[i].name.split('_', 1)[-1] if '_' in rdl_files[i].name else rdl_files[i].name
                        file2_clean = rdl_files[j].name.split('_', 1)[-1] if '_' in rdl_files[j].name else rdl_files[j].name
                        
                        pairs.append({
                            'file1': file1_clean,
                            'file2': file2_clean,
                            'file1_full': rdl_files[i].name,  # Keep full name for backend operations
                            'file2_full': rdl_files[j].name,
                            'similarity': int(similarity_result.get('overall_similarity', 0)),
                            'details': {
                                'data_source_similarity': similarity_result.get('data_source_similarity', 0),
                                'filter_logic_similarity': similarity_result.get('filter_logic_similarity', 0),
                                'business_purpose_similarity': similarity_result.get('business_purpose_similarity', 0),
                                'calculation_similarity': similarity_result.get('calculation_similarity', 0),
                                'parameter_similarity': similarity_result.get('parameter_similarity', 0)
                            },
                            'sql_queries': sql_queries
                        })
                    except Exception as comp_error:
                        print(f"[ERROR] Comparison {comparison_count} failed: {comp_error}")
                        # Continue with other comparisons even if one fails
                        file1_clean = rdl_files[i].name.split('_', 1)[-1] if '_' in rdl_files[i].name else rdl_files[i].name
                        file2_clean = rdl_files[j].name.split('_', 1)[-1] if '_' in rdl_files[j].name else rdl_files[j].name
                        
                        pairs.append({
                            'file1': file1_clean,
                            'file2': file2_clean,
                            'file1_full': rdl_files[i].name,
                            'file2_full': rdl_files[j].name,
                            'similarity': 0,
                            'error': str(comp_error),
                            'details': {
                                'data_source_similarity': 0,
                                'filter_logic_similarity': 0,
                                'business_purpose_similarity': 0,
                                'calculation_similarity': 0,
                                'parameter_similarity': 0
                            }
                        })
            
            print(f"[DEBUG] Analysis completed successfully. Generated {len(pairs)} pairs")
            
            # Store results
            job_results[analysis_job_id] = {
                'status': 'completed',
                'pairs': pairs,
                'total_files': len(rdl_files),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"[DEBUG] Returning results to frontend")
            return jsonify({
                'success': True,
                'job_id': analysis_job_id,
                'results': {
                    'pairs': pairs,
                    'total_files': len(rdl_files)
                },
                'message': f'Analysis completed for {len(rdl_files)} files'
            })
            
        except Exception as analysis_error:
            return jsonify({'success': False, 'error': f'Analysis failed: {str(analysis_error)}'}), 500
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/migrate', methods=['POST'])
def start_migration():
    """Start bulk migration process"""
    try:
        data = request.get_json()
        
        # Handle both old format (job_id) and new format (files array)
        if 'job_id' in data:
            # Old format - batch processing
            job_id = data.get('job_id')
            consolidate = data.get('consolidate', False)
            
            if not job_id:
                return jsonify({'error': 'Job ID required'}), 400
            
            # Get uploaded files from job directory
            upload_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
            rdl_files = list(upload_dir.glob('*.rdl'))
        elif 'files' in data:
            # New format - individual files
            files = data.get('files', [])
            mode = data.get('mode', 'individual')
            consolidate = (mode == 'consolidated')
            
            if len(files) == 0:
                return jsonify({'error': 'No files provided for migration'}), 400
                
            rdl_files = []
            for file_info in files:
                file_path = file_info.get('path')
                if file_path and Path(file_path).exists():
                    rdl_files.append(Path(file_path))
                    
            if len(rdl_files) == 0:
                return jsonify({'error': 'No valid files found for migration'}), 400
                
            # Generate a migration job ID for new format
            job_id = str(uuid.uuid4())
        else:
            return jsonify({'error': 'Either job_id or files array required'}), 400
        
        # Create migration job
        migration_job_id = f"{job_id}_migration"
        job = MigrationJob(migration_job_id, [str(f) for f in rdl_files], 'migration')
        active_jobs[migration_job_id] = job
        
        # Start migration in background thread
        thread = threading.Thread(target=run_bulk_migration, args=(job, rdl_files, consolidate))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': migration_job_id,
            'files_to_migrate': len(rdl_files),
            'consolidation_enabled': consolidate,
            'message': f'Migration started for {len(rdl_files)} files'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/progress/<job_id>')
def get_progress(job_id):
    """Get job progress"""
    if job_id in active_jobs:
        job = active_jobs[job_id]
        return jsonify({
            'job_id': job_id,
            'status': job.status,
            'progress': job.progress,
            'processed': job.processed_files,
            'total': job.total_files,
            'errors': len(job.errors),
            'eta_seconds': job.estimated_completion
        })
    elif job_id in job_results:
        return jsonify(job_results[job_id])
    else:
        return jsonify({'error': 'Job not found'}), 404

@app.route('/api/results/<job_id>')
def get_results(job_id):
    """Get migration results"""
    if job_id in job_results:
        return jsonify(job_results[job_id])
    else:
        return jsonify({'error': 'Results not found'}), 404

@app.route('/api/recent-activity')
def recent_activity():
    """Get recent user activity"""
    try:
        # This is a placeholder - in a real app you'd fetch from a database
        activities = []
        
        # Check if there are any recent uploads or results
        uploads_dir = Path(app.config['UPLOAD_FOLDER'])
        results_dir = Path(app.config['RESULTS_FOLDER'])
        
        # Get recent upload directories (last 5)
        if uploads_dir.exists():
            recent_uploads = sorted(
                [d for d in uploads_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:5]
            
            for upload_dir in recent_uploads:
                files_count = len(list(upload_dir.glob('*.rdl')))
                if files_count > 0:
                    activities.append({
                        'title': f'Uploaded {files_count} RDL files',
                        'description': f'Job ID: {upload_dir.name}',
                        'timestamp': datetime.fromtimestamp(upload_dir.stat().st_mtime).isoformat(),
                        'icon': 'upload'
                    })
        
        return jsonify({
            'success': True,
            'activities': activities
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/results/<job_id>/files')
def get_job_files(job_id):
    """Get list of generated files for a job"""
    try:
        results_dir = Path(app.config['RESULTS_FOLDER']) / job_id
        if not results_dir.exists():
            return jsonify({'error': 'Results not found'}), 404
        
        files = []
        for file_path in results_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(results_dir)
                file_info = {
                    'name': file_path.name,
                    'path': str(relative_path),
                    'full_path': str(file_path),
                    'size': file_path.stat().st_size,
                    'type': file_path.suffix.lower(),
                    'modified': file_path.stat().st_mtime
                }
                files.append(file_info)
        
        # Sort by type and name
        files.sort(key=lambda x: (x['type'], x['name']))
        
        return jsonify({
            'job_id': job_id,
            'files': files,
            'total_files': len(files),
            'total_size': sum(f['size'] for f in files)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<job_id>/preview/<path:file_path>')
def preview_file(job_id, file_path):
    """Preview a specific file"""
    try:
        results_dir = Path(app.config['RESULTS_FOLDER']) / job_id
        target_file = results_dir / file_path
        
        if not target_file.exists() or not target_file.is_file():
            return jsonify({'error': 'File not found'}), 404
        
        # Security check - ensure file is within results directory
        if not str(target_file.resolve()).startswith(str(results_dir.resolve())):
            return jsonify({'error': 'Access denied'}), 403
        
        file_extension = target_file.suffix.lower()
        
        # Read file content based on type
        if file_extension in ['.txt', '.m', '.dax', '.sql', '.py', '.js', '.css', '.json']:
            # Text files
            with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return jsonify({
                'type': 'text',
                'content': content,
                'file_type': file_extension,
                'size': len(content)
            })
        
        elif file_extension == '.md':
            # Markdown files - convert to beautifully formatted HTML
            with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                markdown_content = f.read()
            
            # Check if this is a batch migration guide
            is_batch_guide = 'batch_migration_guide.md' in str(target_file)
            
            # Convert markdown to HTML (content is now pre-populated)
            html_content = convert_markdown_to_html(markdown_content)
            
            # Detect if this is a migration guide
            is_migration_doc = any(keyword in markdown_content.lower() for keyword in [
                'migration guide', 'rdl to power bi', 'power query', 'dax measures', 
                'migration process', 'step-by-step', 'troubleshooting', 'batch migration'
            ])
            
            return jsonify({
                'type': 'html',
                'content': html_content,
                'raw_content': markdown_content,
                'file_type': file_extension,
                'size': len(markdown_content),
                'is_migration_doc': is_migration_doc,
                'preview_title': '📋 Migration Documentation' if is_migration_doc else '📄 Documentation'
            })
        
        elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            # Image files - return base64
            import base64
            with open(target_file, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            return jsonify({
                'type': 'image',
                'content': f"data:image/{file_extension[1:]};base64,{image_data}",
                'file_type': file_extension,
                'size': target_file.stat().st_size
            })
        
        else:
            # Binary or unsupported files
            return jsonify({
                'type': 'binary',
                'content': f'Binary file ({file_extension}) - {target_file.stat().st_size} bytes',
                'file_type': file_extension,
                'size': target_file.stat().st_size,
                'download_url': f'/api/results/{job_id}/download/{file_path}'
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<job_id>/download/<path:file_path>')
def download_single_file(job_id, file_path):
    """Download a single file"""
    try:
        results_dir = Path(app.config['RESULTS_FOLDER']) / job_id
        target_file = results_dir / file_path
        
        if not target_file.exists() or not target_file.is_file():
            return jsonify({'error': 'File not found'}), 404
        
        # Security check
        if not str(target_file.resolve()).startswith(str(results_dir.resolve())):
            return jsonify({'error': 'Access denied'}), 403
        
        return send_file(target_file, as_attachment=True, download_name=target_file.name)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<job_id>')
def download_results(job_id):
    """Download migration results as ZIP"""
    try:
        results_dir = Path(app.config['RESULTS_FOLDER']) / job_id
        if not results_dir.exists():
            return jsonify({'error': 'Results not found'}), 404
        
        # Create ZIP file
        zip_path = results_dir.parent / f"{job_id}_results.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in results_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(results_dir)
                    zipf.write(file_path, arcname)
        
        return send_file(zip_path, as_attachment=True, download_name=f"migration_results_{job_id}.zip")
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/history')
def get_results_history():
    """Get list of all previous migration results"""
    try:
        results_dir = Path(app.config['RESULTS_FOLDER'])
        if not results_dir.exists():
            return jsonify({'results': []})
        
        results = []
        for job_dir in results_dir.iterdir():
            if job_dir.is_dir():
                try:
                    # Get job metadata
                    job_info = {
                        'job_id': job_dir.name,
                        'created': job_dir.stat().st_ctime,
                        'modified': job_dir.stat().st_mtime,
                        'file_count': len(list(job_dir.rglob('*'))),
                        'size': sum(f.stat().st_size for f in job_dir.rglob('*') if f.is_file())
                    }
                    
                    # Try to determine job type from directory structure
                    if any(f.suffix == '.m' for f in job_dir.rglob('*')):
                        job_info['type'] = 'migration'
                    else:
                        job_info['type'] = 'analysis'
                    
                    results.append(job_info)
                except Exception:
                    continue
        
        # Sort by creation time (newest first)
        results.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'results': results,
            'total': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def convert_markdown_to_html(markdown_content):
    """Convert markdown to beautifully formatted HTML for migration documentation"""
    html = markdown_content
    
    # Convert headers with better styling
    html = re.sub(r'^# (.*?)$', r'<h1 class="migration-h1">📋 \1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*?)$', r'<h2 class="migration-h2">🔧 \1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.*?)$', r'<h3 class="migration-h3">📝 \1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.*?)$', r'<h4 class="migration-h4">➤ \1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^##### (.*?)$', r'<h5 class="migration-h5">• \1</h5>', html, flags=re.MULTILINE)
    
    # Enhanced text formatting
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong class="migration-bold">\1</strong>', html)
    html = re.sub(r'\*(.*?)\*', r'<em class="migration-italic">\1</em>', html)
    
    # Code blocks with syntax highlighting classes
    html = re.sub(r'```sql\n(.*?)```', r'<div class="code-block sql-code"><pre><code class="language-sql">\1</code></pre></div>', html, flags=re.DOTALL)
    html = re.sub(r'```m\n(.*?)```', r'<div class="code-block powerquery-code"><pre><code class="language-powerquery">\1</code></pre></div>', html, flags=re.DOTALL)
    html = re.sub(r'```dax\n(.*?)```', r'<div class="code-block dax-code"><pre><code class="language-dax">\1</code></pre></div>', html, flags=re.DOTALL)
    html = re.sub(r'```bash\n(.*?)```', r'<div class="code-block bash-code"><pre><code class="language-bash">\1</code></pre></div>', html, flags=re.DOTALL)
    html = re.sub(r'```(.*?)\n(.*?)```', r'<div class="code-block"><pre><code>\2</code></pre></div>', html, flags=re.DOTALL)
    html = re.sub(r'`(.*?)`', r'<code class="inline-code">\1</code>', html)
    
    # Enhanced lists with icons
    html = re.sub(r'^- \*\*(.*?)\*\*: (.*?)$', r'<li class="migration-step"><strong class="step-title">✓ \1:</strong> <span class="step-desc">\2</span></li>', html, flags=re.MULTILINE)
    html = re.sub(r'^- (.*?)$', r'<li class="migration-item">• \1</li>', html, flags=re.MULTILINE)
    
    # Numbered lists
    html = re.sub(r'^(\d+)\. \*\*(.*?)\*\*: (.*?)$', r'<li class="migration-numbered-step"><span class="step-number">\1</span><strong class="step-title">\2:</strong> <span class="step-desc">\3</span></li>', html, flags=re.MULTILINE)
    html = re.sub(r'^(\d+)\. (.*?)$', r'<li class="migration-numbered-item"><span class="step-number">\1</span>\2</li>', html, flags=re.MULTILINE)
    
    # Convert consecutive list items to proper lists
    html = re.sub(r'(<li class="migration-step">.*?</li>\s*)+', r'<ul class="migration-steps">\g<0></ul>', html, flags=re.DOTALL)
    html = re.sub(r'(<li class="migration-item">.*?</li>\s*)+', r'<ul class="migration-list">\g<0></ul>', html, flags=re.DOTALL)
    html = re.sub(r'(<li class="migration-numbered-step">.*?</li>\s*)+', r'<ol class="migration-numbered-steps">\g<0></ol>', html, flags=re.DOTALL)
    html = re.sub(r'(<li class="migration-numbered-item">.*?</li>\s*)+', r'<ol class="migration-numbered-list">\g<0></ol>', html, flags=re.DOTALL)
    
    # Special callout boxes
    html = re.sub(r'^> \*\*Note:\*\* (.*?)$', r'<div class="callout callout-note">📝 <strong>Note:</strong> \1</div>', html, flags=re.MULTILINE)
    html = re.sub(r'^> \*\*Warning:\*\* (.*?)$', r'<div class="callout callout-warning">⚠️ <strong>Warning:</strong> \1</div>', html, flags=re.MULTILINE)
    html = re.sub(r'^> \*\*Important:\*\* (.*?)$', r'<div class="callout callout-important">❗ <strong>Important:</strong> \1</div>', html, flags=re.MULTILINE)
    html = re.sub(r'^> \*\*Tip:\*\* (.*?)$', r'<div class="callout callout-tip">💡 <strong>Tip:</strong> \1</div>', html, flags=re.MULTILINE)
    html = re.sub(r'^> (.*?)$', r'<div class="callout callout-default">💬 \1</div>', html, flags=re.MULTILINE)
    
    # Tables (basic support)
    html = re.sub(r'^\| (.*?) \|$', r'<tr><td>\1</td></tr>', html, flags=re.MULTILINE)
    html = re.sub(r'(<tr>.*?</tr>\s*)+', r'<table class="migration-table">\g<0></table>', html, flags=re.DOTALL)
    
    # Links
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" class="migration-link" target="_blank">\1 🔗</a>', html)
    
    # Line breaks and paragraphs
    html = html.replace('\n\n', '</p><p class="migration-paragraph">')
    html = f'<div class="migration-document"><p class="migration-paragraph">{html}</p></div>'
    
    # Clean up HTML structure
    html = html.replace('<p class="migration-paragraph"></p>', '')
    html = html.replace('<p class="migration-paragraph"><ul', '<ul')
    html = html.replace('<p class="migration-paragraph"><ol', '<ol')
    html = html.replace('<p class="migration-paragraph"><div class="callout', '<div class="callout')
    html = html.replace('<p class="migration-paragraph"><div class="code-block', '<div class="code-block')
    html = html.replace('<p class="migration-paragraph"><table', '<table')
    html = re.sub(r'</ul></p>', '</ul>', html)
    html = re.sub(r'</ol></p>', '</ol>', html)
    html = re.sub(r'</div></p>', '</div>', html)
    html = re.sub(r'</table></p>', '</table>', html)
    html = re.sub(r'<p class="migration-paragraph"><h([1-5])', r'<h\1', html)
    html = re.sub(r'</h([1-5])></p>', r'</h\1>', html)
    
    # Add CSS styling
    css_styles = """
    <style>
    .migration-document {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 1000px;
        margin: 0 auto;
        padding: 20px;
        background: #fff;
    }
    
    .migration-h1 {
        color: #2563eb;
        border-bottom: 3px solid #2563eb;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
        font-size: 2.2em;
    }
    
    .migration-h2 {
        color: #7c3aed;
        border-left: 4px solid #7c3aed;
        padding-left: 15px;
        margin-top: 25px;
        margin-bottom: 15px;
        font-size: 1.8em;
    }
    
    .migration-h3 {
        color: #059669;
        margin-top: 20px;
        margin-bottom: 12px;
        font-size: 1.4em;
    }
    
    .migration-h4 {
        color: #dc2626;
        margin-top: 15px;
        margin-bottom: 10px;
        font-size: 1.2em;
    }
    
    .migration-h5 {
        color: #6b7280;
        margin-top: 12px;
        margin-bottom: 8px;
        font-size: 1.1em;
    }
    
    .migration-paragraph {
        margin-bottom: 15px;
        text-align: justify;
    }
    
    .migration-bold {
        color: #1f2937;
        font-weight: 600;
    }
    
    .migration-italic {
        color: #4b5563;
        font-style: italic;
    }
    
    .code-block {
        background: #1f2937;
        color: #f9fafb;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        overflow-x: auto;
        border-left: 4px solid #3b82f6;
    }
    
    .sql-code { border-left-color: #ef4444; }
    .powerquery-code { border-left-color: #10b981; }
    .dax-code { border-left-color: #f59e0b; }
    .bash-code { border-left-color: #8b5cf6; }
    
    .code-block pre {
        margin: 0;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 14px;
    }
    
    .inline-code {
        background: #f3f4f6;
        color: #374151;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 0.9em;
    }
    
    .migration-steps {
        background: #f0f9ff;
        border: 1px solid #0ea5e9;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        list-style: none;
    }
    
    .migration-step {
        margin: 8px 0;
        padding: 8px 0;
        border-bottom: 1px solid #e0f2fe;
    }
    
    .migration-step:last-child {
        border-bottom: none;
    }
    
    .step-title {
        color: #0c4a6e;
    }
    
    .step-desc {
        color: #475569;
    }
    
    .migration-list {
        padding-left: 20px;
        margin: 12px 0;
    }
    
    .migration-item {
        margin: 6px 0;
        color: #4b5563;
    }
    
    .migration-numbered-steps, .migration-numbered-list {
        padding-left: 0;
        counter-reset: step-counter;
    }
    
    .migration-numbered-step, .migration-numbered-item {
        position: relative;
        padding-left: 40px;
        margin: 12px 0;
        list-style: none;
    }
    
    .step-number {
        position: absolute;
        left: 0;
        top: 0;
        background: #3b82f6;
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
    }
    
    .callout {
        border-radius: 8px;
        padding: 12px 16px;
        margin: 16px 0;
        border-left: 4px solid;
    }
    
    .callout-note {
        background: #f0f9ff;
        border-left-color: #0ea5e9;
        color: #0c4a6e;
    }
    
    .callout-warning {
        background: #fffbeb;
        border-left-color: #f59e0b;
        color: #92400e;
    }
    
    .callout-important {
        background: #fef2f2;
        border-left-color: #ef4444;
        color: #991b1b;
    }
    
    .callout-tip {
        background: #f0fdf4;
        border-left-color: #22c55e;
        color: #166534;
    }
    
    .callout-default {
        background: #f9fafb;
        border-left-color: #6b7280;
        color: #374151;
    }
    
    .migration-table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    
    .migration-table td {
        padding: 12px;
        border-bottom: 1px solid #e5e7eb;
        vertical-align: top;
    }
    
    .migration-table tr:nth-child(even) {
        background: #f9fafb;
    }
    
    .migration-link {
        color: #2563eb;
        text-decoration: none;
        border-bottom: 1px dotted #2563eb;
    }
    
    .migration-link:hover {
        background: #eff6ff;
        text-decoration: none;
    }
    </style>
    """
    
    return css_styles + html

def run_bulk_migration(job, rdl_files, consolidate=False):
    """Run bulk migration in background with optimized batch-level guide generation"""
    try:
        job.status = 'running'
        job.update_progress(0, "Starting bulk migration...")
        
        # Create results directory
        results_dir = Path(app.config['RESULTS_FOLDER']) / job.job_id.replace('_migration', '')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        converter = PowerBIConverter()
        processed_count = 0
        batch_files_info = []  # Collect info for batch guide generation
        
        for rdl_file in rdl_files:
            try:
                job.update_progress(processed_count, f"Processing {rdl_file.name}...")
                
                # Parse RDL
                parser = RDLParser(str(rdl_file))
                data_sources = parser.extract_data_sources()
                datasets = parser.extract_datasets()
                report_items = parser.extract_report_items()
                
                # Create individual result directory
                file_result_dir = results_dir / rdl_file.stem
                file_result_dir.mkdir(parents=True, exist_ok=True)
                
                # Convert datasets to Power Query
                for dataset in datasets:
                    if dataset['query']:
                        data_source_info = next((ds for ds in data_sources if ds['name'] == dataset.get('data_source_name')), {})
                        power_query = converter.convert_sql_to_powerquery(dataset['query'], dataset['name'], data_source_info)
                        
                        query_file = file_result_dir / f"{dataset['name']}.m"
                        with open(query_file, 'w', encoding='utf-8') as f:
                            f.write(power_query)
                
                # Convert report items to DAX
                for item in report_items:
                    if item.type == 'Table':
                        dax_code = converter.convert_table_to_dax(item)
                        dax_file = file_result_dir / f"{item.name}.dax"
                        with open(dax_file, 'w', encoding='utf-8') as f:
                            f.write(dax_code)
                
                # Collect file information for batch guide generation
                file_info = {
                    'report_name': parser.report_info.get('report_name', rdl_file.stem),
                    'data_sources': data_sources,
                    'datasets': datasets,
                    'datasets_details': datasets,  # Full dataset details for dynamic content
                    'report_items': [{'name': item.name, 'type': item.type, 'properties': item.properties} for item in report_items],
                    'report_items_details': [{'name': item.name, 'type': item.type, 'properties': item.properties} for item in report_items],
                    'file_path': str(file_result_dir)
                }
                batch_files_info.append(file_info)
                
                processed_count += 1
                
            except Exception as e:
                job.errors.append(f"Error processing {rdl_file.name}: {str(e)}")
                processed_count += 1
        
        # Generate single batch-level migration guide (token-optimized)
        try:
            from app import generate_batch_migration_guide
            
            job.update_progress(processed_count, "Generating optimized migration guide...")
            
            # Create batch guides directory
            batch_guides_dir = results_dir / 'guides'
            batch_guides_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare batch info for guide generation
            batch_info = {
                'batch_id': job.job_id.replace('_migration', ''),
                'files': batch_files_info
            }
            
            # Generate token-optimized batch migration guide
            batch_migration_guide = generate_batch_migration_guide(batch_info)
            
            # Save single batch guide
            batch_guide_file = batch_guides_dir / 'batch_migration_guide.md'
            with open(batch_guide_file, 'w', encoding='utf-8') as f:
                f.write(batch_migration_guide)
                
            print(f"Generated optimized batch migration guide with {len(batch_files_info)} files")
                
        except Exception as guide_error:
            print(f"Warning: Could not generate batch migration guide: {guide_error}")
        
        job.status = 'completed'
        job.update_progress(processed_count, "Migration completed!")
        
        # Store results with batch guide info
        job_results[job.job_id] = {
            'job_type': 'migration',
            'status': 'completed',
            'processed_files': processed_count,
            'errors': len(job.errors),
            'results_directory': str(results_dir),
            'batch_guide_generated': True,
            'completed_at': datetime.now().isoformat()
        }
        
        del active_jobs[job.job_id]
        
    except Exception as e:
        job.status = 'error'
        job.errors.append(str(e))

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'message': 'Connected to migration server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    print("🚀 Starting Enterprise RDL Migration Web Application")
    print("📊 Features: Bulk Upload, AI Analysis, Real-time Progress")
    print("🌐 Access at: http://localhost:5000")
    
    # For development with consistent environment loading
    # Use debug=False in production or when environment variables are inconsistent
    debug_mode = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=5000, use_reloader=debug_mode)