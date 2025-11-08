# ============================================================================
# app.py - COMPLETE FLASK APPLICATION
# ============================================================================
# This is the main Flask backend that integrates:
# 1. Startup page (URL input)
# 2. Jupyter notebook pipeline (CSV generation)
# 3. Dashboard display (CSV rendering in HTML)
# 4. Temporary file management
# ============================================================================

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import sys
import shutil
import tempfile
import uuid
import json
import pandas as pd
import subprocess
from datetime import datetime
import threading
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key_change_this_to_random_string_2025'

# ============================================================================
# CONFIGURATION
# ============================================================================

# Temporary directory for storing generated CSVs
TEMP_BASE_DIR = os.path.join(tempfile.gettempdir(), 'amazon_nlp_app')
os.makedirs(TEMP_BASE_DIR, exist_ok=True)

# Path to your Jupyter notebook (containing the scraper + NLP pipeline)
NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), 'Complete_Amazon_Scraper_with_Download.ipynb')

# Maximum session lifetime in seconds (30 minutes)
SESSION_TIMEOUT = 1800

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_notebook_and_generate_csv(product_url, output_dir):
    """
    Execute the Jupyter notebook with the given product URL as input.
    The notebook should create a CSV file in output_dir.
    Returns: path to the generated CSV file, or None if failed.
    """
    try:
        # You can either:
        # 1. Use nbconvert to run the notebook (requires nbconvert)
        # 2. Import and execute the Python code directly (recommended)
        
        # Method 2 (Recommended): Import your pipeline directly as Python code
        # This assumes you've extracted the key functions from the notebook
        
        from nlp_pipeline import generate_csv_from_url
        
        csv_path = generate_csv_from_url(product_url, output_dir)
        return csv_path
    
    except ImportError:
        print("ERROR: Could not import nlp_pipeline. Make sure nlp_pipeline.py exists.")
        return None
    except Exception as e:
        print(f"ERROR in run_notebook_and_generate_csv: {e}")
        return None


def cleanup_old_sessions():
    """
    Delete temporary directories older than SESSION_TIMEOUT.
    This runs in a background thread to avoid blocking the main app.
    """
    def _cleanup():
        while True:
            try:
                current_time = time.time()
                for folder_name in os.listdir(TEMP_BASE_DIR):
                    folder_path = os.path.join(TEMP_BASE_DIR, folder_name)
                    if os.path.isdir(folder_path):
                        folder_age = current_time - os.path.getctime(folder_path)
                        if folder_age > SESSION_TIMEOUT:
                            shutil.rmtree(folder_path, ignore_errors=True)
                            print(f"[CLEANUP] Deleted old session folder: {folder_name}")
            except Exception as e:
                print(f"Cleanup error: {e}")
            
            time.sleep(300)  # Run cleanup every 5 minutes
    
    cleanup_thread = threading.Thread(target=_cleanup, daemon=True)
    cleanup_thread.start()


def read_csv_as_string(csv_path):
    """
    Read CSV file as a string for injection into frontend.
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None


def get_csv_as_json(csv_path, max_rows=100):
    """
    Convert CSV to JSON for easier JavaScript consumption.
    """
    try:
        df = pd.read_csv(csv_path)
        # Limit rows for performance
        df = df.head(max_rows)
        return df.to_json(orient='records', indent=2)
    except Exception as e:
        print(f"Error converting CSV to JSON: {e}")
        return None


def cleanup_session_folder(session_id):
    """
    Immediately delete a session folder after use.
    """
    try:
        folder_path = os.path.join(TEMP_BASE_DIR, session_id)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path, ignore_errors=True)
            print(f"[CLEANUP] Deleted session folder: {session_id}")
    except Exception as e:
        print(f"Error cleaning up session: {e}")


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Startup page: User enters product URL(s).
    Supports single URL or multiple URLs (comma/newline separated).
    """
    if request.method == 'POST':
        urls_raw = request.form.get('product_urls', '').strip()
        
        if not urls_raw:
            flash('❌ Please enter at least one Amazon product URL.', 'error')
            return redirect(url_for('index'))
        
        # Parse multiple URLs (support comma, newline, space separation)
        urls = [url.strip() for url in urls_raw.replace('\n', ',').replace(';', ',').split(',') if url.strip()]
        
        if not urls:
            flash('❌ Invalid URL format. Please check your input.', 'error')
            return redirect(url_for('index'))
        
        # Create unique session folder
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(TEMP_BASE_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Store in Flask session
        session['session_id'] = session_id
        session['urls'] = urls
        session['csv_files'] = []
        
        print(f"[SESSION {session_id}] Created with {len(urls)} URL(s)")
        
        # Generate CSV for each URL (synchronous)
        csv_results = []
        for i, url in enumerate(urls):
            print(f"[SESSION {session_id}] Processing URL {i+1}/{len(urls)}: {url}")
            
            try:
                csv_path = run_notebook_and_generate_csv(url, session_dir)
                if csv_path and os.path.exists(csv_path):
                    csv_results.append({
                        'url': url,
                        'filename': os.path.basename(csv_path),
                        'path': csv_path,
                        'status': 'success'
                    })
                    print(f"[SESSION {session_id}] ✓ CSV generated: {os.path.basename(csv_path)}")
                else:
                    csv_results.append({
                        'url': url,
                        'filename': None,
                        'status': 'failed',
                        'error': 'Pipeline returned None'
                    })
                    print(f"[SESSION {session_id}] ✗ Failed to generate CSV for URL")
            
            except Exception as e:
                csv_results.append({
                    'url': url,
                    'filename': None,
                    'status': 'failed',
                    'error': str(e)
                })
                print(f"[SESSION {session_id}] ✗ Exception: {e}")
        
        # Store results
        session['csv_results'] = csv_results
        session.modified = True
        
        # Redirect to dashboard
        return redirect(url_for('dashboard'))
    
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """
    Dashboard page: Display all generated CSVs with visualization.
    CSV data is injected as a JavaScript variable for the frontend to consume.
    """
    session_id = session.get('session_id')
    
    if not session_id:
        flash('❌ Session expired or not found. Start over.', 'error')
        return redirect(url_for('index'))
    
    session_dir = os.path.join(TEMP_BASE_DIR, session_id)
    csv_results = session.get('csv_results', [])
    
    print(f"[SESSION {session_id}] Loading dashboard with {len(csv_results)} result(s)")
    
    # Prepare data for dashboard
    dashboard_data = []
    
    for result in csv_results:
        if result['status'] == 'success':
            csv_path = result['path']
            
            # Read CSV as string for injection
            csv_string = read_csv_as_string(csv_path)
            
            # Also read as JSON for easier JS consumption
            csv_json = get_csv_as_json(csv_path, max_rows=1000)
            
            if csv_string and csv_json:
                dashboard_data.append({
                    'url': result['url'],
                    'filename': result['filename'],
                    'csv_data': csv_string,
                    'csv_json': csv_json,
                    'status': 'success'
                })
        else:
            dashboard_data.append({
                'url': result['url'],
                'filename': None,
                'error': result.get('error', 'Unknown error'),
                'status': 'failed'
            })
    
    # Optional: Delete session folder immediately after loading (or keep for 30 min)
    # For now, we'll let the cleanup thread handle it
    
    return render_template('Code.html', dashboard_data=dashboard_data)


@app.route('/api/csv/<int:index>')
def api_csv(index):
    """
    API endpoint to get CSV data as JSON for a specific result.
    Useful if frontend wants to fetch data dynamically.
    """
    session_id = session.get('session_id')
    csv_results = session.get('csv_results', [])
    
    if not session_id or index >= len(csv_results):
        return jsonify({'error': 'Invalid request'}), 400
    
    result = csv_results[index]
    
    if result['status'] == 'success':
        csv_json = get_csv_as_json(result['path'])
        return jsonify(json.loads(csv_json) if csv_json else [])
    
    return jsonify({'error': result.get('error', 'Failed')}), 400


@app.route('/cleanup', methods=['POST'])
def cleanup():
    """
    Manual endpoint to trigger immediate session cleanup.
    Called after user leaves the dashboard.
    """
    session_id = session.get('session_id')
    
    if session_id:
        cleanup_session_folder(session_id)
        session.pop('session_id', None)
        session.pop('csv_results', None)
        session.modified = True
    
    return jsonify({'status': 'cleaned'}), 200


@app.route('/start-over')
def start_over():
    """
    Clear session and redirect to home.
    """
    session_id = session.get('session_id')
    
    if session_id:
        cleanup_session_folder(session_id)
    
    session.clear()
    flash('✓ Session cleared. Start fresh!', 'success')
    return redirect(url_for('index'))


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def page_not_found(error):
    return render_template('error.html', error_code=404, error_message='Page not found'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error_code=500, error_message='Internal server error'), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Start cleanup thread
    cleanup_old_sessions()
    
    print("=" * 80)
    print("AMAZON NLP DASHBOARD - FLASK APP")
    print("=" * 80)
    print(f"Temporary directory: {TEMP_BASE_DIR}")
    print(f"Session timeout: {SESSION_TIMEOUT} seconds")
    print("=" * 80)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)