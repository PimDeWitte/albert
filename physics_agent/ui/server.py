#!/usr/bin/env python3
"""
Simple web UI for gravitational theory exploration
"""
import os
import sys
import subprocess
import threading
import queue
import time
import inspect
import shutil
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physics_agent.theory_loader import TheoryLoader
from physics_agent.ui.llm_api import LLMApi
from physics_agent.ui.leaderboard import Leaderboard

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize components
theory_loader = TheoryLoader()
llm_api = LLMApi()
leaderboard = Leaderboard()

# Job queue for theory evaluations
job_queue = queue.Queue()
results_store = {}

def theory_evaluation_worker():
    """Background worker to process theory evaluations"""
    while True:
        try:
            job = job_queue.get()
            if job is None:
                break
                
            job_id = job['id']
            theory_code = job['theory_code']
            theory_name = job['theory_name']
            
            # Update status
            results_store[job_id]['status'] = 'running'
            results_store[job_id]['message'] = 'Evaluating theory...'
            
            # Create a temporary directory in the theories folder
            theories_base_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'physics_agent', 'theories'
            )
            
            # Create a unique subdirectory name
            import uuid
            temp_theory_name = f"ui_custom_{uuid.uuid4().hex[:8]}"
            temp_dir = os.path.join(theories_base_dir, temp_theory_name)
            os.makedirs(temp_dir, exist_ok=True)
            
            theory_file = os.path.join(temp_dir, 'theory.py')
            init_file = os.path.join(temp_dir, '__init__.py')
            
            try:
                # Write the theory code
                with open(theory_file, 'w') as f:
                    f.write(theory_code)
                
                # Create __init__.py
                with open(init_file, 'w') as f:
                    f.write('')
                
                cmd = [
                    sys.executable,
                    '-m', 'physics_agent.theory_engine_core',
                    '--theory-filter', theory_name,
                    '--steps', '5000',  # Shorter run for UI
                    '--no-cache',  # Don't use cached results
                ]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    # Find the results in the latest run
                    latest_run = leaderboard.get_latest_run()
                    if latest_run:
                        # Get theory results
                        theory_results = leaderboard.collect_theory_results(latest_run)
                        for theory in theory_results:
                            if theory['name'] == theory_name:
                                results_store[job_id]['status'] = 'completed'
                                results_store[job_id]['message'] = 'Theory evaluation completed'
                                results_store[job_id]['results'] = theory
                                break
                        else:
                            results_store[job_id]['status'] = 'error'
                            results_store[job_id]['message'] = 'Theory results not found'
                    else:
                        results_store[job_id]['status'] = 'error'
                        results_store[job_id]['message'] = 'No run directory found'
                else:
                    results_store[job_id]['status'] = 'error'
                    results_store[job_id]['message'] = f'Evaluation failed: {result.stderr}'
                    
            except subprocess.TimeoutExpired:
                results_store[job_id]['status'] = 'error'
                results_store[job_id]['message'] = 'Evaluation timed out'
            except Exception as e:
                results_store[job_id]['status'] = 'error'
                results_store[job_id]['message'] = f'Error: {str(e)}'
            finally:
                # Clean up temp directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    
        except Exception as e:
            print(f"Worker error: {e}")
        finally:
            job_queue.task_done()

# Start worker thread
worker_thread = threading.Thread(target=theory_evaluation_worker, daemon=True)
worker_thread.start()

@app.route('/')
def index():
    """Main page showing list of theories"""
    return render_template('index.html')

@app.route('/api/theories')
def list_theories():
    """API endpoint to list all available theories"""
    theories = theory_loader.discover_theories()
    
    # Format for display
    theory_list = []
    for key, info in theories.items():
        theory_list.append({
            'key': key,
            'name': info['class'].__name__ if 'class' in info else key.split('/')[-1],
            'category': info.get('category', 'unknown'),
            'path': info['path'],
            'is_baseline': 'baselines' in info['path']
        })
    
    # Get latest leaderboard rankings
    board = leaderboard.generate_leaderboard()
    rankings = {t['name']: t.get('rank', None) for t in board.get('theories', [])}
    
    # Add rankings to theory list
    for theory in theory_list:
        theory['rank'] = rankings.get(theory['name'], None)
    
    # Sort by rank (None values last)
    theory_list.sort(key=lambda x: (x['rank'] is None, x['rank']))
    
    return jsonify(theory_list)

@app.route('/api/theory/<theory_key>')
def get_theory_code(theory_key):
    """Get the source code of a theory"""
    # Sanitize the key
    theory_key = theory_key.replace('..', '')
    
    theories = theory_loader.discover_theories()
    if theory_key not in theories:
        return jsonify({'error': 'Theory not found'}), 404
    
    theories[theory_key]
    
    # Try to load the source code
    try:
        # Instantiate to get the actual class
        instance = theory_loader.instantiate_theory(theory_key)
        if instance:
            # Get the source code
            source = inspect.getsource(instance.__class__)
            return jsonify({
                'name': instance.name,
                'source': source,
                'category': getattr(instance, 'category', 'unknown')
            })
    except Exception as e:
        return jsonify({'error': f'Could not load source: {str(e)}'}), 500
    
    return jsonify({'error': 'Could not load theory'}), 500

@app.route('/api/modify_theory', methods=['POST'])
def modify_theory():
    """Generate a modified version of a theory based on user input"""
    data = request.json
    base_theory_key = data.get('theory_key')
    modification = data.get('modification')
    
    if not base_theory_key or not modification:
        return jsonify({'error': 'Missing theory_key or modification'}), 400
    
    # Get the base theory code
    theories = theory_loader.discover_theories()
    if base_theory_key not in theories:
        return jsonify({'error': 'Theory not found'}), 404
    
    # Get baseline theory names for the prompt
    baseline_names = []
    for key, info in theories.items():
        if 'baselines' in info['path']:
            instance = theory_loader.instantiate_theory(key)
            if instance:
                baseline_names.append(instance.name)
    
    try:
        # Get the source code
        instance = theory_loader.instantiate_theory(base_theory_key)
        if not instance:
            return jsonify({'error': 'Could not instantiate theory'}), 500
            
        source = inspect.getsource(instance.__class__)
        
        # Generate modified theory
        new_code = llm_api.generate_theory_variation(source, modification, baseline_names)
        
        if new_code:
            # Extract theory name from the generated code
            import re
            name_match = re.search(r'self\.name\s*=\s*["\'](.+?)["\']', new_code)
            theory_name = name_match.group(1) if name_match else f"Modified {instance.name}"
            
            return jsonify({
                'success': True,
                'code': new_code,
                'name': theory_name
            })
        else:
            return jsonify({'error': 'Failed to generate theory'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate_theory', methods=['POST'])
def evaluate_theory():
    """Queue a theory for evaluation"""
    data = request.json
    theory_code = data.get('code')
    theory_name = data.get('name', 'Custom Theory')
    
    if not theory_code:
        return jsonify({'error': 'Missing theory code'}), 400
    
    # Create a job ID
    job_id = f"job_{int(time.time() * 1000)}"
    
    # Initialize job status
    results_store[job_id] = {
        'id': job_id,
        'status': 'queued',
        'message': 'Theory queued for evaluation',
        'theory_name': theory_name,
        'created_at': time.time()
    }
    
    # Queue the job
    job_queue.put({
        'id': job_id,
        'theory_code': theory_code,
        'theory_name': theory_name
    })
    
    return jsonify({
        'job_id': job_id,
        'status': 'queued'
    })

@app.route('/api/job_status/<job_id>')
def get_job_status(job_id):
    """Get the status of an evaluation job"""
    if job_id not in results_store:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(results_store[job_id])

@app.route('/leaderboard')
def show_leaderboard():
    """Display the leaderboard page"""
    return render_template('leaderboard.html')

@app.route('/api/leaderboard')
def get_leaderboard():
    """Get leaderboard data"""
    board = leaderboard.generate_leaderboard()
    return jsonify(board)

@app.route('/api/theory_details/<path:theory_name>')
def get_theory_details(theory_name):
    """Get detailed information about a theory from the leaderboard"""
    details = leaderboard.get_theory_details(theory_name)
    if details:
        return jsonify(details)
    return jsonify({'error': 'Theory not found'}), 404

@app.route('/api/visualization/<path:theory_name>/<filename>')
def get_visualization(theory_name, filename):
    """Serve visualization images"""
    details = leaderboard.get_theory_details(theory_name)
    if not details:
        return jsonify({'error': 'Theory not found'}), 404
    
    # Sanitize filename
    filename = secure_filename(filename)
    
    # Build path to image
    image_path = os.path.join(details['directory'], 'viz', filename)
    
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    
    return jsonify({'error': 'Image not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=8000) 