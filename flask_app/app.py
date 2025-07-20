import os
import atexit
import sys
import glob
import uuid
import time
import threading
import json
import re
import signal
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from werkzeug.utils import secure_filename
import pandas as pd
import shutil
from utils.extract_logic import process_student_files, init_ranking_data, lookup_institution_ranking

UPLOAD_FOLDER = 'uploads'
CUSTOM_RANKING_FOLDER = 'data/custom_rankings'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'jpg', 'jpeg', 'png', 'xlsx'}

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CUSTOM_RANKING_FOLDER'] = CUSTOM_RANKING_FOLDER
app.config['TEXT_FILES_MAX_AGE'] = 24 * 60 * 60

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CUSTOM_RANKING_FOLDER, exist_ok=True)

# Cleanup functions
def clear_uploaded_files():
    """Delete all files in the uploads folder"""
    files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
    for f in files:
        try:
            if os.path.isfile(f):
                os.remove(f)
                print(f"Deleted file: {f}")
        except Exception as e:
            print(f"Error deleting {f}: {str(e)}")

# Combined cleanup function
def clear_on_exit():
    clear_uploaded_files()

# Register the cleanup function
atexit.register(clear_on_exit)

CUSTOM_RANKING_METADATA = os.path.join(CUSTOM_RANKING_FOLDER, 'metadata.json')

def load_custom_rankings():
    if os.path.exists(CUSTOM_RANKING_METADATA):
        try:
            with open(CUSTOM_RANKING_METADATA, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_custom_rankings(metadata):
    with open(CUSTOM_RANKING_METADATA, 'w') as f:
        json.dump(metadata, f)

default_ranking_path = os.path.join("data", "Indianranking2025.xlsx")
default_sheet = "TBS India 25"
ranking_df, institution_list, ranking_dict, normalized_ranking_dict = init_ranking_data(default_ranking_path, default_sheet)

def cleanup_task():
    while True:
        now = time.time()
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.startswith("extracted_") and filename.endswith(".txt"):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.isfile(filepath):
                    file_age = now - os.path.getmtime(filepath)
                    if file_age > app.config['TEXT_FILES_MAX_AGE']:
                        try:
                            os.remove(filepath)
                            print(f"Cleaned up text file: {filename}")
                        except Exception as e:
                            print(f"Error cleaning up {filename}: {e}")
        time.sleep(3600)

cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
cleanup_thread.start()

# Generate a unique startup ID
app.startup_id = str(uuid.uuid4())

@app.before_request
def before_request():
    # Initialize session with startup ID
    if 'startup_id' not in session:
        session['startup_id'] = app.startup_id
    
    # Clear history if this is first session after startup
    if session.get('startup_id') != app.startup_id:
        session['history'] = []
        session['startup_id'] = app.startup_id
    
    # Initialize history if not exists
    if 'history' not in session:
        session['history'] = []
    
    # Initialize ranking preferences
    if 'ranking_prefs' not in session:
        session['ranking_prefs'] = {
            'system': 'default',
            'sheet': default_sheet
        }
    
    # Set session to expire after 24 hours
    session.permanent = True
    app.permanent_session_lifetime = timedelta(hours=24)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    custom_rankings = load_custom_rankings()
    
    if request.method == 'POST':
        transcript = request.files.get('transcript')
        cv = request.files.get('cv')
        reference = request.files.get('reference')
        ranking_selection = request.form.get('ranking_selection', 'default')
        custom_sheet = request.form.get('custom_sheet', '')

        file_paths = {}
        file_names = {}

        for label, file in [('transcript', transcript), ('cv', cv), ('reference', reference)]:
            if file and file.filename:
                filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                file_paths[label] = path
                file_names[label] = filename

        if ranking_selection == 'default':
            ranking_path = default_ranking_path
            sheet_name = default_sheet
        else:
            ranking_path = os.path.join(app.config['CUSTOM_RANKING_FOLDER'], ranking_selection)
            sheet_name = custom_sheet if custom_sheet else custom_rankings.get(ranking_selection, {}).get('sheets', [''])[0]
            
        try:
            ranking_df_local, institution_list_local, ranking_dict_local, normalized_ranking_dict_local = init_ranking_data(
                ranking_path, sheet_name
            )
        except Exception as e:
            flash(f"Error loading ranking data: {str(e)}", 'danger')
            return redirect(url_for('upload'))

        result = process_student_files(
            transcript_path=file_paths.get('transcript'),
            cv_path=file_paths.get('cv'),
            reference_paths=[file_paths['reference']] if 'reference' in file_paths else None,
            ranking_df=ranking_df_local,
            institution_list=institution_list_local,
            ranking_dict=ranking_dict_local,
            normalized_ranking_dict=normalized_ranking_dict_local
        )

        session['source_file'] = result.get("_source")
        session['match_score'] = result.get("_match_score")
        session['match_snippet'] = result.get("_match_snippet", "")
        session['llm_evidence'] = result.get("_llm_evidence", "")
        session['llm_thought_process'] = result.get("_llm_thought_process", "")  # NEW: Store thought process
        session['llm_degree'] = result.get("_llm_degree", "")
        session['match_tool'] = result.get("_match_tool")
        session['evidence'] = result.get("_evidence")
        session['llm_used'] = result.get("_llm_used", False)
        
        raw_text = result.get("_raw_text", "")
        session['raw_text_filename'] = None
        
        text_filename = f"extracted_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        text_filepath = os.path.join(app.config['UPLOAD_FOLDER'], text_filename)
        try:
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            session['raw_text_filename'] = text_filename
        except Exception as e:
            print(f"Error saving text file: {e}")
            session['raw_text_filename'] = None

        if result.get("Name of Institution"):
            session['result'] = {
                "Name of Institution": result.get("Name of Institution"),
                "City": result.get("City"),
                "State": result.get("State"),
                "Tier 1": result.get("Tier 1"),
                "Tier 2": result.get("Tier 2"),
                "Global": result.get("Global")
            }
        else:
            session['result'] = None

        history_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'files': {
                'transcript': file_names.get('transcript'),
                'cv': file_names.get('cv'),
                'reference': file_names.get('reference')
            },
            'result': {
                'institution': result.get("Name of Institution"),
                'city': result.get("City"),
                'state': result.get("State"),
                'match_score': result.get("_match_score"),
                'match_snippet': result.get("_match_snippet", ""),
                'llm_evidence': result.get("_llm_evidence", ""),
                'llm_thought_process': result.get("_llm_thought_process", ""),  # NEW: Store in history
                'global_rank': result.get("Global", {}),
                'ranking_system': ranking_selection,
                'sheet_name': sheet_name
            }
        }

        session['history'].insert(0, history_entry)
        session['history'] = session['history'][:10]
        session.modified = True

        return redirect(url_for('results'))

    custom_rankings = load_custom_rankings()
    return render_template('upload.html', custom_rankings=custom_rankings)

@app.route('/results')
def results():
    result = session.get('result')
    extraction_method = "LLM" if session.get('llm_used') else "Traditional Extraction"
    
    # Get all extraction metadata
    source = session.get('source_file', 'N/A')
    llm_evidence = session.get('llm_evidence', '')
    match_snippet = session.get('match_snippet', '')
    llm_degree = session.get('llm_degree', '')
    
    return render_template('results.html', 
                         result=result,
                         extraction_method=extraction_method,
                         source=source,
                         llm_evidence=llm_evidence,
                         match_snippet=match_snippet,
                         llm_degree=llm_degree)

@app.route('/manual-check')
def manual_check():
    raw_text = ""
    if filename := session.get('raw_text_filename'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except Exception as e:
            print(f"Error reading text file: {e}")
            raw_text = "⚠️ Error loading extracted text"
    
    return render_template('manual_check.html',
                         source=session.get('source_file'),
                         raw_text=raw_text,
                         match_score=session.get('match_score'),
                         match_snippet=session.get('match_snippet', ''),
                         llm_evidence=session.get('llm_evidence', ''),
                         llm_thought_process=session.get('llm_thought_process', ''),  # NEW: Pass to template
                         llm_degree=session.get('llm_degree', ''),
                         extraction_method="LLM" if session.get('llm_used') else "Traditional Extraction")

@app.route("/institution-names")
def institution_names():
    unique_names = sorted(set(name.strip() for name in institution_list))
    return jsonify(unique_names)

@app.route('/search', methods=['GET', 'POST'])
def search():
    # Always load fresh custom rankings
    custom_rankings = load_custom_rankings()
    result = None
    query = None

    if request.method == 'POST':
        query = request.form.get('institution_name', '').strip().lower()
        ranking_selection = request.form.get('ranking_selection', 'default')
        custom_sheet = request.form.get('custom_sheet', '')

        session['ranking_prefs'] = {
            'system': ranking_selection,
            'sheet': custom_sheet if ranking_selection != 'default' else default_sheet
        }
        session.modified = True

        if ranking_selection == 'default':
            ranking_df_local = ranking_df
            institution_list_local = institution_list
            ranking_dict_local = ranking_dict
            normalized_ranking_dict_local = normalized_ranking_dict
        else:
            ranking_path = os.path.join(app.config['CUSTOM_RANKING_FOLDER'], ranking_selection)
            sheet_name = custom_sheet or custom_rankings.get(ranking_selection, {}).get('sheets', [''])[0]
            
            try:
                (ranking_df_local, 
                 institution_list_local, 
                 ranking_dict_local, 
                 normalized_ranking_dict_local) = init_ranking_data(ranking_path, sheet_name)
            except Exception as e:
                flash(f"Error loading ranking data: {str(e)}", 'danger')
                return redirect(url_for('search'))

        if query:
            original_name = ranking_dict_local.get(query)
            if original_name:
                result = lookup_institution_ranking(original_name, ranking_df_local)
            else:
                normalized_name = _normalize_institution_name(query)
                if normalized_name in normalized_ranking_dict_local:
                    result = lookup_institution_ranking(
                        normalized_ranking_dict_local[normalized_name], 
                        ranking_df_local
                    )
                else:
                    matches = ranking_df_local[
                        ranking_df_local['Name of Institution'].str.lower() == query
                    ]
                    if not matches.empty:
                        result = lookup_institution_ranking(
                            matches.iloc[0]['Name of Institution'], 
                            ranking_df_local
                        )
            
            if result:
                result['ranking_system'] = "Default Ranking" if ranking_selection == 'default' else ranking_selection
                result['sheet_name'] = session['ranking_prefs']['sheet'] if ranking_selection != 'default' else None

    ranking_prefs = session.get('ranking_prefs', {'system': 'default', 'sheet': default_sheet})
    
    return render_template('search.html', 
                           result=result, 
                           query=query,
                           custom_rankings=custom_rankings,
                           ranking_prefs=ranking_prefs)

@app.route('/history')
def history():
    return render_template('history.html', history=session.get('history', []))

@app.route('/clear-history', methods=['POST'])
def clear_history():
    session['history'] = []
    session.modified = True
    flash('History cleared successfully', 'success')
    return redirect(url_for('history'))

@app.route('/manage-rankings', methods=['GET', 'POST'])
def manage_rankings():
    custom_rankings = load_custom_rankings()
    
    if request.method == 'POST':
        if 'ranking_file' in request.files:
            file = request.files['ranking_file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                
                if not filename.endswith('.xlsx'):
                    flash('Only Excel (.xlsx) files are allowed', 'danger')
                    return redirect(url_for('manage_rankings'))
                
                file_path = os.path.join(app.config['CUSTOM_RANKING_FOLDER'], filename)
                file.save(file_path)
                
                try:
                    xl = pd.ExcelFile(file_path)
                    sheet_names = xl.sheet_names
                except Exception as e:
                    flash(f"Error reading Excel file: {str(e)}", 'danger')
                    return redirect(url_for('manage_rankings'))
                
                custom_rankings[filename] = {
                    'sheets': sheet_names,
                    'upload_date': datetime.now().strftime("%Y-%m-%d")
                }
                save_custom_rankings(custom_rankings)
                
                flash(f'Ranking file "{filename}" uploaded successfully', 'success')
        
        elif 'delete_ranking' in request.form:
            filename = request.form['delete_ranking']
            file_path = os.path.join(app.config['CUSTOM_RANKING_FOLDER'], filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                if filename in custom_rankings:
                    del custom_rankings[filename]
                    save_custom_rankings(custom_rankings)
                    flash(f'Ranking file "{filename}" deleted', 'info')
    
    return render_template('manage_rankings.html', custom_rankings=custom_rankings)

@app.route('/get-institution-names')
def get_institution_names():
    term = request.args.get('term', '').lower()
    ranking_selection = request.args.get('ranking_system', 'default')
    sheet_name = request.args.get('sheet_name', '')
    
    # Handle default ranking system
    if ranking_selection == 'default':
        try:
            unique_names = sorted(set(name.strip() for name in institution_list))
            if term:
                filtered_names = [name for name in unique_names if term in name.lower()]
                return jsonify(filtered_names)
            return jsonify(unique_names)
        except Exception as e:
            print(f"Error loading default ranking: {e}")
            return jsonify([])
    
    # Handle custom ranking systems
    ranking_path = os.path.join(app.config['CUSTOM_RANKING_FOLDER'], ranking_selection)
    
    if not os.path.exists(ranking_path):
        print(f"Custom ranking file not found: {ranking_path}")
        return jsonify([])
    
    custom_rankings = load_custom_rankings()
    available_sheets = custom_rankings.get(ranking_selection, {}).get('sheets', [])
    
    # Validate sheet name
    if not sheet_name or sheet_name not in available_sheets:
        if available_sheets:
            sheet_name = available_sheets[0]
        else:
            try:
                xl = pd.ExcelFile(ranking_path)
                available_sheets = xl.sheet_names
                sheet_name = available_sheets[0] if available_sheets else ''
            except Exception as e:
                print(f"Error reading custom ranking file: {e}")
                return jsonify([])
    
    # Read institution names from Excel with flexible column names
    try:
        ranking_df_custom = pd.read_excel(ranking_path, sheet_name=sheet_name)
        
        # Try to find institution name column using common variations
        name_columns = [
            'Name of Institution', 'Institution Name', 'College Name',
            'University Name', 'Institution', 'College', 'University'
        ]
        
        institution_column = None
        for col in name_columns:
            if col in ranking_df_custom.columns:
                institution_column = col
                break
        
        if institution_column:
            institution_list_custom = ranking_df_custom[institution_column].dropna().unique().tolist()
            unique_names = sorted(set(name.strip() for name in institution_list_custom))
            
            if term:
                filtered_names = [name for name in unique_names if term in name.lower()]
                return jsonify(filtered_names)
            return jsonify(unique_names)
        else:
            # Try to find a column that contains "name" or "institution"
            possible_columns = []
            for col in ranking_df_custom.columns:
                if 'name' in col.lower() or 'institution' in col.lower() or 'college' in col.lower():
                    possible_columns.append(col)
            
            if possible_columns:
                institution_column = possible_columns[0]
                institution_list_custom = ranking_df_custom[institution_column].dropna().unique().tolist()
                unique_names = sorted(set(name.strip() for name in institution_list_custom))
                
                if term:
                    filtered_names = [name for name in unique_names if term in name.lower()]
                    return jsonify(filtered_names)
                return jsonify(unique_names)
            else:
                print(f"Worksheet '{sheet_name}' missing institution name column. Available columns: {ranking_df_custom.columns.tolist()}")
                return jsonify([])
    except Exception as e:
        print(f"Error loading custom ranking: {e}")
        return jsonify([])

def _normalize_institution_name(name):
    name = re.sub(r'\([^)]*\)', '', name).strip().lower()
    suffixes = ["university", "college", "institute", "institution", 
                "of technology", "school", "academy", "centre", "center"]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:len(name)-len(suffix)].strip()
    return re.split(r',\s*', name)[0]

def check_for_shutdown():
    shutdown_file = os.path.join(app.root_path, 'shutdown.txt')
    if os.path.exists(shutdown_file):
        print("Shutdown signal received. Exiting...")
        os.remove(shutdown_file)
        sys.exit(0)

# Shutdown route to terminate the server
@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Gracefully shutdown the server"""
    try:
        # Perform cleanup operations
        clear_uploaded_files()
        
        # Start a thread to shut down the server after a short delay
        def delayed_shutdown():
            time.sleep(1)  # Give time for response to be sent
            # This will terminate the entire process
            os._exit(0)
            
        threading.Thread(target=delayed_shutdown, daemon=True).start()
        
        return jsonify({'message': 'Server shutdown initiated'}), 200
    except Exception as e:
        return jsonify({'error': f'Shutdown failed: {str(e)}'}), 500

if __name__ == '__main__':
    def shutdown_monitor():
        while True:
            check_for_shutdown()
            time.sleep(1)
    
    monitor_thread = threading.Thread(target=shutdown_monitor, daemon=True)
    monitor_thread.start()
    
    app.run(debug=True)
