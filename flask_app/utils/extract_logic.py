# utils/extract_logic.py (updated)
import pandas as pd
import pytesseract
from PIL import Image
import textract
from pdf2image import convert_from_path
import mimetypes
import os
import re
import nltk
from nltk.corpus import stopwords
import ollama
import time
import traceback
from datetime import datetime

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def verify_llama_setup():
    try:
        response = ollama.generate(
            model='llama3',
            prompt='2+2=',
            options={'temperature': 0.0}
        )
        if response and 'response' in response:
            print("✅ Llama3 is working")
            return True
        return False
    except ollama.ResponseError as e:
        if 'model not found' in str(e).lower():
            print("❌ llama3 model not found")
            print("Download with: ollama pull llama3")
        else:
            print(f"Llama error: {str(e)}")
    except ConnectionError:
        print("Ollama service not running. Start with: ollama serve")
    except Exception as e:
        print(f"Llama verification failed: {str(e)}")
        traceback.print_exc()
    print("Falling back to traditional extraction only")
    return False

LLAMA_AVAILABLE = verify_llama_setup()

def init_ranking_data(excel_path, sheet_name):
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=1)
    if 'Name of Institution' not in df.columns:
        raise ValueError("Ranking sheet must include 'Name of Institution' column.")
    
    institution_list = df['Name of Institution'].dropna().unique().tolist()
    
    ranking_dict = {}
    normalized_ranking_dict = {}
    
    for name in institution_list:
        lower_name = name.lower()
        ranking_dict[lower_name] = name
        
        normalized = _normalize_institution_name(name)
        normalized_ranking_dict[normalized] = name
    
    return df, institution_list, ranking_dict, normalized_ranking_dict

def _normalize_institution_name(name):
    if pd.isna(name):
        return ""
    name = str(name)
    name = re.sub(r'\([^)]*\)', '', name).strip()
    name = re.split(r',|\s+-\s+', name)[0].strip()
    return name.lower()

def correct_ocr_errors(text):
    corrections = {
        r'(\b)un[i1]versity(\b)': r'\1university\2',
        r'(\b)co[l1]lege(\b)': r'\1college\2',
        r'(\b)techno[1l]ogy(\b)': r'\1technology\2',
        r'(\b)institute(\b)': r'\1institute\2',
        r'(\b)v[i1]svesvaraya(\b)': r'\1visvesvaraya\2',
        r'(\b)be[1l]agavi(\b)': r'\1belagavi\2',
        r'(\b)mu[mn][b8]ai(\b)': r'\1mumbai\2',
        r'(\b)pu[rn]e(\b)': r'\1pune\2',
        r'(\b)de[i1]hi(\b)': r'\1delhi\2'
    }
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def extract_text_from_file(file_path):
    file_type, _ = mimetypes.guess_type(file_path)
    print(f"\n🔍 Extracting text from: {file_path}")
    
    try:
        if file_path.lower().endswith(".pdf"):
            try:
                text = textract.process(file_path).decode('utf-8').strip()
                if len(text) > 100:
                    print("✅ Used textract for PDF")
                    return correct_ocr_errors(text)
            except Exception as e:
                print(f"⚠️ Textract failed: {e}, falling back to OCR")
            
            try:
                text = ""
                images = convert_from_path(file_path, dpi=300)
                for i, img in enumerate(images):
                    print(f"  Processing page {i+1}/{len(images)}")
                    text += pytesseract.image_to_string(img) + "\n\n"
                print("✅ Used OCR for PDF")
                return correct_ocr_errors(text)
            except Exception as e:
                print(f"❌ OCR failed: {e}")
                return ""
        
        elif file_path.lower().endswith((".docx", ".doc")):
            try:
                text = textract.process(file_path).decode('utf-8')
                print("✅ Used textract for DOCX")
                return correct_ocr_errors(text)
            except Exception as e:
                print(f"❌ DOCX processing failed: {e}")
                return ""
        
        elif file_type and "image" in file_type:
            try:
                image = Image.open(file_path)
                image = image.convert('L')
                text = pytesseract.image_to_string(image)
                print("✅ Used OCR for image")
                return correct_ocr_errors(text)
            except Exception as e:
                print(f"❌ Image processing failed: {e}")
                return ""
        
        elif file_path.lower().endswith(".txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                print("✅ Read text file directly")
                return text
            except Exception as e:
                print(f"❌ Text file read failed: {e}")
                return ""
        
        print(f"⚠️ Unsupported file type: {file_type}")
        return ""
    
    except Exception as e:
        print(f"❌ Critical error processing file: {e}")
        traceback.print_exc()
        return ""

def llama_extract_institution(text):
    try:
        response = ollama.generate(
            model='llama3',
            prompt=f"""
            ### DOCUMENT EXTRACTION TASK: UNIVERSITY IDENTIFICATION ###
            Follow these rules ABSOLUTELY:

            1. **PRIORITY & SCOPE (STRICT):**
            - Extract ONLY the institution for a BACHELOR'S DEGREE
            - If NO bachelor's exists, only THEN consider master's

            2. **INSTITUTION CLEANING RULES (NEW):**
            - Extract PURE university name (NO locations, campuses, departments)
            - If name contains comma: 
                    • Keep ONLY the segment with "University" or "College" 
                    • If none: keep FIRST segment
            - REMOVE degree names, dates, and honorifics
            - PRESERVE original spelling/capitalization

            3. **AFFILIATION FALLBACK (NEW):**
            - If extracted institute contains "university of [City]" pattern:
                    • Extract ONLY the university portion (e.g., "University of Mumbai")
            - Apply when institute name includes "university of" or "college of"

            4. **DOCUMENT STRUCTURE:**
            - Valid pattern: [Institution] → [Date Range] → [Degree]
            - Degree MUST appear within 5 lines below institution

            5. **EXTRACTION PROTOCOL:**
            a) Scan for ANY bachelor's degree
            b) Find CLOSEST preceding institution
            c) Apply cleaning rules
            d) Apply affiliation fallback if pattern exists
            e) If step (a) fails, repeat for master's

            6. **OUTPUT FORMAT:**
            INSTITUTION: [clean name OR "none"]
            DEGREE: ["Bachelor" OR "Master" OR "none"]
            EVIDENCE: ["EXACT institution text"]

            7. **FAILSAFES:**
            - TERMINATE after first valid bachelor
            - If NO degree link: return "none"

            --- DOCUMENT CONTENT ---
            {text}
            --- END DOCUMENT ---

            YOUR ANALYSIS (REQUIRED):
            Step 1: [Degree at line X] 
            Step 2: [Raw institution] 
            Step 3: [Cleaned name] 
            Step 4: [Affiliation fallback?] 
            Step 5: [Final institution]
            """,
            options={
                'temperature': 0.1,
                'num_ctx': 4096
            },
            stream=False
        )
        response_text = response['response'].strip()
        
        institution_name = None
        degree_level = None
        evidence = None
        
        inst_match = re.search(r'INSTITUTION:\s*(.*?)$', response_text, re.IGNORECASE | re.MULTILINE)
        degree_match = re.search(r'DEGREE:\s*(.*?)$', response_text, re.IGNORECASE | re.MULTILINE)
        
        if inst_match:
            institution_name = inst_match.group(1).strip()
            if institution_name.lower() in ["none", "not found", "n/a"]:
                institution_name = None
            else:
                if institution_name.lower().startswith("the "):
                    institution_name = institution_name[4:]
                if institution_name.endswith("."):
                    institution_name = institution_name[:-1]
        
        if degree_match:
            degree_level = degree_match.group(1).strip()
        
        evidence_match = re.search(r'EVIDENCE:\s*(.*?)(?:\n\n|\Z)', response_text, re.IGNORECASE | re.DOTALL)
        if evidence_match:
            evidence = evidence_match.group(1).strip()
            if evidence.lower() == "none":
                evidence = None
        
        return institution_name, evidence, degree_level, response_text
        
    except Exception as e:
        print(f"Llama extraction error: {e}")
        return None, None, None, None

def _find_institution_fallback(text, ranking_dict, normalized_ranking_dict):
    normalized_text = ' '.join(text.lower().split()).replace('-', ' ').replace('.', '')
    evidence = ""
    
    for name in ranking_dict.values():
        if name.lower() in normalized_text:
            for line in text.split('\n'):
                if name.lower() in line.lower():
                    evidence = line.strip()
                    break
            return name, evidence
    
    text_normalized = _normalize_institution_name(normalized_text)
    for normalized, original in normalized_ranking_dict.items():
        if normalized in text_normalized:
            for line in text.split('\n'):
                if original.lower() in line.lower() or normalized in line.lower():
                    evidence = line.strip()
                    break
            return original, evidence
    
    text_tokens = set(text_normalized.split())
    for normalized, original in normalized_ranking_dict.items():
        inst_tokens = set(normalized.split())
        if len(inst_tokens & text_tokens) / len(inst_tokens) > 0.8:
            best_line = ""
            best_score = 0
            for line in text.split('\n'):
                line_tokens = set(_normalize_institution_name(line).split())
                similarity = len(inst_tokens & line_tokens) / len(inst_tokens)
                if similarity > best_score:
                    best_score = similarity
                    best_line = line.strip()
            return original, best_line
    
    return None, ""

def lookup_institution_ranking(name, ranking_df, extracted_text=None):
    name_lower = name.lower()
    matches = ranking_df[ranking_df['Name of Institution'].str.lower() == name_lower]

    if matches.empty:
        norm_name = _normalize_institution_name(name)
        norm_matches = ranking_df[
            ranking_df['Name of Institution'].str.lower().apply(_normalize_institution_name) == norm_name
        ]
        
        if not norm_matches.empty:
            matches = norm_matches
        else:
            name_tokens = set(name_lower.split())
            ranking_df['token_match'] = ranking_df['Name of Institution'].str.lower().apply(
                lambda x: len(set(x.split()) & name_tokens) / len(set(x.split())))
            token_matches = ranking_df[ranking_df['token_match'] > 0.7].sort_values('token_match', ascending=False)
            
            if not token_matches.empty:
                matches = token_matches.head(1)

    if matches.empty:
        return None

    if len(matches) == 1:
        row = matches.iloc[0]
    else:
        filtered = matches.copy()
        if extracted_text:
            extracted_text = extracted_text.lower()
            if 'City' in ranking_df.columns:
                city_col = [c for c in ranking_df.columns if c.lower() == 'city'][0]
                filtered = filtered[filtered[city_col].astype(str).str.lower().apply(lambda c: c in extracted_text)]
            
            if 'State' in ranking_df.columns and len(filtered) > 1:
                state_col = [c for c in ranking_df.columns if c.lower() == 'state'][0]
                filtered = filtered[filtered[state_col].astype(str).str.lower().apply(lambda s: s in extracted_text)]
        
        row = filtered.iloc[0] if not filtered.empty else matches.iloc[0]

    tier_1_cols = [col for col in ranking_df.columns if "top 100" in col.lower()]
    tier_2_cols = [col for col in ranking_df.columns if "101-200" in col.lower()]
    global_cols = [col for col in ranking_df.columns if "global" in col.lower()]

    tier_1 = {col: row[col] for col in tier_1_cols if col in row and pd.notna(row[col])}
    tier_2 = {col: row[col] for col in tier_2_cols if col in row and pd.notna(row[col])}
    global_rank = {col: row[col] for col in global_cols if col in row and pd.notna(row[col])}

    result = {
        "Name of Institution": row["Name of Institution"],
        "City": row.get("CITY") or row.get("City"),
        "State": row.get("STATE") or row.get("State"),
    }
    
    if tier_1: result["Tier 1"] = tier_1
    if tier_2: result["Tier 2"] = tier_2
    if global_rank: result["Global"] = global_rank
    
    return result

def process_student_files(transcript_path=None, cv_path=None, reference_paths=None, 
                          ranking_df=None, institution_list=None, ranking_dict=None,
                          normalized_ranking_dict=None):
    source = None
    matched_name = None
    raw_text = ""
    llm_used = False
    llm_evidence = None
    llm_thought_process = None  # NEW: Variable to store thought process
    llm_degree = None
    extraction_method = "Traditional"
    match_snippet = ""
    
    docs = [
        ("Transcript", transcript_path),
        ("CV", cv_path)
    ]
    if reference_paths:
        docs.extend([(f"Reference {i+1}", path) for i, path in enumerate(reference_paths)])
    
    # Cache for extracted text to avoid reprocessing
    doc_texts = {}
    
    if LLAMA_AVAILABLE:
        for doc_type, path in docs:
            if not path:
                continue
                
            # Cache text extraction per file
            if path not in doc_texts:
                doc_texts[path] = extract_text_from_file(path)
            text = doc_texts[path]
            
            if not raw_text:
                raw_text = text
                
            print(f"Trying Llama extraction on {doc_type}...")
            start_time = time.time()
            
            # SINGLE LLM CALL PER DOCUMENT
            institution_name, evidence, degree_level, full_response = llama_extract_institution(text)
            llm_thought_process = full_response  # NEW: Always store the latest thought process
            
            # Print the thought process for debugging
            if full_response:
                print("\n" + "="*50)
                print(f"Llama's Thought Process for {doc_type}:")
                print(full_response)
                print("="*50 + "\n")
            
            llm_evidence = evidence
            llm_degree = degree_level
            
            if institution_name:
                print(f"Llama extracted: {institution_name}")
                if evidence:
                    print(f"Evidence: {evidence}")
                print(f"Extraction took {time.time()-start_time:.2f} seconds")
                
                # Use advanced ranking lookup
                result_from_llama = lookup_institution_ranking(
                    institution_name, 
                    ranking_df, 
                    text  # Pass current document text
                )
                
                if result_from_llama:
                    # Found match in ranking data - return immediately
                    llm_used = True
                    extraction_method = "Llama"
                    result_from_llama["_source"] = f"{doc_type} (Llama)"
                    result_from_llama["_raw_text"] = raw_text
                    result_from_llama["_llm_used"] = llm_used
                    result_from_llama["_llm_evidence"] = evidence
                    result_from_llama["_llm_thought_process"] = full_response  # NEW: Store in result
                    result_from_llama["_llm_degree"] = degree_level
                    result_from_llama["_match_snippet"] = ""
                    result_from_llama["_method"] = extraction_method
                    return result_from_llama
                else:
                    print(f"⚠️ Institution '{institution_name}' not found in ranking list")

    # Traditional matching fallback
    print("Falling back to traditional matching...")
    extraction_method = "Traditional"
    
    for doc_type, path in docs:
        if not path or matched_name:
            continue
            
        # Use cached text if available
        if path not in doc_texts:
            doc_texts[path] = extract_text_from_file(path)
        text = doc_texts[path]
        
        if not raw_text:
            raw_text = text
            
        matched_name, snippet = _find_institution_fallback(
            text, 
            ranking_dict, 
            normalized_ranking_dict
        )
        
        if matched_name:
            source = doc_type
            match_snippet = snippet
            break

    if not matched_name:
        result = {
            "_raw_text": raw_text,
            "_source": source if source else "No match",
            "_llm_used": llm_used,
            "_llm_evidence": llm_evidence,
            "_llm_thought_process": llm_thought_process,  # NEW: Include thought process
            "_llm_degree": llm_degree,
            "_match_snippet": match_snippet,
            "_method": extraction_method
        }
        return result

    result = lookup_institution_ranking(matched_name.lower(), ranking_df, raw_text)
    if result:
        result["_source"] = source
        result["_raw_text"] = raw_text
        result["_llm_used"] = llm_used
        result["_llm_evidence"] = llm_evidence
        result["_llm_thought_process"] = llm_thought_process
        result["_llm_degree"] = llm_degree
        result["_match_snippet"] = match_snippet
        result["_method"] = extraction_method
    return result
