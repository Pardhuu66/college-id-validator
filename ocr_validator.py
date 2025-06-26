import easyocr
import re
import csv
from thefuzz import fuzz, process
import os
from PIL import Image
import io
import numpy as np

# --- Configuration ---
COLLEGE_NAME_SIMILARITY_THRESHOLD = 80
STUDENT_NAME_KEYWORDS = ["name", "student name", "student", "holder name", "name of student", "s/o", "d/o", "w/o"]
ROLL_NUMBER_KEYWORDS = ["roll no", "roll number", "id no", "reg no", "registration no", "enrollment no", "student id", "admission no", "sr no"]
NON_GENERAL_FIELD_INDICATORS = [
    "college", "university", "institute", "vidyalaya", "polytechnic", "school", "vidyapeeth",
    "department", "dept", "branch", "faculty", "programme", "course", "stream",
    "card", "identity", "session", "academic year", "valid till", "date of birth", "dob", "issue date", "expiry date",
    "address", "city", "state", "pin", "email", "phone", "mobile", "contact",
    "principal", "director", "dean", "signature", "authorized", "controller", "examinations",
    "library", "hostel", "batch", "year", "semester", "class", "degree", "diploma", "certificate",
    "permanent", "temporary", "affiliation", "affiliated", "government", "india", "tech", "autonomous"
]
EXACT_LINE_EXCLUSIONS_FOR_NAME = ["identity card", "student card", "student id card", "id card", "student identity card"]
COMMON_INTERMEDIATE_HEADERS = ["identity card", "student id", "student card", "id card", "student identity card", "office copy", "student copy", "details"]

# Initialize OCR
'''try:
    OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
except Exception as e:
    OCR_READER = None
    print(f"Error initializing EasyOCR: {e}")'''
try:
    OCR_READER = easyocr.Reader(['en'], model_storage_directory='ocr_models/model', gpu=False, verbose=False)
except Exception as e:
    OCR_READER = None
    print(f"Error initializing EasyOCR: {e}")


def ocr_char_fixer(text, is_likely_roll_no=False):
    if not text: return text
    if any(char.isdigit() for char in text) or is_likely_roll_no:
        text = text.replace('O', '0').replace('o', '0')
        text = text.replace('I', '1').replace('l', '1')
        text = text.replace('S', '5').replace('s', '5')
        text = text.replace('B', '8')
        text = text.replace('Z', '2')
        if is_likely_roll_no: text = text.replace('G', '6')
    return text

def extract_text_from_image_bytes(image_bytes):
    if OCR_READER is None: return [], []
    try:
        results = OCR_READER.readtext(image_bytes, detail=1, paragraph=False, batch_size=4, decoder='beamsearch', beamWidth=10)
        return [res[1] for res in results], results
    except Exception as e:
        print(f"Error during OCR: {e}")
        return [], []

def validate_college_name(ocr_texts, known_colleges, threshold=COLLEGE_NAME_SIMILARITY_THRESHOLD):
    if not known_colleges: return False, None, [], "Known colleges list is empty."
    if not ocr_texts: return False, None, [], "No text for college name validation."
    potential_matches = []
    max_lines_for_college = min(len(ocr_texts), 4)
    for i in range(max_lines_for_college):
        ocr_line = ocr_texts[i]
        ocr_line_lower = ocr_line.lower()
        if len(ocr_line_lower) < 5: continue
        match = process.extractOne(ocr_line_lower, known_colleges, scorer=fuzz.token_set_ratio)
        if match:
            score = match[1]
            known_college_matched = match[0]
            potential_matches.append({
                "score": score,
                "ocr_text": ocr_line,
                "known_text": known_college_matched,
                "line_indices": [i]
            })
    for num_lines_to_combine in range(2, max_lines_for_college + 1):
        if len(ocr_texts) >= num_lines_to_combine:
            current_segment_lines = ocr_texts[:num_lines_to_combine]
            combined_ocr_text = " ".join(current_segment_lines)
            combined_ocr_text_lower = combined_ocr_text.lower()
            if len(combined_ocr_text_lower) < 10: continue
            match = process.extractOne(combined_ocr_text_lower, known_colleges, scorer=fuzz.token_set_ratio)
            if match:
                score = match[1]
                known_college_matched = match[0]
                potential_matches.append({
                    "score": score,
                    "ocr_text": combined_ocr_text,
                    "known_text": known_college_matched,
                    "line_indices": list(range(num_lines_to_combine))
                })
    if not potential_matches:
        return False, None, [], "No potential college name segments found or matched."
    valid_matches = [m for m in potential_matches if m["score"] >= threshold]
    if not valid_matches:
        best_below_threshold = sorted(potential_matches, key=lambda x: x['score'], reverse=True)[0]
        msg = f"College name not matched above threshold (best score: {best_below_threshold['score']}% for '{best_below_threshold['known_text']}' vs OCR '{best_below_threshold['ocr_text']}')."
        return False, None, [], msg
    best_valid_match = sorted(valid_matches, key=lambda x: (x['score'], len(x['line_indices'])), reverse=True)[0]
    return True, best_valid_match["known_text"].title(), best_valid_match["line_indices"], \
           f"Matched '{best_valid_match['known_text'].title()}' (score {best_valid_match['score']}%) from OCR '{best_valid_match['ocr_text']}'."

def collect_field_candidates(ocr_texts, claimed_line_indices, keywords, pattern_regex, field_type, general_exclusions=None, exact_line_exclusions=None, college_line_indices=None):
    candidates = []
    if general_exclusions is None: general_exclusions = []
    if exact_line_exclusions is None: exact_line_exclusions = []
    for i, text_line in enumerate(ocr_texts):
        if i in claimed_line_indices: continue
        cleaned_line = text_line.strip()
        line_lower = cleaned_line.lower()
        if not cleaned_line or line_lower in exact_line_exclusions: continue
        is_keyword_match_for_this_line = False
        for kw_idx, kw in enumerate(keywords):
            kw_lower = kw.lower()
            if line_lower.startswith(kw_lower):
                value_candidate_text = cleaned_line[len(kw):].strip(": ").strip()
                if value_candidate_text and pattern_regex.fullmatch(value_candidate_text):
                    candidates.append({"text": value_candidate_text, "score": 100 + (len(keywords)-kw_idx), "line_idx": i, "method": f"{field_type}_keyword_current"})
                    is_keyword_match_for_this_line = True
                    break 
            if i > 0 and (i-1) not in claimed_line_indices:
                prev_line_content = ocr_texts[i-1].strip().lower().rstrip(':').strip()
                if prev_line_content == kw_lower and pattern_regex.fullmatch(cleaned_line):
                    candidates.append({"text": cleaned_line, "score": 95 + (len(keywords)-kw_idx), "line_idx": i, "method": f"{field_type}_keyword_prev"})
                    is_keyword_match_for_this_line = True
                    break
        if is_keyword_match_for_this_line: continue
        is_general_header = False
        num_excluded_words = sum(1 for word in line_lower.split() if word in general_exclusions)
        total_words_in_line = len(line_lower.split())
        if total_words_in_line > 0 and (num_excluded_words / total_words_in_line) > 0.6: is_general_header = True
        if cleaned_line.isupper() and total_words_in_line <= 3 and num_excluded_words >=1: is_general_header = True
        if is_general_header: continue
        if pattern_regex.fullmatch(cleaned_line):
            score = 80
            if field_type == "student_name" and college_line_indices and i == max(college_line_indices, default=-1) + 1:
                if line_lower not in (general_exclusions + exact_line_exclusions + COMMON_INTERMEDIATE_HEADERS): score = 85
                else: continue 
            candidates.append({"text": cleaned_line, "score": score, "line_idx": i, "method": f"{field_type}_pattern"})
    return candidates

def validate_id_card_fields(image_bytes, known_colleges_list):
    results = {
        "college_name_found": False, "college_name_valid": False, "matched_college_name": None,
        "student_name_found": False, "extracted_student_name": None,
        "roll_number_found": False, "extracted_roll_number": None,
        "overall_status": "REJECTED", "reasons": [], "ocr_texts": [], "claimed_line_indices": set()
    }
    known_colleges = [c.lower() for c in known_colleges_list] if known_colleges_list else []
    ocr_texts, _ = extract_text_from_image_bytes(image_bytes)
    results["ocr_texts"] = ocr_texts
    if not ocr_texts and not results["reasons"]:
        results["reasons"].append("OCR failed to extract any text.")
    
    is_college_ok, college_str, college_line_indices, college_msg = validate_college_name(ocr_texts, known_colleges)
    results["college_name_found"] = bool(college_str)
    results["college_name_valid"] = is_college_ok
    results["matched_college_name"] = college_str
    if is_college_ok:
        [results["claimed_line_indices"].add(idx) for idx in college_line_indices]
    else:
        results["reasons"].append(college_msg or "College name validation failed.")

    name_pattern = re.compile(r"^(?:[A-Z][a-zA-Z'.\-]+)(?:\s+[A-Z][a-zA-Z'.\-]+){0,3}$")
    student_name_candidates = collect_field_candidates(
        ocr_texts, results["claimed_line_indices"], STUDENT_NAME_KEYWORDS, name_pattern, "student_name",
        NON_GENERAL_FIELD_INDICATORS, EXACT_LINE_EXCLUSIONS_FOR_NAME, college_line_indices
    )
    if student_name_candidates:
        best_name_candidate = sorted(student_name_candidates, key=lambda x: x["score"], reverse=True)[0]
        results["student_name_found"] = True
        results["extracted_student_name"] = best_name_candidate["text"]
        results["claimed_line_indices"].add(best_name_candidate["line_idx"])
    else:
        results["reasons"].append("Student name not identified.")
    
    roll_pattern_list = [
        re.compile(patt) for patt in [
            r"^(?:[A-Z0-9]{2,4}\d{2,7}[A-Z0-9]{0,7})$",
            r"^(?:\d{2,4}[A-Z0-9]{2,5}\d{2,7}[A-Z0-9]{0,2})$",
            r"^(?=.*\d)[A-Z0-9]{6,15}$",
            r"^(?=[A-Z0-9]*[0-9])(?:[A-Z0-9]{2,7}(?:\s?[A-Z0-9]{2,7}){1,3})$",
            r"^(?:\d{7,12})$"
        ]
    ]
    roll_candidates = []
    just_letters_pattern_roll = re.compile(r"^[A-Z]{5,}$")
    date_like_pattern_roll = re.compile(r"\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4})\b")
    for i, text_line in enumerate(ocr_texts):
        if i in results["claimed_line_indices"]: continue
        cleaned_line = text_line.strip()
        if not cleaned_line or len(cleaned_line) < 3 or date_like_pattern_roll.search(cleaned_line): continue
        line_lower = cleaned_line.lower()
        keyword_found_this_line = False
        for kw_idx, kw in enumerate(ROLL_NUMBER_KEYWORDS):
            kw_lower = kw.lower()
            current_line_val_for_kw = ""
            from_prev_line_val_for_kw = ""
            if line_lower.startswith(kw_lower):
                current_line_val_for_kw = cleaned_line[len(kw):].strip(": ").strip()
            elif i > 0 and (i-1) not in results["claimed_line_indices"] and ocr_texts[i-1].strip().lower().rstrip(':').strip() == kw_lower:
                from_prev_line_val_for_kw = cleaned_line
            val_to_check = current_line_val_for_kw or from_prev_line_val_for_kw
            score_base = 100 if current_line_val_for_kw else 95
            if val_to_check:
                normalized_val = ocr_char_fixer(val_to_check.upper().replace(" ", ""), is_likely_roll_no=True)
                if not (just_letters_pattern_roll.fullmatch(normalized_val) and len(normalized_val) < 7):
                    for rp in roll_pattern_list:
                        if rp.fullmatch(normalized_val) and len(normalized_val) >=6:
                            roll_candidates.append({"text": normalized_val, "score": score_base + (10-kw_idx), "line_idx": i, "method": "roll_keyword"})
                            keyword_found_this_line = True
                            break
            if keyword_found_this_line: break
        if keyword_found_this_line: continue
        normalized_line = ocr_char_fixer(cleaned_line.upper().rstrip().replace(" ", ""), is_likely_roll_no=True)
        if len(normalized_line) < 6 or len(normalized_line) > 15 or \
           (just_letters_pattern_roll.fullmatch(normalized_line) and len(normalized_line) < 7): continue
        for rp in roll_pattern_list:
            if rp.fullmatch(normalized_line):
                roll_candidates.append({"text": normalized_line, "score": 70, "line_idx": i, "method": "roll_pattern_full_line"})
                break
    if roll_candidates:
        best_roll = sorted(roll_candidates, key=lambda x: x["score"], reverse=True)[0]
        results["roll_number_found"] = True
        results["extracted_roll_number"] = best_roll["text"]
        results["claimed_line_indices"].add(best_roll["line_idx"])
    else:
        results["reasons"].append("Roll number not identified.")

    if results["college_name_valid"] and results["student_name_found"] and results["roll_number_found"]:
        results["overall_status"] = "ACCEPTED"
    else:
        results["overall_status"] = "REJECTED"
        if not results["reasons"]:
            results["reasons"].append("One or more required fields failed validation.")
    if "claimed_line_indices" in results:
        del results["claimed_line_indices"]
    return results