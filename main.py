#!/usr/bin/env python3
"""
DYNAMIC DOCUMENT TRANSLATOR BACKEND
Real OCR processing, translation, and image manipulation
"""

import easyocr
import os
import ollama
from PIL import Image, ImageDraw, ImageFont
import webbrowser
import base64
from io import BytesIO
import numpy as np
import textwrap
import json
import tempfile
import cv2
from flask import Flask, request, jsonify, send_file
import threading
import time

app = Flask(__name__)

# Global variables to store processing data
current_image = None
current_bboxes = []
current_translated_text = []
current_image_path = None
current_target_language = 'te'

def process_image_with_ocr(image_path):
    """Process image with EasyOCR and return bounding boxes and text"""
    print(f"Processing image: {image_path}")
    
    # Try multiple language combinations for better detection
    language_combinations = [
        ['en'],  # English only
        ['en', 'hi'],  # English + Hindi
        ['en', 'te'],  # English + Telugu
        ['en', 'hi', 'te'],  # English + Hindi + Telugu
        ['en', 'hi', 'te', 'ta', 'kn', 'ml', 'gu', 'pa', 'bn', 'or'],  # All major Indian languages
    ]
    
    bboxes = []
    
    for lang_combo in language_combinations:
        try:
            print(f"Trying OCR with languages: {lang_combo}")
            reader = easyocr.Reader(lang_combo)
            results = reader.readtext(image_path, detail=1, paragraph=True)
            
            for res in results:
                if len(res) >= 2:
                    bbox_info = {
                        "bbox": res[0], 
                        "text": res[1],
                        "confidence": res[2] if len(res) >= 3 else 0.9
                    }
                    bboxes.append(bbox_info)
            
            if bboxes:
                print(f"Found {len(bboxes)} text regions with {lang_combo}")
                break
                
        except Exception as e:
            print(f"OCR failed with {lang_combo}: {e}")
            continue
    
    # If still no text found, try without paragraph mode
    if not bboxes:
        print("Trying OCR without paragraph mode...")
        try:
            reader = easyocr.Reader(['en'])
            results = reader.readtext(image_path, detail=1, paragraph=False)
            
            for res in results:
                if len(res) >= 2:
                    bbox_info = {
                        "bbox": res[0], 
                        "text": res[1],
                        "confidence": res[2] if len(res) >= 3 else 0.9
                    }
                    bboxes.append(bbox_info)
            
            print(f"Found {len(bboxes)} text regions without paragraph mode")
        except Exception as e:
            print(f"OCR failed without paragraph mode: {e}")
    
    if not bboxes:
        print("⚠️ No text detected! The image might be:")
        print("   - Too blurry or low quality")
        print("   - Contains only images/logos without text")
        print("   - Text is too small or unclear")
        print("   - Contains handwritten text (EasyOCR works best with printed text)")
    
    return bboxes

def translate_text(text, model='gemma3-legal-samanantar-pro:latest', target_language='te'):
    """Translate text using Ollama with fallback"""
    print("Translating text...")
    
    # Get language name and prompt
    lang_info = get_language_info(target_language)
    
    if target_language == 'en':
        # For English, improve the text rather than translate
        prompt = f"""Improve this government document text for better clarity and formal tone while maintaining all legal meaning:

{text}

Provide improved English version:"""
    else:
        # For other languages, translate
        prompt = f"""Translate this government document to {lang_info['name']} ({lang_info['native']}) maintaining:

1. Exact legal meaning and structure
2. Formal government document tone  
3. All dates, numbers, and references intact
4. Complete sentences and proper grammar

Document:
{text}

Provide complete translation in {lang_info['name']}:"""
    
    try:
        translated_text = ollama.generate(
            model=model,
            prompt=prompt
        )['response']
        print(f"Translation to {lang_info['name']} completed")
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        print("Using fallback translation method...")
        return fallback_translation(text, target_language)

def fallback_translation(text, target_language):
    """Fallback translation when Ollama is unavailable"""
    print(f"Using fallback translation for {target_language}")
    
    # Simple fallback translations for common legal terms
    fallback_translations = {
        'te': {  # Telugu
            'OFFICE OF THE REGISTRAR GENERAL': 'రిజిస్ట్రార్ జనరల్ కార్యాలయం',
            'Government of India': 'భారత ప్రభుత్వం',
            'Ministry of Home Affairs': 'గృహ వ్యవహారాల మంత్రిత్వ శాఖ',
            'TENDER ENQUIRY NOTICE': 'టెండర్ విచారణ నోటీసు',
            'Subject:': 'విషయం:',
            'Dated:': 'తేదీ:',
            'No.': 'సంఖ్య:',
            'Procurement': 'కొనుగోలు',
            'Supply': 'సరఫరా',
            'Services': 'సేవలు',
            'Contract': 'ఒప్పందం',
            'Agreement': 'ఒప్పందం',
            'Terms and Conditions': 'నిబంధనలు మరియు షరతులు'
        },
        'hi': {  # Hindi
            'OFFICE OF THE REGISTRAR GENERAL': 'रजिस्ट्रार जनरल का कार्यालय',
            'Government of India': 'भारत सरकार',
            'Ministry of Home Affairs': 'गृह मंत्रालय',
            'TENDER ENQUIRY NOTICE': 'निविदा पूछताछ नोटिस',
            'Subject:': 'विषय:',
            'Dated:': 'दिनांक:',
            'No.': 'संख्या:',
            'Procurement': 'खरीद',
            'Supply': 'आपूर्ति',
            'Services': 'सेवाएं',
            'Contract': 'अनुबंध',
            'Agreement': 'समझौता',
            'Terms and Conditions': 'नियम और शर्तें'
        },
        'ta': {  # Tamil
            'OFFICE OF THE REGISTRAR GENERAL': 'பதிவாளர் ஜெனரல் அலுவலகம்',
            'Government of India': 'இந்திய அரசு',
            'Ministry of Home Affairs': 'உள்துறை அமைச்சகம்',
            'TENDER ENQUIRY NOTICE': 'டெண்டர் விசாரணை அறிவிப்பு',
            'Subject:': 'பொருள்:',
            'Dated:': 'தேதி:',
            'No.': 'எண்:',
            'Procurement': 'கொள்முதல்',
            'Supply': 'வழங்கல்',
            'Services': 'சேவைகள்',
            'Contract': 'ஒப்பந்தம்',
            'Agreement': 'ஒப்பந்தம்',
            'Terms and Conditions': 'விதிமுறைகள் மற்றும் நிபந்தனைகள்'
        }
    }
    
    # Get translations for the target language
    translations = fallback_translations.get(target_language, {})
    
    # Apply translations
    translated_text = text
    for english_term, translated_term in translations.items():
        translated_text = translated_text.replace(english_term, translated_term)
    
    # If no translations were applied, return original text with a note
    if translated_text == text:
        translated_text = f"[{target_language.upper()}] {text} [Translation unavailable - Ollama service required]"
    
    return translated_text

def translate_text_with_agents(text, model='gemma3-legal-samanantar-pro:latest', target_language='te'):
    """Translate text using simplified legal translation"""
    print("⚖️ Starting Legal Translation...")
    
    lang_info = get_language_info(target_language)
    
    try:
        # Simplified but effective legal translation approach
        print(f"🔍 Translating to {lang_info['name']}...")
        
        # Direct legal translation prompt
        prompt = f"""You are a professional legal translator. Translate this government document to {lang_info['name']} ({lang_info['native']}) maintaining:

1. EXACT legal meaning and structure
2. Formal government document tone
3. All dates, numbers, and references intact
4. Legal terminology accuracy
5. Complete sentences and proper grammar

Document to translate:
{text}

Provide the complete translation in {lang_info['name']}:"""
        
        translated_text = ollama.generate(model=model, prompt=prompt)['response']
        
        print(f"✅ Translation to {lang_info['name']} completed")
        return translated_text
        
    except Exception as e:
        print(f"Legal translation error: {e}")
        print("Using fallback translation method...")
        return fallback_translation(text, target_language)

def execute_context_agent(text, model, target_language):
    """Agent 1: Context Agent - Legal Specialist"""
    lang_info = get_language_info(target_language)
    
    prompt = f"""You are a Legal Context Specialist. Analyze this document and identify:

1. DOCUMENT TYPE: What specific legal document is this? (Contract, NDA, SLA, Court Filing, Patent, etc.)
2. LEGAL JURISDICTION: Which legal system? (US, EU, India, UK, etc.)
3. LEGAL DOMAIN: What area of law? (Corporate, IP, Employment, Real Estate, etc.)
4. FORMALITY LEVEL: How formal/technical is the language?
5. CRITICAL ELEMENTS: What legal concepts are central?

Document to analyze:
{text}

Respond with structured analysis for {lang_info['name']} translation:"""
    
    return ollama.generate(model=model, prompt=prompt)['response']

def execute_terms_agent(text, model, context_result):
    """Agent 2: Unknown Terms Agent - Legal Jargon Spotter"""
    
    prompt = f"""You are a Legal Terms Spotter. Based on this context analysis:

{context_result}

Identify ALL legal "terms of art" in this document:

1. LATIN TERMS: (res judicata, force majeure, etc.)
2. DEFINED TERMS: ("Hereinafter, 'The Company'...")
3. LEGAL VERBS: (indemnify, warrant, covenant, etc.)
4. LEGAL PHRASES: (notwithstanding, subject to, etc.)
5. TECHNICAL TERMS: (specific to the legal domain)

Document:
{text}

List each term with its legal significance and translation priority:"""
    
    return ollama.generate(model=model, prompt=prompt)['response']

def execute_knowledge_agent(terms_result, model, target_language):
    """Agent 3: Knowledge Agent - Legal Glossary Guardian"""
    lang_info = get_language_info(target_language)
    
    prompt = f"""You are a Legal Glossary Guardian. Based on these identified terms:

{terms_result}

Provide the SINGLE, CORRECT, NON-NEGOTIABLE translation for each legal term in {lang_info['name']}:

1. Use established legal terminology
2. Maintain exact legal meaning
3. No creative interpretation
4. Preserve legal weight and enforceability

Format as: TERM → CORRECT TRANSLATION (LEGAL SIGNIFICANCE)

For terms without established translations, mark as [REQUIRES LEGAL REVIEW]:"""
    
    return ollama.generate(model=model, prompt=prompt)['response']

def execute_translation_agent(text, model, target_language, context_result, knowledge_result):
    """Agent 4: Translation Agent - Precision Engine"""
    lang_info = get_language_info(target_language)
    
    prompt = f"""You are a Legal Translation Precision Engine. Translate this document to {lang_info['name']} with STRICT LEGAL PRECISION:

CONTEXT ANALYSIS:
{context_result}

LEGAL TERMS GLOSSARY:
{knowledge_result}

TRANSLATION RULES:
1. Preserve EXACT sentence structure and dependent clauses
2. Maintain ALL conditional logic ("if-then" statements)
3. Keep ALL deadlines and time references intact
4. Preserve ALL legal obligations and rights
5. NO stylistic changes that could alter legal meaning
6. Use ONLY the approved legal terms from the glossary

Document to translate:
{text}

Provide precise legal translation:"""
    
    return ollama.generate(model=model, prompt=prompt)['response']

def execute_causality_agent(original_text, translated_text, model, target_language):
    """Agent 5: Causality Judge Agent - Guardian of Obligations"""
    lang_info = get_language_info(target_language)
    
    prompt = f"""You are a Legal Causality Judge. Verify that ALL legal obligations and conditions are preserved:

ORIGINAL TEXT:
{original_text}

TRANSLATED TEXT:
{translated_text}

Check for:
1. CONDITIONAL CLAUSES: Every "if-then" statement preserved
2. LIABILITY STATEMENTS: All obligations maintained
3. DEADLINES: All time references intact ("within 30 days", etc.)
4. LOGICAL FLOW: Cause-and-effect relationships preserved
5. LEGAL CONSEQUENCES: All penalties/consequences maintained

If ANY legal causality is altered, provide corrected translation. If preserved, confirm "LEGAL CAUSALITY VERIFIED":"""
    
    return ollama.generate(model=model, prompt=prompt)['response']

def execute_consistency_agent(text, model, target_language, terms_result):
    """Agent 6: Language Consistency Agent - Enforcer of Definitions"""
    lang_info = get_language_info(target_language)
    
    prompt = f"""You are a Legal Consistency Enforcer. Ensure ALL defined terms are translated EXACTLY the same way every time:

IDENTIFIED TERMS:
{terms_result}

TRANSLATED TEXT:
{text}

Check for:
1. DEFINED TERMS: Every instance uses identical translation
2. LEGAL REFERENCES: Consistent terminology throughout
3. CROSS-REFERENCES: All internal references maintained
4. DEFINITION CONSISTENCY: No variations in legal definitions

If inconsistencies found, provide corrected version. If consistent, confirm "LEGAL CONSISTENCY VERIFIED":"""
    
    return ollama.generate(model=model, prompt=prompt)['response']

def execute_validation_agent(original_text, translated_text, model, target_language):
    """Agent 7: Validation Agent - Final Legal Review"""
    lang_info = get_language_info(target_language)
    
    prompt = f"""You are a Legal Validation Specialist. Perform final legal review:

ORIGINAL TEXT:
{original_text}

TRANSLATED TEXT:
{translated_text}

FINAL VALIDATION CHECKLIST:
1. LEGAL ENFORCEABILITY: Does translated document have same legal effect?
2. PRECISION: Are all legal obligations preserved?
3. CONSISTENCY: Are all terms used consistently?
4. COMPLETENESS: Is no legal content lost or altered?
5. FORMALITY: Is appropriate legal tone maintained?

Provide final validated translation or mark areas requiring human legal review:"""
    
    return ollama.generate(model=model, prompt=prompt)['response']

def get_language_info(lang_code):
    """Get language information by code"""
    languages = {
        'te': {'name': 'Telugu', 'native': 'తెలుగు'},
        'hi': {'name': 'Hindi', 'native': 'हिन्दी'},
        'ta': {'name': 'Tamil', 'native': 'தமிழ்'},
        'kn': {'name': 'Kannada', 'native': 'ಕನ್ನಡ'},
        'ml': {'name': 'Malayalam', 'native': 'മലയാളം'},
        'en': {'name': 'English', 'native': 'English'},
        'gu': {'name': 'Gujarati', 'native': 'ગુજરાતી'},
        'pa': {'name': 'Punjabi', 'native': 'ਪੰਜਾਬੀ'},
        'bn': {'name': 'Bengali', 'native': 'বাংলা'},
        'or': {'name': 'Odia', 'native': 'ଓଡ଼ିଆ'}
    }
    return languages.get(lang_code, {'name': 'Telugu', 'native': 'తెలుగు'})

def create_processed_html(image_path, bboxes, translated_lines, user_actions, target_language='te'):
    """Create processed HTML document based on user actions"""
    print("Creating processed HTML document...")
    
    # Get language info for font selection
    lang_info = get_language_info(target_language)
    
    # Convert image to base64
    with open(image_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode()
    
    # Get image dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size
    
    # Create text overlays based on user actions
    text_overlays = []
    for i, bbox in enumerate(bboxes):
        action = user_actions.get(str(i), 'preserve')
        
        if action == 'whiteout':
            # Skip whiteout regions (they won't be rendered)
            print(f"Whiteout region {i}")
            continue
            
        elif action == 'translate' and i < len(translated_lines):
            text_to_draw = translated_lines[i].strip()
            print(f"Translate region {i}")
        else:
            text_to_draw = bbox['text']
            print(f"Preserve region {i}")
        
        # Convert bbox coordinates to percentages
        bbox_coords = bbox['bbox']
        x_coords = [point[0] for point in bbox_coords]
        y_coords = [point[1] for point in bbox_coords]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Convert to percentages
        left_pct = (x_min / img_width) * 100
        top_pct = (y_min / img_height) * 100
        width_pct = ((x_max - x_min) / img_width) * 100
        height_pct = ((y_max - y_min) / img_height) * 100
        
        # Calculate font size based on height
        font_size = max(8, min(16, int(height_pct * 0.8)))
        
        text_overlays.append({
            'text': text_to_draw,
            'left': left_pct,
            'top': top_pct,
            'width': width_pct,
            'height': height_pct,
            'font_size': font_size,
            'action': action
        })
    
    # Get appropriate font for the language
    font_family = get_font_family(target_language)
    
    # Create HTML with precise positioning
    overlays_html = ""
    for i, overlay in enumerate(text_overlays):
        overlays_html += f"""
        <div class="text-overlay {overlay['action']}" style="
            left: {overlay['left']}%;
            top: {overlay['top']}%;
            width: {overlay['width']}%;
            height: {overlay['height']}%;
            font-size: {overlay['font_size']}px;
            box-sizing: border-box;
            overflow: hidden;
            word-wrap: break-word;
        ">
            {overlay['text']}
        </div>
        """
    
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Processed Document - {lang_info['name']}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family={font_family}:wght@400;700&display=swap');
        
        body {{
            margin: 0;
            padding: 0;
            font-family: '{font_family}', Arial, sans-serif;
            background: white;
            position: relative;
        }}
        
        .document-container {{
            position: relative;
            width: 100%;
            height: 100vh;
            background: white;
        }}
        
        .background-image {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 1.0;
            z-index: 1;
            pointer-events: none;
        }}
        
        .text-overlay {{
            position: absolute;
            z-index: 2;
            background: rgba(255, 255, 255, 1.0);
            border-radius: 2px;
            display: flex;
            align-items: flex-start;
            justify-content: flex-start;
            line-height: 1.2;
            padding: 3px;
            box-shadow: none;
            border: none;
        }}
        
        .text-overlay.translate {{
            background: rgba(255, 255, 255, 1.0);
            border: none;
        }}
        
        .text-overlay.preserve {{
            background: rgba(255, 255, 255, 1.0);
            border: none;
        }}
        
        .text-overlay.whiteout {{
            background: rgba(255, 255, 255, 1.0);
            border: none;
        }}
        
        /* Print styles */
        @media print {{
            body {{ margin: 0; padding: 0; }}
            .text-overlay {{ background: white; }}
        }}
        
        /* Responsive adjustments */
        @media (max-width: 768px) {{
            .text-overlay {{
                font-size: 10px !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="document-container">
        <!-- Background image with full opacity to preserve logo/signatures -->
        <img src="data:image/png;base64,{img_data}" class="background-image" alt="Original Document">
        
        <!-- Text overlays positioned exactly like original -->
        {overlays_html}
    </div>
</body>
</html>
"""
    
    return html_template

def get_font_family(lang_code):
    """Get appropriate Google Font family for the language"""
    font_families = {
        'te': 'Noto+Sans+Telugu',
        'hi': 'Noto+Sans+Devanagari',
        'ta': 'Noto+Sans+Tamil',
        'kn': 'Noto+Sans+Kannada',
        'ml': 'Noto+Sans+Malayalam',
        'gu': 'Noto+Sans+Gujarati',
        'pa': 'Noto+Sans+Gurmukhi',
        'bn': 'Noto+Sans+Bengali',
        'or': 'Noto+Sans+Oriya',
        'en': 'Roboto'
    }
    return font_families.get(lang_code, 'Noto+Sans+Telugu')

@app.route('/')
def index():
    """Serve the main HTML page"""
    return create_dynamic_ui()

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and OCR processing"""
    global current_image, current_bboxes, current_translated_text, current_image_path, current_target_language
    
    try:
        # Get uploaded file
        file = request.files['image']
        if not file:
            return jsonify({'error': 'No image uploaded'}), 400
        
        # Save uploaded file
        filename = file.filename
        file_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(file_path)
        
        current_image_path = file_path
        
        # Process with OCR
        bboxes = process_image_with_ocr(file_path)
        current_bboxes = bboxes
        
        if not bboxes:
            return jsonify({
                'success': False,
                'error': 'No text detected in the image. Please try with a clearer image or one with printed text.',
                'suggestions': [
                    'Make sure the image is clear and not blurry',
                    'Ensure the text is printed (not handwritten)',
                    'Try a higher resolution image',
                    'Check if the image contains readable text'
                ]
            }), 400
        
        # Extract text for translation
        original_text = '\n'.join([bbox['text'] for bbox in bboxes])
        print(f"DEBUG: Original text extracted: {original_text[:100]}...")
        
        # Get model, agent mode, and target language from request
        model = request.form.get('model', 'gemma3-legal-samanantar-pro:latest')
        agent_mode = request.form.get('agent_mode', 'false').lower() == 'true'
        target_language = request.form.get('target_language', 'te')
        
        print(f"DEBUG: Model={model}, AgentMode={agent_mode}, TargetLang={target_language}")
        
        # Translate each text region individually
        translated_lines = []
        for i, bbox in enumerate(bboxes):
            region_text = bbox['text']
            print(f"DEBUG: Translating region {i+1}: '{region_text[:50]}...'")
            
            if agent_mode:
                region_translated = translate_text_with_agents(region_text, model, target_language)
            else:
                region_translated = translate_text(region_text, model, target_language)
            
            print(f"DEBUG: Region {i+1} translated: '{region_translated[:50]}...'")
            translated_lines.append(region_translated)
        
        current_translated_text = translated_lines
        
        # Store target language
        current_target_language = target_language
        
        # Convert image to base64 for display
        with open(file_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        # Prepare response data
        response_data = {
            'success': True,
            'image': img_data,
            'text_regions': []
        }
        
        # Add text regions with bounding boxes
        for i, bbox in enumerate(bboxes):
            bbox_coords = bbox['bbox']
            x_coords = [point[0] for point in bbox_coords]
            y_coords = [point[1] for point in bbox_coords]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            region_data = {
                'id': i,
                'title': f'Text Region {i+1}',
                'text': bbox['text'],
                'translated': translated_lines[i] if i < len(translated_lines) else bbox['text'],
                'bbox': {
                    'x': x_min,
                    'y': y_min,
                    'width': x_max - x_min,
                    'height': y_max - y_min
                },
                'action': 'preserve'  # Default action
            }
            response_data['text_regions'].append(region_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_document():
    """Process document with user actions"""
    global current_image_path, current_bboxes, current_translated_text, current_target_language
    
    try:
        data = request.get_json()
        user_actions = data.get('actions', {})
        
        if not current_image_path or not current_bboxes:
            return jsonify({'error': 'No image processed'}), 400
        
        # Create processed HTML document
        processed_html = create_processed_html(
            current_image_path, 
            current_bboxes, 
            current_translated_text, 
            user_actions,
            current_target_language
        )
        
        # Save processed HTML
        output_path = os.path.join(tempfile.gettempdir(), 'processed_document.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_html)
        
        return jsonify({
            'success': True,
            'processed_html': processed_html,
            'download_url': '/download_html'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_html')
def download_html_document():
    """Download the processed HTML document"""
    try:
        output_path = os.path.join(tempfile.gettempdir(), 'processed_document.html')
        if os.path.exists(output_path):
            return send_file(output_path, as_attachment=True, download_name='translated_document.html')
        else:
            return jsonify({'error': 'No processed document available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ollama/models', methods=['GET'])
def get_ollama_models():
    """Get available Ollama models"""
    try:
        # Get list of available models
        models = ollama.list()
        model_list = []
        
        for model in models['models']:
            model_info = {
                'name': model.model,
                'size': model.size,
                'modified_at': str(model.modified_at) if model.modified_at else '',
                'family': model.details.family if model.details else 'unknown'
            }
            model_list.append(model_info)
        
        return jsonify({
            'success': True,
            'models': model_list
        })
        
    except Exception as e:
        print(f"Error getting Ollama models: {e}")
        # Fallback to default models if Ollama is not available
        fallback_models = [
            {'name': 'gemma3-legal-samanantar-pro:latest', 'family': 'gemma', 'size': 0},
            {'name': 'llama3.1:8b', 'family': 'llama', 'size': 0},
            {'name': 'gemma3:4b', 'family': 'gemma', 'size': 0},
            {'name': 'gaganyatri/sarvam-2b-v0.5:latest', 'family': 'sarvam', 'size': 0}
        ]
        return jsonify({
            'success': True,
            'models': fallback_models,
            'fallback': True,
            'message': 'Ollama service unavailable - using fallback models'
        })

@app.route('/debug/translation', methods=['POST'])
def debug_translation():
    """Debug translation endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_language = data.get('target_language', 'te')
        model = data.get('model', 'gemma3-legal-samanantar-pro:latest')
        
        print(f"Debug: Translating '{text[:50]}...' to {target_language} using {model}")
        
        # Test translation
        result = translate_text(text, model, target_language)
        
        return jsonify({
            'success': True,
            'original': text,
            'translated': result,
            'target_language': target_language,
            'model': model
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ollama', methods=['POST'])
def ollama_api():
    """Ollama API endpoint for agent framework"""
    try:
        data = request.get_json()
        model = data.get('model', 'gemma3-legal-samanantar-pro:latest')
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Call Ollama
        translated_text = ollama.generate(
            model=model,
            prompt=prompt
        )['response']
        
        return jsonify({
            'success': True,
            'response': translated_text,
            'model': model
        })
        
    except Exception as e:
        print(f"Ollama API error: {e}")
        # Return a fallback response
        return jsonify({
            'success': False,
            'error': str(e),
            'fallback': True,
            'message': 'Ollama service unavailable - please check Ollama installation and service status'
        }), 503

@app.route('/download')
def download_document():
    """Download the processed document"""
    try:
        output_path = os.path.join(tempfile.gettempdir(), 'processed_document.png')
        if os.path.exists(output_path):
            return send_file(output_path, as_attachment=True, download_name='translated_document.png')
        else:
            return jsonify({'error': 'No processed document available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_dynamic_ui():
    """Create the dynamic UI with real backend integration"""
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Dynamic Document Translator</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@400;700&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #000;
            color: #fff;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 2px solid #fff;
            padding-bottom: 20px;
        }
        
        .header h1 {
            font-size: 2.5em;
            font-weight: 300;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.8;
        }
        
        .upload-section {
            background: #111;
            border: 2px dashed #333;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: #666;
            background: #1a1a1a;
        }
        
        .upload-section.dragover {
            border-color: #fff;
            background: #222;
        }
        
        .upload-icon {
            font-size: 3em;
            margin-bottom: 20px;
            opacity: 0.6;
        }
        
        .upload-text {
            font-size: 1.2em;
            margin-bottom: 20px;
        }
        
        .file-input {
            display: none;
        }
        
        .upload-btn {
            background: #fff;
            color: #000;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .upload-btn:hover {
            background: #ccc;
        }
        
        .model-selection-section {
            background: #111;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .model-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .model-card {
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .model-card:hover {
            border-color: #666;
            background: #222;
        }
        
        .model-card.selected {
            border-color: #fff;
            background: #333;
        }
        
        .model-icon {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .model-name {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .model-desc {
            font-size: 0.9em;
            opacity: 0.7;
        }
        
        .agent-mode-toggle {
            text-align: center;
            margin-top: 20px;
        }
        
        .toggle-label {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            cursor: pointer;
        }
        
        .toggle-label input[type="checkbox"] {
            display: none;
        }
        
        .toggle-slider {
            width: 60px;
            height: 30px;
            background: #333;
            border-radius: 15px;
            position: relative;
            transition: all 0.3s ease;
        }
        
        .toggle-slider::before {
            content: '';
            position: absolute;
            width: 26px;
            height: 26px;
            background: #fff;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: all 0.3s ease;
        }
        
        .toggle-label input[type="checkbox"]:checked + .toggle-slider {
            background: #4CAF50;
        }
        
        .toggle-label input[type="checkbox"]:checked + .toggle-slider::before {
            transform: translateX(30px);
        }
        
        .toggle-text {
            font-size: 1.1em;
            font-weight: 500;
        }
        
        /* Loading Screen Styles */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }
        
        .loading-content {
            text-align: center;
            color: white;
        }
        
        .loading-spinner {
            width: 60px;
            height: 60px;
            border: 4px solid #333;
            border-top: 4px solid #fff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-text {
            font-size: 1.2em;
            margin-bottom: 10px;
        }
        
        .loading-subtext {
            font-size: 0.9em;
            opacity: 0.7;
        }
        
        .model-loading {
            opacity: 0.5;
            pointer-events: none;
        }
        
        .model-loading::after {
            content: '⏳';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 2em;
        }
        
        .model-loading-card {
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            grid-column: 1 / -1;
        }
        
        .model-loading-card .loading-spinner {
            width: 40px;
            height: 40px;
            margin-bottom: 15px;
        }
        
        .model-loading-card .loading-text {
            font-size: 1em;
            margin-bottom: 5px;
        }
        
        .translation-options {
            margin-top: 20px;
        }
        
        .language-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .language-card {
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .language-card:hover {
            border-color: #666;
            background: #222;
        }
        
        .language-card.selected {
            border-color: #4CAF50;
            background: #2a2a2a;
        }
        
        .language-icon {
            font-size: 1.5em;
            margin-bottom: 8px;
        }
        
        .language-name {
            font-size: 0.9em;
            font-weight: bold;
            margin-bottom: 3px;
        }
        
        .language-desc {
            font-size: 0.8em;
            opacity: 0.7;
        }
        
        .processing-section {
            display: none;
            background: #111;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .processing-step {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        
        .processing-step.active {
            background: #222;
        }
        
        .processing-step.completed {
            background: #0a0;
        }
        
        .step-icon {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: #333;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            font-weight: bold;
        }
        
        .step-icon.active {
            background: #fff;
            color: #000;
        }
        
        .step-icon.completed {
            background: #0f0;
            color: #000;
        }
        
        .step-text {
            flex: 1;
        }
        
        .main-content {
            display: none;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .image-section {
            background: #111;
            border-radius: 10px;
            padding: 20px;
        }
        
        .image-container {
            position: relative;
            display: inline-block;
        }
        
        .document-image {
            max-width: 100%;
            border: 1px solid #333;
            border-radius: 5px;
        }
        
        .bbox-overlay {
            position: absolute;
            border: 2px solid #fff;
            background: rgba(255, 255, 255, 0.1);
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .bbox-overlay:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        
        .bbox-overlay.translate {
            border-color: #fff;
            background: rgba(255, 255, 255, 0.2);
        }
        
        .bbox-overlay.preserve {
            border-color: #0f0;
            background: rgba(0, 255, 0, 0.2);
        }
        
        .bbox-overlay.whiteout {
            border-color: #f00;
            background: rgba(255, 0, 0, 0.2);
        }
        
        .controls-section {
            background: #111;
            border-radius: 10px;
            padding: 20px;
        }
        
        .section-title {
            font-size: 1.5em;
            margin-bottom: 20px;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }
        
        .text-region {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .text-region:hover {
            border-color: #666;
            background: #222;
        }
        
        .text-region.selected {
            border-color: #fff;
            background: #333;
        }
        
        .text-region.translate {
            border-color: #fff;
            background: #333;
        }
        
        .text-region.preserve {
            border-color: #0f0;
            background: #0a0;
        }
        
        .text-region.whiteout {
            border-color: #f00;
            background: #a00;
        }
        
        .region-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .region-title {
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .region-controls {
            display: flex;
            gap: 10px;
        }
        
        .control-btn {
            background: #333;
            color: #fff;
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s ease;
        }
        
        .control-btn:hover {
            background: #555;
        }
        
        .control-btn.active {
            background: #fff;
            color: #000;
        }
        
        .region-text {
            font-size: 0.9em;
            opacity: 0.8;
            line-height: 1.4;
        }
        
        .action-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 30px;
        }
        
        .action-btn {
            background: #fff;
            color: #000;
            border: none;
            padding: 15px 30px;
            border-radius: 5px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .action-btn:hover {
            background: #ccc;
        }
        
        .action-btn:disabled {
            background: #333;
            color: #666;
            cursor: not-allowed;
        }
        
        .preview-section {
            display: none;
            background: #111;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
        }
        
        .preview-image {
            max-width: 100%;
            border: 1px solid #333;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .download-btn {
            background: #0f0;
            color: #000;
            border: none;
            padding: 12px 25px;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .download-btn:hover {
            background: #0a0;
        }
        
        .status-message {
            text-align: center;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: none;
        }
        
        .status-message.success {
            background: #0a0;
            color: #fff;
        }
        
        .status-message.error {
            background: #a00;
            color: #fff;
        }
        
        .status-message.info {
            background: #006;
            color: #fff;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Dynamic Document Translator</h1>
            <p>Upload any document image and control every text region</p>
        </div>
        
        <div class="upload-section" id="uploadSection">
            <div class="upload-icon">📄</div>
            <div class="upload-text">Drag & drop your document image here or click to browse</div>
            <input type="file" id="fileInput" class="file-input" accept="image/*">
            <button class="upload-btn" onclick="document.getElementById('fileInput').click()">Choose File</button>
        </div>
        
        <div class="model-selection-section">
            <div class="section-title">🤖 Choose Translation Model</div>
            <div class="model-grid" id="modelGrid">
                <div class="model-loading-card">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">Loading available models...</div>
                </div>
            </div>
            <div class="translation-options">
                <div class="section-title">🌐 Translation Options</div>
                <div class="language-grid">
                    <div class="language-card" data-lang="te">
                        <div class="language-icon">📜</div>
                        <div class="language-name">Telugu</div>
                        <div class="language-desc">తెలుగు</div>
                    </div>
                    <div class="language-card" data-lang="hi">
                        <div class="language-icon">📖</div>
                        <div class="language-name">Hindi</div>
                        <div class="language-desc">हिन्दी</div>
                    </div>
                    <div class="language-card" data-lang="ta">
                        <div class="language-icon">📚</div>
                        <div class="language-name">Tamil</div>
                        <div class="language-desc">தமிழ்</div>
                    </div>
                    <div class="language-card" data-lang="kn">
                        <div class="language-icon">📝</div>
                        <div class="language-name">Kannada</div>
                        <div class="language-desc">ಕನ್ನಡ</div>
                    </div>
                    <div class="language-card" data-lang="ml">
                        <div class="language-icon">📄</div>
                        <div class="language-name">Malayalam</div>
                        <div class="language-desc">മലയാളം</div>
                    </div>
                    <div class="language-card" data-lang="en">
                        <div class="language-icon">📰</div>
                        <div class="language-name">English</div>
                        <div class="language-desc">Improved</div>
                    </div>
                </div>
            </div>
            <div class="agent-mode-toggle">
                <label class="toggle-label">
                    <input type="checkbox" id="agentModeToggle" checked>
                    <span class="toggle-slider"></span>
                    <span class="toggle-text">🚀 Agentic Framework (Multi-Agent Pipeline)</span>
                </label>
            </div>
        </div>
        
        <!-- Loading Overlay -->
        <div class="loading-overlay" id="loadingOverlay">
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="loading-text" id="loadingText">Processing...</div>
                <div class="loading-subtext" id="loadingSubtext">Please wait</div>
            </div>
        </div>
        
        <div class="processing-section" id="processingSection">
            <div class="section-title">Processing Document</div>
            <div class="processing-step" id="step1">
                <div class="step-icon">1</div>
                <div class="step-text">Extracting text with OCR...</div>
            </div>
            <div class="processing-step" id="step2">
                <div class="step-icon">2</div>
                <div class="step-text">Translating text...</div>
            </div>
            <div class="processing-step" id="step3">
                <div class="step-icon">3</div>
                <div class="step-text">Preparing layout controls...</div>
            </div>
        </div>
        
        <div class="main-content" id="mainContent">
            <div class="image-section">
                <div class="section-title">Document Preview</div>
                <div class="image-container" id="imageContainer">
                    <img id="documentImage" class="document-image" alt="Document Preview">
                </div>
            </div>
            
            <div class="controls-section">
                <div class="section-title">Text Region Controls</div>
                <div class="status-message" id="statusMessage"></div>
                <div id="textRegions"></div>
                <div class="action-buttons">
                    <button class="action-btn" id="previewBtn" onclick="previewDocument()">Preview Document</button>
                    <button class="action-btn" id="downloadBtn" onclick="downloadDocument()">Download Result</button>
                </div>
            </div>
        </div>
        
        <div class="preview-section" id="previewSection">
            <div class="section-title">Final Document Preview</div>
            <img id="previewImage" class="preview-image" alt="Final Document Preview">
            <button class="download-btn" onclick="downloadDocument()">📥 Download HTML Document</button>
        </div>
    </div>
    
    <script>
        let documentData = null;
        let textRegions = [];
        let translatedText = [];
        let userActions = {};
        let selectedModel = 'gemma3-legal-samanantar-pro:latest';
        let agentMode = true;
        let selectedLanguage = 'te'; // Default to Telugu
        
        // Loading functions
        function showLoading(text = 'Processing...', subtext = 'Please wait') {
            const overlay = document.getElementById('loadingOverlay');
            const loadingText = document.getElementById('loadingText');
            const loadingSubtext = document.getElementById('loadingSubtext');
            
            loadingText.textContent = text;
            loadingSubtext.textContent = subtext;
            overlay.style.display = 'flex';
        }
        
        function hideLoading() {
            const overlay = document.getElementById('loadingOverlay');
            overlay.style.display = 'none';
        }
        
        // Model selection handling
        document.addEventListener('DOMContentLoaded', function() {
            loadAvailableModels();
            setupLanguageSelection();
            
            // Agent mode toggle
            const agentToggle = document.getElementById('agentModeToggle');
            if (agentToggle) {
                agentToggle.addEventListener('change', function() {
                    agentMode = this.checked;
                    console.log('Agent mode:', agentMode ? 'enabled' : 'disabled');
                });
            }
        });
        
        function setupLanguageSelection() {
            // Set default language selection
            const defaultLangCard = document.querySelector('[data-lang="te"]');
            if (defaultLangCard) {
                defaultLangCard.classList.add('selected');
            }
            
            // Language card selection
            document.querySelectorAll('.language-card').forEach(card => {
                card.addEventListener('click', function() {
                    document.querySelectorAll('.language-card').forEach(c => c.classList.remove('selected'));
                    this.classList.add('selected');
                    selectedLanguage = this.dataset.lang;
                    console.log('Selected language:', selectedLanguage);
                });
            });
        }
        
        async function loadAvailableModels() {
            try {
                const response = await fetch('/api/ollama/models');
                const data = await response.json();
                
                if (data.success) {
                    renderModels(data.models);
                } else {
                    console.error('Failed to load models:', data.error);
                    renderFallbackModels();
                }
            } catch (error) {
                console.error('Error loading models:', error);
                renderFallbackModels();
            }
        }
        
        function renderModels(models) {
            const modelGrid = document.getElementById('modelGrid');
            modelGrid.innerHTML = '';
            
            // Sort models by family and name
            models.sort((a, b) => {
                if (a.family !== b.family) {
                    return a.family.localeCompare(b.family);
                }
                return a.name.localeCompare(b.name);
            });
            
            models.forEach((model, index) => {
                const modelCard = document.createElement('div');
                modelCard.className = 'model-card';
                modelCard.dataset.model = model.name;
                
                // Get icon based on family
                const icon = getModelIcon(model.family);
                const sizeText = model.size > 0 ? formatBytes(model.size) : '';
                
                modelCard.innerHTML = `
                    <div class="model-icon">${icon}</div>
                    <div class="model-name">${model.name}</div>
                    <div class="model-desc">${model.family} ${sizeText}</div>
                `;
                
                // Add click handler
                modelCard.addEventListener('click', function() {
                    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
                    this.classList.add('selected');
                    selectedModel = this.dataset.model;
                    console.log('Selected model:', selectedModel);
                });
                
                modelGrid.appendChild(modelCard);
            });
            
            // Select first model by default
            if (models.length > 0) {
                const firstCard = modelGrid.querySelector('.model-card');
                if (firstCard) {
                    firstCard.classList.add('selected');
                    selectedModel = firstCard.dataset.model;
                }
            }
        }
        
        function renderFallbackModels() {
            const modelGrid = document.getElementById('modelGrid');
            modelGrid.innerHTML = '';
            
            const fallbackModels = [
                { name: 'gemma3-legal-samanantar-pro:latest', family: 'gemma', icon: '⚖️' },
                { name: 'llama3.1:7b', family: 'llama', icon: '🦙' },
                { name: 'qwen2.5:7b', family: 'qwen', icon: '🧠' },
                { name: 'mistral:7b', family: 'mistral', icon: '🌪️' }
            ];
            
            fallbackModels.forEach(model => {
                const modelCard = document.createElement('div');
                modelCard.className = 'model-card';
                modelCard.dataset.model = model.name;
                
                modelCard.innerHTML = `
                    <div class="model-icon">${model.icon}</div>
                    <div class="model-name">${model.name}</div>
                    <div class="model-desc">${model.family} (fallback)</div>
                `;
                
                modelCard.addEventListener('click', function() {
                    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
                    this.classList.add('selected');
                    selectedModel = this.dataset.model;
                    console.log('Selected model:', selectedModel);
                });
                
                modelGrid.appendChild(modelCard);
            });
            
            // Select first model by default
            const firstCard = modelGrid.querySelector('.model-card');
            if (firstCard) {
                firstCard.classList.add('selected');
                selectedModel = firstCard.dataset.model;
            }
        }
        
        function getModelIcon(family) {
            const icons = {
                'gemma': '⚖️',
                'llama': '🦙',
                'qwen': '🧠',
                'mistral': '🌪️',
                'phi': 'Φ',
                'codellama': '💻',
                'default': '🤖'
            };
            return icons[family] || icons.default;
        }
        
        function formatBytes(bytes) {
            if (bytes === 0) return '';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return `(${(bytes / Math.pow(k, i)).toFixed(1)}${sizes[i]})`;
        }
        
        // Agent Framework Implementation
        class AgentFramework {
            constructor(model) {
                this.model = model;
                this.agents = new Map();
                this.dataPipeline = [];
                this.agentStates = new Map();
                this.initializeAgents();
            }

            initializeAgents() {
                // Register all agents with their roles and capabilities
                this.agents.set('contextAnalyzer', {
                    agent: new ContextAgent(this.model),
                    role: 'context_analysis',
                    priority: 1,
                    required: true,
                    timeout: 30000
                });

                this.agents.set('translator', {
                    agent: new TranslationAgent(this.model),
                    role: 'translation',
                    priority: 2,
                    required: true,
                    timeout: 45000
                });

                this.agents.set('validator', {
                    agent: new ValidationAgent(this.model),
                    role: 'validation',
                    priority: 3,
                    required: true,
                    timeout: 30000
                });

                this.agents.set('qualityAssurance', {
                    agent: new QualityAgent(this.model),
                    role: 'quality_improvement',
                    priority: 4,
                    required: false,
                    timeout: 45000
                });

                this.agents.set('languageConsistencyChecker', {
                    agent: new LanguageConsistencyAgent(this.model),
                    role: 'language_consistency',
                    priority: 5,
                    required: true,
                    timeout: 20000
                });
            }

            async executeTranslationPipeline(originalText, sourceLang, targetLang, progressCallback) {
                console.log('🚀 Starting Agent Framework Pipeline');
                this.dataPipeline = [];
                this.agentStates.clear();

                const pipelineData = {
                    originalText,
                    sourceLang,
                    targetLang,
                    context: null,
                    translatedText: null,
                    validation: null,
                    consistencyCheck: null,
                    finalText: null,
                    errors: [],
                    metadata: {
                        startTime: Date.now(),
                        agentResults: {}
                    }
                };

                try {
                    // Step 1: Context Analysis
                    progressCallback('🔍 Agent 1/5: Context Analysis', 1, 'active');
                    pipelineData.context = await this.executeAgent('contextAnalyzer', {
                        text: originalText,
                        sourceLang,
                        targetLang
                    });
                    pipelineData.metadata.agentResults.contextAnalysis = pipelineData.context;
                    progressCallback('✅ Context Analysis Complete', 1, 'completed');

                    // Step 2: Translation
                    progressCallback('🔄 Agent 2/5: Translation', 2, 'active');
                    pipelineData.translatedText = await this.executeAgent('translator', {
                        text: originalText,
                        sourceLang,
                        targetLang,
                        context: pipelineData.context
                    });
                    pipelineData.metadata.agentResults.translation = pipelineData.translatedText;
                    progressCallback('✅ Translation Complete', 2, 'completed');

                    // Step 3: Validation
                    progressCallback('✅ Agent 3/5: Validation', 3, 'active');
                    pipelineData.validation = await this.executeAgent('validator', {
                        originalText,
                        translatedText: pipelineData.translatedText,
                        sourceLang,
                        targetLang
                    });
                    pipelineData.metadata.agentResults.validation = pipelineData.validation;
                    progressCallback('✅ Validation Complete', 3, 'completed');

                    // Step 4: Language Consistency Check
                    progressCallback('🔍 Agent 4/5: Language Consistency', 4, 'active');
                    pipelineData.consistencyCheck = await this.executeAgent('languageConsistencyChecker', {
                        text: pipelineData.translatedText,
                        targetLang,
                        originalText
                    });
                    pipelineData.metadata.agentResults.consistencyCheck = pipelineData.consistencyCheck;
                    progressCallback('✅ Language Consistency Check Complete', 4, 'completed');

                    // Step 5: Quality Improvement (if needed)
                    if (pipelineData.validation.status === 'needs_revision' || 
                        pipelineData.validation.status === 'invalid' ||
                        !pipelineData.consistencyCheck.isConsistent) {
                        
                        progressCallback('🔧 Agent 5/5: Quality Improvement', 5, 'active');
                        pipelineData.finalText = await this.executeAgent('qualityAssurance', {
                            originalText,
                            translatedText: pipelineData.translatedText,
                            sourceLang,
                            targetLang,
                            context: pipelineData.context,
                            validation: pipelineData.validation,
                            consistencyCheck: pipelineData.consistencyCheck
                        });
                        pipelineData.metadata.agentResults.qualityImprovement = pipelineData.finalText;
                        progressCallback('✅ Quality Improvement Complete', 5, 'completed');
                    } else {
                        pipelineData.finalText = pipelineData.translatedText;
                        progressCallback('✅ Quality Check Passed', 5, 'completed');
                    }

                    // Final validation
                    if (pipelineData.finalText !== pipelineData.translatedText) {
                        progressCallback('🔄 Final Validation', 3, 'active');
                        const finalValidation = await this.executeAgent('validator', {
                            originalText,
                            translatedText: pipelineData.finalText,
                            sourceLang,
                            targetLang
                        });
                        pipelineData.metadata.agentResults.finalValidation = finalValidation;
                        progressCallback('✅ Final Validation Complete', 3, 'completed');
                    }

                    pipelineData.metadata.endTime = Date.now();
                    pipelineData.metadata.duration = pipelineData.metadata.endTime - pipelineData.metadata.startTime;

                    console.log('🎯 Agent Framework Pipeline Complete:', pipelineData.metadata);
                    return pipelineData.finalText || pipelineData.translatedText;

                } catch (error) {
                    console.error('❌ Agent Framework Pipeline Error:', error);
                    pipelineData.errors.push(error.message);
                    throw error;
                }
            }

            async executeAgent(agentName, inputData) {
                const agentConfig = this.agents.get(agentName);
                if (!agentConfig) {
                    throw new Error(`Agent ${agentName} not found`);
                }

                console.log(`🤖 Executing Agent: ${agentName}`, inputData);
                
                const startTime = Date.now();
                this.agentStates.set(agentName, {
                    status: 'running',
                    startTime,
                    inputData
                });

                try {
                    let result;
                    
                    switch (agentName) {
                        case 'contextAnalyzer':
                            result = await agentConfig.agent.analyzeContext(
                                inputData.text, 
                                inputData.sourceLang, 
                                inputData.targetLang
                            );
                            break;
                            
                        case 'translator':
                            result = await agentConfig.agent.translate(
                                inputData.text, 
                                inputData.sourceLang, 
                                inputData.targetLang, 
                                inputData.context
                            );
                            break;
                            
                        case 'validator':
                            result = await agentConfig.agent.validateTranslation(
                                inputData.originalText, 
                                inputData.translatedText, 
                                inputData.sourceLang, 
                                inputData.targetLang
                            );
                            break;
                            
                        case 'qualityAssurance':
                            result = await agentConfig.agent.improveTranslation(
                                inputData.originalText, 
                                inputData.translatedText, 
                                inputData.sourceLang, 
                                inputData.targetLang, 
                                inputData.context
                            );
                            break;
                            
                        case 'languageConsistencyChecker':
                            result = await agentConfig.agent.checkConsistency(
                                inputData.text, 
                                inputData.targetLang, 
                                inputData.originalText
                            );
                            break;
                            
                        default:
                            throw new Error(`Unknown agent: ${agentName}`);
                    }

                    const endTime = Date.now();
                    this.agentStates.set(agentName, {
                        status: 'completed',
                        startTime,
                        endTime,
                        duration: endTime - startTime,
                        inputData,
                        result
                    });

                    console.log(`✅ Agent ${agentName} completed in ${endTime - startTime}ms`);
                    return result;

                } catch (error) {
                    const endTime = Date.now();
                    this.agentStates.set(agentName, {
                        status: 'error',
                        startTime,
                        endTime,
                        duration: endTime - startTime,
                        inputData,
                        error: error.message
                    });

                    console.error(`❌ Agent ${agentName} failed:`, error);
                    throw error;
                }
            }

            getAgentStates() {
                return this.agentStates;
            }

            getPipelineData() {
                return this.dataPipeline;
            }
        }

        // Individual Agent Classes
        class ContextAgent {
            constructor(model) {
                this.model = model;
            }

            async analyzeContext(text, sourceLang, targetLang) {
                const targetLanguageName = this.getLanguageName(targetLang);
                
                const prompt = `You are a Legal Context Specialist. Analyze this document and identify:

1. DOCUMENT TYPE: What specific legal document is this? (Contract, NDA, SLA, Court Filing, Patent, etc.)
2. LEGAL JURISDICTION: Which legal system? (US, EU, India, UK, etc.)
3. LEGAL DOMAIN: What area of law? (Corporate, IP, Employment, Real Estate, etc.)
4. FORMALITY LEVEL: How formal/technical is the language?
5. CRITICAL ELEMENTS: What legal concepts are central?

Document to analyze:
${text}

Respond with structured analysis for ${targetLanguageName} translation:`;

                return await this.callOllama(prompt);
            }

            getLanguageName(langCode) {
                const names = {
                    'en': 'English', 'te': 'Telugu', 'kn': 'Kannada', 'ta': 'Tamil',
                    'hi': 'Hindi', 'bn': 'Bengali', 'gu': 'Gujarati', 'pa': 'Punjabi',
                    'mr': 'Marathi', 'or': 'Odia', 'as': 'Assamese', 'ne': 'Nepali',
                    'ur': 'Urdu', 'ml': 'Malayalam', 'si': 'Sinhala', 'my': 'Burmese',
                    'th': 'Thai', 'km': 'Khmer', 'lo': 'Lao', 'vi': 'Vietnamese',
                    'es': 'Spanish', 'fr': 'French', 'de': 'German', 'ja': 'Japanese',
                    'ko': 'Korean', 'zh': 'Chinese'
                };
                return names[langCode] || langCode;
            }

            async callOllama(prompt) {
                const response = await fetch('/api/ollama', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.model,
                        prompt: prompt
                    })
                });
                
                if (!response.ok) throw new Error(`Context agent error: ${response.status}`);
                const data = await response.json();
                return data.response.trim();
            }
        }

        class TranslationAgent {
            constructor(model) {
                this.model = model;
            }

            async translate(text, sourceLang, targetLang, context) {
                const targetLanguageName = this.getLanguageName(targetLang);
                
                const prompt = `You are a Legal Translation Precision Engine. Translate this document to ${targetLanguageName} with STRICT LEGAL PRECISION:

CONTEXT ANALYSIS:
${context}

TRANSLATION RULES:
1. Preserve EXACT sentence structure and dependent clauses
2. Maintain ALL conditional logic ("if-then" statements)
3. Keep ALL deadlines and time references intact
4. Preserve ALL legal obligations and rights
5. NO stylistic changes that could alter legal meaning

Document to translate:
${text}

Provide precise legal translation:`;

                return await this.callOllama(prompt);
            }

            getLanguageName(langCode) {
                const names = {
                    'en': 'English', 'te': 'Telugu', 'kn': 'Kannada', 'ta': 'Tamil',
                    'hi': 'Hindi', 'bn': 'Bengali', 'gu': 'Gujarati', 'pa': 'Punjabi',
                    'mr': 'Marathi', 'or': 'Odia', 'as': 'Assamese', 'ne': 'Nepali',
                    'ur': 'Urdu', 'ml': 'Malayalam', 'si': 'Sinhala', 'my': 'Burmese',
                    'th': 'Thai', 'km': 'Khmer', 'lo': 'Lao', 'vi': 'Vietnamese',
                    'es': 'Spanish', 'fr': 'French', 'de': 'German', 'ja': 'Japanese',
                    'ko': 'Korean', 'zh': 'Chinese'
                };
                return names[langCode] || langCode;
            }

            async callOllama(prompt) {
                const response = await fetch('/api/ollama', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.model,
                        prompt: prompt
                    })
                });
                
                if (!response.ok) throw new Error(`Translation agent error: ${response.status}`);
                const data = await response.json();
                return data.response.trim();
            }
        }

        class ValidationAgent {
            constructor(model) {
                this.model = model;
            }

            async validateTranslation(originalText, translatedText, sourceLang, targetLang) {
                const targetLanguageName = this.getLanguageName(targetLang);
                
                const prompt = `You are a Legal Validation Specialist. Perform legal review:

ORIGINAL TEXT:
${originalText}

TRANSLATED TEXT:
${translatedText}

FINAL VALIDATION CHECKLIST:
1. LEGAL ENFORCEABILITY: Does translated document have same legal effect?
2. PRECISION: Are all legal obligations preserved?
3. CONSISTENCY: Are all terms used consistently?
4. COMPLETENESS: Is no legal content lost or altered?
5. FORMALITY: Is appropriate legal tone maintained?

Respond with JSON:
{
    "status": "valid/invalid/needs_revision",
    "score": 0-100,
    "issues": ["issue1", "issue2"],
    "recommendations": ["rec1", "rec2"]
}`;

                const result = await this.callOllama(prompt);
                
                try {
                    return JSON.parse(result);
                } catch (error) {
                    return {
                        status: 'needs_revision',
                        score: 50,
                        issues: ['JSON parsing failed'],
                        recommendations: ['Manual review required']
                    };
                }
            }

            getLanguageName(langCode) {
                const names = {
                    'en': 'English', 'te': 'Telugu', 'kn': 'Kannada', 'ta': 'Tamil',
                    'hi': 'Hindi', 'bn': 'Bengali', 'gu': 'Gujarati', 'pa': 'Punjabi',
                    'mr': 'Marathi', 'or': 'Odia', 'as': 'Assamese', 'ne': 'Nepali',
                    'ur': 'Urdu', 'ml': 'Malayalam', 'si': 'Sinhala', 'my': 'Burmese',
                    'th': 'Thai', 'km': 'Khmer', 'lo': 'Lao', 'vi': 'Vietnamese',
                    'es': 'Spanish', 'fr': 'French', 'de': 'German', 'ja': 'Japanese',
                    'ko': 'Korean', 'zh': 'Chinese'
                };
                return names[langCode] || langCode;
            }

            async callOllama(prompt) {
                const response = await fetch('/api/ollama', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.model,
                        prompt: prompt
                    })
                });
                
                if (!response.ok) throw new Error(`Validation agent error: ${response.status}`);
                const data = await response.json();
                return data.response.trim();
            }
        }

        class QualityAgent {
            constructor(model) {
                this.model = model;
            }

            async improveTranslation(originalText, translatedText, sourceLang, targetLang, context) {
                const targetLanguageName = this.getLanguageName(targetLang);
                
                const prompt = `You are a Legal Quality Assurance Specialist. Improve this translation:

ORIGINAL TEXT:
${originalText}

CURRENT TRANSLATION:
${translatedText}

CONTEXT:
${context}

Provide improved translation maintaining legal precision:`;

                return await this.callOllama(prompt);
            }

            getLanguageName(langCode) {
                const names = {
                    'en': 'English', 'te': 'Telugu', 'kn': 'Kannada', 'ta': 'Tamil',
                    'hi': 'Hindi', 'bn': 'Bengali', 'gu': 'Gujarati', 'pa': 'Punjabi',
                    'mr': 'Marathi', 'or': 'Odia', 'as': 'Assamese', 'ne': 'Nepali',
                    'ur': 'Urdu', 'ml': 'Malayalam', 'si': 'Sinhala', 'my': 'Burmese',
                    'th': 'Thai', 'km': 'Khmer', 'lo': 'Lao', 'vi': 'Vietnamese',
                    'es': 'Spanish', 'fr': 'French', 'de': 'German', 'ja': 'Japanese',
                    'ko': 'Korean', 'zh': 'Chinese'
                };
                return names[langCode] || langCode;
            }

            async callOllama(prompt) {
                const response = await fetch('/api/ollama', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.model,
                        prompt: prompt
                    })
                });
                
                if (!response.ok) throw new Error(`Quality agent error: ${response.status}`);
                const data = await response.json();
                return data.response.trim();
            }
        }

        // Enhanced Language Consistency Agent
        class LanguageConsistencyAgent {
            constructor(model) {
                this.model = model;
            }

            async checkConsistency(text, targetLang, originalText) {
                const targetLanguageName = this.getLanguageName(targetLang);
                
                const prompt = `You are a language consistency expert. Analyze the following text for language mixing.

Original text:
${originalText}

Translated text (should be in ${targetLanguageName}):
${text}

CRITICAL ANALYSIS REQUIRED:
1. Check if the translation uses ONLY ${targetLanguageName} language
2. Identify any words from other languages (English, Telugu, Tamil, Hindi, etc.)
3. Look for mixed language patterns
4. Verify consistency in terminology

Respond with a JSON object:
{
    "isConsistent": true/false,
    "mixedWords": ["word1", "word2"],
    "mixedLanguages": ["language1", "language2"],
    "consistencyScore": 0-100,
    "recommendations": ["recommendation1", "recommendation2"]
}`;

                const result = await this.callOllama(prompt);
                
                try {
                    const analysis = JSON.parse(result);
                    return {
                        isConsistent: analysis.isConsistent,
                        mixedWords: analysis.mixedWords || [],
                        mixedLanguages: analysis.mixedLanguages || [],
                        consistencyScore: analysis.consistencyScore || 0,
                        recommendations: analysis.recommendations || [],
                        targetLanguage: targetLanguageName
                    };
                } catch (error) {
                    // Fallback parsing
                    return {
                        isConsistent: !result.toLowerCase().includes('inconsistent'),
                        mixedWords: [],
                        mixedLanguages: [],
                        consistencyScore: 50,
                        recommendations: ['Manual review recommended'],
                        targetLanguage: targetLanguageName
                    };
                }
            }

            getLanguageName(langCode) {
                const names = {
                    'en': 'English', 'te': 'Telugu', 'kn': 'Kannada', 'ta': 'Tamil',
                    'hi': 'Hindi', 'bn': 'Bengali', 'gu': 'Gujarati', 'pa': 'Punjabi',
                    'mr': 'Marathi', 'or': 'Odia', 'as': 'Assamese', 'ne': 'Nepali',
                    'ur': 'Urdu', 'ml': 'Malayalam', 'si': 'Sinhala', 'my': 'Burmese',
                    'th': 'Thai', 'km': 'Khmer', 'lo': 'Lao', 'vi': 'Vietnamese',
                    'es': 'Spanish', 'fr': 'French', 'de': 'German', 'ja': 'Japanese',
                    'ko': 'Korean', 'zh': 'Chinese'
                };
                return names[langCode] || langCode;
            }

            async callOllama(prompt) {
                const response = await fetch('/api/ollama', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.model,
                        prompt: prompt
                    })
                });
                
                if (!response.ok) throw new Error(`Language consistency agent error: ${response.status}`);
                const data = await response.json();
                return data.response.trim();
            }
        }
        
        // File upload handling
        const uploadSection = document.getElementById('uploadSection');
        const fileInput = document.getElementById('fileInput');
        
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });
        
        function handleFileUpload(file) {
            if (!file.type.startsWith('image/')) {
                showStatus('Please upload an image file.', 'error');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', file);
            formData.append('model', selectedModel);
            formData.append('agent_mode', agentMode.toString());
            formData.append('target_language', selectedLanguage);
            
            showLoading('Processing Document', 'OCR and Translation in progress...');
            showProcessingStep(1);
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.success) {
                    showProcessingStep(2);
                    setTimeout(() => {
                        showProcessingStep(3);
                        setTimeout(() => {
                            displayResults(data);
                        }, 1000);
                    }, 2000);
                } else {
                    showStatus('Error: ' + data.error, 'error');
                    if (data.suggestions) {
                        console.log('Suggestions:', data.suggestions);
                        // You could display these suggestions in the UI
                    }
                }
            })
            .catch(error => {
                hideLoading();
                showStatus('Error uploading file: ' + error, 'error');
            });
        }
        
        function displayResults(data) {
            document.getElementById('processingSection').style.display = 'none';
            document.getElementById('mainContent').style.display = 'grid';
            
            // Display image
            const img = document.getElementById('documentImage');
            img.src = 'data:image/png;base64,' + data.image;
            
            // Store data
            textRegions = data.text_regions;
            userActions = {};
            
            // Create bounding box overlays
            createBoundingBoxes(data.text_regions);
            
            // Render text region controls
            renderTextRegions();
            
            showStatus('Document processed successfully! Select actions for each text region.', 'success');
        }
        
        function createBoundingBoxes(regions) {
            const container = document.getElementById('imageContainer');
            const img = document.getElementById('documentImage');
            
            // Clear existing overlays
            container.querySelectorAll('.bbox-overlay').forEach(overlay => overlay.remove());
            
            // Wait for image to load
            img.onload = () => {
                const imgRect = img.getBoundingClientRect();
                const scaleX = imgRect.width / img.naturalWidth;
                const scaleY = imgRect.height / img.naturalHeight;
                
                regions.forEach(region => {
                    const overlay = document.createElement('div');
                    overlay.className = 'bbox-overlay preserve';
                    overlay.style.left = (region.bbox.x * scaleX) + 'px';
                    overlay.style.top = (region.bbox.y * scaleY) + 'px';
                    overlay.style.width = (region.bbox.width * scaleX) + 'px';
                    overlay.style.height = (region.bbox.height * scaleY) + 'px';
                    overlay.dataset.regionId = region.id;
                    
                    overlay.addEventListener('click', () => {
                        selectRegion(region.id);
                    });
                    
                    container.appendChild(overlay);
                });
            };
        }
        
        function selectRegion(regionId) {
            // Highlight the region
            document.querySelectorAll('.text-region').forEach(region => {
                region.classList.remove('selected');
            });
            document.querySelector(`[data-region-id="${regionId}"]`).parentElement.querySelector('.text-region').classList.add('selected');
        }
        
        function showProcessingStep(stepNumber) {
            for (let i = 1; i <= 3; i++) {
                const step = document.getElementById(`step${i}`);
                const icon = step.querySelector('.step-icon');
                
                if (i < stepNumber) {
                    step.classList.add('completed');
                    icon.classList.add('completed');
                    icon.textContent = '✓';
                } else if (i === stepNumber) {
                    step.classList.add('active');
                    icon.classList.add('active');
                } else {
                    step.classList.remove('active', 'completed');
                    icon.classList.remove('active', 'completed');
                    icon.textContent = i;
                }
            }
        }
        
        function renderTextRegions() {
            const container = document.getElementById('textRegions');
            container.innerHTML = '';
            
            textRegions.forEach(region => {
                const regionDiv = document.createElement('div');
                regionDiv.className = `text-region preserve`;
                regionDiv.innerHTML = `
                    <div class="region-header">
                        <div class="region-title">Text Region ${region.id + 1}</div>
                        <div class="region-controls">
                            <button class="control-btn ${userActions[region.id] === 'translate' ? 'active' : ''}" 
                                    onclick="setRegionAction(${region.id}, 'translate')">Translate</button>
                            <button class="control-btn ${userActions[region.id] === 'preserve' ? 'active' : ''}" 
                                    onclick="setRegionAction(${region.id}, 'preserve')">Preserve</button>
                            <button class="control-btn ${userActions[region.id] === 'whiteout' ? 'active' : ''}" 
                                    onclick="setRegionAction(${region.id}, 'whiteout')">Whiteout</button>
                        </div>
                    </div>
                    <div class="region-text">
                        <strong>Original:</strong> ${region.text}<br>
                        <strong>Translated:</strong> ${region.translated}
                    </div>
                `;
                container.appendChild(regionDiv);
            });
        }
        
        function setRegionAction(regionId, action) {
            userActions[regionId] = action;
            
            // Update region styling
            const regionDiv = document.querySelector(`[data-region-id="${regionId}"]`);
            if (regionDiv) {
                regionDiv.className = `bbox-overlay ${action}`;
            }
            
            // Update control buttons
            const regionElement = document.querySelector(`[data-region-id="${regionId}"]`).parentElement.querySelector('.text-region');
            regionElement.className = `text-region ${action}`;
            
            // Update control buttons
            const buttons = regionElement.querySelectorAll('.control-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            buttons.forEach(btn => {
                if (btn.textContent.toLowerCase() === action) {
                    btn.classList.add('active');
                }
            });
        }
        
        function previewDocument() {
            showLoading('Generating HTML Preview', 'Creating layout-preserving document...');
            
            fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ actions: userActions })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.success) {
                    document.getElementById('mainContent').style.display = 'none';
                    document.getElementById('previewSection').style.display = 'block';
                    
                    // Create a new window with the HTML content
                    const newWindow = window.open('', '_blank');
                    newWindow.document.write(data.processed_html);
                    newWindow.document.close();
                    
                    showStatus('HTML preview opened in new window!', 'success');
                } else {
                    showStatus('Error: ' + data.error, 'error');
                    if (data.suggestions) {
                        console.log('Suggestions:', data.suggestions);
                        // You could display these suggestions in the UI
                    }
                }
            })
            .catch(error => {
                hideLoading();
                showStatus('Error generating preview: ' + error, 'error');
            });
        }
        
        function downloadDocument() {
            // First process the document, then download the HTML
            showLoading('Preparing Download', 'Generating final HTML document...');
            
            fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ actions: userActions })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.success) {
                    // Create a blob with the HTML content and download it
                    const blob = new Blob([data.processed_html], { type: 'text/html' });
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'translated_document.html';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                    
                    showStatus('HTML document downloaded successfully!', 'success');
                } else {
                    showStatus('Error: ' + data.error, 'error');
                    if (data.suggestions) {
                        console.log('Suggestions:', data.suggestions);
                        // You could display these suggestions in the UI
                    }
                }
            })
            .catch(error => {
                hideLoading();
                showStatus('Error downloading document: ' + error, 'error');
            });
        }
        
        function showStatus(message, type) {
            const statusDiv = document.getElementById('statusMessage');
            statusDiv.textContent = message;
            statusDiv.className = `status-message ${type}`;
            statusDiv.style.display = 'block';
            
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 3000);
        }
    </script>
</body>
</html>
"""
    
    return html_content

def main():
    print("Starting Dynamic Document Translator...")
    
    # Create the HTML file
    html_content = create_dynamic_ui()
    html_path = 'DYNAMIC_DOCUMENT_TRANSLATOR.html'
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Dynamic UI saved as: {html_path}")
    
    # Start Flask server
    print("🚀 Starting Flask server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    
    # Open in browser
    try:
        webbrowser.open('http://localhost:5000')
        print("✅ Browser opened")
    except Exception as e:
        print(f"❌ Could not open browser: {e}")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()
