import json
import io
import logging
import re
import requests
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import traceback
import time

# PDF Processing Libraries
import pdfplumber
import fitz  # PyMuPDF

# Excel Processing Libraries
import pandas as pd
import openpyxl
from openpyxl import load_workbook

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimetableExtractor:
    """
    ULTRA-FAST LLM-based timetable extraction system using Cerebras Inference API
    Drop-in replacement for Gemini extractor with 10x faster token generation
    """

    def __init__(self, cerebras_api_key: str, endpoint_url: str = None):
        """
        Initialize the extractor with Cerebras API key
        
        Args:
            cerebras_api_key: Cerebras API key
            endpoint_url: Cerebras endpoint URL (optional, uses default if not provided)
        """
        self.cerebras_api_key = cerebras_api_key
        self.endpoint_url = endpoint_url or "https://api.cerebras.ai/v1/chat/completions"
        
        # Test API connection
        self._test_connection()
        
        # Load the extraction prompt
        self.extraction_prompt = self._get_extraction_prompt()
        
        logger.info("✅ Cerebras LLM extractor initialized successfully")

    def _test_connection(self):
        """Test Cerebras API connection"""
        try:
            headers = {
                "Authorization": f"Bearer {self.cerebras_api_key}",
                "Content-Type": "application/json"
            }
            
            # Simple test request
            test_payload = {
                "model": "gpt-oss-120b",  # Default fast model
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 10,
                "temperature": 0.0
            }
            
            response = requests.post(
                self.endpoint_url, 
                headers=headers, 
                json=test_payload, 
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ Cerebras API connection successful")
            else:
                logger.warning(f"⚠️ Cerebras API test failed: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"⚠️ Cerebras API connection test failed: {e}")

    def _get_extraction_prompt(self) -> str:
        """Load extraction prompt from file or return built-in prompt"""
        try:
            with open('prompt.md', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("prompt.md not found, using built-in prompt")
            return self._get_builtin_prompt()

    def _get_builtin_prompt(self) -> str:
        """Return optimized built-in extraction prompt for Cerebras"""
        return """
# Timetable Data Extractor Prompt

You are an expert timetable data extractor. Your task is to analyze the provided timetable document and extract structured information.

## CRITICAL INSTRUCTIONS:

1. You MUST output ONLY valid JSON - no additional text, explanations, or markdown formatting.
2. The entire response must be a single JSON object.
3. Do NOT use code blocks or markdown syntax like ```json.
4. Extract ALL available information from the document.
5. Create comprehensive mappings for subjects, faculty, and sections.
6. Generate appropriate IDs where missing.
7. Infer reasonable defaults for missing information.

## REQUIRED JSON STRUCTURE:

```json
{{
  "college_info": {{
    "name": "College Name from Document",
    "session": "2025-26",
    "effective_date": "2025-09-15"
  }},
  "time_slots": {{
    "periods": [
      {{"id": 1, "start_time": "08:00", "end_time": "08:50"}},
      {{"id": 2, "start_time": "08:50", "end_time": "09:40"}},
      {{"id": 3, "start_time": "09:40", "end_time": "10:30"}},
      {{"id": 4, "start_time": "10:45", "end_time": "11:35"}},
      {{"id": 5, "start_time": "11:35", "end_time": "12:25"}},
      {{"id": 6, "start_time": "12:25", "end_time": "13:15"}},
      {{"id": 7, "start_time": "14:15", "end_time": "15:05"}},
      {{"id": 8, "start_time": "15:05", "end_time": "15:55"}}
    ],
    "working_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "break_periods": [4, 6],
    "lunch_period": 6,
    "mentorship_period": 4
  }},
  "departments": [
    {{
      "dept_id": "CSE",
      "name": "Computer Science Engineering",
      "sections": [
        {{
          "section_id": "CSE_3_A",
          "name": "CSE-3A",
          "semester": 3,
          "year": 2,
          "room": "CL-301",
          "student_count": 60,
          "coordinator": "HOD CSE",
          "specialization": ""
        }}
      ]
    }}
  ],
  "subjects": [
    {{
      "subject_id": "CS301",
      "name": "Data Structures",
      "type": "Theory",
      "credits": 4,
      "lectures_per_week": 4,
      "semester": 3,
      "departments": ["CSE"],
      "specialization": "",
      "min_classes_per_week": 3,
      "max_classes_per_day": 2,
      "tutorial_sessions": 0
    }}
  ],
  "labs": [
    {{
      "lab_id": "CS301L",
      "name": "Data Structures Lab",
      "type": "Lab",
      "credits": 2,
      "sessions_per_week": 1,
      "duration_hours": 2,
      "semester": 3,
      "departments": ["CSE"],
      "specialization": "",
      "lab_rooms": ["LAB-1", "LAB-2"]
    }}
  ],
  "faculty": [
    {{
      "faculty_id": "F001",
      "name": "Dr. Faculty Name",
      "department": "CSE",
      "designation": "Professor",
      "subjects": ["CS301", "CS301L"],
      "max_hours_per_week": 20,
      "avg_leaves_per_month": 2,
      "preferred_time_slots": [1, 2, 3, 5, 7, 8]
    }}
  ],
  "rooms": [
    {{
      "room_id": "CL-301",
      "name": "Classroom 301",
      "type": "Classroom",
      "capacity": 70,
      "department": "CSE",
      "equipment": ["Projector", "Smart Board"]
    }}
  ],
  "subject_name_mapping": {{
    "DS": "CS301",
    "Data Structures": "CS301"
  }},
  "constraints": {{
    "hard_constraints": {{
      "no_faculty_clash": true,
      "no_room_clash": true,
      "no_section_clash": true,
      "break_periods_fixed": [4, 6],
      "lunch_period_fixed": 6,
      "max_classes_per_day_per_section": 7,
      "lab_duration_consecutive": true
    }},
    "soft_constraints": {{
      "balanced_daily_load": {{"weight": 0.3, "max_deviation": 2}},
      "faculty_preference_slots": {{"weight": 0.2}},
      "minimize_faculty_travel": {{"weight": 0.15}}
    }}
  }},
  "special_requirements": {{
    "mentorship_break": {{
      "period": 4,
      "duration": 1,
      "all_sections": true
    }}
  }},
  "genetic_algorithm_params": {{
    "population_size": 50,
    "generations": 50,
    "mutation_rate": 0.2,
    "crossover_rate": 0.8,
    "elite_size": 5,
    "early_stopping_patience": 10
    }}
}}
````

## EXTRACTION RULES:

1.  Look for time slots (periods) - extract start/end times from the document
2.  Find all subjects mentioned - create subject IDs and names
3.  Identify faculty names - generate faculty IDs (F001, F002, etc.)
4.  Extract section names - create section IDs
5.  Find room information - create room IDs
6.  Map subjects to faculty based on timetable assignments, SHOULD NOT BE WRONG.
7.  Infer department information from section names
8.  Create appropriate constraints and requirements
9.  Don't use ... to skip parts, use proper delimiters

## IMPORTANT:

  - Extract REAL data from the document, don't just use the template values
  - If you find specific time periods, use those instead of the template times
  - Map actual faculty names from the document
  - Use actual subject names and codes from the document
  - Create realistic section names based on what you see
  - MAKE SURE TO PROVIDE VALID JSON

## Document content to analyze:

{document_text}

REMEMBER: Output ONLY the JSON structure with extracted data. No explanations, no code blocks, no markdown formatting.
"""

    def _generate_with_cerebras(self, prompt: str, max_tokens: int = 70000) -> tuple:
        """Generate response using Cerebras API"""
        headers = {
            "Authorization": f"Bearer {self.cerebras_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-oss-120b",  # Fast, capable model
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,  # Deterministic for consistent JSON
            "top_p": 1.0,        # Very focused generation
            "stream": False      # Get complete response
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                self.endpoint_url, 
                headers=headers, 
                json=payload,
                timeout=60  # 1 minute timeout
            )
            
            latency = time.time() - start_time
            
            if response.status_code != 200:
                raise Exception(f"Cerebras API error: {response.status_code} - {response.text}")
            
            data = response.json()
            
            # Extract content from response
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                return content, latency
            else:
                raise Exception("Invalid response format from Cerebras API")
                
        except requests.exceptions.Timeout:
            raise Exception("Cerebras API request timed out")
        except Exception as e:
            raise Exception(f"Cerebras API request failed: {str(e)}")

    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """
        Extract text from PDF using multiple methods for better coverage
        """
        text_content = ""
        try:
            # Method 1: pdfplumber (best for tables and layout)
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\\n=== PAGE {page_num + 1} ===\\n{page_text}\\n"
                    
                    # Extract tables if any
                    tables = page.extract_tables()
                    if tables:
                        text_content += f"\\n=== TABLES ON PAGE {page_num + 1} ===\\n"
                        for table_num, table in enumerate(tables):
                            text_content += f"\\nTable {table_num + 1}:\\n"
                            for row_num, row in enumerate(table):
                                if row:
                                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                                    text_content += f"Row {row_num + 1}: {' | '.join(clean_row)}\\n"
                            text_content += "\\n"
        
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            try:
                # Method 2: PyMuPDF as fallback
                doc = fitz.open(stream=file_content, filetype="pdf")
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    if page_text:
                        text_content += f"\\n=== PAGE {page_num + 1} (PyMuPDF) ===\\n{page_text}\\n"
                doc.close()
            
            except Exception as e2:
                logger.error(f"All PDF extraction methods failed: {e2}")
                raise Exception("Failed to extract text from PDF")
        
        return text_content.strip()

    def extract_text_from_excel(self, file_content: bytes, filename: str) -> str:
        """
        Extract text from Excel files
        """
        text_content = ""
        try:
            # Use pandas for better handling of different Excel formats
            if filename.endswith('.xlsx') or filename.endswith('.xlsm'):
                engine = 'openpyxl'
            elif filename.endswith('.xls'):
                engine = 'xlrd'
            else:
                engine = 'openpyxl'
            
            # Read all sheets
            excel_file = pd.ExcelFile(io.BytesIO(file_content), engine=engine)
            for sheet_name in excel_file.sheet_names:
                text_content += f"\\n=== SHEET: {sheet_name} ===\\n"
                df = pd.read_excel(io.BytesIO(file_content), 
                                 sheet_name=sheet_name, 
                                 engine=engine,
                                 header=None)
                
                for idx, row in df.iterrows():
                    row_values = []
                    for val in row:
                        if pd.notna(val):
                            row_values.append(str(val).strip())
                        else:
                            row_values.append("")
                    row_text = " | ".join(row_values)
                    if row_text.strip():
                        text_content += f"Row {idx + 1}: {row_text}\\n"
                text_content += "\\n"
        
        except Exception as e:
            logger.error(f"Excel extraction failed: {e}")
            raise Exception(f"Failed to extract text from Excel: {str(e)}")
        
        return text_content.strip()

    def clean_and_validate_json(self, response_text: str) -> Dict:
        """
        Clean and validate JSON response with multiple strategies
        """
        try:
            # Find JSON boundaries
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            response_text = re.sub(r'^```json\\s*', '', response_text)
            response_text = re.sub(r'^```\\s*', '', response_text)
            response_text = re.sub(r'\\s*```$', '', response_text)
            
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                raise ValueError("No valid JSON found in response")
            
            json_content = response_text[start_idx:end_idx + 1]
            
            # Multiple cleaning strategies
            cleaning_strategies = [
                lambda x: x,  # Try original first
                lambda x: re.sub(r',\\s*([}\\]])', r'\\1', x),  # Remove trailing commas
                lambda x: re.sub(r'\\.\\.\\.+', '', x),  # Remove ellipsis
                lambda x: re.sub(r'(?<=[{,])\\s*([a-zA-Z_][a-zA-Z0-9_]*)\\s*:', r'"\\1":', x),  # Quote keys
            ]
            
            for i, strategy in enumerate(cleaning_strategies):
                try:
                    cleaned = strategy(json_content)
                    result = json.loads(cleaned)
                    if i > 0:
                        logger.info(f"✅ JSON cleaned with strategy {i}")
                    return result
                except json.JSONDecodeError:
                    continue
            
            # Last resort: extract first complete JSON object
            brace_count = 0
            for i, char in enumerate(json_content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            result = json.loads(json_content[:i+1])
                            logger.info("Extracted first complete JSON object")
                            return result
                        except:
                            continue
            
            raise ValueError("All JSON cleaning strategies failed")
            
        except Exception as e:
            logger.error(f"JSON cleaning failed: {e}")
            logger.error(f"Response preview: {response_text[:200]}...")
            raise ValueError(f"Invalid JSON response: {str(e)}")

    def enhance_extracted_data(self, data: Dict) -> Dict:
        """
        Enhance and validate extracted data with proper defaults
        """
        if not data:
            data = {}
        
        # Ensure required structure exists
        if 'college_info' not in data:
            data['college_info'] = {
                "name": "Extracted College",
                "session": "2025-26",
                "effective_date": "2025-09-15"
            }
        
        if 'time_slots' not in data or not data['time_slots'].get('periods'):
            data['time_slots'] = {
                "periods": [
                    {"id": 1, "start_time": "08:30", "end_time": "09:20"},
                    {"id": 2, "start_time": "09:20", "end_time": "10:10"},
                    {"id": 3, "start_time": "10:10", "end_time": "11:00"},
                    {"id": 4, "start_time": "11:00", "end_time": "11:50"},
                    {"id": 5, "start_time": "11:50", "end_time": "12:40"},
                    {"id": 6, "start_time": "12:40", "end_time": "13:30"},
                    {"id": 7, "start_time": "13:30", "end_time": "14:20"},
                    {"id": 8, "start_time": "14:20", "end_time": "15:10"}
                ],
                "working_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "break_periods": [3, 6],
                "lunch_period": 6,
                "mentorship_period": 3
            }
        
        # Add other required sections with defaults
        data.setdefault('departments', [])
        data.setdefault('subjects', [])
        data.setdefault('labs', [])
        data.setdefault('faculty', [])
        data.setdefault('rooms', [])
        data.setdefault('subject_name_mapping', {})
        
        data.setdefault('constraints', {
            "hard_constraints": {
                "no_faculty_clash": True,
                "no_room_clash": True,
                "no_section_clash": True,
                "break_periods_fixed": data['time_slots'].get('break_periods', [3, 6]),
                "lunch_period_fixed": data['time_slots'].get('lunch_period', 6),
                "max_classes_per_day_per_section": 7,
                "lab_duration_consecutive": True
            },
            "soft_constraints": {
                "balanced_daily_load": {"weight": 0.3, "max_deviation": 2},
                "faculty_preference_slots": {"weight": 0.2},
                "minimize_faculty_travel": {"weight": 0.15}
            }
        })
        
        data.setdefault('special_requirements', {
            "mentorship_break": {
                "period": data['time_slots'].get('mentorship_period', 3),
                "duration": 1,
                "all_sections": True
            }
        })
        
        data.setdefault('genetic_algorithm_params', {
            "population_size": 50,
            "generations": 50,
            "mutation_rate": 0.2,
            "crossover_rate": 0.8,
            "elite_size": 5,
            "early_stopping_patience": 10
        })
        
        return data

    def extract_from_text(self, document_text: str, max_retries: int = 2) -> Dict:
        """
        ULTRA-FAST: Extract structured timetable data using Cerebras
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Aggressive text reduction for ultra-fast processing
                max_text_length = 100000  # Even smaller for Cerebras speed
                if len(document_text) > max_text_length:
                    document_text = document_text[:max_text_length] + "\\n[TRUNCATED FOR SPEED]"
                    logger.warning(f"Document truncated to {max_text_length} chars for ultra-fast processing")
                
                # Build prompt
                full_prompt = self.extraction_prompt.format(document_text=document_text)
                
                logger.info(f"Attempt {attempt + 1}: Cerebras ultra-fast extraction")
                
                # Generate with Cerebras (ultra-fast!)
                response_text, latency = self._generate_with_cerebras(full_prompt)
                
                logger.info(f"Cerebras API response time: {latency:.2f}s")
                
                if not response_text:
                    raise Exception("Empty response from Cerebras API")
                
                # Clean and validate JSON
                parsed_data = self.clean_and_validate_json(response_text)
                
                # Enhance with defaults
                enhanced_data = self.enhance_extracted_data(parsed_data)
                
                logger.info(f"Ultra-fast extraction successful on attempt {attempt + 1}")
                return enhanced_data
            
            except Exception as e:
                last_error = e
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info("Retrying in 1 second...")
                    time.sleep(1)
        
        # All attempts failed - return enhanced default
        logger.error(f"All attempts failed. Last error: {last_error}")
        logger.info("Returning enhanced default structure")
        
        default_data = {
            "college_info": {
                "name": "Extracted College",
                "session": "2025-26",
                "effective_date": "2025-09-15"
            }
        }
        return self.enhance_extracted_data(default_data)

    def extract_timetable_data(self, 
                             file_content: bytes,
                             filename: str,
                             college_name: Optional[str] = None,
                             session: Optional[str] = None) -> Dict:
        """
        Main method to extract timetable data - ULTRA-FAST with Cerebras
        """
        try:
            logger.info(f"🚀 Starting ULTRA-FAST Cerebras extraction for: {filename}")
            total_start = time.time()
            
            # Determine file type and extract text
            if filename.lower().endswith('.pdf'):
                document_text = self.extract_text_from_pdf(file_content)
                logger.info("PDF text extraction completed")
            elif filename.lower().endswith(('.xlsx', '.xls', '.xlsm')):
                document_text = self.extract_text_from_excel(file_content, filename)
                logger.info("Excel text extraction completed")
            else:
                raise Exception(f"Unsupported file format: {filename}")
            
            if not document_text.strip():
                raise Exception("No text content extracted from file")
            
            logger.info(f"Extracted {len(document_text)} characters of text")
            
            # Extract structured data using ultra-fast Cerebras
            structured_data = self.extract_from_text(document_text)
            
            # Override college info if provided
            if college_name:
                structured_data['college_info']['name'] = college_name
            if session:
                structured_data['college_info']['session'] = session
            
            # Add extraction metadata
            structured_data['extraction_info'] = {
                'extracted_at': datetime.now().isoformat(),
                'source_file': filename,
                'extracted_text_length': len(document_text),
                'extraction_method': 'cerebras_ultra_fast',
                'model_used': 'gpt-oss-120b',
                'total_time_seconds': round(time.time() - total_start, 2)
            }
            
            total_time = time.time() - total_start
            logger.info(f"ULTRA-FAST Cerebras extraction completed in {total_time:.2f}s")
            logger.info(f"Extracted: {len(structured_data.get('subjects', []))} subjects, "
                       f"{len(structured_data.get('faculty', []))} faculty, "
                       f"{len(structured_data.get('departments', []))} departments")
            
            return structured_data
        
        except Exception as e:
            logger.error(f"Cerebras extraction failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise Exception(f"Timetable extraction failed: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Test with your Cerebras API key
    # extractor = TimetableExtractor("your_cerebras_api_key")
    # with open("sample_timetable.pdf", "rb") as f:
    #     result = extractor.extract_timetable_data(f.read(), "sample_timetable.pdf")
    #     print(json.dumps(result, indent=2))
    pass