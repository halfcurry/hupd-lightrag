import os
import argparse
import shutil
import concurrent.futures
import orjson  # Using the faster orjson library
from typing import Dict, Any, List
from dotenv import load_dotenv # Import the dotenv library

def has_g06_main_ipcr_label(patent_data: Dict[str, Any]) -> bool:
    """Check if patent has specific G06 subcategory main_ipcr_label"""
    main_ipcr_label = patent_data.get('main_ipcr_label', '')
    if isinstance(main_ipcr_label, str):
        # Look for specific G06 subcategories: G06N, G06V
        target_labels = ["G06N", "G06V"]
        return any(main_ipcr_label.startswith(target) for target in target_labels)
    return False

def process_file(filename, input_path, output_path, field_limits, total_word_limit):
    """
    Processes a single JSON file. It truncates individual text fields based on
    word limits and then ensures the total word count does not exceed a total
    limit by truncating 'full_description'.
    """
    json_filepath = os.path.join(input_path, filename)
    try:
        # Open the file in binary mode ('rb') and use orjson for faster parsing
        with open(json_filepath, 'rb') as f:
            json_data = orjson.loads(f.read())

        if has_g06_main_ipcr_label(json_data):
            # --- MODIFIED: Part 1 - Truncate individual fields based on their limits ---
            for field, limit in field_limits.items():
                if limit < 0:  # Skip if limit is negative (i.e., not set or disabled)
                    continue
                
                text = json_data.get(field, '')
                if isinstance(text, str):
                    words = text.split()
                    if len(words) > limit:
                        json_data[field] = ' '.join(words[:limit])

            # --- MODIFIED: Part 2 - Apply total word limit logic ---
            base_fields = ['abstract', 'claims', 'background', 'summary']
            
            # Calculate the combined word count for the base fields (using potentially truncated text)
            base_word_count = 0
            for field in base_fields:
                text = json_data.get(field, '')
                if isinstance(text, str):
                    base_word_count += len(text.split())

            # Safely get the full_description text and its word count
            full_description_text = json_data.get('full_description', '')
            if not isinstance(full_description_text, str):
                full_description_text = ''
            full_description_word_count = len(full_description_text.split())

            # ✂️ Check if the total word count exceeds the limit and clip full_description if necessary
            if base_word_count + full_description_word_count > total_word_limit:
                # Calculate how many words are allowed for the full_description
                allowed_words_for_desc = total_word_limit - base_word_count
                
                if allowed_words_for_desc > 0:
                    # Clip the full_description to fit the remaining word budget
                    clipped_words = full_description_text.split()[:allowed_words_for_desc]
                    json_data['full_description'] = ' '.join(clipped_words)
                else:
                    # If base fields already exceed the limit, remove full_description entirely
                    json_data.pop('full_description', None)

            # --- MODIFIED: Write the JSON data in a formatted way ---
            destination_filepath = os.path.join(output_path, filename)
            with open(destination_filepath, 'wb') as f_out:
                # Use OPT_INDENT_2 for pretty-printing the JSON output
                f_out.write(orjson.dumps(json_data, option=orjson.OPT_INDENT_2))
            
            return ('COPIED', filename)
        else:
            return ('SKIPPED', filename)

    except Exception as e:
        # Any error during file processing is caught here
        print(f"Error processing file '{filename}': {e}")
        return ('ERROR', filename)

def main():
    """
    Main function to parse arguments and coordinate the parallel processing of JSON files.
    """
    # --- MODIFIED: Load environment variables from a .env file ---
    load_dotenv()

    # --- MODIFIED: Updated parser description to explain .env file usage ---
    parser = argparse.ArgumentParser(
        description="""Rapidly copy and process JSON patent files based on IPC labels and word counts.

This script requires the 'python-dotenv' library. Install it using:
pip install python-dotenv

Filters for patents with 'G06N' or 'G06V' main_ipcr_label.
Truncates text fields to meet word count limits.

Word limits can be configured via a .env file in the same directory or by setting
environment variables. If a variable is not set, no limit is applied to that field.

Example .env file:
--------------------------
ABSTRACT_WORD_LIMIT=500
CLAIMS_WORD_LIMIT=2500
BACKGROUND_WORD_LIMIT=1000
SUMMARY_WORD_LIMIT=1000
TOTAL_WORD_LIMIT=8000
--------------------------

- ABSTRACT_WORD_LIMIT: Max words for the 'abstract' field.
- CLAIMS_WORD_LIMIT:   Max words for the 'claims' field.
- BACKGROUND_WORD_LIMIT: Max words for the 'background' field.
- SUMMARY_WORD_LIMIT:    Max words for the 'summary' field.
- TOTAL_WORD_LIMIT:      Max total words for the document (defaults to 10000).
                         The 'full_description' field is truncated to meet this limit.

Example Usage:
python your_script_name.py 1000 ./input ./output
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "max_files",
        type=int,
        help="Maximum number of JSON files to copy to the output directory."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the folder containing JSON files."
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the folder where accepted JSON files will be copied."
    )

    args = parser.parse_args()

    # --- MODIFIED: Read configuration from environment after loading .env ---
    field_limits = {
        'abstract': int(os.environ.get('ABSTRACT_WORD_LIMIT', -1)),
        'claims': int(os.environ.get('CLAIMS_WORD_LIMIT', -1)),
        'background': int(os.environ.get('BACKGROUND_WORD_LIMIT', -1)),
        'summary': int(os.environ.get('SUMMARY_WORD_LIMIT', -1)),
    }
    TOTAL_WORD_LIMIT = int(os.environ.get('TOTAL_WORD_LIMIT', 10000))

    input_path = os.path.abspath(args.input_folder)
    output_path = os.path.abspath(args.output_folder)
    max_files_to_copy = args.max_files

    os.makedirs(output_path, exist_ok=True)

    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"🚀 Starting high-speed processing with orjson...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Max files to copy: {max_files_to_copy}")

    try:
        with os.scandir(input_path) as it:
            files = sorted([entry.name for entry in it if entry.is_file() and entry.name.endswith('.json')])

        if not files:
            print("No JSON files found in the input directory.")
            return

        with concurrent.futures.ProcessPoolExecutor() as executor:
            # --- MODIFIED: Pass configuration to each worker process ---
            future_to_file = {executor.submit(process_file, f, input_path, output_path, field_limits, TOTAL_WORD_LIMIT): f for f in files}

            for future in concurrent.futures.as_completed(future_to_file):
                if copied_count >= max_files_to_copy:
                    # Attempt to cancel pending futures to stop processing early
                    for f in future_to_file:
                        if not f.done():
                            f.cancel()
                    break

                try:
                    status, filename = future.result()
                    if status == 'COPIED':
                        copied_count += 1
                        print(f"✅ Copied '{filename}' ({copied_count}/{max_files_to_copy})")
                    elif status == 'SKIPPED':
                        skipped_count += 1
                    elif status == 'ERROR':
                        error_count += 1
                except concurrent.futures.CancelledError:
                    # This is expected when we cancel futures
                    pass
            
            if copied_count >= max_files_to_copy:
                print(f"\nReached maximum file copy limit of {max_files_to_copy}. Halting.")


    except FileNotFoundError:
        print(f"Error: Input folder '{input_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    total_processed = copied_count + skipped_count + error_count
    print(f"\n🏁 Finished. Processed {total_processed} files. Copied {copied_count}, Skipped {skipped_count}, Errors {error_count}.")


if __name__ == "__main__":
    main()
