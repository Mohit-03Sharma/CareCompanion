import sys
import json
sys.path.insert(0, 'src')
from extractor import process_note

test_files = [
    "data/sample_notes/diabetes.txt",
    "data/sample_notes/hypertension.txt",
    "data/sample_notes/copd.txt",
]

for filepath in test_files:
    print(f"\n{'='*50}")
    print(f"Processing: {filepath}")
    print('='*50)

    result = process_note(file_path=filepath)

    if "error" in result:
        print(f"Error: {result['error']}")
        continue

    print(f"Patient: {result.get('patient_age')}yo {result.get('patient_gender')}")
    print(f"Chief complaint: {result.get('chief_complaint')}")
    print(f"\nDiagnoses:")
    for d in result.get('diagnoses', []):
        print(f"  - {d}")
    print(f"\nDischarge medications:")
    for m in result.get('discharge_medications', []):
        new = " [NEW]" if m.get('is_new') else ""
        print(f"  - {m.get('name')} {m.get('dose')}{new}")
    print(f"\nFollow-up actions:")
    for f in result.get('follow_up_actions', []):
        print(f"  - {f}")
    print(f"\nWarning signs:")
    for w in result.get('warning_signs', []):
        print(f"  - {w}")
    print(f"\nValidation warnings:")
    warnings = result.get('validation_warnings', [])
    if warnings:
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("  None — all diagnoses found in knowledge base")