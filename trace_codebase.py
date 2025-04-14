"""
trace_codebase.py

Estrae e salva il contenuto di tutti i file Python all'interno della directory 'src/',
in un formato leggibile e utilizzabile per chatbot AI come contesto di progetto.

Output:
- code_trace.json  → struttura JSON { path: codice }
- code_trace.md    → struttura Markdown leggibile

Usage:
    python trace_codebase.py
"""

import os
import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
OUTPUT_JSON = Path(__file__).resolve().parent / "trace/code_trace.json"
OUTPUT_MD = Path(__file__).resolve().parent / "trace/code_trace.md"

def list_python_files(src_dir: Path) -> list:
    return sorted([f for f in src_dir.rglob("*.py") if f.is_file()])

def read_code(files: list) -> dict:
    code_map = {}
    for file in files:
        relative_path = file.relative_to(SRC_DIR)
        try:
            content = file.read_text(encoding="utf-8")
            code_map[str(relative_path)] = content
        except Exception as e:
            print(f"⚠️ Errore durante la lettura di {file}: {e}")
    return code_map

def save_json(code_map: dict, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(code_map, f, indent=2, ensure_ascii=False)

def save_markdown(code_map: dict, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for path, code in code_map.items():
            f.write(f"\n\n---\n\n### 📄 `{path}`\n\n```python\n{code}\n```\n")

def main():
    print(f"📁 Scanning Python files in: {SRC_DIR}")
    py_files = list_python_files(SRC_DIR)
    print(f"📦 Trovati {len(py_files)} file Python.")

    code_map = read_code(py_files)

    save_json(code_map, OUTPUT_JSON)
    save_markdown(code_map, OUTPUT_MD)

    print(f"✅ Esportazione completata:")
    print(f"- JSON: {OUTPUT_JSON}")
    print(f"- Markdown: {OUTPUT_MD}")

if __name__ == "__main__":
    main()
