import sys
import os
import subprocess

def docx_to_pdf(input_path, output_dir=None):
    if not output_dir:
        output_dir = os.path.dirname(os.path.abspath(input_path))

    command = [
        "soffice",
        "--headless",
        "--convert-to", "pdf",
        "--outdir", output_dir,
        input_path
    ]
    
    print(f"Converting '{input_path}' to PDF in '{output_dir}'...")
    subprocess.run(command, check=True)
    print(f"✅ Conversion complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python docx_to_pdf.py <input.docx> [output_dir]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    docx_to_pdf(input_file, output_dir)
