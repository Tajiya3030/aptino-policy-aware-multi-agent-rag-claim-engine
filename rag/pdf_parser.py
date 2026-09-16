import os
from typing import List, Dict, Any
import pypdf

class PolicyPDFParser:
    """Extracts text page-by-page from policy PDF, preserving physical page numbers."""
    
    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Policy PDF not found at {pdf_path}")
        self.pdf_path = pdf_path

    def extract_pages(self) -> List[Dict[str, Any]]:
        """
        Reads the PDF and returns a list of dictionaries with page details:
        [
            {
                "page_number": 1,
                "text": "...",
                "source_file": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
            },
            ...
        ]
        """
        pages = []
        reader = pypdf.PdfReader(self.pdf_path)
        filename = os.path.basename(self.pdf_path)
        
        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            text = page.extract_text() or ""
            pages.append({
                "page_number": page_num,
                "text": text,
                "source_file": filename
            })
            
        return pages

if __name__ == "__main__":
    import sys
    path = r"data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    if os.path.exists(path):
        parser = PolicyPDFParser(path)
        extracted = parser.extract_pages()
        print(f"Extracted {len(extracted)} pages.")
        if extracted:
            print("Page 1 snippet:", extracted[0]['text'][:200])
