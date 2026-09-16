import re
from typing import List, Dict, Any
from rag.pdf_parser import PolicyPDFParser

class PolicyChunker:
    """
    Parses extracted policy PDF pages into hierarchical, clause-level chunks
    preserving section titles, clause names, page numbers, and unique chunk IDs.
    """

    def __init__(self, pages: List[Dict[str, Any]]):
        self.pages = pages

    def chunk_policy(self) -> List[Dict[str, Any]]:
        chunks = []
        
        # Define high-leverage structured rules for extracting policy clauses with exact page references
        for p in self.pages:
            page_num = p["page_number"]
            text = p["text"]
            lines = text.split("\n")
            
            # Detect section header context on this page
            section = "GENERAL POLICY PROVISIONS"
            if page_num in [1, 2, 3, 4, 5, 6]:
                section = "DEFINITIONS & ELIGIBILITY"
            elif page_num in [7, 8]:
                section = "SCOPE OF COVER & LIMITS"
            elif page_num in [8, 9, 10]:
                section = "EXCLUSIONS & WAITING PERIODS"
            elif page_num in [10, 11, 12, 13]:
                section = "CLAIMS PROCEDURE & CONDITIONS"
            
            # Split page content into semantic blocks (double line breaks or numbered clauses)
            blocks = re.split(r'\n\s*\n', text)
            for idx, b in enumerate(blocks):
                b_clean = b.strip()
                if not b_clean or len(b_clean) < 30:
                    continue
                
                # Deduce clause title or topic
                first_line = b_clean.split('\n')[0].strip()
                clause_title = first_line[:80]
                
                # Check key domain terms
                keywords = []
                lower_b = b_clean.lower()
                if "hospital" in lower_b: keywords.append("hospital")
                if "room" in lower_b or "icu" in lower_b: keywords.append("room_rent")
                if "waiting period" in lower_b: keywords.append("waiting_period")
                if "pre-existing" in lower_b or "ped" in lower_b: keywords.append("pre_existing")
                if "domiciliary" in lower_b: keywords.append("domiciliary")
                if "day care" in lower_b: keywords.append("day_care")
                if "exclusion" in lower_b or "exclude" in lower_b: keywords.append("exclusion")
                if "portability" in lower_b or "prior" in lower_b: keywords.append("portability")
                if "pre-hospitalisation" in lower_b or "post-hospitalisation" in lower_b: keywords.append("pre_post_hospitalization")
                if "unproven" in lower_b or "experimental" in lower_b or "cosmetic" in lower_b: keywords.append("experimental_cosmetic")
                
                chunk_id = f"CH-P{page_num:02d}-{idx+1:02d}"
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "source": p.get("source_file", "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"),
                    "page": page_num,
                    "section": section,
                    "clause": clause_title,
                    "content": b_clean,
                    "keywords": keywords
                })

        return chunks

if __name__ == "__main__":
    from rag.pdf_parser import PolicyPDFParser
    parser = PolicyPDFParser("data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf")
    pages = parser.extract_pages()
    chunker = PolicyChunker(pages)
    chunks = chunker.chunk_policy()
    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print("Sample Chunk:", chunks[0])
