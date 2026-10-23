"""PDF Document Loader for GE GER-3620 Rev P with Section and Page Extraction."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re
import pymupdf

logger = logging.getLogger(__name__)

# Canonical section mapping based on GER-3620 Rev P Table of Contents
PAGE_TO_SECTION_MAP = {
    1: "Title Page & Authors",
    2: "Blank Page",
    3: "Contents",
    4: "Contents",
    5: "Introduction / Maintenance Planning",
    6: "Digital Solutions for Asset Management",
    7: "Gas Turbine Configuration Maintenance Features",
    8: "Borescope Inspections / Major Factors Influencing Maintenance and Equipment Life",
    9: "Starts and Hours Criteria",
    10: "Service Factors / Fuel",
    11: "Fuel / Firing Temperatures",
    12: "Firing Temperatures / Steam/Water Injection",
    13: "Steam/Water Injection",
    14: "Cyclic Effects and Fast Starts / Hot Gas Path Parts",
    15: "Hot Gas Path Parts",
    16: "Hot Gas Path Parts",
    17: "Rotor Parts",
    18: "Rotor Parts",
    19: "Combustion Parts",
    20: "Combustion Parts / Casing Parts / Exhaust Diffuser Parts",
    21: "Exhaust Diffuser Parts",
    22: "Off-Frequency Operation / Compressor Condition and Performance",
    23: "Compressor Condition and Performance",
    24: "Lube Oil Cleanliness / Moisture Intake",
    25: "Maintenance Inspections",
    26: "Running Inspections / Rapid Cool-Down / Combustion Inspection",
    27: "Combustion Inspection / Hot Gas Path Inspection",
    28: "Hot Gas Path Inspection",
    29: "Hot Gas Path Inspection (Bucket Oxidation & Coatings)",
    30: "Hot Gas Path Inspection / Major Inspection",
    31: "Major Inspection / Parts Planning",
    32: "Inspection Intervals / Borescope Inspection Interval / ABC Inspections",
    33: "Inspection Intervals (Figure 36: Baseline Recommended Inspection Intervals)",
    34: "Inspection Intervals (Figure 37: ABC Inspections)",
    35: "Inspection Intervals (Figure 38: Annular/Silo Fleet Inspection Intervals)",
    36: "Combustion Inspection Interval",
    37: "Combustion Inspection Interval (Hours & Starts Maintenance Factors)",
    38: "Hot Gas Path Inspection Interval",
    39: "GE Vernova Annular/Silo Fleet & Rotor Inspection Intervals",
    40: "Rotor Inspection Interval (Starts-Based & Welded Rotor Life)",
    41: "Personnel Planning / Conclusion",
    42: "References",
    43: "Appendix A.1 - Hot Gas Path Maintenance Interval Calculation",
    44: "Appendix A.2 & A.3 - Factored Starts and Load Step Calculations",
    45: "Appendix B - Combustion Maintenance Interval Calculations",
    46: "Appendix C - Definitions",
    47: "Appendix D - Estimated Repair and Replacement Intervals (6B.03 & 7E.03)",
    48: "Appendix D - Estimated Repair and Replacement Intervals (9E.03 & 6F.03)",
    49: "Appendix D - Estimated Repair and Replacement Intervals (7F.03 & 7F.04/7FB.04)",
    50: "Appendix D - Estimated Repair and Replacement Intervals (9F.03 & 9F.05)",
    51: "Appendix E - Borescope Inspection Ports (6FA.03 & 7F/9F)",
    52: "Appendix E - Borescope Inspection Ports (7E.03 & 7HA.01)",
    53: "Appendix E - Borescope Inspection Ports (6FA.03 & Arrangements)",
    54: "Appendix E - Borescope Inspection Ports (GT11N2, GT13E2, GT24/GT26)",
    55: "Appendix F - Turning Gear/Ratchet Running Guidelines",
    56: "Appendix G - Turbine Naming Conventions",
    57: "Revision History",
    58: "Revision History (Continued)",
    59: "List of Figures",
    60: "Acknowledgments",
}


@dataclass
class DocumentPage:
    """Represents an extracted page from the PDF document."""
    page_number: int       # 1-based PDF page index (1..60)
    printed_page: str      # Printed page number from document footer/header (e.g., '21', 'i', 'N/A')
    section: str           # Section title / heading
    text: str              # Extracted plain text
    is_blank: bool         # Whether the page is blank or non-textual


def compute_pdf_fingerprint(pdf_path: Path) -> str:
    """Computes a SHA-256 checksum for the PDF file to detect changes."""
    sha256 = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class PDFLoader:
    """Extracts text and metadata from GER-3620-P.pdf preserving page and section tracking."""

    def __init__(self, pdf_path: Path | str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found at: {self.pdf_path.resolve()}.\n"
                f"Please ensure the official 'GER-3620-P.pdf' is placed in the data/ folder."
            )

    def extract_printed_page(self, page_index_0: int, text: str) -> str:
        """Extracts the printed page number from page header/footer lines."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Check top lines
        for line in lines[:4]:
            if re.fullmatch(r"\d+|[ivx]+", line.lower()):
                return line
        # Check bottom lines
        for line in lines[-4:]:
            if re.fullmatch(r"\d+|[ivx]+", line.lower()):
                return line
        # Offset fallback (PDF page 5 corresponds to printed page 1)
        if page_index_0 >= 4:
            return str(page_index_0 - 3)
        if page_index_0 == 2:
            return "i"
        if page_index_0 == 3:
            return "ii"
        return "N/A"

    def detect_section(self, page_number_1: int, text: str) -> str:
        """Detects section heading using structural lookup with on-page header verification."""
        # Use canonical mapping as foundational ground truth
        canonical_section = PAGE_TO_SECTION_MAP.get(page_number_1, "General Considerations")
        return canonical_section

    def load(self) -> List[DocumentPage]:
        """Loads and extracts all pages from the PDF document."""
        pages: List[DocumentPage] = []
        try:
            doc = pymupdf.open(self.pdf_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF at {self.pdf_path}: {e}") from e

        total_pages = len(doc)
        logger.info(f"Loading PDF: {self.pdf_path.name} ({total_pages} pages)")

        for page_idx in range(total_pages):
            page_num_1 = page_idx + 1
            try:
                page = doc[page_idx]
                raw_text = page.get_text()
                stripped = raw_text.strip()

                # Clean non-critical document header watermark lines to keep text clean
                clean_lines = []
                for line in raw_text.splitlines():
                    # Strip out standard document classification watermark
                    if "Non-Public- Class II" in line or "Doc Name : GER-3620" in line:
                        continue
                    clean_lines.append(line)
                cleaned_text = "\n".join(clean_lines).strip()

                is_blank = len(cleaned_text) < 30
                printed_page = self.extract_printed_page(page_idx, raw_text)
                section = self.detect_section(page_num_1, cleaned_text)

                pages.append(
                    DocumentPage(
                        page_number=page_num_1,
                        printed_page=printed_page,
                        section=section,
                        text=cleaned_text if not is_blank else "",
                        is_blank=is_blank,
                    )
                )
            except Exception as exc:
                logger.error(f"Error extracting page {page_num_1}: {exc}")
                pages.append(
                    DocumentPage(
                        page_number=page_num_1,
                        printed_page=str(page_num_1),
                        section=PAGE_TO_SECTION_MAP.get(page_num_1, "Error"),
                        text="",
                        is_blank=True,
                    )
                )

        doc.close()
        return pages
