"""Generate small local sample PDFs for RAGiT demos and evals.

The generator uses only the Python standard library so it works before the
backend dependencies are installed. The PDFs are intentionally plain text,
which makes retrieval behavior easy to inspect.
"""

from __future__ import annotations

from pathlib import Path


DOCS: dict[str, list[list[str]]] = {
    "Employee_Handbook.pdf": [
        [
            "Employee Handbook",
            "Section: Leave Policy",
            "Full-time employees are eligible for annual leave after completing probation.",
            "Annual leave requests should be submitted through the HR portal at least five business days in advance.",
            "Managers approve leave based on team coverage and business continuity.",
        ],
        [
            "Section: Benefits Eligibility",
            "Permanent full-time employees are eligible for medical, dental, and retirement benefits.",
            "Contractors and temporary staff are not eligible unless a written agreement states otherwise.",
            "Benefits enrollment must be completed within thirty days of the employee start date.",
        ],
        [
            "Section: Travel Reimbursement",
            "Business travel expenses are reimbursable when pre-approved by a manager.",
            "Eligible expenses include airfare, lodging, ground transportation, and reasonable meals.",
            "Receipts must be submitted within fourteen days after the trip.",
        ],
    ],
    "Vendor_Onboarding_Guide.pdf": [
        [
            "Vendor Onboarding Guide",
            "Section: Onboarding Steps",
            "Vendor onboarding starts with business owner sponsorship and procurement intake.",
            "The vendor must complete due diligence, security review, tax setup, and contract approval.",
            "Procurement creates the vendor record only after approvals are complete.",
        ],
        [
            "Section: Required Documents",
            "Required vendor documents include a signed contract, tax form, bank details, insurance certificate, and security questionnaire.",
            "High-risk vendors must also complete a data protection impact assessment.",
        ],
    ],
    "SLA_Support_Process.pdf": [
        [
            "SLA Support Process",
            "Section: Escalation Path",
            "The SLA escalation path is documented in the Support Operations runbook.",
            "Priority one incidents escalate to the duty manager after fifteen minutes without acknowledgement.",
            "Priority two incidents escalate to the service owner after four business hours.",
        ],
        [
            "Section: Response Targets",
            "Priority one response target is fifteen minutes.",
            "Priority two response target is four business hours.",
            "Priority three response target is two business days.",
        ],
    ],
    "Security_Access_Control_Policy.pdf": [
        [
            "Security Access Control Policy",
            "Section: Access Control Requirements",
            "All production access requires unique user accounts, multi-factor authentication, and manager approval.",
            "Privileged access must be reviewed quarterly by system owners.",
            "Shared accounts are prohibited unless a documented exception is approved by security.",
        ],
        [
            "Section: Incident Reporting",
            "Employees must report suspected security incidents to the security operations mailbox immediately.",
            "Incidents involving customer data must also be escalated to legal and privacy teams within twenty-four hours.",
        ],
        [
            "Section: Data Retention",
            "Business records must be retained according to the data retention schedule.",
            "Security logs are retained for at least one year unless legal hold requires a longer period.",
        ],
    ],
}


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _page_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 12 Tf", "72 742 Td", "16 TL"]
    for line in lines:
        commands.append(f"({_pdf_escape(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return "\n".join(commands).encode("ascii")


def write_pdf(path: Path, pages: list[list[str]]) -> None:
    objects: list[bytes] = []
    page_object_ids: list[int] = []

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for page in pages:
        stream = _page_stream(page)
        content_id = len(objects) + 2
        page_id = len(objects) + 1
        page_object_ids.append(page_id)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        )

    kids = " ".join(f"{page_id} 0 R" for page_id in page_object_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_ids)} >>".encode("ascii")

    path.parent.mkdir(parents=True, exist_ok=True)
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{idx} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_start = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )

    path.write_bytes(bytes(output))


def main() -> None:
    root = Path(__file__).resolve().parent
    for filename, pages in DOCS.items():
        write_pdf(root / filename, pages)
        print(f"Wrote {root / filename}")


if __name__ == "__main__":
    main()
