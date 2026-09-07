"""Automated clinical referral report compiler for AI-assisted screening (Zero Third-Party Dependencies)."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def compile_text_summary(encounter_data: Dict[str, Any]) -> str:
    """Generates an ASCII clinical screening summary."""
    img_id = encounter_data.get("image_id", "Unknown")
    q_data = encounter_data.get("quality") or {}
    q_status = q_data.get("status", "UNKNOWN")
    cls_data = encounter_data.get("classification") or {}
    rec_data = encounter_data.get("recommendation") or {}

    grade = cls_data.get("predicted_grade", "N/A")
    conf = cls_data.get("confidence", 0.0)
    referable = "REFERABLE" if cls_data.get("referable") else "NON-REFERABLE"
    action = rec_data.get("action", "UNKNOWN")
    reason = rec_data.get("reason", "")

    lines = [
        "=" * 65,
        "     NETRARAKSHAK AI: CLINICAL TRIAGE SCREENING REPORT",
        "=" * 65,
        f"Encounter ID:        {img_id}",
        f"Timestamp:           {encounter_data.get('metadata', {}).get('timestamp', 'N/A')}",
        f"Quality Gate Status: {q_status}",
    ]
    if q_status == "FAIL":
        lines.append(f"Rejection Reasons:   {'; '.join(q_data.get('failed_checks', []))}")
        lines.append(f"Triage Action:       {action}")
        lines.append(f"Patient Notice:      {q_data.get('message', '')}")
    else:
        lines.extend([
            f"Predicted Severity:  ICDR Grade {grade}",
            f"Referral Status:     {referable}",
            f"Calibrated Conf.:    {conf * 100:.2f}%",
            f"Triage Action:       {action}",
            f"Clinical Rationale:  {reason}",
        ])

    lines.extend([
        "-" * 65,
        "MANDATORY SCIENTIFIC & REGULATORY NOTICE:",
        "This tool provides AI-assisted screening evidence, NOT a definitive",
        "ophthalmological diagnosis. Independent clinical review remains necessary.",
        "=" * 65,
    ])
    return "\n".join(lines)


def _generate_pure_pdf(encounter_data: Dict[str, Any], output_path: Path) -> Path:
    """Compiles a compliant PDF 1.4 document using pure standard Python."""
    img_id = encounter_data.get("image_id", "Unknown")
    q_data = encounter_data.get("quality") or {}
    q_status = q_data.get("status", "UNKNOWN")
    cls_data = encounter_data.get("classification") or {}
    rec_data = encounter_data.get("recommendation") or {}
    meta = encounter_data.get("metadata") or {}

    grade = cls_data.get("predicted_grade", "N/A")
    conf = cls_data.get("confidence", 0.0)
    referable = "REFERABLE" if cls_data.get("referable") else "NON-REFERABLE"
    action = rec_data.get("action", "UNKNOWN")
    reason = rec_data.get("reason", "")
    timestamp = meta.get("timestamp", "N/A")

    def escape_pdf(s: str) -> str:
        return str(s).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    lines: List[Tuple[str, int, str]] = [
        ("F2", 16, "NetraRakshakAI: Diabetic Retinopathy Screening Report"),
        ("F1", 10, "Automated AI-assisted triage report | Model: E007 (Frozen)"),
        ("F1", 10, "-" * 60),
        ("F2", 11, f"Encounter ID:        {escape_pdf(img_id)}"),
        ("F1", 10, f"Timestamp (UTC):     {escape_pdf(timestamp)}"),
        ("F2", 11, f"Quality Assessment:  {escape_pdf(q_status)}"),
    ]

    if q_status == "FAIL":
        checks = "; ".join(q_data.get("failed_checks", []))
        lines.extend([
            ("F2", 10, f"Rejection Reasons:   {escape_pdf(checks)}"),
            ("F2", 11, f"Recommended Action:  {escape_pdf(action)}"),
            ("F1", 10, f"Patient Notice:      {escape_pdf(q_data.get('message', 'Recapture required.'))}"),
        ])
    else:
        lines.extend([
            ("F2", 11, f"Predicted Severity:  ICDR Grade {escape_pdf(grade)}"),
            ("F2", 11, f"Referral Decision:   {escape_pdf(referable)}"),
            ("F1", 10, f"Calibrated Conf.:    {conf * 100:.2f}% (T=0.7785)"),
            ("F2", 11, f"Recommended Action:  {escape_pdf(action)}"),
            ("F1", 10, f"Triage Rationale:    {escape_pdf(reason)}"),
        ])

    lines.extend([
        ("F1", 10, "-" * 60),
        ("F2", 9, "MANDATORY SCIENTIFIC & REGULATORY NOTICE:"),
        ("F1", 8, "This tool provides AI-assisted screening evidence, NOT a definitive"),
        ("F1", 8, "ophthalmological diagnosis. Independent clinical review remains necessary."),
    ])

    stream_ops = ["BT", "50 740 Td"]
    for font, size, text in lines:
        stream_ops.append(f"/{font} {size} Tf")
        stream_ops.append(f"({text}) Tj")
        stream_ops.append("0 -18 Td")
    stream_ops.append("ET")

    stream_str = "\n".join(stream_ops)
    stream_bytes = stream_str.encode("latin1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length " + str(len(stream_bytes)).encode("latin1") + b" >>\nstream\n" + stream_bytes + b"\nendstream",
    ]

    out_bytes = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out_bytes))
        out_bytes.extend(f"{i} 0 obj\n".encode("latin1"))
        out_bytes.extend(obj)
        out_bytes.extend(b"\nendobj\n")

    xref_offset = len(out_bytes)
    out_bytes.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin1"))
    out_bytes.extend(b"0000000000 65535 f \n")
    for off in offsets:
        out_bytes.extend(f"{off:010d} 00000 n \n".encode("latin1"))

    out_bytes.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin1")
    )

    output_path.write_bytes(out_bytes)
    return output_path


def compile_screening_report(
    encounter_data: Dict[str, Any],
    output_path: Union[str, Path]
) -> Path:
    """Compiles a clinical screening report as PDF, text, or JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if out_p.suffix.lower() == ".txt":
        out_p.write_text(compile_text_summary(encounter_data), encoding="utf-8")
        return out_p
    elif out_p.suffix.lower() == ".json":
        clean_data = json.loads(json.dumps(encounter_data, default=lambda o: None))
        out_p.write_text(json.dumps(clean_data, indent=2), encoding="utf-8")
        return out_p
    elif out_p.suffix.lower() == ".pdf":
        return _generate_pure_pdf(encounter_data, out_p)
    else:
        out_p.write_text(compile_text_summary(encounter_data), encoding="utf-8")
        return out_p
