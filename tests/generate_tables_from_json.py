"""
Generate PDF tables from LFW evaluation results JSON file.
This script reads the evaluation results and generates publication-ready tables.
"""

import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


def generate_tables(results_json_path: Path, output_path: Path):
    """Generate PDF tables from evaluation results JSON."""
    
    # Load results
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    # Create PDF
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph("<b>LFW Protocol Evaluation Results</b>", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.3 * inch))
    
    # Table 1: Verification Results
    verification_title = Paragraph("<b>Table 1: Verification Accuracy (%)</b>", styles['Heading2'])
    story.append(verification_title)
    story.append(Spacer(1, 0.1 * inch))
    
    # Build verification table data
    header = ['Method'] + ['16×16', '24×24', '32×32', '112×112 (HR)']
    bicubic_row = ['Bicubic']
    dsr_row = ['DSR']
    
    for res in ['16x16', '24x24', '32x32', '112x112']:
        if res == '112x112':
            # HR baseline - stored under 'bicubic' key
            acc = results['table1_verification'][res]['bicubic']['accuracy']
            bicubic_row.append(f"{acc:.2f}")
            dsr_row.append('-')
        else:
            # VLR resolutions - have both methods
            bicubic_acc = results['table1_verification'][res]['bicubic']['accuracy']
            dsr_acc = results['table1_verification'][res]['dsr']['accuracy']
            bicubic_row.append(f"{bicubic_acc:.2f}")
            dsr_row.append(f"{dsr_acc:.2f}")
    
    table1_data = [header, bicubic_row, dsr_row]
    
    table1 = Table(table1_data, colWidths=[1.5*inch] + [1.2*inch] * 4)
    table1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    story.append(table1)
    story.append(Spacer(1, 0.5 * inch))
    
    # Table 2: Identification Results (Rank-1)
    identification_title = Paragraph("<b>Table 2: Identification Rank-1 Accuracy (%)</b>", styles['Heading2'])
    story.append(identification_title)
    story.append(Spacer(1, 0.1 * inch))
    
    # Build identification table data
    bicubic_row = ['Bicubic']
    dsr_row = ['DSR']
    
    for res in ['16x16', '24x24', '32x32', '112x112']:
        if res == '112x112':
            # HR baseline - stored under 'bicubic' key
            rank1 = results['table2_identification'][res]['bicubic']['rank1']
            bicubic_row.append(f"{rank1:.2f}")
            dsr_row.append('-')
        else:
            # VLR resolutions - have both methods
            bicubic_rank1 = results['table2_identification'][res]['bicubic']['rank1']
            dsr_rank1 = results['table2_identification'][res]['dsr']['rank1']
            bicubic_row.append(f"{bicubic_rank1:.2f}")
            dsr_row.append(f"{dsr_rank1:.2f}")
    
    table2_data = [header, bicubic_row, dsr_row]
    
    table2 = Table(table2_data, colWidths=[1.5*inch] + [1.2*inch] * 4)
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    story.append(table2)
    
    # Build PDF
    doc.build(story)
    print(f"Tables generated successfully at {output_path}")


if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent.parent.parent
    results_json = project_root / "technical" / "evaluation_results" / "lfw_evaluation_results.json"
    output_pdf = project_root / "technical" / "evaluation_results" / "lfw_tables.pdf"
    
    # Ensure output directory exists
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate tables
    generate_tables(results_json, output_pdf)
