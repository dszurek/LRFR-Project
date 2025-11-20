"""
Generate formatted PDF tables from LFW evaluation results
"""

import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import numpy as np


def generate_tables(results_json_path: Path, output_pdf_path: Path):
    """Generate formatted tables from evaluation results"""
    
    # Load results
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    with PdfPages(output_pdf_path) as pdf:
        # ============================================================
        # TABLE 1: VERIFICATION RATES
        # ============================================================
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axis('off')
        
        # Prepare table data
        # Get actual resolution keys from results
        res_keys = sorted(results['table1_verification'].keys(), 
                         key=lambda x: int(x.split('x')[0]))
        headers = ['Method'] + [k.replace('x', '×') for k in res_keys]
        
        table_data = [headers]
        
        # Bicubic row
        bicubic_row = ['Bicubic']
        for res in res_keys:
            acc = results['table1_verification'][res]['bicubic']['accuracy']
            bicubic_row.append(f"{acc:.2f}%")
        table_data.append(bicubic_row)
        
        # DSR row
        dsr_row = ['DSR']
        for res in res_keys:
            acc = results['table1_verification'][res]['dsr']['accuracy']
            dsr_row.append(f"{acc:.2f}%")
        table_data.append(dsr_row)
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style method column
        for i in range(1, 3):
            table[(i, 0)].set_facecolor('#E8E8E8')
            table[(i, 0)].set_text_props(weight='bold')
        
        ax.set_title('Table 1: Verification Accuracy (%) on LFW\n(EdgeFace Network, DSR Super-Resolution)',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============================================================
        # TABLE 2: RANK-1 IDENTIFICATION RATES
        # ============================================================
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axis('off')
        
        # Prepare table data
        res_keys = sorted(results['table2_identification'].keys(), 
                         key=lambda x: int(x.split('x')[0]))
        headers = ['Method'] + [k.replace('x', '×') for k in res_keys]
        
        table_data = [headers]
        
        # Bicubic row
        bicubic_row = ['Bicubic']
        for res in res_keys:
            rank1 = results['table2_identification'][res]['bicubic']['rank1']
            bicubic_row.append(f"{rank1:.2f}%")
        table_data.append(bicubic_row)
        
        # DSR row
        dsr_row = ['DSR']
        for res in res_keys:
            rank1 = results['table2_identification'][res]['dsr']['rank1']
            dsr_row.append(f"{rank1:.2f}%")
        table_data.append(dsr_row)
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style method column
        for i in range(1, 3):
            table[(i, 0)].set_facecolor('#E8E8E8')
            table[(i, 0)].set_text_props(weight='bold')
        
        ax.set_title('Table 2: Rank-1 Identification Rates (%) on LFW\n(EdgeFace Network, Closed-Set)',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============================================================
        # TABLE 3: ROC-AUC AND EER (VERIFICATION)
        # ============================================================
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.axis('off')
        
        # Prepare table data
        res_keys = sorted(results['table1_verification'].keys(), 
                         key=lambda x: int(x.split('x')[0]))
        
        table_data = [
            ['Method', 'Resolution', 'ROC-AUC', 'EER (%)', 'Best Threshold']
        ]
        
        for method in ['bicubic', 'dsr']:
            method_label = 'Bicubic' if method == 'bicubic' else 'DSR'
            for res_key in res_keys:
                res_data = results['table1_verification'][res_key][method]
                res_display = res_key.replace('x', '×')
                
                row = [
                    method_label,
                    res_display,
                    f"{res_data['roc_auc']:.4f}",
                    f"{res_data['eer']:.2f}",
                    f"{res_data['threshold']:.3f}"
                ]
                table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')
        
        ax.set_title('Table 3: Detailed Verification Metrics\n(ROC-AUC, Equal Error Rate, Threshold)',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============================================================
        # TABLE 4: RANK-K IDENTIFICATION RATES
        # ============================================================
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')
        
        # Prepare table data
        res_keys = sorted(results['table2_identification'].keys(), 
                         key=lambda x: int(x.split('x')[0]))
        
        table_data = [
            ['Method', 'Resolution', 'Rank-1 (%)', 'Rank-5 (%)', 'Rank-10 (%)']
        ]
        
        for method in ['bicubic', 'dsr']:
            method_label = 'Bicubic' if method == 'bicubic' else 'DSR'
            for res_key in res_keys:
                res_data = results['table2_identification'][res_key][method]
                res_display = res_key.replace('x', '×')
                
                row = [
                    method_label,
                    res_display,
                    f"{res_data['rank1']:.2f}",
                    f"{res_data['rank5']:.2f}",
                    f"{res_data['rank10']:.2f}"
                ]
                table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')
        
        ax.set_title('Table 4: Detailed Identification Metrics\n(Rank-1, Rank-5, Rank-10 Accuracy)',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"\n✓ PDF tables saved to: {output_pdf_path}")


def main():
    """Main function"""
    project_root = Path(__file__).resolve().parents[2]
    
    results_json = project_root / "technical" / "evaluation_results" / "lfw_evaluation_results.json"
    output_pdf = project_root / "technical" / "evaluation_results" / "lfw_evaluation_tables.pdf"
    
    if not results_json.exists():
        print(f"ERROR: Results file not found: {results_json}")
        return
    
    print(f"Loading results from: {results_json}")
    generate_tables(results_json, output_pdf)


if __name__ == "__main__":
    main()
