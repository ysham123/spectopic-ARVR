import argparse
import json
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="Homework 3 — MNIST CNN (PyTorch)")
    parser.add_argument("--outfile", type=str, default="Homework3_Report.pdf")
    args = parser.parse_args()

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(args.outfile, pagesize=A4)

    flow = []
    flow.append(Paragraph(args.title, styles["Title"]))
    flow.append(Spacer(1, 12))
    flow.append(Paragraph("1) Network Architecture", styles["Heading2"]))
    try:
        with open("architecture.txt","r") as f:
            arch_txt = f.read().replace("\n","<br/>")
        flow.append(Paragraph(arch_txt, styles["Code"]))
    except:
        flow.append(Paragraph("architecture.txt not found. Run the training script first.", styles["Normal"]))

    flow.append(Spacer(1, 12))
    flow.append(Paragraph("2) Code", styles["Heading2"]))
    flow.append(Paragraph("See the attached file: mnist_cnn_pytorch.py", styles["Normal"]))

    flow.append(Spacer(1, 12))
    flow.append(Paragraph("3) Training & Validation Curves", styles["Heading2"]))
    try:
        flow.append(Image("curves_loss.png", width=400, height=300))
        flow.append(Image("curves_acc.png", width=400, height=300))
    except:
        flow.append(Paragraph("Curves not found. After training, curves_loss.png and curves_acc.png will appear.", styles["Normal"]))

    flow.append(Spacer(1, 12))
    flow.append(Paragraph("4) Test Results", styles["Heading2"]))
    try:
        with open("metrics.json","r") as f:
            m = json.load(f)
        data = [["Metric","Value"],
                ["Best Val Loss", f"{m.get('best_val_loss', 0.0):.4f}"],
                ["Test Loss", f"{m.get('test_loss', 0.0):.4f}"],
                ["Test Accuracy", f"{m.get('test_acc', 0.0):.4f}"],
                ["Trainable Params", f"{m.get('params',0):,}"]]
        table = Table(data, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.lightgrey),
            ("BOX", (0,0), (-1,-1), 0.5, colors.black),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("ALIGN",(1,1),(-1,-1),"RIGHT")
        ]))
        flow.append(table)
    except Exception as e:
        flow.append(Paragraph("metrics.json not found. Run the training script to generate metrics.", styles["Normal"]))

    flow.append(Spacer(1, 12))
    flow.append(Paragraph("Sample Predictions", styles["Heading3"]))
    try:
        flow.append(Image("samples.png", width=300, height=300))
    except:
        flow.append(Paragraph("samples.png not found.", styles["Normal"]))

    flow.append(Spacer(1, 12))
    flow.append(Paragraph("Confusion Matrix", styles["Heading3"]))
    try:
        flow.append(Image("confusion_matrix.png", width=350, height=350))
    except:
        flow.append(Paragraph("confusion_matrix.png not found.", styles["Normal"]))

    doc.build(flow)
    print(f"Wrote {args.outfile}")

if __name__ == "__main__":
    main()
