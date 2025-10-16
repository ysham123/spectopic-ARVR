
# Homework 3 — MNIST CNN (PyTorch)

# Sections required by assignment
1. Network architecture (see `architecture.txt` once you run training)  
2. Code (`mnist_cnn_pytorch.py`)  
3. Training/testing information (see `log.csv`, curves images)  
4. Test results (see `metrics.json`, confusion matrix, and samples)

## How to reproduce
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python mnist_cnn_pytorch.py --epochs 8 --batch-size 128
python report_builder.py --title "Homework 3"
```

This will generate:
- `model.pt`, `metrics.json`, `log.csv`
- `curves_loss.png`, `curves_acc.png`, `confusion_matrix.png`, `samples.png`
- `Homework3_Report.pdf`
