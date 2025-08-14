<p><img src="./assets/interspeech.jpeg" width="800"></p>

<h1 align="center">
    <a href="https://arxiv.org/abs/2505.14723" style="color:#825987">
        QUADS: QUAntized Distillation Framework for Efficient Speech Language Understanding
    </a>
</h1>

<img src="./assets/QUADS_main_diagram.png" width="800" />

## 🚀 Main Results
**Comparison of QUADS and Prior Methods on the SLURP and FSC Datasets.** We report accuracy and F1-score for model performance, alongside GMACs and model size, to evaluate efficiency.
<img src="./assets/main_result_table.png" width="800" />

## 🛠️ Requirements and Installation
```bash
git clone https://github.com/BASHLab/QUADS.git
cd QUADS
pythom3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🗝️ Training & Evaluation
```bash
python train.py
```

## Citation
```
@article{biswas2025quads,
@inproceedings{biswas25b_interspeech,
  title     = {{QUADS: Quantized Distillation Framework for Efficient Speech Language Understanding}},
  author    = {{Subrata Biswas and Mohammad Nur Hossain Khan and Bashima Islam}},
  year      = {{2025}},
  booktitle = {{Interspeech 2025}},
  pages     = {{4098--4102}},
  doi       = {{10.21437/Interspeech.2025-532}},
}
```
