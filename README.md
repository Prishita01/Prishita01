# Prishita Ghanathe Krishna

MS Computer Science (AI/ML), Case Western Reserve University. I build ML systems end-to-end — inference pipelines, orchestration, monitoring, retraining. Not just models.

---

## Projects

**[perception-av](https://github.com/Prishita01/perception-av)** — Production AV perception pipeline on nuScenes. YOLO26 detection (2,221 detections, 104ms/frame), Depth Anything V2 depth maps in S3, SAM2 pixel masks in PostgreSQL. Airflow 2.9 across 7 Docker containers, Evidently AI drift monitoring, hard case mining, MLflow tracking, YOLO26n fine-tuning (+251% mAP50 on held-out val), dbt + Grafana analytics layer.

**[arxiv-rag](https://github.com/Prishita01/arxiv-rag)** — Production RAG over arXiv papers. Hybrid retrieval: BM25 sparse + sentence-transformer dense, fused with RRF. Cross-encoder reranking, Llama 3.2 via Ollama, FastAPI, Docker Compose.

**[ReconStructAI3D](https://github.com/Prishita01/ReconStructAI3D)** — Monocular 3D reconstruction from smartphone photos. SAM segmentation, COLMAP SfM, OpenMVS dense reconstruction, Open3D refinement. PSNR 24.3 dB, SSIM 0.985, no depth sensors.

**[Triton-ResNet50](https://github.com/Prishita01/Triton-ResNet50)** — Three custom Triton GPU convolution kernels benchmarked across 50+ models on NVIDIA H100 NVL and L40S. ScalarMM outperforms cuDNN ~29% on ResNet-family layers. Validated at ~1e-5 numerical correctness end-to-end.

**[FairCredit](https://github.com/Prishita01/FairCredit)** — Responsible AI framework for credit risk. 68.7% bias reduction at 2.3% accuracy cost across three datasets. SHAP explainability, post-processing threshold optimization.

**[StockNet HPC](https://github.com/Prishita01/Stock-Prediction-GPU)** — Distributed LSTM training with PyTorch DDP + SLURM. 72.6x CPU-to-GPU speedup (5515s to 76s) on CWRU Markov HPC cluster.

---

## Stack

Python, PyTorch, Apache Airflow, Docker, AWS S3/RDS, YOLO26, SAM2, Depth Anything V2, MLflow, Evidently AI, dbt, Grafana, FastAPI, PostgreSQL, OpenAI Triton, CUDA, COLMAP, OpenMVS, SLURM, Hugging Face

---

## Research

Prishita G K. *Eye Tracking: Gaze Detection in Market Research.* IJIRCCE Vol. 11(5), 2023.  
Prishita G K. *AutoML: Automated Machine Learning using Streamlit.* IJIRCCE Vol. 11(5), 2023.

---

prishita.ghanathe@gmail.com · [linkedin.com/in/prishitagk](https://linkedin.com/in/prishitagk)  
Research Assistant @ CWRU KLab-AI3 · open to ML Engineer / CV Engineer / MLOps roles
