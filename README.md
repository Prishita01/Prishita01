# Prishita Ghanathe Krishna

MS Computer Science (AI/ML), Case Western Reserve University. I build ML systems that run end-to-end in production — not just models, but the pipelines, infra, and monitoring that keep them working.

---

## What I've built

**perception-av** — AV perception pipeline on nuScenes: YOLO26 detection (2,221 detections), Depth Anything V2 depth estimation, SAM2 segmentation (1,889 RLE masks), Evidently AI drift monitoring, hard case mining, MLflow experiment tracking, and auto-retraining with a BranchPythonOperator DAG. Fine-tuned YOLO26n on 323 frames, mAP50 improved +251% on a held-out val set. Orchestrated with Airflow across 7 Docker containers, analytics via dbt + Grafana.

**arxiv-rag** — Production RAG over arXiv papers. Hybrid retrieval: BM25 sparse + sentence-transformer dense, fused with RRF. Cross-encoder reranking, Llama 3.2 via Ollama, FastAPI backend, Docker Compose.

**ReconStructAI3D** — Monocular 3D reconstruction from smartphone photos. SAM segmentation → COLMAP SfM → OpenMVS dense reconstruction → Open3D mesh refinement. PSNR 24.3 dB, SSIM 0.985, no depth sensors.

**Triton-ResNet50** — Three custom Triton GPU convolution kernels (ScalarMM, KN2Row, Im2Col) benchmarked layer-wise across 50+ torchvision models on NVIDIA H100 NVL and L40S. ScalarMM outperforms cuDNN ~29% on ResNet family. Validated at ~1e-5 numerical correctness.

**FairCredit** — Responsible AI framework for credit risk. 68.7% bias reduction at 2.3% accuracy cost across German Credit, Portuguese Bank Marketing, and Synthetic US Credit datasets. SHAP explainability, post-processing threshold optimization.

**StockNet HPC** — Distributed LSTM training with PyTorch DDP + SLURM across 2 GPUs on CWRU's Markov HPC cluster. 72.6x CPU-to-GPU speedup (5515s → 76s).

---

## Stack

Python, PyTorch, Triton, CUDA, Apache Airflow, Docker, AWS (S3, RDS), YOLO, SAM2, Depth Anything V2, MLflow, Evidently AI, dbt, Grafana, FastAPI, PostgreSQL, COLMAP, OpenMVS, Hugging Face, SLURM

---

## Research

Prishita G K. *Eye Tracking: Gaze Detection in Market Research.* IJIRCCE Vol. 11(5), 2023.  
Prishita G K. *AutoML: Automated Machine Learning using Streamlit.* IJIRCCE Vol. 11(5), 2023.

---

prishita.ghanathe@gmail.com · [linkedin.com/in/prishitagk](https://linkedin.com/in/prishitagk)  
Research Assistant @ CWRU KLab-AI3 · actively interviewing for ML Engineer / CV Engineer roles
