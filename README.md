# food-delivery-kpt-predictor
Synthetic data generation and visual analytics for de-noising restaurant prep time signals and improving delivery ETAs.

# Optimizing Kitchen Prep Time (KPT) Prediction via Signal Enrichment

## 📌 Project Overview
This repository contains the simulation code, synthetic data generation scripts, and visual analytics for our hackathon submission addressing Kitchen Prep Time (KPT) optimization. 

Instead of a purely algorithmic approach, our solution focuses on improving signal quality and denoising. We propose a tiered architecture (POS webhooks for large cloud kitchens and an AI-driven ambient audio proxy for small kiosks) to capture the true "Live Kitchen Rush"—including in-store dining and multi-app delivery traffic—which is currently a blind spot in baseline prediction models.

## 🗂️ Repository Structure
* `/data`: Contains the synthesized operational dataset (`kpt_simulation_data.csv`) featuring baseline "flawed" manual signals vs. our proposed "de-noised" AI signals.
* `/scripts`: Pure Python scripts (`.py`) developed for data engineering logic, simulation, and outputting the analytical visualizations.
* `/visuals`: Exported high-resolution `.png` charts demonstrating business impact.

## ⚙️ Setup & Execution
This project is built using standard Python scripts. To reproduce our synthetic dataset and visualizations locally, clone this repository and install the dependencies:

```bash
pip install -r requirements.txt
