# 📚 Archive - Original Notebooks

This folder contains the original Jupyter notebooks that were used in the initial prototype development.

## Original Notebooks

### 01_generate_data.ipynb
- **Purpose**: Initial synthetic data generation
- **Scale**: 3,900 transactions for single user
- **Status**: Replaced by `final_demo_production_scale.py` (1M+ transactions)

### 02_Prepare_SFT_Data.ipynb  
- **Purpose**: Convert CSV data to SFT (Supervised Fine-Tuning) format
- **Status**: Functionality integrated into training scripts

### 03_LoRA_Train_TinyLlama_FinanceCopilot.ipynb
- **Purpose**: LoRA fine-tuning with TinyLlama model
- **Model**: TinyLlama-1.1B-Chat-v1.0
- **Status**: Replaced by `improved_model_trainer.py` (GPT2 + LoRA)

### 04_Chat_With_LoRA_FinanceBot.ipynb
- **Purpose**: Basic chat interface with LoRA model
- **Status**: Replaced by `hybrid_financial_ai.py` (hybrid system)

## Evolution Summary

**Original → Enhanced:**
- Data: 3,900 → 1,000,000+ transactions (256x improvement)
- Model: TinyLlama → GPT2 (better CPU compatibility)
- Architecture: Basic chat → Hybrid AI system (100% reliability)
- Deployment: Notebook → Production Python scripts

These notebooks represent the foundation that led to the current production-ready system.