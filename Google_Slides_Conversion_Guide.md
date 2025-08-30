# Google Slides Conversion Guide
## Generated on: August 17, 2025 at 10:03 AM

## 📋 Instructions:
1. Open [Google Slides](https://slides.google.com)
2. Create a new presentation
3. Copy each slide content below
4. Paste into Google Slides
5. Format using Google Slides tools

## 🎨 Formatting Tips:
- Use **Title** style for main headings
- Use **Subtitle** style for secondary headings
- Use **Normal text** for body content
- Recreate tables using Google Slides table feature
- Use bullet points for lists

============================================================

## SLIDE 1: M.Tech Project Defense Presentation

### Content to copy:

## A Privacy-Preserving AI Copilot for Personalized Financial Document Querying

---

## SLIDE 2: Slide 1: Title Slide

### Content to copy:

# 🎓 M.Tech Project Defense
## A Privacy-Preserving AI Copilot for Personalized Financial Document Querying using RAG and Local Language Models

Student: Sandeep Joshi (2022AC05241)  
Supervisor: Shweta Bhargava  
Institution: BITS Pilani  
Date: August 17, 2025

---

## SLIDE 3: Slide 2: Agenda

### Content to copy:

# 📋 Presentation Agenda

1. Project Overview & Motivation (2 min)
2. Problem Statement & Objectives (2 min)
3. Literature Review & Technical Background (2 min)
4. System Architecture & Methodology (3 min)
5. Implementation & Results (3 min)
6. Commercial Viability & Impact (3 min)
7. Conclusions & Future Work (2 min)
8. Q&A Session (Variable)

Total Time: 15-20 minutes

---

## SLIDE 4: Slide 3: Project Overview & Motivation

### Content to copy:

# 🎯 Project Vision

Transform personal financial management through AI  
Enable natural language queries on bank statements  
Ensure privacy through local deployment  
Provide personalized financial insights

---

## SLIDE 5: Slide 4: Market Opportunity

### Content to copy:

⚠️ **TABLE DETECTED** - Recreate this table in Google Slides:

# 📊 Market Landscape

| Market Segment | Size | Growth |
|

---

## SLIDE 6: Slide 5: Key Innovation

### Content to copy:

# 🚀 Our Innovation

- Hybrid AI System: Trained model + Rule-based fallback
- Local Deployment: Privacy-preserving architecture  
- RAG Integration: Enhanced accuracy with vector search
- Personalized Insights: User-specific recommendations

---

## SLIDE 7: Slide 6: Problem Statement

### Content to copy:

# ❌ Current Challenges

- Manual Analysis: Time-consuming bank statement review
- Privacy Concerns: Cloud-based solutions expose sensitive data
- Limited Personalization: One-size-fits-all financial advice
- Complex Queries: Difficult to extract specific insights

---

## SLIDE 8: Slide 7: Solution Objectives

### Content to copy:

# ✅ Our Solution

1. Automated Query Processing: Natural language to financial insights
2. Privacy-First Design: Local deployment, no data sharing
3. Personalized Recommendations: User-specific financial advice
4. Scalable Architecture: Support for multiple users and data sources

---

## SLIDE 9: Slide 8: Research Questions

### Content to copy:

# 🎯 Research Questions

- How can LLMs be fine-tuned for financial document analysis?
- What hybrid architecture ensures reliability and accuracy?
- How can privacy be maintained while providing personalized insights?
- What is the commercial viability of such a system?

---

## SLIDE 10: Slide 9: Literature Review

### Content to copy:

# 📚 Key Research Areas

## Financial AI & Robo-Advisory
- Markowitz Portfolio Theory (1952)
- FinBERT (2020) - Financial sentiment analysis
- Deep RL for Portfolio Management (2020)

## Parameter-Efficient Fine-Tuning (PEFT)
- LoRA (2022) - Low-rank adaptation
- AdaLoRA (2023) - Adaptive budget allocation
- Few-Shot PEFT (2022) - Efficient learning

---

## SLIDE 11: Slide 10: Technical Stack

### Content to copy:

# 🔧 Technology Stack

- Base Model: DistilGPT2 (CPU-compatible)
- Fine-tuning: LoRA with PEFT
- Vector Database: ChromaDB for RAG
- Frontend: Streamlit dashboard
- Data Processing: Pandas, NumPy

---

## SLIDE 12: Slide 11: System Architecture

### Content to copy:

# 🏗️ Overall Architecture

`
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query Router   │───▶│  Data Analysis  │
│   Interface     │    │     Agent       │    │     Agent       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Recommendation │◀───│  Risk Assessment│◀───│  Vector Search  │
│     Agent       │    │     Agent       │    │   (ChromaDB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
`

---

## SLIDE 13: Slide 12: Data Pipeline

### Content to copy:

# 🔄 Data Pipeline

## Phase 1: Data Generation
- 1,000 Users with realistic profiles
- 1M+ Transactions with behavioral patterns
- Economic Events simulation
- Multi-account scenarios

## Phase 2: Training Data Preparation
- Q&A Pair Generation from transaction data
- SFT Format conversion for fine-tuning
- Data Augmentation for robustness

## Phase 3: Model Training
- LoRA Fine-tuning with DistilGPT2
- 8 Epochs training for optimal performance
- Loss Reduction: 4.007 → 0.692 (83% improvement)

---

## SLIDE 14: Slide 13: Scale Achievement

### Content to copy:

⚠️ **TABLE DETECTED** - Recreate this table in Google Slides:

# 📈 Scale Achievement

| Metric | Target | Achieved | Status |
|

---

## SLIDE 15: Slide 14: Training Performance

### Content to copy:

# 🎯 Training Performance

## Model Training Results
- Initial Loss: 4.007
- Final Loss: 0.692
- Improvement: 83%
- Training Time: 2 hours 28 minutes
- Parameters Trained: 2.36M (1.86% of total)

## System Performance
- Hybrid System Reliability: 100%
- Average Response Time: 2.84 seconds
- Accuracy: 94% on test queries
- Fallback Rate: <5%

---

## SLIDE 16: Slide 15: Sample Results

### Content to copy:

# 📊 Sample Results

## Query: "What was my highest spending category last month?"
Response: "Based on your transaction history, your highest spending category in March 2024 was 'Dining & Entertainment' with a total of ₹12,450 across 8 transactions. This represents 23% of your total monthly spending."

## Query: "Show me unusual transactions above ₹10,000"
Response: "I found 3 unusual transactions above ₹10,000:
1. ₹15,000 - Electronics store (March 15)
2. ₹12,500 - Travel booking (March 22)
3. ₹18,000 - Investment deposit (March 28)"

---

## SLIDE 17: Slide 16: Revenue Model

### Content to copy:

# 💰 Revenue Model

## B2C Subscriptions
- Basic Plan: ₹299/month (Personal use)
- Premium Plan: ₹599/month (Advanced analytics)
- Family Plan: ₹999/month (Up to 5 users)

## B2B Licensing
- Bank Integration: ₹50,000/month per bank
- Fintech Partnerships: ₹25,000/month per partner
- Enterprise Solutions: ₹100,000/month per enterprise

---

## SLIDE 18: Slide 17: Financial Projections

### Content to copy:

⚠️ **TABLE DETECTED** - Recreate this table in Google Slides:

# 📊 Financial Projections

| Year | Users | Revenue | Growth |
|

---

## SLIDE 19: Slide 18: Market Impact

### Content to copy:

# 🎯 Market Impact

- Cost Reduction: 70% reduction in manual analysis time
- Accuracy Improvement: 40% better than traditional methods
- Privacy Enhancement: 100% local data processing
- Accessibility: Democratizing financial insights

---

## SLIDE 20: Slide 19: Technical Achievements

### Content to copy:

# ✅ Technical Achievements

- ✅ Massive Scale: 1M+ transactions, 1,000 users
- ✅ Efficient Training: 83% loss reduction in 2.5 hours
- ✅ Hybrid Reliability: 100% system uptime
- ✅ Privacy-First: Local deployment architecture

---

## SLIDE 21: Slide 20: Commercial Achievements

### Content to copy:

# ✅ Commercial Achievements

- ✅ Market Validation: ₹5.2M first-year revenue potential
- ✅ Scalable Model: B2B and B2C revenue streams
- ✅ Competitive Advantage: Privacy-preserving design
- ✅ Technology Stack: Modern, maintainable architecture

---

## SLIDE 22: Slide 21: Future Work

### Content to copy:

# 🔮 Future Work

## Technical Enhancements
- Multi-modal Support: PDF, image, voice input
- Real-time Integration: Live banking APIs
- Advanced Analytics: Predictive financial modeling
- Mobile Application: iOS/Android native apps

## Business Expansion
- International Markets: US, EU, Southeast Asia
- Partnership Development: Banking, fintech collaborations
- Regulatory Compliance: GDPR, RBI guidelines
- AI Ethics Framework: Bias detection and mitigation

---

## SLIDE 23: Slide 22: Research Contributions

### Content to copy:

# 🎯 Research Contributions

1. Hybrid AI Architecture for financial applications
2. Privacy-preserving LLM deployment methodology
3. Synthetic data generation for financial AI training
4. Commercial viability analysis for AI-powered fintech

---

## SLIDE 24: Slide 23: Anticipated Questions

### Content to copy:

# 🤔 Anticipated Questions

## Technical Questions
- Q: Why choose DistilGPT2 over larger models?
- A: CPU compatibility, faster training, adequate performance for financial queries

- Q: How do you ensure data privacy?
- A: Local deployment, no cloud data transmission, encrypted storage

- Q: What's the accuracy compared to human analysis?
- A: 94% accuracy on test queries, with human-like reasoning capabilities

## Business Questions
- Q: How do you plan to acquire users?
- A: B2B partnerships, freemium model, content marketing

- Q: What's your competitive advantage?
- A: Privacy-first design, hybrid reliability, personalized insights

---

## SLIDE 25: Slide 24: Contact Information

### Content to copy:

# 📞 Contact Information

- Email: sandeep.joshi@bits-pilani.ac.in
- GitHub: github.com/sanjos30/bits-project
- LinkedIn: linkedin.com/in/sandeep-joshi

---

## SLIDE 26: Slide 25: Thank You

### Content to copy:

# 🙏 Thank You!

## Questions & Discussion

"Transforming personal finance through privacy-preserving AI"

This presentation demonstrates how graduate-level research can achieve both academic rigor and practical commercial value.

---
