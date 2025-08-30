
# M.Tech Project Defense Script
## Speaking Notes for Presentation

---

## Slide 1: Title Slide
**Speaking Time:** 30 seconds

"Good morning everyone. I'm Sandeep Joshi, and today I'll be presenting my M.Tech project on 'A Privacy-Preserving AI Copilot for Personalized Financial Document Querying using RAG and Local Language Models.' This project was completed under the supervision of Mrs. Shweta Bhargava."

---

## Slide 2: Agenda
**Speaking Time:** 30 seconds

"Let me outline what I'll cover today. We'll start with the project overview and motivation, then discuss the problem statement, literature review, system architecture, implementation results, commercial viability, and finally conclusions and future work."

---

## Slide 3: Project Overview & Motivation
**Speaking Time:** 2 minutes

"Our project aims to transform personal financial management through AI. The vision is to enable natural language queries on bank statements while ensuring privacy through local deployment.

The market opportunity is significant - the global robo-advisory market is expected to reach $1.2 trillion by 2027, with the Indian market showing 40% CAGR. Digital banking adoption has grown 80% post-COVID, creating a perfect storm for AI-powered financial tools.

Our key innovation is a hybrid AI system that combines a trained model with rule-based fallback, ensuring both accuracy and reliability. We deploy everything locally to preserve privacy."

---

## Slide 4: Problem Statement & Objectives
**Speaking Time:** 2 minutes

"Current challenges in personal finance include manual analysis being time-consuming, privacy concerns with cloud-based solutions, limited personalization, and difficulty in extracting specific insights from complex financial data.

Our solution objectives are fourfold: First, automated query processing that converts natural language to financial insights. Second, a privacy-first design with local deployment. Third, personalized recommendations based on user behavior. And fourth, a scalable architecture supporting multiple users.

Our research questions focus on how to fine-tune LLMs for financial analysis, what hybrid architecture ensures reliability, how to maintain privacy while providing insights, and the commercial viability of such systems."

---

## Slide 5: Literature Review & Technical Background
**Speaking Time:** 2 minutes

"Our literature review covered three key areas. In financial AI, we built upon Markowitz's portfolio theory, FinBERT for sentiment analysis, and recent work on deep reinforcement learning for portfolio management.

For parameter-efficient fine-tuning, we leveraged LoRA from 2022, AdaLoRA from 2023, and few-shot PEFT techniques. This allowed us to train efficiently with limited computational resources.

For synthetic data generation, we used the Faker library with behavioral modeling through Markov chains and economic event simulation to create realistic financial scenarios.

Our technical stack includes DistilGPT2 as the base model for CPU compatibility, LoRA with PEFT for fine-tuning, ChromaDB for vector search, and Streamlit for the dashboard interface."

---

## Slide 6: System Architecture & Methodology
**Speaking Time:** 3 minutes

"Our system architecture follows a multi-agent approach. The query router agent receives user queries and directs them to appropriate specialized agents. The data analysis agent processes financial data, while the risk assessment agent evaluates transaction patterns. The recommendation agent provides personalized insights.

Our data pipeline has three phases. Phase 1 generates 1,000 users with 1 million transactions, including realistic behavioral patterns and economic events. Phase 2 prepares training data by converting transactions into Q&A pairs. Phase 3 involves LoRA fine-tuning with 8 epochs, achieving an 83% improvement in loss reduction."

---

## Slide 7: Implementation & Results
**Speaking Time:** 3 minutes

"Let me highlight our key achievements. We successfully generated 1,000 users with over 1 million transactions, meeting our scale targets. Training completed in 2 hours 28 minutes, well under our 3-hour target. The model loss improved from 4.007 to 0.692, an 83% improvement.

Our hybrid system achieved 100% reliability with an average response time of 2.84 seconds. The system maintains 94% accuracy on test queries with less than 5% fallback to rule-based responses.

Here are some sample results. When asked about highest spending categories, the system provides detailed analysis with percentages and transaction counts. For unusual transactions, it identifies specific amounts and dates with context."

---

## Slide 8: Commercial Viability & Impact
**Speaking Time:** 3 minutes

"Our revenue model includes both B2C subscriptions and B2B licensing. B2C plans range from ₹299 to ₹999 per month, while B2B licensing targets banks at ₹50,000 per month and enterprises at ₹100,000 per month.

Financial projections show strong growth potential. Year 1 targets 10,000 users with ₹5.2 million revenue, growing to 500,000 users and ₹280 million by year 5.

The market impact is significant - we achieve 70% reduction in manual analysis time, 40% better accuracy than traditional methods, 100% local data processing for privacy, and democratized access to financial insights."

---

## Slide 9: Conclusions & Future Work
**Speaking Time:** 2 minutes

"In conclusion, we've achieved both technical and commercial milestones. Technically, we've demonstrated massive scale, efficient training, hybrid reliability, and privacy-first architecture. Commercially, we've validated market potential, developed scalable revenue streams, and established competitive advantages.

Future work includes multi-modal support for PDFs and images, real-time banking integration, advanced predictive analytics, mobile applications, international expansion, and regulatory compliance frameworks.

Our research contributions include hybrid AI architecture for financial applications, privacy-preserving LLM deployment methodology, synthetic data generation techniques, and commercial viability analysis."

---

## Slide 10: Q&A Session
**Speaking Time:** Variable

"Thank you for your attention. I'm now ready to answer your questions about the technical implementation, business model, or any other aspects of the project."

---

## Key Points to Remember:

1. **Emphasize the hybrid approach** - it's unique and addresses reliability concerns
2. **Highlight privacy** - this is a key differentiator
3. **Show scale** - 1M+ transactions is impressive
4. **Demonstrate commercial viability** - ₹5.2M first-year revenue
5. **Be prepared for technical questions** about LoRA, RAG, and model training
6. **Have backup slides ready** for detailed technical explanations

## Tips for Delivery:

- **Speak clearly and at moderate pace**
- **Make eye contact with committee members**
- **Use hand gestures to emphasize key points**
- **Pause briefly between sections**
- **Have water ready**
- **Practice the timing** - aim for 15-20 minutes total
- **Be confident** - you've done excellent work!

---

*Good luck with your defense! 🚀*
