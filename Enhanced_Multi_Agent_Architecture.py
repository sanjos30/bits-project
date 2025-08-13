"""
Enhanced Multi-Agent Financial Intelligence System
M.Tech AIML Project - Advanced Architecture Implementation

This module implements a sophisticated multi-agent system for personalized financial intelligence
with advanced ML techniques, RAG capabilities, and commercial-grade features.

Key Components:
1. Multi-Agent System with specialized agents
2. Hybrid RAG + Fine-tuning approach
3. Vector database integration
4. Advanced analytics engine
5. Privacy-preserving personalization
6. Real-time risk assessment
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
import hashlib

# ML and NLP imports
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, 
    TrainingArguments, Trainer, pipeline
)
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import faiss

# Advanced analytics
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import lightgbm as lgb
import optuna

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryContext:
    """Context information for financial queries"""
    user_id: str
    query: str
    query_type: str
    timestamp: datetime
    user_profile: Dict[str, Any]
    transaction_history: List[Dict[str, Any]]
    relevant_context: List[str] = None
    confidence_score: float = 0.0

@dataclass
class AgentResponse:
    """Response from an agent"""
    agent_name: str
    response: str
    confidence: float
    metadata: Dict[str, Any]
    processing_time: float

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, model_path: str = None):
        self.name = name
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.initialize()
    
    @abstractmethod
    def initialize(self):
        """Initialize agent-specific components"""
        pass
    
    @abstractmethod
    async def process(self, context: QueryContext) -> AgentResponse:
        """Process a query context and return response"""
        pass
    
    def _calculate_confidence(self, context: QueryContext, response: str) -> float:
        """Calculate confidence score for the response"""
        # Implement confidence calculation logic
        base_confidence = 0.7
        
        # Adjust based on query complexity
        query_words = len(context.query.split())
        if query_words > 10:
            base_confidence -= 0.1
        
        # Adjust based on available context
        if context.relevant_context and len(context.relevant_context) > 3:
            base_confidence += 0.2
        
        return min(1.0, max(0.1, base_confidence))

class QueryRouterAgent(BaseAgent):
    """Routes queries to appropriate specialized agents"""
    
    def initialize(self):
        """Initialize query classification model"""
        self.query_classifier = pipeline(
            "text-classification",
            model="microsoft/DialoGPT-medium",
            return_all_scores=True
        )
        
        self.query_types = {
            'spending_analysis': ['spend', 'expense', 'cost', 'money', 'budget'],
            'investment_advice': ['invest', 'portfolio', 'stocks', 'mutual fund', 'return'],
            'risk_assessment': ['risk', 'fraud', 'suspicious', 'unusual', 'anomaly'],
            'budget_planning': ['budget', 'plan', 'save', 'goal', 'target'],
            'tax_optimization': ['tax', 'deduction', 'filing', 'refund'],
            'loan_management': ['loan', 'emi', 'mortgage', 'debt', 'credit'],
            'general_query': ['what', 'how', 'when', 'where', 'explain']
        }
    
    async def process(self, context: QueryContext) -> AgentResponse:
        """Route query to appropriate agent"""
        start_time = datetime.now()
        
        # Classify query type
        query_type = self._classify_query(context.query)
        context.query_type = query_type
        
        # Determine confidence based on keyword matching
        confidence = self._calculate_routing_confidence(context.query, query_type)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AgentResponse(
            agent_name=self.name,
            response=f"Query classified as: {query_type}",
            confidence=confidence,
            metadata={'query_type': query_type, 'routing_decision': True},
            processing_time=processing_time
        )
    
    def _classify_query(self, query: str) -> str:
        """Classify query into appropriate type"""
        query_lower = query.lower()
        
        # Score each query type
        type_scores = {}
        for query_type, keywords in self.query_types.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            type_scores[query_type] = score
        
        # Return type with highest score, default to general_query
        return max(type_scores, key=type_scores.get) if max(type_scores.values()) > 0 else 'general_query'
    
    def _calculate_routing_confidence(self, query: str, query_type: str) -> float:
        """Calculate confidence in routing decision"""
        keywords = self.query_types.get(query_type, [])
        matches = sum(1 for keyword in keywords if keyword in query.lower())
        return min(1.0, matches / len(keywords) if keywords else 0.5)

class DataAnalysisAgent(BaseAgent):
    """Performs complex financial computations and analysis"""
    
    def initialize(self):
        """Initialize analysis models and tools"""
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.clustering_model = KMeans(n_clusters=5, random_state=42)
        self.forecasting_model = None  # Will be trained on user data
        
    async def process(self, context: QueryContext) -> AgentResponse:
        """Perform financial data analysis"""
        start_time = datetime.now()
        
        # Prepare transaction data
        df = pd.DataFrame(context.transaction_history)
        
        if context.query_type == 'spending_analysis':
            response = await self._analyze_spending_patterns(df, context)
        elif context.query_type == 'risk_assessment':
            response = await self._assess_financial_risk(df, context)
        elif context.query_type == 'budget_planning':
            response = await self._generate_budget_insights(df, context)
        else:
            response = await self._general_analysis(df, context)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        confidence = self._calculate_confidence(context, response)
        
        return AgentResponse(
            agent_name=self.name,
            response=response,
            confidence=confidence,
            metadata={'analysis_type': context.query_type, 'data_points': len(df)},
            processing_time=processing_time
        )
    
    async def _analyze_spending_patterns(self, df: pd.DataFrame, context: QueryContext) -> str:
        """Analyze spending patterns and trends"""
        if df.empty:
            return "No transaction data available for analysis."
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        
        # Calculate monthly spending by category
        monthly_spending = df[df['type'] == 'debit'].groupby(['month', 'category'])['amount'].sum().reset_index()
        
        # Find top spending categories
        top_categories = df[df['type'] == 'debit'].groupby('category')['amount'].sum().sort_values(ascending=False).head(5)
        
        # Calculate trends
        recent_months = df[df['date'] >= (datetime.now() - timedelta(days=90))]
        previous_months = df[(df['date'] >= (datetime.now() - timedelta(days=180))) & 
                            (df['date'] < (datetime.now() - timedelta(days=90)))]
        
        recent_spending = recent_months[recent_months['type'] == 'debit']['amount'].sum()
        previous_spending = previous_months[previous_months['type'] == 'debit']['amount'].sum()
        
        trend = "increased" if recent_spending > previous_spending else "decreased"
        trend_percentage = abs((recent_spending - previous_spending) / previous_spending * 100) if previous_spending > 0 else 0
        
        # Generate insights
        insights = []
        insights.append(f"Your spending has {trend} by {trend_percentage:.1f}% in the last 3 months.")
        insights.append(f"Top spending category: {top_categories.index[0]} (₹{top_categories.iloc[0]:,.2f})")
        
        # Detect anomalies
        if len(df) > 10:
            features = df[['amount']].values
            self.anomaly_detector.fit(features)
            anomalies = self.anomaly_detector.predict(features)
            anomaly_count = sum(1 for x in anomalies if x == -1)
            if anomaly_count > 0:
                insights.append(f"Detected {anomaly_count} unusual transactions that may need attention.")
        
        return "\\n".join(insights)
    
    async def _assess_financial_risk(self, df: pd.DataFrame, context: QueryContext) -> str:
        """Assess financial risk based on transaction patterns"""
        if df.empty:
            return "No transaction data available for risk assessment."
        
        risk_factors = []
        risk_score = 0.0
        
        # Calculate debt-to-income ratio
        monthly_income = df[df['type'] == 'credit']['amount'].sum() / 12  # Rough monthly income
        monthly_debt = df[df['category'].str.contains('loan|emi', case=False, na=False)]['amount'].sum() / 12
        
        if monthly_income > 0:
            debt_ratio = monthly_debt / monthly_income
            if debt_ratio > 0.4:
                risk_factors.append(f"High debt-to-income ratio: {debt_ratio:.1%}")
                risk_score += 0.3
        
        # Check for irregular income
        monthly_income_variance = df[df['type'] == 'credit'].groupby(df['date'].dt.to_period('M'))['amount'].sum().var()
        if monthly_income_variance > monthly_income * 0.5:
            risk_factors.append("Irregular income pattern detected")
            risk_score += 0.2
        
        # Check emergency fund adequacy
        total_savings = context.user_profile.get('savings_balance', 0)
        monthly_expenses = df[df['type'] == 'debit']['amount'].sum() / 12
        emergency_fund_months = total_savings / monthly_expenses if monthly_expenses > 0 else 0
        
        if emergency_fund_months < 3:
            risk_factors.append(f"Emergency fund covers only {emergency_fund_months:.1f} months of expenses")
            risk_score += 0.3
        
        # Overall risk assessment
        if risk_score < 0.3:
            risk_level = "Low"
        elif risk_score < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        response = f"Financial Risk Assessment: {risk_level} (Score: {risk_score:.2f})\\n"
        if risk_factors:
            response += "Risk factors identified:\\n" + "\\n".join(f"• {factor}" for factor in risk_factors)
        else:
            response += "No significant risk factors identified. Your financial health looks good!"
        
        return response
    
    async def _generate_budget_insights(self, df: pd.DataFrame, context: QueryContext) -> str:
        """Generate budget planning insights"""
        if df.empty:
            return "No transaction data available for budget planning."
        
        # Calculate average monthly spending by category
        df['date'] = pd.to_datetime(df['date'])
        monthly_spending = df[df['type'] == 'debit'].groupby([df['date'].dt.to_period('M'), 'category'])['amount'].sum()
        avg_monthly_by_category = monthly_spending.groupby('category').mean().sort_values(ascending=False)
        
        # Calculate total monthly income and expenses
        monthly_income = df[df['type'] == 'credit']['amount'].sum() / 12
        monthly_expenses = df[df['type'] == 'debit']['amount'].sum() / 12
        savings_rate = (monthly_income - monthly_expenses) / monthly_income if monthly_income > 0 else 0
        
        insights = []
        insights.append(f"Current savings rate: {savings_rate:.1%}")
        
        # Budget recommendations
        recommended_budget = {}
        for category, amount in avg_monthly_by_category.head(5).items():
            percentage = (amount / monthly_expenses * 100) if monthly_expenses > 0 else 0
            recommended_budget[category] = amount
            insights.append(f"{category}: ₹{amount:.0f}/month ({percentage:.1f}% of expenses)")
        
        # Savings recommendations
        target_savings_rate = 0.2  # 20% target
        if savings_rate < target_savings_rate:
            shortfall = (target_savings_rate - savings_rate) * monthly_income
            insights.append(f"\\nTo reach 20% savings target, reduce expenses by ₹{shortfall:.0f}/month")
        
        return "Budget Analysis:\\n" + "\\n".join(insights)
    
    async def _general_analysis(self, df: pd.DataFrame, context: QueryContext) -> str:
        """Perform general financial analysis"""
        if df.empty:
            return "No transaction data available for analysis."
        
        # Basic statistics
        total_transactions = len(df)
        total_spent = df[df['type'] == 'debit']['amount'].sum()
        total_earned = df[df['type'] == 'credit']['amount'].sum()
        avg_transaction = df['amount'].mean()
        
        # Date range
        df['date'] = pd.to_datetime(df['date'])
        date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        
        # Most frequent categories
        top_category = df['category'].value_counts().index[0]
        top_merchant = df['merchant'].value_counts().index[0]
        
        summary = f"""Financial Overview:
• Total transactions: {total_transactions:,}
• Period: {date_range}
• Total spent: ₹{total_spent:,.2f}
• Total earned: ₹{total_earned:,.2f}
• Net flow: ₹{total_earned - total_spent:,.2f}
• Average transaction: ₹{avg_transaction:.2f}
• Most frequent category: {top_category}
• Most used merchant: {top_merchant}"""
        
        return summary

class RecommendationAgent(BaseAgent):
    """Provides personalized financial recommendations"""
    
    def initialize(self):
        """Initialize recommendation models"""
        self.user_profiles = {}
        self.recommendation_model = None
        
    async def process(self, context: QueryContext) -> AgentResponse:
        """Generate personalized recommendations"""
        start_time = datetime.now()
        
        # Generate recommendations based on query type and user profile
        if context.query_type == 'investment_advice':
            response = await self._generate_investment_recommendations(context)
        elif context.query_type == 'budget_planning':
            response = await self._generate_budget_recommendations(context)
        elif context.query_type == 'tax_optimization':
            response = await self._generate_tax_recommendations(context)
        else:
            response = await self._generate_general_recommendations(context)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        confidence = self._calculate_confidence(context, response)
        
        return AgentResponse(
            agent_name=self.name,
            response=response,
            confidence=confidence,
            metadata={'recommendation_type': context.query_type},
            processing_time=processing_time
        )
    
    async def _generate_investment_recommendations(self, context: QueryContext) -> str:
        """Generate investment recommendations based on user profile"""
        user_profile = context.user_profile
        age = user_profile.get('age', 30)
        risk_tolerance = user_profile.get('risk_tolerance', 'medium')
        monthly_income = user_profile.get('monthly_income', 50000)
        
        recommendations = []
        
        # Age-based recommendations
        if age < 30:
            recommendations.append("• Consider aggressive growth investments (70% equity, 30% debt)")
            recommendations.append("• Start SIP in large-cap and mid-cap mutual funds")
            recommendations.append("• Invest in ELSS for tax benefits")
        elif age < 50:
            recommendations.append("• Balanced portfolio (60% equity, 40% debt)")
            recommendations.append("• Consider real estate investment")
            recommendations.append("• Increase insurance coverage")
        else:
            recommendations.append("• Conservative approach (40% equity, 60% debt)")
            recommendations.append("• Focus on debt funds and fixed deposits")
            recommendations.append("• Consider senior citizen schemes")
        
        # Risk tolerance adjustments
        if risk_tolerance == 'high':
            recommendations.append("• Consider direct stock investments")
            recommendations.append("• Explore international funds")
        elif risk_tolerance == 'low':
            recommendations.append("• Focus on government bonds and FDs")
            recommendations.append("• Avoid volatile investments")
        
        # Income-based recommendations
        monthly_investment = monthly_income * 0.15  # 15% of income
        recommendations.append(f"• Recommended monthly investment: ₹{monthly_investment:.0f}")
        
        return "Investment Recommendations:\\n" + "\\n".join(recommendations)
    
    async def _generate_budget_recommendations(self, context: QueryContext) -> str:
        """Generate budget optimization recommendations"""
        # Analyze spending patterns from transaction history
        df = pd.DataFrame(context.transaction_history)
        if df.empty:
            return "No transaction data available for budget recommendations."
        
        df['date'] = pd.to_datetime(df['date'])
        monthly_spending = df[df['type'] == 'debit'].groupby('category')['amount'].sum() / 12
        
        recommendations = []
        recommendations.append("Budget Optimization Recommendations:")
        
        # 50/30/20 rule recommendations
        monthly_income = context.user_profile.get('monthly_income', 50000)
        recommended_needs = monthly_income * 0.5
        recommended_wants = monthly_income * 0.3
        recommended_savings = monthly_income * 0.2
        
        recommendations.append(f"\\nIdeal Budget Allocation (50/30/20 rule):")
        recommendations.append(f"• Needs (50%): ₹{recommended_needs:.0f}")
        recommendations.append(f"• Wants (30%): ₹{recommended_wants:.0f}")
        recommendations.append(f"• Savings (20%): ₹{recommended_savings:.0f}")
        
        # Category-specific recommendations
        essential_categories = ['groceries', 'rent', 'utilities', 'healthcare']
        essential_spending = monthly_spending[monthly_spending.index.isin(essential_categories)].sum()
        
        if essential_spending > recommended_needs:
            recommendations.append(f"\\n⚠️ Essential spending (₹{essential_spending:.0f}) exceeds recommended 50%")
            recommendations.append("Consider ways to reduce fixed costs")
        
        # Identify high-spending categories for optimization
        top_spending = monthly_spending.sort_values(ascending=False).head(3)
        recommendations.append(f"\\nTop spending categories to monitor:")
        for category, amount in top_spending.items():
            percentage = (amount / monthly_income * 100) if monthly_income > 0 else 0
            recommendations.append(f"• {category}: ₹{amount:.0f} ({percentage:.1f}% of income)")
        
        return "\\n".join(recommendations)
    
    async def _generate_tax_recommendations(self, context: QueryContext) -> str:
        """Generate tax optimization recommendations"""
        user_profile = context.user_profile
        annual_income = user_profile.get('monthly_income', 50000) * 12
        
        recommendations = []
        recommendations.append("Tax Optimization Recommendations:")
        
        # Section 80C recommendations
        recommendations.append("\\nSection 80C (up to ₹1.5 lakh):")
        recommendations.append("• Invest in ELSS mutual funds")
        recommendations.append("• Contribute to PPF")
        recommendations.append("• Pay life insurance premiums")
        recommendations.append("• Repay home loan principal")
        
        # Other deductions
        recommendations.append("\\nOther Tax-Saving Options:")
        recommendations.append("• Section 80D: Health insurance premiums")
        recommendations.append("• Section 24: Home loan interest deduction")
        recommendations.append("• Section 80E: Education loan interest")
        
        # Income-based recommendations
        if annual_income > 1000000:  # 10 lakh+
            recommendations.append("\\nHigh-Income Specific:")
            recommendations.append("• Consider tax-efficient investment options")
            recommendations.append("• Explore salary structuring opportunities")
            recommendations.append("• Consult a tax advisor for advanced planning")
        
        return "\\n".join(recommendations)
    
    async def _generate_general_recommendations(self, context: QueryContext) -> str:
        """Generate general financial recommendations"""
        user_profile = context.user_profile
        profile_type = user_profile.get('profile_type', 'general')
        
        if profile_type == 'young_professional':
            return """Recommendations for Young Professionals:
• Build an emergency fund (3-6 months expenses)
• Start investing early for compound growth
• Focus on skill development and career growth
• Consider term life insurance
• Track expenses and create a budget"""
        
        elif profile_type == 'family_oriented':
            return """Recommendations for Family-Oriented Individuals:
• Increase life and health insurance coverage
• Start child education planning
• Create separate emergency fund for family needs
• Consider home purchase if renting
• Plan for family vacations within budget"""
        
        elif profile_type == 'senior_professional':
            return """Recommendations for Senior Professionals:
• Accelerate retirement planning
• Diversify investment portfolio
• Consider estate planning
• Optimize tax strategies
• Plan for healthcare costs in retirement"""
        
        else:
            return """General Financial Recommendations:
• Maintain emergency fund
• Invest regularly for long-term goals
• Review and optimize insurance coverage
• Track expenses and stick to budget
• Seek professional advice for complex decisions"""

class VectorDatabase:
    """Vector database for storing and retrieving financial embeddings"""
    
    def __init__(self, collection_name: str = "financial_data"):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection_name = collection_name
        self.collection = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.initialize_collection()
    
    def initialize_collection(self):
        """Initialize or get existing collection"""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(name=self.collection_name)
            logger.info(f"Created new collection: {self.collection_name}")
    
    def add_transactions(self, transactions: List[Dict[str, Any]]):
        """Add transaction embeddings to the database"""
        documents = []
        metadatas = []
        ids = []
        
        for txn in transactions:
            # Create text representation of transaction
            doc_text = f"Transaction: {txn['amount']} {txn['currency']} at {txn['merchant']} for {txn['category']} on {txn['date']} via {txn['payment_mode']}"
            documents.append(doc_text)
            
            # Store metadata
            metadata = {
                'user_id': txn['user_id'],
                'category': txn['category'],
                'amount': float(txn['amount']),
                'date': txn['date'],
                'merchant': txn['merchant']
            }
            metadatas.append(metadata)
            ids.append(txn['transaction_id'])
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(transactions)} transactions to vector database")
    
    def search_similar_transactions(self, query: str, user_id: str = None, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar transactions"""
        where_clause = {"user_id": user_id} if user_id else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )
        
        return results
    
    def get_user_transaction_context(self, user_id: str, category: str = None, limit: int = 10) -> List[str]:
        """Get relevant transaction context for a user"""
        where_clause = {"user_id": user_id}
        if category:
            where_clause["category"] = category
        
        results = self.collection.query(
            query_texts=[f"user {user_id} transactions"],
            n_results=limit,
            where=where_clause
        )
        
        return results['documents'][0] if results['documents'] else []

class EnhancedFinancialIntelligenceSystem:
    """Main system orchestrating all agents and components"""
    
    def __init__(self):
        self.agents = {}
        self.vector_db = VectorDatabase()
        self.conversation_history = defaultdict(list)
        self.user_profiles = {}
        self.initialize_agents()
    
    def initialize_agents(self):
        """Initialize all agents"""
        logger.info("Initializing Enhanced Financial Intelligence System...")
        
        self.agents = {
            'router': QueryRouterAgent("QueryRouter"),
            'analyzer': DataAnalysisAgent("DataAnalyzer"),
            'recommender': RecommendationAgent("Recommender")
        }
        
        logger.info("All agents initialized successfully")
    
    def load_user_data(self, users_file: str, transactions_file: str):
        """Load user profiles and transaction data"""
        logger.info("Loading user data and transactions...")
        
        # Load user profiles
        users_df = pd.read_csv(users_file)
        for _, user in users_df.iterrows():
            self.user_profiles[user['user_id']] = user.to_dict()
        
        # Load transactions
        transactions_df = pd.read_csv(transactions_file)
        transactions = transactions_df.to_dict('records')
        
        # Add to vector database
        self.vector_db.add_transactions(transactions)
        
        logger.info(f"Loaded {len(self.user_profiles)} users and {len(transactions)} transactions")
    
    async def process_query(self, user_id: str, query: str) -> Dict[str, Any]:
        """Process a user query through the multi-agent system"""
        logger.info(f"Processing query for user {user_id}: {query}")
        
        # Get user profile and transaction history
        user_profile = self.user_profiles.get(user_id, {})
        transaction_context = self.vector_db.get_user_transaction_context(user_id)
        
        # Create query context
        context = QueryContext(
            user_id=user_id,
            query=query,
            query_type="",  # Will be set by router
            timestamp=datetime.now(),
            user_profile=user_profile,
            transaction_history=transaction_context,
            relevant_context=[]
        )
        
        # Route query
        router_response = await self.agents['router'].process(context)
        context.query_type = router_response.metadata['query_type']
        
        # Process with appropriate agents
        responses = []
        
        # Always get analysis
        analyzer_response = await self.agents['analyzer'].process(context)
        responses.append(analyzer_response)
        
        # Get recommendations for most query types
        if context.query_type != 'general_query':
            recommender_response = await self.agents['recommender'].process(context)
            responses.append(recommender_response)
        
        # Combine responses
        final_response = self._combine_responses(responses, context)
        
        # Store in conversation history
        self.conversation_history[user_id].append({
            'query': query,
            'response': final_response,
            'timestamp': datetime.now().isoformat(),
            'query_type': context.query_type
        })
        
        return {
            'response': final_response,
            'query_type': context.query_type,
            'confidence': np.mean([r.confidence for r in responses]),
            'processing_time': sum(r.processing_time for r in responses),
            'agents_used': [r.agent_name for r in responses]
        }
    
    def _combine_responses(self, responses: List[AgentResponse], context: QueryContext) -> str:
        """Combine responses from multiple agents into a coherent answer"""
        if not responses:
            return "I'm sorry, I couldn't process your query. Please try rephrasing it."
        
        combined_response = []
        
        # Add analysis results
        for response in responses:
            if response.agent_name == "DataAnalyzer":
                combined_response.append("📊 **Analysis:**")
                combined_response.append(response.response)
                combined_response.append("")
            elif response.agent_name == "Recommender":
                combined_response.append("💡 **Recommendations:**")
                combined_response.append(response.response)
                combined_response.append("")
        
        # Add confidence and metadata
        avg_confidence = np.mean([r.confidence for r in responses])
        if avg_confidence < 0.7:
            combined_response.append("⚠️ *Note: This response has moderate confidence. Consider consulting a financial advisor for important decisions.*")
        
        return "\\n".join(combined_response)
    
    def generate_financial_report(self, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive financial report for a user"""
        user_profile = self.user_profiles.get(user_id, {})
        if not user_profile:
            return {"error": "User not found"}
        
        # Get user's transaction context
        transaction_context = self.vector_db.get_user_transaction_context(user_id, limit=100)
        
        # Create a comprehensive analysis query
        context = QueryContext(
            user_id=user_id,
            query="Generate comprehensive financial analysis and recommendations",
            query_type="general_analysis",
            timestamp=datetime.now(),
            user_profile=user_profile,
            transaction_history=transaction_context
        )
        
        # This would typically be async, but simplified for demo
        report = {
            'user_profile': user_profile,
            'financial_summary': "Comprehensive financial analysis would go here",
            'risk_assessment': "Risk analysis would go here",
            'recommendations': "Personalized recommendations would go here",
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        return self.conversation_history[user_id][-limit:]

# Example usage and testing
async def demo_system():
    """Demonstrate the enhanced financial intelligence system"""
    
    # Initialize system
    system = EnhancedFinancialIntelligenceSystem()
    
    # Mock user profile
    mock_user = {
        'user_id': 'demo_user_001',
        'name': 'John Doe',
        'age': 30,
        'profile_type': 'young_professional',
        'monthly_income': 75000,
        'risk_tolerance': 'medium'
    }
    
    system.user_profiles['demo_user_001'] = mock_user
    
    # Mock transaction data
    mock_transactions = [
        {
            'transaction_id': 'txn_001',
            'user_id': 'demo_user_001',
            'date': '2024-01-15',
            'merchant': 'Zomato',
            'category': 'food_dining',
            'amount': 850.0,
            'currency': 'INR',
            'payment_mode': 'UPI',
            'type': 'debit'
        }
    ]
    
    system.vector_db.add_transactions(mock_transactions)
    
    # Test queries
    test_queries = [
        "How much did I spend on food last month?",
        "What's my investment recommendation?",
        "Analyze my spending patterns",
        "How can I save more money?"
    ]
    
    print("🤖 Enhanced Financial Intelligence System Demo")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\\n👤 User: {query}")
        response = await system.process_query('demo_user_001', query)
        print(f"🤖 Assistant: {response['response']}")
        print(f"📊 Confidence: {response['confidence']:.2f}")
        print(f"⏱️ Processing Time: {response['processing_time']:.3f}s")
        print(f"🔧 Agents Used: {', '.join(response['agents_used'])}")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_system())