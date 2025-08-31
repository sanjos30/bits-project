#!/usr/bin/env python3
"""
AI-Based Financial System for M.Tech Demo
Uses the trained LoRA model for actual AI responses
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import re

class AIFinancialSystem:
    def __init__(self):
        """Initialize AI-based financial system"""
        print("🤖 Initializing AI-Based Financial System...")
        
        # Load data
        self.load_data()
        
        # Load AI model
        self.load_ai_model()
        
        print("✅ AI System Ready!")
    
    def load_data(self):
        """Load user and transaction data"""
        print("📊 Loading financial data...")
        
        try:
            self.users_df = pd.read_csv('data/production_users.csv')
            self.transactions_df = pd.read_csv('data/production_transactions.csv')
            print(f"✅ Loaded PRODUCTION data: {len(self.users_df):,} users, {len(self.transactions_df):,} transactions")
        except FileNotFoundError:
            self.users_df = pd.read_csv('data/demo_users.csv')
            self.transactions_df = pd.read_csv('data/demo_transactions.csv')
            print(f"✅ Loaded DEMO data: {len(self.users_df):,} users, {len(self.transactions_df):,} transactions")
        
        self.current_user = None
        self.user_transactions = None
    
    def load_ai_model(self):
        """Load the trained AI model"""
        print("🧠 Loading trained AI model...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=torch.float32,
                device_map=None
            )
            
            # Load trained LoRA adapter
            self.model = PeftModel.from_pretrained(
                self.base_model,
                "models/improved_financial_lora"
            )
            
            print("✅ Trained AI model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading AI model: {e}")
            self.model = None
    
    def select_user(self, user_id):
        """Select a specific user for personalized responses"""
        user = self.users_df[self.users_df['user_id'] == user_id]
        if user.empty:
            print(f"❌ User {user_id} not found")
            return False
        
        self.current_user = user.iloc[0].to_dict()
        self.user_transactions = self.transactions_df[
            self.transactions_df['user_id'] == user_id
        ].copy()
        
        print(f"👤 Selected User: {self.current_user['name']} (ID: {user_id})")
        print(f"💰 Monthly Income: ₹{self.current_user['monthly_income']:,.0f}")
        print(f"📊 Transactions: {len(self.user_transactions):,}")
        
        return True
    
    def calculate_user_summary(self):
        """Calculate user financial summary"""
        if self.user_transactions is None:
            return {}
        
        total_spent = self.user_transactions[
            self.user_transactions['type'] == 'debit'
        ]['amount'].sum()
        
        total_earned = self.user_transactions[
            self.user_transactions['type'] == 'credit'
        ]['amount'].sum()
        
        net_balance = total_earned - total_spent
        
        category_spending = self.user_transactions[
            self.user_transactions['type'] == 'debit'
        ].groupby('category')['amount'].sum().sort_values(ascending=False)
        
        return {
            'total_spent': total_spent,
            'total_earned': total_earned,
            'net_balance': net_balance,
            'category_spending': category_spending.to_dict()
        }
    
    def create_ai_prompt(self, query):
        """Create prompt for AI model"""
        user_summary = self.calculate_user_summary()
        
        # Create simple, clear prompt
        prompt = f"""User: {self.current_user['name']}
Age: {self.current_user['age']}
Income: ₹{self.current_user['monthly_income']:,.0f}
Total Spent: ₹{user_summary['total_spent']:,.0f}
Total Earned: ₹{user_summary['total_earned']:,.0f}
Net Balance: ₹{user_summary['net_balance']:,.0f}

Question: {query}
Answer:"""
        
        return prompt
    
    def generate_ai_response(self, prompt):
        """Generate response using AI model"""
        if self.model is None:
            return "AI model not available."
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            response = response[len(prompt):].strip()
            
            return response if response else "I don't have enough information to answer that question."
            
        except Exception as e:
            print(f"❌ AI generation error: {e}")
            return "AI model encountered an error."
    
    def process_query(self, query):
        """Process a financial query using AI"""
        if self.current_user is None:
            return {
                'response': 'Please select a user first.',
                'validation': {'confidence': 0.0, 'status': 'No user selected'}
            }
        
        print(f"🔍 Processing AI query: '{query}'")
        
        # Create prompt
        prompt = self.create_ai_prompt(query)
        
        # Generate AI response
        ai_response = self.generate_ai_response(prompt)
        
        # Simple validation
        validation = self.validate_response(query, ai_response)
        
        return {
            'response': ai_response,
            'validation': validation,
            'prompt_used': prompt
        }
    
    def validate_response(self, query, response):
        """Simple validation of AI response"""
        user_summary = self.calculate_user_summary()
        
        # Check for financial terms
        has_financial_info = any(term in response.lower() for term in ['₹', 'rupee', 'amount', 'spent', 'earned', 'balance'])
        has_user_info = self.current_user['name'] in response
        
        confidence = 0.0
        if has_financial_info and has_user_info:
            confidence = 0.8
        elif has_financial_info or has_user_info:
            confidence = 0.5
        else:
            confidence = 0.2
        
        status = "Accurate" if confidence > 0.7 else "Needs Review" if confidence > 0.3 else "Inaccurate"
        
        return {
            'confidence': confidence,
            'status': status
        }

def main():
    """Main function for testing"""
    print("🎯 AI-BASED FINANCIAL SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize system
    ai_system = AIFinancialSystem()
    
    # Select a user
    ai_system.select_user('prod_user_0496')  # Wayne Morgan
    
    # Test queries
    test_queries = [
        "How much did I spend on travel?",
        "What is my total income?",
        "What's my net balance?",
        "What is my monthly income?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 40)
        
        result = ai_system.process_query(query)
        
        print(f"🤖 AI Response: {result['response']}")
        print(f"✅ Validation: {result['validation']['status']} (Confidence: {result['validation']['confidence']:.2f})")
        
        # Show real data for comparison
        user_summary = ai_system.calculate_user_summary()
        if 'travel' in query.lower():
            travel_spending = user_summary['category_spending'].get('travel', 0)
            print(f"📊 Real Travel Spending: ₹{travel_spending:,.2f}")
        elif 'income' in query.lower() and 'monthly' in query.lower():
            print(f"📊 Real Monthly Income: ₹{ai_system.current_user['monthly_income']:,.2f}")
        elif 'income' in query.lower():
            print(f"📊 Real Total Earned: ₹{user_summary['total_earned']:,.2f}")
        elif 'balance' in query.lower():
            print(f"📊 Real Net Balance: ₹{user_summary['net_balance']:,.2f}")
        
        print()

if __name__ == "__main__":
    main()
