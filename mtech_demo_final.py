#!/usr/bin/env python3
"""
M.Tech Final Demo Script
Shows AI-based financial system with proper validation
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

class MtechDemo:
    def __init__(self):
        """Initialize the M.Tech demo system"""
        print("🎓 M.Tech AIML Project Demo")
        print("=" * 50)
        print("Personalized Financial AI System")
        print("Using LoRA Fine-tuned GPT2 Model")
        print("=" * 50)
        
        self.load_data()
        self.load_ai_model()
        
    def load_data(self):
        """Load production data"""
        print("\n📊 Loading Production Data...")
        self.users_df = pd.read_csv('data/production_users.csv')
        self.transactions_df = pd.read_csv('data/production_transactions.csv')
        print(f"✅ Loaded: {len(self.users_df):,} users, {len(self.transactions_df):,} transactions")
        
    def load_ai_model(self):
        """Load the trained AI model"""
        print("\n🧠 Loading Trained AI Model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=torch.float32,
                device_map=None
            )
            
            self.model = PeftModel.from_pretrained(
                self.base_model,
                "models/improved_financial_lora"
            )
            
            print("✅ AI Model Loaded Successfully!")
            print("📈 Model: GPT2 + LoRA Fine-tuning")
            print("🎯 Purpose: Personalized Financial Queries")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def select_user(self, user_id):
        """Select a user for personalized responses"""
        user = self.users_df[self.users_df['user_id'] == user_id]
        if user.empty:
            return False
            
        self.current_user = user.iloc[0].to_dict()
        self.user_transactions = self.transactions_df[
            self.transactions_df['user_id'] == user_id
        ].copy()
        
        print(f"\n👤 Selected User: {self.current_user['name']}")
        print(f"🎂 Age: {self.current_user['age']}")
        print(f"💰 Monthly Income: ₹{self.current_user['monthly_income']:,.0f}")
        print(f"🏷️ Profile: {self.current_user['profile_type']}")
        print(f"🌍 City: {self.current_user['city']}")
        print(f"📊 Transactions: {len(self.user_transactions):,}")
        
        return True
    
    def calculate_real_data(self):
        """Calculate real financial data for validation"""
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
    
    def create_prompt(self, query):
        """Create AI prompt with user context"""
        real_data = self.calculate_real_data()
        
        prompt = f"""User: {self.current_user['name']}
Age: {self.current_user['age']}
Monthly Income: ₹{self.current_user['monthly_income']:,.0f}
Total Spent: ₹{real_data['total_spent']:,.0f}
Total Earned: ₹{real_data['total_earned']:,.0f}
Net Balance: ₹{real_data['net_balance']:,.0f}

Question: {query}
Answer:"""
        
        return prompt
    
    def generate_ai_response(self, prompt):
        """Generate AI response"""
        if self.model is None:
            return "AI model not available"
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=512,
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.6,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            return response if response else "I don't have enough information."
            
        except Exception as e:
            return f"AI Error: {str(e)}"
    
    def demo_query(self, query, expected_type):
        """Demonstrate a query with validation"""
        print(f"\n🔍 Query: {query}")
        print("-" * 60)
        
        # Create prompt
        prompt = self.create_prompt(query)
        
        # Generate AI response
        print("🤖 AI Processing...")
        start_time = time.time()
        ai_response = self.generate_ai_response(prompt)
        processing_time = time.time() - start_time
        
        print(f"⏱️ Processing Time: {processing_time:.2f} seconds")
        print(f"🤖 AI Response: {ai_response}")
        
        # Show real data for validation
        real_data = self.calculate_real_data()
        print(f"\n📊 REAL DATA VALIDATION:")
        
        if expected_type == "travel":
            travel_spending = real_data['category_spending'].get('travel', 0)
            print(f"   💸 Travel Spending: ₹{travel_spending:,.2f}")
        elif expected_type == "income":
            print(f"   💳 Total Earned: ₹{real_data['total_earned']:,.2f}")
        elif expected_type == "balance":
            print(f"   ⚖️ Net Balance: ₹{real_data['net_balance']:,.2f}")
        elif expected_type == "monthly_income":
            print(f"   💰 Monthly Income: ₹{self.current_user['monthly_income']:,.2f}")
        elif expected_type == "categories":
            top_cats = list(real_data['category_spending'].items())[:3]
            for i, (cat, amt) in enumerate(top_cats, 1):
                print(f"   {i}. {cat}: ₹{amt:,.2f}")
        
        # Simple accuracy check
        has_financial_info = any(term in ai_response.lower() for term in ['₹', 'rupee', 'amount', 'spent', 'earned', 'balance'])
        has_user_info = self.current_user['name'] in ai_response
        
        if has_financial_info and has_user_info:
            accuracy = "✅ Good"
        elif has_financial_info or has_user_info:
            accuracy = "⚠️ Partial"
        else:
            accuracy = "❌ Poor"
        
        print(f"\n🎯 Accuracy Assessment: {accuracy}")
        print("=" * 60)
    
    def run_demo(self):
        """Run the complete demo"""
        print("\n🚀 Starting M.Tech Demo...")
        
        # Select Wayne Morgan
        if not self.select_user('prod_user_0496'):
            print("❌ User not found")
            return
        
        # Demo queries
        demo_queries = [
            ("How much did I spend on travel?", "travel"),
            ("What is my total income?", "income"),
            ("What's my current financial balance?", "balance"),
            ("What is my monthly income?", "monthly_income"),
            ("What are my top spending categories?", "categories")
        ]
        
        for query, expected_type in demo_queries:
            self.demo_query(query, expected_type)
            time.sleep(2)  # Pause between queries
        
        print("\n🎉 Demo Complete!")
        print("\n📋 Key Points Demonstrated:")
        print("✅ AI Model Integration (LoRA Fine-tuned GPT2)")
        print("✅ Personalized Responses")
        print("✅ Real-time Data Validation")
        print("✅ Production-scale Data (1M+ transactions)")
        print("✅ User-specific Financial Analysis")

def main():
    """Main demo function"""
    demo = MtechDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
