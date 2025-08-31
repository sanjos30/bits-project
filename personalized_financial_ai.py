#!/usr/bin/env python3
"""
Personalized Financial AI System
Addresses user identification, context, and validation issues
"""

import pandas as pd
import numpy as np
import torch
import json
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class PersonalizedFinancialAI:
    """Personalized Financial AI with proper user identification and validation"""
    
    def __init__(self):
        self.users_df = None
        self.transactions_df = None
        self.current_user = None
        self.user_transactions = None
        self.user_summary = None
        self.load_data()
        
    def load_data(self):
        """Load production dataset"""
        try:
            self.users_df = pd.read_csv('data/production_users.csv')
            self.transactions_df = pd.read_csv('data/production_transactions.csv')
            print(f"✅ Loaded {len(self.users_df):,} users and {len(self.transactions_df):,} transactions")
        except FileNotFoundError:
            print("❌ Production data not found. Please run data generation first.")
            return False
        return True
    
    def select_user(self, user_id: Optional[str] = None) -> Dict:
        """Select a specific user or random user"""
        if user_id:
            user_data = self.users_df[self.users_df['user_id'] == user_id]
            if len(user_data) == 0:
                print(f"❌ User {user_id} not found. Selecting random user.")
                user_data = self.users_df.sample(n=1)
        else:
            user_data = self.users_df.sample(n=1)
        
        self.current_user = user_data.iloc[0].to_dict()
        self.user_transactions = self.transactions_df[
            self.transactions_df['user_id'] == self.current_user['user_id']
        ]
        self.user_summary = self.calculate_user_summary()
        
        print(f"👤 Selected User: {self.current_user['name']} (ID: {self.current_user['user_id']})")
        print(f"💰 Monthly Income: ₹{self.current_user['monthly_income']:,}")
        print(f"📊 Transactions: {len(self.user_transactions):,}")
        
        return self.current_user
    
    def calculate_user_summary(self) -> Dict:
        """Calculate comprehensive user financial summary"""
        if self.user_transactions is None or len(self.user_transactions) == 0:
            return {}
        
        # Basic metrics
        total_spent = self.user_transactions[self.user_transactions['type'] == 'debit']['amount'].sum()
        total_earned = self.user_transactions[self.user_transactions['type'] == 'credit']['amount'].sum()
        net_balance = total_earned - total_spent
        
        # Category analysis
        category_spending = self.user_transactions[
            self.user_transactions['type'] == 'debit'
        ].groupby('category')['amount'].sum().sort_values(ascending=False)
        
        # Monthly spending pattern
        try:
            # Convert date column to datetime if it's not already
            if 'date' in self.user_transactions.columns:
                self.user_transactions['date'] = pd.to_datetime(self.user_transactions['date'])
                monthly_spending = self.user_transactions[
                    self.user_transactions['type'] == 'debit'
                ].groupby(self.user_transactions['date'].dt.to_period('M'))['amount'].sum()
            else:
                monthly_spending = pd.Series()
        except:
            monthly_spending = pd.Series()
        
        # Top merchants
        top_merchants = self.user_transactions.groupby('merchant')['amount'].sum().sort_values(ascending=False).head(10)
        
        return {
            'total_spent': total_spent,
            'total_earned': total_earned,
            'net_balance': net_balance,
            'category_spending': category_spending.to_dict(),
            'monthly_spending': monthly_spending.to_dict(),
            'top_merchants': top_merchants.to_dict(),
            'transaction_count': len(self.user_transactions),
            'avg_transaction': self.user_transactions['amount'].mean(),
            'savings_rate': (net_balance / self.current_user['monthly_income']) * 100 if self.current_user['monthly_income'] > 0 else 0
        }
    
    def create_personalized_prompt(self, query: str) -> str:
        """Create personalized prompt with user context"""
        if not self.current_user or not self.user_summary:
            return f"Q: {query} A:"
        
        # Build user context
        context_parts = [
            f"User: {self.current_user['name']} (Age: {self.current_user['age']}, Income: ₹{self.current_user['monthly_income']:,})",
            f"Profile: {self.current_user['profile_type']}",
            f"Total Spent: ₹{self.user_summary['total_spent']:,.0f}",
            f"Net Balance: ₹{self.user_summary['net_balance']:,.0f}",
            f"Savings Rate: {self.user_summary['savings_rate']:.1f}%"
        ]
        
        # Add top spending categories
        if self.user_summary['category_spending']:
            top_categories = list(self.user_summary['category_spending'].items())[:3]
            category_info = ", ".join([f"{cat}: ₹{amt:,.0f}" for cat, amt in top_categories])
            context_parts.append(f"Top Categories: {category_info}")
        
        user_context = " | ".join(context_parts)
        
        return f"Context: {user_context}\nQ: {query} A:"
    
    def validate_response_accuracy(self, query: str, response: str) -> Dict:
        """Validate if response matches actual user data"""
        validation = {
            'accurate': False,
            'confidence': 0.0,
            'issues': [],
            'data_verification': {}
        }
        
        query_lower = query.lower()
        
        # Validate spending amounts
        if 'spend' in query_lower or 'expense' in query_lower:
            validation.update(self.validate_spending_response(response))
        
        # Validate savings rate
        elif 'save' in query_lower or 'savings' in query_lower:
            validation.update(self.validate_savings_response(response))
        
        # Validate investment recommendations
        elif 'invest' in query_lower or 'allocation' in query_lower:
            validation.update(self.validate_investment_response(response))
        
        return validation
    
    def validate_spending_response(self, response: str) -> Dict:
        """Validate spending-related responses"""
        actual_total = self.user_summary['total_spent']
        actual_categories = self.user_summary['category_spending']
        
        # Extract amounts from response
        import re
        amounts = re.findall(r'₹([\d,]+)', response)
        
        validation = {
            'accurate': False,
            'confidence': 0.0,
            'issues': [],
            'data_verification': {
                'actual_total_spent': actual_total,
                'actual_categories': actual_categories,
                'extracted_amounts': amounts
            }
        }
        
        # Check if total spending is mentioned correctly
        if f"₹{actual_total:,.0f}" in response or f"₹{int(actual_total):,}" in response:
            validation['accurate'] = True
            validation['confidence'] += 0.4
        
        # Check if top categories are mentioned
        if actual_categories:
            top_category = list(actual_categories.keys())[0]
            if top_category.lower() in response.lower():
                validation['confidence'] += 0.3
        
        # Check for reasonable amounts
        for amount_str in amounts:
            try:
                amount = float(amount_str.replace(',', ''))
                if 0 < amount <= actual_total * 1.5:  # Allow some tolerance
                    validation['confidence'] += 0.1
                else:
                    validation['issues'].append(f"Unrealistic amount: ₹{amount:,.0f}")
            except:
                validation['issues'].append(f"Invalid amount format: {amount_str}")
        
        validation['confidence'] = min(1.0, validation['confidence'])
        
        return validation
    
    def validate_savings_response(self, response: str) -> Dict:
        """Validate savings-related responses"""
        actual_savings_rate = self.user_summary['savings_rate']
        actual_net_balance = self.user_summary['net_balance']
        
        validation = {
            'accurate': False,
            'confidence': 0.0,
            'issues': [],
            'data_verification': {
                'actual_savings_rate': actual_savings_rate,
                'actual_net_balance': actual_net_balance
            }
        }
        
        # Check if savings rate is mentioned correctly
        if f"{actual_savings_rate:.1f}%" in response or f"{int(actual_savings_rate)}%" in response:
            validation['accurate'] = True
            validation['confidence'] += 0.6
        
        # Check if net balance is mentioned
        if f"₹{actual_net_balance:,.0f}" in response or f"₹{int(actual_net_balance):,}" in response:
            validation['confidence'] += 0.4
        
        validation['confidence'] = min(1.0, validation['confidence'])
        
        return validation
    
    def validate_investment_response(self, response: str) -> Dict:
        """Validate investment-related responses"""
        age = self.current_user['age']
        income = self.current_user['monthly_income']
        
        # Calculate expected allocation
        expected_equity = max(20, min(80, 100 - age))
        expected_debt = 100 - expected_equity
        expected_sip = income * 0.2
        
        validation = {
            'accurate': False,
            'confidence': 0.0,
            'issues': [],
            'data_verification': {
                'user_age': age,
                'user_income': income,
                'expected_equity': expected_equity,
                'expected_debt': expected_debt,
                'expected_sip': expected_sip
            }
        }
        
        # Check if age-appropriate allocation is mentioned
        if f"{expected_equity}%" in response or f"{expected_debt}%" in response:
            validation['accurate'] = True
            validation['confidence'] += 0.5
        
        # Check if reasonable SIP amount is mentioned
        if f"₹{int(expected_sip):,}" in response or f"₹{expected_sip:,.0f}" in response:
            validation['confidence'] += 0.3
        
        # Check if age is considered
        if str(age) in response:
            validation['confidence'] += 0.2
        
        validation['confidence'] = min(1.0, validation['confidence'])
        
        return validation
    
    def process_query(self, query: str, user_id: Optional[str] = None) -> Dict:
        """Process query with full personalization and validation"""
        # Select user if not already selected or if specific user requested
        if not self.current_user or (user_id and self.current_user['user_id'] != user_id):
            self.select_user(user_id)
        
        # Create personalized prompt
        personalized_prompt = self.create_personalized_prompt(query)
        
        # Generate response (simulate model response for now)
        response = self.generate_response(query, personalized_prompt)
        
        # Validate response accuracy
        validation = self.validate_response_accuracy(query, response)
        
        return {
            'user_id': self.current_user['user_id'],
            'user_name': self.current_user['name'],
            'query': query,
            'personalized_prompt': personalized_prompt,
            'response': response,
            'validation': validation,
            'user_summary': self.user_summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_response(self, query: str, prompt: str) -> str:
        """Generate response based on user data"""
        query_lower = query.lower()
        
        # Debug: Print what we're processing
        print(f"DEBUG: Processing query: '{query}'")
        print(f"DEBUG: Query lower: '{query_lower}'")
        
        # Income questions
        if any(word in query_lower for word in ['income', 'earn', 'salary', 'monthly']):
            income = self.current_user['monthly_income']
            return f"Your monthly income is ₹{income:,}. This is your base salary as a {self.current_user['profile_type']}."
        
        # Education spending
        elif 'education' in query_lower:
            if 'education' in self.user_summary['category_spending']:
                amount = self.user_summary['category_spending']['education']
                total_spent = self.user_summary['total_spent']
                percentage = (amount / total_spent) * 100
                return f"You spent ₹{amount:,.2f} on education, which is {percentage:.1f}% of your total spending."
            else:
                return "You have no education spending recorded in your transactions."
        
        # Travel spending
        elif 'travel' in query_lower:
            if 'travel' in self.user_summary['category_spending']:
                amount = self.user_summary['category_spending']['travel']
                total_spent = self.user_summary['total_spent']
                percentage = (amount / total_spent) * 100
                return f"You spent ₹{amount:,.2f} on travel, which is {percentage:.1f}% of your total spending."
            else:
                return "You have no travel spending recorded in your transactions."
        
        # Healthcare spending
        elif 'healthcare' in query_lower or 'health' in query_lower:
            if 'healthcare' in self.user_summary['category_spending']:
                amount = self.user_summary['category_spending']['healthcare']
                total_spent = self.user_summary['total_spent']
                percentage = (amount / total_spent) * 100
                return f"You spent ₹{amount:,.2f} on healthcare, which is {percentage:.1f}% of your total spending."
            else:
                return "You have no healthcare spending recorded in your transactions."
        
        # Groceries spending
        elif 'grocery' in query_lower or 'groceries' in query_lower:
            if 'groceries' in self.user_summary['category_spending']:
                amount = self.user_summary['category_spending']['groceries']
                total_spent = self.user_summary['total_spent']
                percentage = (amount / total_spent) * 100
                return f"You spent ₹{amount:,.2f} on groceries, which is {percentage:.1f}% of your total spending."
            else:
                return "You have no groceries spending recorded in your transactions."
        
        # Total spending
        elif any(word in query_lower for word in ['total', 'spend', 'expense']) and 'category' not in query_lower:
            total_spent = self.user_summary['total_spent']
            top_category = list(self.user_summary['category_spending'].keys())[0]
            top_amount = list(self.user_summary['category_spending'].values())[0]
            return f"You have spent ₹{total_spent:,.0f} in total. Your highest spending category is {top_category} at ₹{top_amount:,.0f}."
        
        # Savings questions
        elif any(word in query_lower for word in ['save', 'savings']):
            savings_rate = self.user_summary['savings_rate']
            net_balance = self.user_summary['net_balance']
            return f"Your current savings rate is {savings_rate:.1f}% with a net balance of ₹{net_balance:,.0f}. {'Good job!' if savings_rate >= 20 else 'Consider reducing expenses to improve savings.'}"
        
        # Investment spending (check if asking about spending first)
        elif 'invest' in query_lower and any(word in query_lower for word in ['spend', 'spent', 'amount', 'how much']):
            if 'investments' in self.user_summary['category_spending']:
                amount = self.user_summary['category_spending']['investments']
                total_spent = self.user_summary['total_spent']
                percentage = (amount / total_spent) * 100
                return f"You spent ₹{amount:,.2f} on investments, which is {percentage:.1f}% of your total spending."
            else:
                return "You have no investment spending recorded in your transactions."
        
        # Investment recommendations
        elif any(word in query_lower for word in ['invest', 'allocation']) and not any(word in query_lower for word in ['spend', 'spent', 'amount', 'how much']):
            age = self.current_user['age']
            income = self.current_user['monthly_income']
            equity_pct = max(20, min(80, 100 - age))
            debt_pct = 100 - equity_pct
            sip = income * 0.2
            return f"Based on your age ({age}), I recommend {equity_pct}% equity and {debt_pct}% debt allocation. Consider a monthly SIP of ₹{sip:,.0f}."
        
        # Balance questions
        elif any(word in query_lower for word in ['balance', 'net']):
            net_balance = self.user_summary['net_balance']
            total_earned = self.user_summary['total_earned']
            total_spent = self.user_summary['total_spent']
            return f"Your net balance is ₹{net_balance:,.0f}. You earned ₹{total_earned:,.0f} and spent ₹{total_spent:,.0f}."
        
        # Financial health questions
        elif any(word in query_lower for word in ['health', 'score']):
            savings_rate = self.user_summary['savings_rate']
            if savings_rate >= 20:
                return f"Your financial health is good with a savings rate of {savings_rate:.1f}%. Keep up the good work!"
            elif savings_rate >= 10:
                return f"Your financial health is fair with a savings rate of {savings_rate:.1f}%. Consider increasing savings."
            else:
                return f"Your financial health needs improvement with a savings rate of {savings_rate:.1f}%. Focus on reducing expenses."
        
        # Budget questions
        elif any(word in query_lower for word in ['budget', 'improve']):
            top_category = list(self.user_summary['category_spending'].keys())[0]
            top_amount = list(self.user_summary['category_spending'].values())[0]
            return f"To improve your budget, focus on reducing {top_category} spending (₹{top_amount:,.0f}). This is your highest expense category."
        
        # Top categories
        elif 'category' in query_lower or 'categories' in query_lower:
            total_spent = self.user_summary['total_spent']
            top_categories = list(self.user_summary['category_spending'].items())[:3]
            category_list = ', '.join([f'{cat}: ₹{amt:,.0f}' for cat, amt in top_categories])
            return f"Your top spending categories are: {category_list}. Total spending: ₹{total_spent:,.0f}."
        
        # Individual transactions with type filtering
        elif any(word in query_lower for word in ['transaction', 'transactions']) and any(word in query_lower for word in ['top', 'highest', 'biggest', 'largest']):
            # Extract number from query (default to 5 if not specified)
            import re
            number_match = re.search(r'(\d+)', query_lower)
            num_transactions = int(number_match.group(1)) if number_match else 5
            
            # Filter by transaction type if specified
            if any(word in query_lower for word in ['debit', 'spent', 'expense', 'payment']):
                filtered_transactions = self.user_transactions[self.user_transactions['type'] == 'debit']
                transaction_type = "debit"
            elif any(word in query_lower for word in ['credit', 'earned', 'income', 'received']):
                filtered_transactions = self.user_transactions[self.user_transactions['type'] == 'credit']
                transaction_type = "credit"
            else:
                filtered_transactions = self.user_transactions
                transaction_type = "all"
            
            # Get top transactions
            top_transactions = filtered_transactions.nlargest(num_transactions, 'amount')
            
            if transaction_type == "all":
                response = f"Your top {num_transactions} individual transactions:\n"
            else:
                response = f"Your top {num_transactions} {transaction_type} transactions:\n"
            
            for i, (_, txn) in enumerate(top_transactions.iterrows(), 1):
                response += f"{i}. ₹{txn['amount']:,.2f} - {txn['category']} ({txn['merchant']}) - {txn['date']}\n"
            return response
        
        # Default response with user context
        else:
            return f"Hello {self.current_user['name']}! I can help you with spending analysis, savings advice, and investment recommendations. What would you like to know about your finances?"

def main():
    """Demo the personalized system"""
    print("🎯 PERSONALIZED FINANCIAL AI SYSTEM")
    print("=" * 50)
    
    ai = PersonalizedFinancialAI()
    
    # Demo queries
    demo_queries = [
        "How much did I spend in total?",
        "What's my savings rate?",
        "What investment allocation do you recommend?",
        "What are my top spending categories?"
    ]
    
    for query in demo_queries:
        print(f"\n🔍 Query: {query}")
        result = ai.process_query(query)
        
        print(f"👤 User: {result['user_name']} (ID: {result['user_id']})")
        print(f"💬 Response: {result['response']}")
        print(f"✅ Validation: {'Accurate' if result['validation']['accurate'] else 'Needs Review'} (Confidence: {result['validation']['confidence']:.2f})")
        
        if result['validation']['issues']:
            print(f"⚠️ Issues: {', '.join(result['validation']['issues'])}")

if __name__ == "__main__":
    main()
