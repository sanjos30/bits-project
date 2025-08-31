#!/usr/bin/env python3
"""
Show Real Raw Data for Random User
Display actual user profile and transaction data
"""

import pandas as pd
import random
import json
from datetime import datetime

def main():
    """Show raw data for a random user"""
    print("🔍 SHOWING REAL RAW DATA FOR RANDOM USER")
    print("=" * 60)
    
    # Load production data
    try:
        users_df = pd.read_csv('data/production_users.csv')
        transactions_df = pd.read_csv('data/production_transactions.csv')
        print(f"✅ Loaded {len(users_df):,} users and {len(transactions_df):,} transactions")
    except FileNotFoundError:
        print("❌ Production data not found. Please run data generation first.")
        return
    
    # Pick a random user
    random_user = users_df.sample(n=1).iloc[0]
    user_id = random_user['user_id']
    
    print(f"\n🎯 SELECTED RANDOM USER:")
    print(f"👤 User ID: {user_id}")
    print(f"📋 Name: {random_user['name']}")
    print(f"🎂 Age: {random_user['age']}")
    print(f"💰 Monthly Income: ₹{random_user['monthly_income']:,}")
    print(f"🏷️ Profile Type: {random_user['profile_type']}")
    print(f"🌍 City: {random_user['city']}")
    print(f"⚖️ Risk Tolerance: {random_user['risk_tolerance']}")
    
    # Get user's transactions
    user_transactions = transactions_df[transactions_df['user_id'] == user_id]
    
    print(f"\n📊 TRANSACTION SUMMARY:")
    print(f"📈 Total Transactions: {len(user_transactions):,}")
    print(f"💸 Total Debits: {len(user_transactions[user_transactions['type'] == 'debit']):,}")
    print(f"💳 Total Credits: {len(user_transactions[user_transactions['type'] == 'credit']):,}")
    
    # Calculate financial metrics
    total_spent = user_transactions[user_transactions['type'] == 'debit']['amount'].sum()
    total_earned = user_transactions[user_transactions['type'] == 'credit']['amount'].sum()
    net_balance = total_earned - total_spent
    
    print(f"\n💰 FINANCIAL METRICS:")
    print(f"💸 Total Spent: ₹{total_spent:,.2f}")
    print(f"💳 Total Earned: ₹{total_earned:,.2f}")
    print(f"⚖️ Net Balance: ₹{net_balance:,.2f}")
    print(f"📈 Savings Rate: {(net_balance / random_user['monthly_income']) * 100:.1f}%")
    
    # Category analysis
    print(f"\n📂 SPENDING BY CATEGORY:")
    category_spending = user_transactions[user_transactions['type'] == 'debit'].groupby('category')['amount'].sum().sort_values(ascending=False)
    
    for category, amount in category_spending.head(10).items():
        percentage = (amount / total_spent) * 100
        print(f"   {category:20} ₹{amount:12,.2f} ({percentage:5.1f}%)")
    
    # Show sample transactions
    print(f"\n📋 SAMPLE TRANSACTIONS (First 20):")
    print(f"{'Date':<12} {'Type':<8} {'Category':<15} {'Merchant':<25} {'Amount':<12}")
    print("-" * 80)
    
    sample_transactions = user_transactions.head(20)
    for _, txn in sample_transactions.iterrows():
        date_str = str(txn['date'])[:10] if pd.notna(txn['date']) else 'N/A'
        print(f"{date_str:<12} {txn['type']:<8} {txn['category']:<15} {txn['merchant']:<25} ₹{txn['amount']:>10,.2f}")
    
    # Show transaction distribution
    print(f"\n📊 TRANSACTION DISTRIBUTION:")
    
    # Amount ranges
    amount_ranges = [
        (0, 1000, "₹0 - ₹1,000"),
        (1000, 5000, "₹1,000 - ₹5,000"),
        (5000, 10000, "₹5,000 - ₹10,000"),
        (10000, 50000, "₹10,000 - ₹50,000"),
        (50000, 100000, "₹50,000 - ₹1,00,000"),
        (100000, float('inf'), "₹1,00,000+")
    ]
    
    for min_amt, max_amt, range_name in amount_ranges:
        if max_amt == float('inf'):
            count = len(user_transactions[user_transactions['amount'] >= min_amt])
        else:
            count = len(user_transactions[(user_transactions['amount'] >= min_amt) & (user_transactions['amount'] < max_amt)])
        
        if count > 0:
            percentage = (count / len(user_transactions)) * 100
            print(f"   {range_name:20} {count:>6} transactions ({percentage:5.1f}%)")
    
    # Payment methods (check if column exists)
    print(f"\n💳 PAYMENT METHODS:")
    if 'payment_method' in user_transactions.columns:
        payment_methods = user_transactions['payment_method'].value_counts()
        for method, count in payment_methods.items():
            percentage = (count / len(user_transactions)) * 100
            print(f"   {method:15} {count:>6} transactions ({percentage:5.1f}%)")
    else:
        print("   Payment method data not available in this dataset")
    
    # Monthly spending pattern
    print(f"\n📅 MONTHLY SPENDING PATTERN:")
    try:
        user_transactions['date'] = pd.to_datetime(user_transactions['date'])
        monthly_spending = user_transactions[user_transactions['type'] == 'debit'].groupby(user_transactions['date'].dt.to_period('M'))['amount'].sum()
        
        for month, amount in monthly_spending.head(12).items():
            print(f"   {month} ₹{amount:>12,.2f}")
    except:
        print("   Monthly pattern not available")
    
    # Top merchants
    print(f"\n🏪 TOP MERCHANTS:")
    top_merchants = user_transactions.groupby('merchant')['amount'].sum().sort_values(ascending=False).head(10)
    
    for merchant, amount in top_merchants.items():
        print(f"   {merchant:30} ₹{amount:>12,.2f}")
    
    # Save detailed data to file
    output_file = f"user_{user_id}_raw_data.json"
    
    detailed_data = {
        'user_profile': random_user.to_dict(),
        'transaction_summary': {
            'total_transactions': len(user_transactions),
            'total_spent': float(total_spent),
            'total_earned': float(total_earned),
            'net_balance': float(net_balance),
            'savings_rate': float((net_balance / random_user['monthly_income']) * 100)
        },
        'category_spending': category_spending.to_dict(),
        'sample_transactions': user_transactions.head(50).to_dict('records'),
        'payment_methods': payment_methods.to_dict() if 'payment_method' in user_transactions.columns else {},
        'top_merchants': top_merchants.to_dict()
    }
    
    with open(output_file, 'w') as f:
        json.dump(detailed_data, f, indent=2, default=str)
    
    print(f"\n💾 Detailed data saved to: {output_file}")
    print(f"\n{'='*60}")
    print(f"🎯 This is REAL RAW DATA for user {user_id}")
    print(f"📊 All amounts, categories, and transactions are actual generated data")
    print(f"🔍 You can verify this data matches the AI responses")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
