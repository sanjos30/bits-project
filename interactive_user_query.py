#!/usr/bin/env python3
"""
Interactive User Query System
Ask any question for Wayne Morgan and verify against real data
"""

from personalized_financial_ai import PersonalizedFinancialAI
import json

def main():
    """Interactive query system for Wayne Morgan"""
    print("🎯 INTERACTIVE QUERY SYSTEM FOR WAYNE MORGAN")
    print("=" * 60)
    
    # Initialize AI system with Wayne Morgan
    user_id = "prod_user_0496"  # Wayne Morgan
    ai = PersonalizedFinancialAI()
    user = ai.select_user(user_id)
    
    print(f"👤 Current User: {user['name']} (ID: {user_id})")
    print(f"🎂 Age: {user['age']}, 💰 Income: ₹{user['monthly_income']:,}")
    print(f"🏷️ Profile: {user['profile_type']}, 🌍 City: {user['city']}")
    
    # Show quick data summary
    print(f"\n📊 QUICK DATA SUMMARY:")
    print(f"💸 Total Spent: ₹{ai.user_summary['total_spent']:,.2f}")
    print(f"💳 Total Earned: ₹{ai.user_summary['total_earned']:,.2f}")
    print(f"⚖️ Net Balance: ₹{ai.user_summary['net_balance']:,.2f}")
    print(f"📈 Savings Rate: {ai.user_summary['savings_rate']:.1f}%")
    
    # Show top categories
    print(f"\n📂 TOP SPENDING CATEGORIES:")
    category_items = list(ai.user_summary['category_spending'].items())[:5]
    for category, amount in category_items:
        percentage = (amount / ai.user_summary['total_spent']) * 100
        print(f"   {category:15} ₹{amount:>12,.2f} ({percentage:5.1f}%)")
    
    print(f"\n{'='*60}")
    print("💬 ASK ANY QUESTION ABOUT WAYNE'S FINANCES")
    print("Type 'quit' to exit, 'data' to see raw data, 'help' for suggestions")
    print(f"{'='*60}")
    
    while True:
        try:
            # Get user input
            query = input("\n🔍 Your Question: ").strip()
            
            if query.lower() == 'quit':
                print("👋 Goodbye!")
                break
                
            elif query.lower() == 'data':
                show_raw_data(ai)
                continue
                
            elif query.lower() == 'help':
                show_help_suggestions()
                continue
                
            elif not query:
                print("❌ Please enter a question.")
                continue
            
            print(f"\n🤖 PROCESSING: {query}")
            print("-" * 50)
            
            # Get AI response
            result = ai.process_query(query, user_id)
            
            # Display response
            print(f"💬 AI Response: {result['response']}")
            print(f"✅ Validation: {'Accurate' if result['validation']['accurate'] else 'Needs Review'} (Confidence: {result['validation']['confidence']:.2f})")
            
            # Show data verification
            if 'data_verification' in result['validation']:
                verification = result['validation']['data_verification']
                print(f"\n📊 REAL DATA VERIFICATION:")
                
                if 'actual_total_spent' in verification:
                    print(f"   💸 Real Total Spent: ₹{verification['actual_total_spent']:,.2f}")
                
                if 'actual_savings_rate' in verification:
                    print(f"   📈 Real Savings Rate: {verification['actual_savings_rate']:.1f}%")
                
                if 'actual_net_balance' in verification:
                    print(f"   ⚖️ Real Net Balance: ₹{verification['actual_net_balance']:,.2f}")
                
                if 'actual_categories' in verification:
                    print(f"   📂 Real Top Categories:")
                    for i, (category, amount) in enumerate(list(verification['actual_categories'].items())[:3]):
                        print(f"      {i+1}. {category}: ₹{amount:,.2f}")
                
                if 'user_age' in verification:
                    print(f"   🎂 User Age: {verification['user_age']}")
                
                if 'user_income' in verification:
                    print(f"   💰 User Income: ₹{verification['user_income']:,}")
            
            # Show personalized prompt used
            print(f"\n🎯 PERSONALIZED PROMPT USED:")
            print(f"{'─'*50}")
            print(result['personalized_prompt'])
            print(f"{'─'*50}")
            
            # Show any validation issues
            if result['validation']['issues']:
                print(f"\n⚠️ VALIDATION ISSUES:")
                for issue in result['validation']['issues']:
                    print(f"   - {issue}")
            
            print(f"\n{'='*60}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")

def show_raw_data(ai):
    """Show raw data for Wayne Morgan"""
    print(f"\n📋 WAYNE MORGAN'S RAW DATA")
    print("=" * 50)
    
    user = ai.current_user
    summary = ai.user_summary
    
    print(f"👤 Profile:")
    print(f"   Name: {user['name']}")
    print(f"   Age: {user['age']}")
    print(f"   Income: ₹{user['monthly_income']:,}")
    print(f"   Profile: {user['profile_type']}")
    print(f"   City: {user['city']}")
    print(f"   Risk Tolerance: {user['risk_tolerance']}")
    
    print(f"\n💰 Financial Summary:")
    print(f"   Total Spent: ₹{summary['total_spent']:,.2f}")
    print(f"   Total Earned: ₹{summary['total_earned']:,.2f}")
    print(f"   Net Balance: ₹{summary['net_balance']:,.2f}")
    print(f"   Savings Rate: {summary['savings_rate']:.1f}%")
    print(f"   Transaction Count: {summary['transaction_count']:,}")
    print(f"   Average Transaction: ₹{summary['avg_transaction']:,.2f}")
    
    print(f"\n📂 All Spending Categories:")
    for category, amount in summary['category_spending'].items():
        percentage = (amount / summary['total_spent']) * 100
        print(f"   {category:20} ₹{amount:>12,.2f} ({percentage:5.1f}%)")
    
    print(f"\n🏪 Top Merchants:")
    for merchant, amount in list(summary['top_merchants'].items())[:10]:
        print(f"   {merchant:30} ₹{amount:>12,.2f}")

def show_help_suggestions():
    """Show suggested questions"""
    print(f"\n💡 SUGGESTED QUESTIONS:")
    print("=" * 40)
    print("💰 Spending Questions:")
    print("   - How much did I spend on travel?")
    print("   - What's my total spending?")
    print("   - Which category do I spend most on?")
    print("   - How much did I spend on healthcare?")
    print("   - What are my top 3 spending categories?")
    
    print(f"\n📈 Financial Health:")
    print("   - What's my savings rate?")
    print("   - How is my financial health?")
    print("   - What's my net balance?")
    print("   - Am I overspending?")
    
    print(f"\n🎯 Investment & Planning:")
    print("   - What investment allocation do you recommend?")
    print("   - How much should I save monthly?")
    print("   - Should I invest in mutual funds?")
    print("   - What's my recommended SIP amount?")
    
    print(f"\n📊 Analysis:")
    print("   - Analyze my spending patterns")
    print("   - What are my spending trends?")
    print("   - How can I improve my budget?")
    print("   - What's my financial score?")
    
    print(f"\n🔍 Specific Categories:")
    print("   - How much did I spend on groceries?")
    print("   - What's my entertainment spending?")
    print("   - How much do I spend on utilities?")
    print("   - What's my education spending?")

if __name__ == "__main__":
    main()
