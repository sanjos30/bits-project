#!/usr/bin/env python3
"""
Test script for the trained LoRA adapter
Demonstrates the trained financial AI model capabilities
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

class FinancialAITester:
    """Test the trained financial AI model"""
    
    def __init__(self, model_path="./models/financial_ai_lora"):
        """Initialize tester"""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        print("🧪 FINANCIAL AI MODEL TESTER")
        print("=" * 50)
        print(f"📱 Device: {self.device}")
        print(f"📂 Model path: {model_path}")
        
    def load_model(self):
        """Load the trained model and tokenizer"""
        print("\n🔄 Loading trained model...")
        
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found at {self.model_path}")
            print("💡 Please run: python train_production_model.py first")
            return False
        
        try:
            # Load base model and tokenizer
            base_model_name = "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_response(self, instruction, input_context="", max_length=200):
        """Generate response for a financial query"""
        # Format input
        prompt = f"### Instruction: {instruction}\n### Input: {input_context}\n### Response:"
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split("### Response:")[-1].strip()
        
        return response
    
    def run_demo_queries(self):
        """Run demo queries to test the model"""
        print("\n🎯 RUNNING DEMO QUERIES")
        print("-" * 50)
        
        # Demo user context
        user_context = "User: John Doe, Age: 30, Profile: young_professional, Monthly Income: ₹85,000"
        
        test_queries = [
            {
                "instruction": "How much did I spend on groceries this month?",
                "input": user_context,
                "category": "Spending Analysis"
            },
            {
                "instruction": "What's my recommended investment allocation?",
                "input": f"{user_context}, Risk Tolerance: Medium",
                "category": "Investment Advice"
            },
            {
                "instruction": "Analyze my spending patterns",
                "input": f"{user_context}, Total Spent: ₹65,000",
                "category": "Pattern Analysis"
            },
            {
                "instruction": "How can I improve my savings rate?",
                "input": f"{user_context}, Current Savings Rate: 15%",
                "category": "Financial Planning"
            },
            {
                "instruction": "Are there any suspicious transactions?",
                "input": f"{user_context}, Recent high-value transaction: ₹50,000",
                "category": "Risk Assessment"
            }
        ]
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[Query {i}/{len(test_queries)}] {query['category']}")
            print(f"🔹 Question: {query['instruction']}")
            print(f"📝 Context: {query['input']}")
            
            # Generate response
            response = self.generate_response(query['instruction'], query['input'])
            
            print(f"🤖 AI Response: {response}")
            print("-" * 50)
            
            results.append({
                "query": query['instruction'],
                "context": query['input'],
                "response": response,
                "category": query['category']
            })
        
        return results
    
    def interactive_mode(self):
        """Interactive mode for testing"""
        print("\n💬 INTERACTIVE MODE")
        print("-" * 50)
        print("Ask financial questions! Type 'quit' to exit.")
        
        user_context = "User: Demo User, Age: 35, Profile: family_oriented, Monthly Income: ₹120,000"
        print(f"📝 Using context: {user_context}")
        
        while True:
            query = input("\n🔹 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print("🤖 AI Response:", end=" ")
            response = self.generate_response(query, user_context)
            print(response)
    
    def benchmark_performance(self):
        """Benchmark model performance"""
        print("\n📊 PERFORMANCE BENCHMARK")
        print("-" * 50)
        
        import time
        
        test_query = "What's my total spending this month?"
        test_context = "User financial data available"
        
        # Warmup
        self.generate_response(test_query, test_context)
        
        # Benchmark
        times = []
        for _ in range(5):
            start_time = time.time()
            response = self.generate_response(test_query, test_context)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        
        print(f"⏱️ Average Response Time: {avg_time:.2f} seconds")
        print(f"🎯 Performance Target: <2 seconds")
        print(f"📊 Status: {'✅ PASS' if avg_time < 2.0 else '⚠️ NEEDS OPTIMIZATION'}")
        
        return avg_time
    
    def save_test_results(self, results, performance_time):
        """Save test results"""
        test_summary = {
            "model_path": self.model_path,
            "test_timestamp": torch.datetime.now().isoformat(),
            "performance": {
                "average_response_time": performance_time,
                "target_met": performance_time < 2.0
            },
            "test_results": results,
            "model_info": {
                "base_model": "microsoft/DialoGPT-medium",
                "adapter_type": "LoRA",
                "training_data": "Production financial transactions"
            }
        }
        
        with open('MODEL_TEST_RESULTS.json', 'w') as f:
            json.dump(test_summary, f, indent=2)
        
        print(f"📄 Test results saved to: MODEL_TEST_RESULTS.json")

def main():
    """Main testing pipeline"""
    print("🎯 FINANCIAL AI MODEL TESTING")
    print("=" * 50)
    
    # Initialize tester
    tester = FinancialAITester()
    
    # Load model
    if not tester.load_model():
        print("\n💡 To train the model first, run:")
        print("   python train_production_model.py")
        return
    
    # Run demo queries
    results = tester.run_demo_queries()
    
    # Benchmark performance
    performance_time = tester.benchmark_performance()
    
    # Save results
    tester.save_test_results(results, performance_time)
    
    # Interactive mode option
    print("\n🎯 Would you like to try interactive mode? (y/n)")
    if input().lower().startswith('y'):
        tester.interactive_mode()
    
    print("\n✅ Model testing complete!")
    print("🎓 Your financial AI model is ready for M.Tech evaluation!")

if __name__ == "__main__":
    main()