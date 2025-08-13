#!/bin/bash

# Enhanced M.Tech Financial AI Project - Quick Launcher
# This script sets up and runs the complete demo

echo "🚀 M.TECH FINANCIAL AI PROJECT - QUICK LAUNCHER"
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    echo "Please install pip and try again."
    exit 1
fi

echo "✅ pip found: $(pip3 --version)"

# Install minimal required packages
echo ""
echo "📦 Installing minimal required packages..."
pip3 install pandas numpy faker matplotlib --quiet

if [ $? -eq 0 ]; then
    echo "✅ Packages installed successfully"
else
    echo "⚠️ Some packages may have failed to install, but continuing..."
fi

# Check if demo file exists
if [ ! -f "quick_demo.py" ]; then
    echo "❌ quick_demo.py not found in current directory"
    echo "Please ensure you're in the project directory"
    exit 1
fi

# Run the demo
echo ""
echo "🎯 Starting Enhanced Financial AI Demo..."
echo "========================================="
python3 quick_demo.py

# Check if demo completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Demo completed successfully!"
    echo ""
    echo "📁 Generated files:"
    ls -la *.csv *.json 2>/dev/null || echo "   No output files found"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Review the generated files"
    echo "   2. Check SETUP_AND_RUN_GUIDE.md for full installation"
    echo "   3. Run individual components for deeper exploration"
    echo ""
    echo "✅ Your M.Tech project is ready for enhancement!"
else
    echo "❌ Demo encountered an error"
    echo "💡 Try installing additional dependencies:"
    echo "   pip3 install pandas numpy faker matplotlib seaborn plotly"
    exit 1
fi