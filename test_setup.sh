#!/bin/bash
# Quick test script for I2V dataset setup

echo "=========================================="
echo "I2V Dataset Setup - Quick Test"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "setup_i2v_datasets.py" ]; then
    echo "❌ Error: Please run this script from the i2v directory"
    exit 1
fi

echo "1. Checking Python version..."
python3 --version
echo ""

echo "2. Checking dependencies..."
python3 setup_i2v_datasets.py --check_deps
echo ""

echo "3. Checking existing data..."
echo "   - first_frames/: $(ls first_frames/*.jpg 2>/dev/null | wc -l) frames"
echo "   - DL3DV videos: $(find datasets/dl3dv -name '*.mp4' 2>/dev/null | wc -l) videos"
echo "   - RealEstate videos: $(ls datasets/realestate10k/videos/*.mp4 2>/dev/null | wc -l) videos"
echo ""

if [ -f "generated_prompts.json" ]; then
    echo "   - generated_prompts.json: ✓ exists"
    prompt_count=$(python3 -c "import json; print(len(json.load(open('generated_prompts.json'))))" 2>/dev/null || echo "0")
    echo "     Prompts: $prompt_count entries"
else
    echo "   - generated_prompts.json: ✗ not found"
fi
echo ""

echo "=========================================="
echo "Quick Test Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Install missing dependencies if any"
echo "  2. Run: python setup_i2v_datasets.py"
echo "  3. Check USAGE.md for detailed instructions"
echo ""
