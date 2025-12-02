# 1. Run tests (already done ✅)
python3 scripts/test_pipeline.py

# 2. Start pipeline
cd /home/member2/tomoooo/IndonesiaLipReading_GNN
nohup bash scripts/run_storage_efficient_pipeline.sh > pipeline.log 2>&1 &

# 3. Monitor
tail -f pipeline.log

# 4. Check progress anytime
find results/ -name "best.pth" | wc -l  # Should reach 90

3529349