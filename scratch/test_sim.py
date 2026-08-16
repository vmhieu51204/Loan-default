import sys, os
sys.path.insert(0, os.path.abspath('.'))

from src.scorecard.simulation_engine import get_simulation_engine
engine = get_simulation_engine()
m = engine.evaluate_cutoff(480, lgd=3.73, interest_margin=0.15)
print('Evaluation at Cutoff 480:')
print('ApprovalRate:', f"{m['approval_rate']:.2f}")
print('ApprovedCount:', f"{m['approved_count']/1000:.0f}K ({m['approved_count']:,})")
print('ExpectedDefaults:', f"{m['expected_defaults']/1000:.2f}K ({m['expected_defaults']:.2f})")
print('ExpectedLoss:', f"{m['expected_loss']/1e9:.2f}bn (${m['expected_loss']:,.2f})")
print('ExpectedProfit:', f"{m['expected_profit']/1e9:.2f}bn (${m['expected_profit']:,.2f})")

df_b, totals = engine.get_score_bucket_breakdown(lgd=3.73)
print('\nScore Buckets:')
for _, r in df_b.iterrows():
    print(r['score_bucket'], f"count={r['Bucket_Count']:,}", f"avg_pd={r['Bucket_Avg_PD']:.2%}", f"loss=${r['Bucket_Expected_Loss']:,.2f}")
print(f"Total: {totals['Total_Count']:,} | {totals['Total_Avg_PD']:.2%} | ${totals['Total_Expected_Loss']:,.2f}")
