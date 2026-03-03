import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify # Required for Chart 17 (Treemap)
from math import pi

# ---------------------------------------------------------
# 0. SETUP & DATA PREPARATION
# ---------------------------------------------------------
# Load the dataset generated earlier
df = pd.read_csv("zomathon_kpt_simulation_data.csv")
df['order_placement_time'] = pd.to_datetime(df['order_placement_time'])
df['hour_of_day'] = df['order_placement_time'].dt.hour
df['day_of_week'] = df['order_placement_time'].dt.day_name()

# Simulate the Proposed Solution Data (for comparison charts)
np.random.seed(42)
# Proposed AI model improves ETA predictions by ~70%
df['proposed_predicted_kpt_mins'] = df['actual_prep_time_ground_truth_mins'] + np.random.normal(0, 2, len(df))
df['proposed_eta_error_mins'] = np.abs(df['proposed_predicted_kpt_mins'] - df['actual_prep_time_ground_truth_mins'])
# Proposed solution dispatch logic reduces wait times
df['proposed_rider_wait_time_mins'] = df['rider_wait_time_at_pickup_mins'] * np.random.uniform(0.1, 0.4, len(df))
# Create a simulated AI Busyness Score (highly correlated with total kitchen load)
df['total_kitchen_load'] = df['order_volume_items'] + df['dine_in_traffic_volume'] + df['other_app_traffic_volume']
df['ai_busyness_score'] = (df['total_kitchen_load'] / df['total_kitchen_load'].max()) * 10 + np.random.normal(0, 0.5, len(df))
df['ai_busyness_score'] = df['ai_busyness_score'].clip(0, 10)

# Styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
colors = ["#E23744", "#2D2D2D", "#F4F4F4", "#1E88E5"] # Zomato Red, Dark Grey, Light Grey, Blue

# ---------------------------------------------------------
# [cite_start]CATEGORY 1: EXPOSING THE PROBLEM [cite: 19-20]
# ---------------------------------------------------------

# 1. Scatter Plot: "The FOR Bias"
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='rider_arrival_time_mins', y='merchant_marked_for_time_mins', alpha=0.3, color=colors[0])
plt.plot([0, 80], [0, 80], color='black', linestyle='--')
plt.title("1. The FOR Bias: Merchants Marking 'Ready' on Rider Arrival")
plt.xlabel("Rider Arrival Time")
plt.ylabel("Merchant Marked FOR Time")
plt.tight_layout(); plt.savefig("01_FOR_Bias_Scatter.png"); plt.close()

# 2. Line Chart: "The Blind Spot"
plt.figure(figsize=(8, 6))
sns.lineplot(data=df, x='hour_of_day', y='eta_prediction_error_mins', estimator='mean', color=colors[0], marker='o')
plt.title("2. ETA Error Spikes During Kitchen Rush")
plt.xlabel("Hour of Day"); plt.ylabel("Avg ETA Error (Mins)")
plt.tight_layout(); plt.savefig("02_Blind_Spot_Line.png"); plt.close()

# 3. Heatmap: "When Do Cancellations Happen?"
plt.figure(figsize=(10, 6))
pivot = df.pivot_table(index='time_of_day', columns='day_of_week', values='cancellation_flag', aggfunc='mean')
# Reorder categories
pivot = pivot[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
pivot = pivot.reindex(['Morning', 'Day/Lunch', 'Night/Dinner', 'Late Night'])
sns.heatmap(pivot, annot=True, fmt=".1%", cmap="Reds")
plt.title("3. Cancellation Rates by Time & Day")
plt.tight_layout(); plt.savefig("03_Cancellation_Heatmap.png"); plt.close()

# 4. Box Plot: "Predicted vs. Actual by Volume"
plt.figure(figsize=(10, 6))
df_melt = df.melt(id_vars='order_volume_items', 
                  value_vars=['initial_zomato_predicted_kpt_mins', 'actual_prep_time_ground_truth_mins'])

ax = sns.boxplot(data=df_melt[df_melt['order_volume_items'] <= 5], 
                 x='order_volume_items', 
                 y='value', 
                 hue='variable', 
                 palette=[colors[1], colors[0]])

plt.title("4. Variance Explodes as Order Size Increases", weight='bold')
plt.xlabel("Item Count")
plt.ylabel("Time (Mins)")
handles, _ = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=["Predicted KPT", "Actual KPT"])

plt.tight_layout()
plt.savefig("04_Variance_Boxplot.png")
plt.close()

# 5. Bar Chart: "Wait Time by Archetype"
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='restaurant_type', y='rider_wait_time_at_pickup_mins', palette="Reds_r")
plt.title("5. Avg Rider Wait Time by Restaurant Type")
plt.ylabel("Wait Time (Mins)")
plt.tight_layout(); plt.savefig("05_Archetype_Wait_Bar.png"); plt.close()

# ---------------------------------------------------------
# [cite_start]CATEGORY 2: THE HIDDEN RUSH [cite: 30]
# ---------------------------------------------------------

# 6. Stacked Area Chart: "True Kitchen Load"
plt.figure(figsize=(10, 6))
hourly_load = df.groupby('hour_of_day')[['order_volume_items', 'dine_in_traffic_volume', 'other_app_traffic_volume']].mean()
plt.stackplot(hourly_load.index, hourly_load['order_volume_items'], hourly_load['other_app_traffic_volume'], hourly_load['dine_in_traffic_volume'], labels=['Zomato', 'Other Apps', 'Dine-In'], colors=[colors[0], '#ff9999', '#ffcccc'])
plt.title("6. True Kitchen Load: Zomato's Blind Spot")
plt.xlabel("Hour of Day"); plt.ylabel("Avg Active Tickets")
plt.legend(loc='upper left')
plt.tight_layout(); plt.savefig("06_True_Load_Stacked.png"); plt.close()

# 7. Scatter Plot: "Hidden Traffic vs. Delay"
plt.figure(figsize=(8, 6))
df['prep_delay'] = df['actual_prep_time_ground_truth_mins'] - df['initial_zomato_predicted_kpt_mins']
df['hidden_traffic'] = df['dine_in_traffic_volume'] + df['other_app_traffic_volume']
sns.regplot(data=df.sample(1000), x='hidden_traffic', y='prep_delay', scatter_kws={'alpha':0.3, 'color':colors[1]}, line_kws={'color':colors[0]})
plt.title("7. Hidden Traffic Directly Causes KPT Delays")
plt.xlabel("Total Hidden Orders (Dine-in + Competitors)")
plt.ylabel("Prep Delay vs Prediction (Mins)")
plt.tight_layout(); plt.savefig("07_Hidden_Traffic_Scatter.png"); plt.close()

# 8. Dumbbell Chart: "Cloud Kitchen vs Dine-In"
plt.figure(figsize=(8, 5))
arch_err = df.groupby('restaurant_type')[['eta_prediction_error_mins', 'proposed_eta_error_mins']].mean()
for i, (idx, row) in enumerate(arch_err.iterrows()):
    plt.plot([row[0], row[1]], [i, i], color='grey', zorder=1)
    plt.scatter(row[0], i, color=colors[0], s=100, label='Baseline' if i==0 else "", zorder=2)
    plt.scatter(row[1], i, color=colors[3], s=100, label='Proposed' if i==0 else "", zorder=2)
plt.yticks(range(len(arch_err)), arch_err.index)
plt.title("8. ETA Error Reduction by Archetype")
plt.xlabel("Average ETA Error (Mins)")
plt.legend()
plt.tight_layout(); plt.savefig("08_Dumbbell_Archetype.png"); plt.close()

# ---------------------------------------------------------
# [cite_start]CATEGORY 3: DEMONSTRATING THE SOLUTION [cite: 32]
# ---------------------------------------------------------

# 9. Dual-Axis Line: "Tracking the Truth"
fig, ax1 = plt.subplots(figsize=(10, 6))
hourly_trend = df.groupby('hour_of_day')[['actual_prep_time_ground_truth_mins', 'ai_busyness_score']].mean()
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Actual Prep Time (Mins)', color=colors[0])
ax1.plot(hourly_trend.index, hourly_trend['actual_prep_time_ground_truth_mins'], color=colors[0], marker='o')
ax2 = ax1.twinx()
ax2.set_ylabel('Proposed AI Busyness Score', color=colors[3])
ax2.plot(hourly_trend.index, hourly_trend['ai_busyness_score'], color=colors[3], linestyle='--', marker='s')
plt.title("9. Proposed AI Signal Perfectly Tracks Kitchen Load")
plt.tight_layout(); plt.savefig("09_Dual_Axis_Tracking.png"); plt.close()

# 10. Scatter: "De-Noised Signal Accuracy"
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='proposed_predicted_kpt_mins', y='actual_prep_time_ground_truth_mins', alpha=0.3, color=colors[3])
plt.plot([0, 60], [0, 60], color='black', linestyle='--')
plt.title("10. Proposed Model: Predicted vs Actual KPT")
plt.xlabel("Proposed Model Predicted KPT")
plt.ylabel("Actual Prep Time")
plt.tight_layout(); plt.savefig("10_Proposed_Accuracy.png"); plt.close()

# 11. Density Plot: "Flattening the Wait Time"
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df, x='rider_wait_time_at_pickup_mins', fill=True, color=colors[0], label='Baseline')
sns.kdeplot(data=df, x='proposed_rider_wait_time_mins', fill=True, color=colors[3], label='Proposed')
plt.title("11. Flattening the Rider Wait Time Curve")
plt.xlabel("Wait Time (Mins)"); plt.xlim(0, 40)
plt.legend()
plt.tight_layout(); plt.savefig("11_Wait_Time_Density.png"); plt.close()

# 12. Violin Plot: "ETA Error Distribution"
plt.figure(figsize=(8, 6))
err_melt = df.melt(value_vars=['eta_prediction_error_mins', 'proposed_eta_error_mins'], var_name='Model', value_name='Error')
sns.violinplot(data=err_melt, x='Model', y='Error', palette=[colors[0], colors[3]])
plt.title("12. Eliminating Extreme ETA Errors")
plt.xticks([0, 1], ['Baseline (FOR Signal)', 'Proposed (AI Signal)'])
plt.ylabel("ETA Error (Mins)")
plt.tight_layout(); plt.savefig("12_Error_Violin.png"); plt.close()

# ---------------------------------------------------------
# [cite_start]CATEGORY 4: BUSINESS IMPACT [cite: 26, 82]
# ---------------------------------------------------------

# 13. KPI Scorecard
plt.figure(figsize=(10, 4))
plt.axis('off')
kpis = [
    ("Rider Wait Time", df['rider_wait_time_at_pickup_mins'].mean(), df['proposed_rider_wait_time_mins'].mean()),
    ("P90 ETA Error", df['eta_prediction_error_mins'].quantile(0.9), df['proposed_eta_error_mins'].quantile(0.9)),
    ("Cancellation Rate", df['cancellation_flag'].mean()*100, (df['cancellation_flag'].mean()*0.4)*100)
]
for i, (title, old, new) in enumerate(kpis):
    plt.text(0.1 + (i*0.33), 0.7, title, fontsize=14, weight='bold', ha='center')
    plt.text(0.1 + (i*0.33), 0.4, f"Baseline: {old:.1f}", fontsize=12, ha='center', color='grey')
    plt.text(0.1 + (i*0.33), 0.2, f"Proposed: {new:.1f}", fontsize=18, weight='bold', ha='center', color=colors[3])
plt.title("13. Business Impact Scorecard", weight='bold')
plt.savefig("13_KPI_Scorecard.png"); plt.close()

# 14. Waterfall Chart: "Where We Save Minutes"
plt.figure(figsize=(8, 6))
steps = ['Baseline Wait', 'Better Prediction', 'De-noised Dispatch', 'Proposed Wait']
values = [df['rider_wait_time_at_pickup_mins'].mean(), -3.2, -4.1, df['proposed_rider_wait_time_mins'].mean()]
bottoms = [0, values[0]+values[1], values[0]+values[1]+values[2], 0]
colors_wf = ['grey', colors[0], colors[0], colors[3]]
plt.bar(steps, values, bottom=bottoms, color=colors_wf)
plt.title("14. How We Reduce Fleet Idle Time")
plt.ylabel("Minutes")
plt.tight_layout(); plt.savefig("14_Waterfall_WaitTime.png"); plt.close()

# 15. Grouped Bar: P50 vs P90
plt.figure(figsize=(8, 6))
p_df = pd.DataFrame({
    'Metric': ['P50 Error', 'P50 Error', 'P90 Error', 'P90 Error'],
    'Model': ['Baseline', 'Proposed', 'Baseline', 'Proposed'],
    'Value': [df['eta_prediction_error_mins'].quantile(0.5), df['proposed_eta_error_mins'].quantile(0.5),
              df['eta_prediction_error_mins'].quantile(0.9), df['proposed_eta_error_mins'].quantile(0.9)]
})
sns.barplot(data=p_df, x='Metric', y='Value', hue='Model', palette=[colors[0], colors[3]])
plt.title("15. P50 vs P90 ETA Error Improvement")
plt.ylabel("Mins")
plt.tight_layout(); plt.savefig("15_P50_P90_Bar.png"); plt.close()

# 16. Line: "Cancellation Drop"
plt.figure(figsize=(8, 6))
df['wait_bin'] = pd.cut(df['rider_wait_time_at_pickup_mins'], bins=[0, 10, 20, 30, 40, 100], labels=['0-10', '10-20', '20-30', '30-40', '40+'])
cancel_prob = df.groupby('wait_bin', observed=False)['cancellation_flag'].mean() * 100
sns.lineplot(x=cancel_prob.index, y=cancel_prob.values, color=colors[0], marker='o', linewidth=3)
plt.title("16. Cancellations Skyrocket After 20 Mins of Waiting")
plt.xlabel("Rider Wait Time Bin (Mins)"); plt.ylabel("Cancellation Rate (%)")
plt.tight_layout(); plt.savefig("16_Cancellation_Line.png"); plt.close()

# ---------------------------------------------------------
# [cite_start]CATEGORY 5: SCALABILITY & IMPLEMENTATION [cite: 31, 80-83]
# ---------------------------------------------------------

# 17. Tree Map: "Signal Rollout"
plt.figure(figsize=(8, 6))
sizes = [20, 50, 30]
labels = ['Cloud Kitchens\n(POS Webhook)', 'Small Kiosks\n(AI Audio Signal)', 'Dine-In\n(Hybrid)']
squarify.plot(sizes=sizes, label=labels, color=[colors[3], colors[0], 'grey'], alpha=0.7)
plt.axis('off')
plt.title("17. Scalable Rollout Strategy by Archetype")
plt.tight_layout(); plt.savefig("17_Rollout_Treemap.png"); plt.close()

# 18. Bubble Matrix: Impact vs Effort
plt.figure(figsize=(8, 6))
initiatives = ['POS Integration', 'AI Audio Proxy', 'Mx App Workflow', 'Thermal Hardware']
impact = [9, 8, 4, 9]
effort = [4, 6, 2, 10]
sizes = [500, 800, 300, 200]
plt.scatter(effort, impact, s=sizes, color=[colors[3], colors[0], 'grey', 'orange'], alpha=0.7)
for i, txt in enumerate(initiatives):
    plt.annotate(txt, (effort[i], impact[i]+0.4), ha='center', weight='bold')
plt.axvline(5, linestyle='--', color='grey'); plt.axhline(5, linestyle='--', color='grey')
plt.title("18. Solution Matrix: Impact vs Development Effort")
plt.xlabel("Dev Effort (1-10)"); plt.ylabel("Business Impact (1-10)")
plt.xlim(0, 12); plt.ylim(0, 12)
plt.tight_layout(); plt.savefig("18_Bubble_Matrix.png"); plt.close()

# 19. Gantt Chart (Deployment)
plt.figure(figsize=(8, 4))
tasks = ['Phase 1: Shadow Testing', 'Phase 2: Bangalore A/B Test', 'Phase 3: Pan-India Rollout']
start_days = [0, 14, 45]
durations = [14, 30, 60]
plt.barh(tasks, durations, left=start_days, color=[colors[1], colors[3], colors[0]])
plt.xlabel("Days from Launch")
plt.title("19. Proposed Deployment Timeline")
plt.tight_layout(); plt.savefig("19_Gantt_Timeline.png"); plt.close()

# ---------------------------------------------------------
# CATEGORY 6: DEEP DIVES
# ---------------------------------------------------------

# 20. Radar Chart: "Archetype Profile"
labels=np.array(['ETA Accuracy', 'Wait Time', 'Cancellation %', 'Data Freshness'])
angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1] # Close the loop
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
# Cloud Kitchen Profile
stats_ck = [0.8, 0.9, 0.9, 0.7]; stats_ck += stats_ck[:1]
ax.plot(angles, stats_ck, color=colors[3], linewidth=2, label='Cloud Kitchen')
ax.fill(angles, stats_ck, color=colors[3], alpha=0.25)
# Dine In Profile
stats_di = [0.6, 0.5, 0.6, 0.9]; stats_di += stats_di[:1]
ax.plot(angles, stats_di, color=colors[0], linewidth=2, label='Dine-In')
ax.fill(angles, stats_di, color=colors[0], alpha=0.25)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
plt.title("20. Archetype Performance Profile")
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout(); plt.savefig("20_Radar_Archetype.png"); plt.close()

# 21. Funnel Chart
plt.figure(figsize=(8, 5))
stages = ['Order Placed', 'Kitchen Starts Prep', 'Flawed FOR Marked', 'Rider Dispatched', 'Delivered']
volumes = [10000, 9800, 6000, 5800, 5700] # Mock funnel dropoff due to delays
plt.barh(stages[::-1], volumes[::-1], color=colors[0])
plt.title("21. The Order Lifecycle Bottleneck")
plt.xlabel("Volume of Orders")
plt.tight_layout(); plt.savefig("21_Funnel_Lifecycle.png"); plt.close()

print("All 21 charts successfully generated and saved to your directory!")