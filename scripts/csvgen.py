import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

# Configuration
NUM_ORDERS = 10000

def generate_zomathon_dataset(num_orders):
    # 1. Base Identifiers & Context
    order_ids = [f"ORD_{i:06d}" for i in range(num_orders)]
    
    # Randomly generate order placement times over a 30-day period
    start_date = datetime(2024, 2, 1)
    placement_times = [start_date + timedelta(days=np.random.randint(0, 30), 
                                              minutes=np.random.randint(0, 1440)) for _ in range(num_orders)]
    
    # Derived Temporal Features
    days_of_week = [t.weekday() for t in placement_times] # 0=Mon, 6=Sun
    is_weekend = [1 if d >= 5 else 0 for d in days_of_week]
    
    def get_time_of_day(hour):
        if 6 <= hour < 11: return 'Morning'
        elif 11 <= hour < 16: return 'Day/Lunch'
        elif 16 <= hour < 22: return 'Night/Dinner'
        else: return 'Late Night'
        
    time_of_day = [get_time_of_day(t.hour) for t in placement_times]
    
    # 2. Restaurant & Order Characteristics
    restaurant_types = np.random.choice(['Dine-in Restaurant', 'Cloud Kitchen', 'Quick Service/Kiosk'], size=num_orders, p=[0.5, 0.3, 0.2])
    order_volume_items = np.random.randint(1, 10, size=num_orders) # Number of items ordered
    is_specialty_item = np.random.choice([0, 1], size=num_orders, p=[0.7, 0.3]) # 1 = ordering the specialty
    
    # 3. The "Hidden" Kitchen Traffic (What Zomato CANNOT see)
    # Dine-in traffic only exists for Dine-in restaurants, spiking during lunch/dinner and weekends
    dine_in_traffic = []
    other_delivery_traffic = [] # Swiggy, direct orders, etc.
    
    for i in range(num_orders):
        hour = placement_times[i].hour
        is_rush = hour in [13, 14, 19, 20, 21]
        traffic_multiplier = 2.0 if is_rush else 1.0
        traffic_multiplier += 0.5 if is_weekend[i] else 0
        
        if restaurant_types[i] == 'Dine-in Restaurant':
            dine_in_traffic.append(int(np.random.randint(5, 30) * traffic_multiplier))
            other_delivery_traffic.append(int(np.random.randint(2, 15) * traffic_multiplier))
        elif restaurant_types[i] == 'Cloud Kitchen':
            dine_in_traffic.append(0) # Cloud kitchens have no dine-in
            other_delivery_traffic.append(int(np.random.randint(10, 40) * traffic_multiplier)) # Heavy multi-app reliance
        else:
            dine_in_traffic.append(int(np.random.randint(0, 10) * traffic_multiplier))
            other_delivery_traffic.append(int(np.random.randint(0, 5) * traffic_multiplier))
            
    # 4. Processing Times & Ground Truth (in minutes relative to placement time)
    # Zomato's flawed prediction only looks at item count and ignores hidden traffic
    predicted_kpt_mins = order_volume_items * 4.0 + np.random.normal(0, 2, num_orders)
    
    # Actual prep time considers everything
    actual_prep_time_mins = []
    for i in range(num_orders):
        base_prep = order_volume_items[i] * 4.0
        # Specialty items take less time (batch prepared)
        if is_specialty_item[i]:
            base_prep *= 0.7 
        # Hidden traffic delays the kitchen
        rush_delay = (dine_in_traffic[i] + other_delivery_traffic[i]) * 0.3
        actual_prep_time_mins.append(round(base_prep + rush_delay, 1))
        
    # 5. Rider & Flawed Signals
    rider_arrival_time_mins = predicted_kpt_mins + np.random.normal(2, 5, num_orders) # Rider arrives based on predicted KPT
    merchant_marked_for_mins = []
    restaurant_closed_after_arrival = []
    
    for i in range(num_orders):
        actual = actual_prep_time_mins[i]
        rider_arr = rider_arrival_time_mins[i]
        total_rush = dine_in_traffic[i] + other_delivery_traffic[i]
        
        # Flawed FOR Logic: Rider-influenced marking
        if total_rush > 20 and rider_arr > actual:
            # Too busy to press button; pressed when rider arrives and asks
            merchant_marked_for_mins.append(rider_arr + np.random.uniform(0.5, 2.0))
        else:
            # Normal marking, slightly after actual prep
            merchant_marked_for_mins.append(actual + np.random.uniform(0.0, 3.0))
            
        # Edge case: Restaurant closed after arrival (happens mostly late night)
        if time_of_day[i] == 'Late Night' and rider_arr > actual + 30:
            restaurant_closed_after_arrival.append(np.random.choice([0, 1], p=[0.95, 0.05]))
        else:
            restaurant_closed_after_arrival.append(0)

    # Dispatch time is when the food is ready AND the rider is there
    dispatch_time_mins = [max(actual_prep_time_mins[i], rider_arrival_time_mins[i]) for i in range(num_orders)]
    delivery_time_mins = [dispatch_time_mins[i] + np.random.randint(10, 30) for i in range(num_orders)]

    # 6. Compile DataFrame
    df = pd.DataFrame({
        'order_id': order_ids,
        'order_placement_time': placement_times,
        'time_of_day': time_of_day,
        'is_weekend': is_weekend,
        'restaurant_type': restaurant_types,
        'is_specialty_item': is_specialty_item,
        'order_volume_items': order_volume_items,
        'dine_in_traffic_volume': dine_in_traffic,
        'other_app_traffic_volume': other_delivery_traffic,
        # Our target feature to fix
        'initial_zomato_predicted_kpt_mins': np.round(predicted_kpt_mins, 1),
        'actual_prep_time_ground_truth_mins': actual_prep_time_mins,
        'rider_arrival_time_mins': np.round(rider_arrival_time_mins, 1),
        # The flawed signal Zomato currently uses
        'merchant_marked_for_time_mins': np.round(merchant_marked_for_mins, 1),
        'dispatch_time_mins': np.round(dispatch_time_mins, 1),
        'delivery_time_mins': delivery_time_mins,
        'restaurant_closed_after_arrival': restaurant_closed_after_arrival
    })

    # 7. Calculate Hackathon Success Metrics
    df['rider_wait_time_at_pickup_mins'] = np.maximum(0, df['actual_prep_time_ground_truth_mins'] - df['rider_arrival_time_mins'])
    df['eta_prediction_error_mins'] = np.abs(df['initial_zomato_predicted_kpt_mins'] - df['actual_prep_time_ground_truth_mins'])
    
    # Cancellation Logic: High wait time + high error = cancellation
    df['cancellation_flag'] = np.where(
        (df['rider_wait_time_at_pickup_mins'] > 20) | (df['restaurant_closed_after_arrival'] == 1), 1, 0
    )

    return df

# Generate and view data
dataset = generate_zomathon_dataset(NUM_ORDERS)
dataset.to_csv("zomathon_kpt_simulation_data.csv", index=False)
print(dataset.head())