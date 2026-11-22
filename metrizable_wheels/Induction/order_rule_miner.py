
import itertools
import os

def parse_orders_file(filename):
    """Parses the text file and extracts good and bad orders."""
    good_orders = []
    bad_orders = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Good #"):
                order_str = line.split(":", 1)[1].strip()
                order = [x.strip() for x in order_str.split(">")]
                good_orders.append(order)
            elif line.startswith("Bad #"):
                order_str = line.split(":", 1)[1].strip()
                order = [x.strip() for x in order_str.split(">")]
                bad_orders.append(order)
                
    return good_orders, bad_orders

def mine_generic_rules(filename, max_rule_size=4):
    if not os.path.exists(filename):
        print(f"Cannot find file: {filename}")
        return
        
    good_orders, bad_orders = parse_orders_file(filename)
    if not good_orders or not bad_orders:
        print("Not enough data to mine rules (need both Good and Bad orders).")
        return
        
    all_deltas = set(good_orders[0])
    num_good = len(good_orders)
    
    print(f"Analyzing {num_good} Good orders vs {len(bad_orders)} Bad orders...")
    print(f"Total unique deltas: {len(all_deltas)}\n")
    
    # 1. Generate all possible pairs (X > Y)
    all_pairs = list(itertools.permutations(all_deltas, 2))
    
    # 2. Filter for "Safe Features" (Pairs that NEVER appear in Bad orders)
    safe_features = []
    for x, y in all_pairs:
        # Check if x comes before y in ANY bad order
        is_safe = True
        for order in bad_orders:
            if order.index(x) < order.index(y):
                is_safe = False
                break
        if is_safe:
            safe_features.append((x, y))
            
    print(f"Found {len(safe_features)} completely 'safe' pairwise rules (Never happen in Bad orders).")
    
    # 3. Calculate coverage for each safe feature
    # coverage[feature] = set of indices of good orders that satisfy this feature
    feature_coverage = {}
    for feature in safe_features:
        x, y = feature
        covered_indices = set()
        for i, order in enumerate(good_orders):
            if order.index(x) < order.index(y):
                covered_indices.add(i)
        feature_coverage[feature] = covered_indices

    target_coverage = set(range(num_good))
    
    print("Searching for minimal generic rule...\n")
    
    # 4. Search for the smallest combination of features that covers ALL Good orders
    for k in range(1, max_rule_size + 1):
        for combo in itertools.combinations(safe_features, k):
            # Calculate union of coverages
            combined_coverage = set()
            for feature in combo:
                combined_coverage.update(feature_coverage[feature])
                
            # Did we cover everything?
            if len(combined_coverage) == num_good:
                print("=" * 60)
                print(f"🎯 FOUND PERFECT RULE (Size {k}):")
                print("=" * 60)
                
                # Format positive rule
                good_conditions = " OR ".join([f"({x} > {y})" for x, y in combo])
                print(f"\n✅ An order is GOOD (Solvable) if and only if:")
                print(f"   {good_conditions}")
                
                # Format negative rule (contrapositive)
                bad_conditions = " AND ".join([f"({y} > {x})" for x, y in combo])
                print(f"\n❌ An order is BAD (Infeasible) if and only if:")
                print(f"   {bad_conditions}")
                print("=" * 60 + "\n")
                
                # We return immediately because we want the MINIMAL rule
                return
                
    print(f"Could not find a perfect rule using up to {max_rule_size} combined pairs.")
    print("The rule might be more complex, or involve triplets.")

if __name__ == "__main__":
    # החלף פה את שם הקובץ לקובץ שאתה רוצה לנתח!
    FILENAME = "invalid_delta_orders_n5.txt"
    mine_generic_rules(FILENAME, max_rule_size=4)