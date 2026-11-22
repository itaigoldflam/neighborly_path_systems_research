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
                # מחלץ את שורת הדלתאות אחרי הנקודתיים
                order_str = line.split(":", 1)[1].strip()
                # מפריד לפי סימן '>', ומוריד רווחים
                order = [x.strip() for x in order_str.split(">")]
                good_orders.append(order)
            elif line.startswith("Bad #"):
                order_str = line.split(":", 1)[1].strip()
                order = [x.strip() for x in order_str.split(">")]
                bad_orders.append(order)
                
    return good_orders, bad_orders

def get_relative_order(full_order, subset):
    """
    Given a full order (list) and a subset of elements (set),
    returns the relative order of just those elements as a tuple.
    Example: full=[A,B,C,D], subset={B,D} -> returns (B,D)
    """
    return tuple(item for item in full_order if item in subset)

def find_forbidden_patterns(good_orders, bad_orders, max_k=4):
    """
    Searches for minimal sub-orders that appear in bad_orders but NEVER in good_orders.
    max_k is the maximum length of the sub-order to check.
    """
    if not good_orders or not bad_orders:
        print("Error: Didn't find enough data in the file.")
        return
        
    # אוסף את כל הדלתאות האפשריות מהסדר הראשון
    all_deltas = set(good_orders[0])
    print(f"Total unique deltas found: {len(all_deltas)}")
    print(f"Analyzing {len(good_orders)} Good orders vs {len(bad_orders)} Bad orders...\n")
    
    forbidden_patterns = set()
    
    # בודק אורכים מ-2 עד max_k
    for k in range(2, max_k + 1):
        print(f"--- Searching for forbidden rules of length {k} ---")
        new_rules_found = 0
        
        # עובר על כל תתי-הקבוצות בגודל k
        for subset in itertools.combinations(all_deltas, k):
            subset_set = set(subset)
            
            # אוסף את כל הסידורים של התת-קבוצה הזו שמופיעים בסדרים הטובים
            good_patterns = set()
            for order in good_orders:
                good_patterns.add(get_relative_order(order, subset_set))
                
            # אוסף את כל הסידורים בזרים הרעים
            bad_patterns = set()
            for order in bad_orders:
                bad_patterns.add(get_relative_order(order, subset_set))
                
            # A forbidden pattern is one that appears in the bad orders but not in the good orders
            forbidden_candidates = bad_patterns - good_patterns
            
            for pattern in forbidden_candidates:
                # Ensure this is a minimal pattern (i.e. no proper sub-pattern is already forbidden)
                is_minimal = True
                for i in range(2, k):
                    for sub_pattern in itertools.combinations(pattern, i):
                        if sub_pattern in forbidden_patterns:
                            is_minimal = False
                            break
                    if not is_minimal:
                        break
                        
                if is_minimal:
                    forbidden_patterns.add(pattern)
                    new_rules_found += 1
                    print(f"RULE: {' > '.join(pattern)}")
                    
        print(f"-> Found {new_rules_found} new minimal rules of length {k}.\n")

if __name__ == "__main__":
    # Make sure the filename matches the file you previously generated
    filename = "invalid_delta_orders_n5.txt" 
    
    if not os.path.exists(filename):
        print(f"Cannot find file: {filename}")
    else:
        good, bad = parse_orders_file(filename)
        # Search for rules up to length 4 (number of involved deltas). You can increase to 5 if wanted.
        find_forbidden_patterns(good, bad, max_k=5)