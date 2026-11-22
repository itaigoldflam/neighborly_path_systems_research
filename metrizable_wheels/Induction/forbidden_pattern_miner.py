import itertools
import os

def parse_orders_file(filename):
    """
    קורא קובץ תוצאות קיים (עם בלוקים של Good ו-Bad)
    ומחלץ את הסדרים עבור כל מערכת R בנפרד.
    """
    systems_data = {}
    current_R = None
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("SYSTEM: R ="):
                current_R = line.split("=")[1].strip()
                systems_data[current_R] = {'good': [], 'bad': []}
            elif line.startswith("Good #") and current_R:
                order_str = line.split(":", 1)[1].strip()
                order = [x.strip() for x in order_str.split(">")]
                systems_data[current_R]['good'].append(order)
            elif line.startswith("Bad #") and current_R:
                order_str = line.split(":", 1)[1].strip()
                order = [x.strip() for x in order_str.split(">")]
                systems_data[current_R]['bad'].append(order)
                
    return systems_data

def order_to_pairs(order):
    """הופך סדר מלא (רשימה) לקבוצה של כל הזוגות (X > Y) שמתקיימים בו"""
    pairs = set()
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            pairs.add((order[i], order[j]))
    return frozenset(pairs)

def mine_forbidden_patterns(good_orders, bad_orders, max_pattern_size=3):
    """
    מחפש קבוצה מינימלית של זוגות (למשל A>B וגם B>C)
    שמופיעה ברעים, אבל לעולם לא מופיעה בטובים.
    """
    if not good_orders or not bad_orders:
        return []
        
    print(f"  * Analyzing {len(good_orders)} Good vs {len(bad_orders)} Bad orders...")
    
    # ממיר כל סדר לקבוצת הזוגות שבו (כדי לבדוק הכלה במהירות)
    good_sets = [order_to_pairs(o) for o in good_orders]
    bad_sets = [order_to_pairs(o) for o in bad_orders]
    
    all_possible_pairs = list(bad_sets[0]) # כל זוג אפשרי של דלתאות
    
    forbidden_patterns = []
    
    # נחפש תבניות בגודל 1 (זוג בודד), 2 (שני זוגות = שלישייה או שני צלבים), ו-3
    for size in range(1, max_pattern_size + 1):
        print(f"  * Scanning for fatal combinations of size {size}...")
        
        for candidate_pattern in itertools.combinations(all_possible_pairs, size):
            candidate_set = set(candidate_pattern)
            
            # בדיקה: האם התבנית הזו כבר מכילה תבנית קטנה יותר שאסורה? (מוודא מינימליות)
            is_minimal = True
            for known_forbidden in forbidden_patterns:
                if set(known_forbidden).issubset(candidate_set):
                    is_minimal = False
                    break
            if not is_minimal:
                continue
                
            # בדיקה קריטית: האם התבנית מופיעה באפילו סדר **טוב** אחד?
            appears_in_good = False
            for g_set in good_sets:
                if candidate_set.issubset(g_set):
                    appears_in_good = True
                    break
                    
            if not appears_in_good:
                # התבנית הזו מעולם לא הצליחה לפתור את המערכת!
                # נוודא שהיא בכלל קיימת בתוך קבוצת הרעים (כדי שתהיה רלוונטית)
                for b_set in bad_sets:
                    if candidate_set.issubset(b_set):
                        forbidden_patterns.append(candidate_pattern)
                        break

    return forbidden_patterns

def analyze_file(filename):
    if not os.path.exists(filename):
        print(f"File '{filename}' not found.")
        return
        
    systems_data = parse_orders_file(filename)
    
    for R, data in systems_data.items():
        if not data['bad']:
            continue # סדר בטוח, מדלגים
            
        print(f"\n{'='*60}")
        print(f"MINING FORBIDDEN PATTERNS FOR SYSTEM: R = {R}")
        print(f"{'='*60}")
        
        forbidden = mine_forbidden_patterns(data['good'], data['bad'], max_pattern_size=3)
        
        if not forbidden:
            print("\nCould not find minimal forbidden patterns up to size 3.")
            print("The rules governing this system might be exceedingly complex.")
            continue
            
        print("\n🔥 FATAL RULES FOUND! An order is INFEASIBLE if it contains ANY of these patterns:\n")
        
        for idx, pattern in enumerate(forbidden):
            # Format nicely
            conds = []
            for (x, y) in pattern:
                conds.append(f"({x} > {y})")
                
            print(f"  Rule #{idx + 1}: If {' AND '.join(conds)} --> FAILURE")
            
            # אם מדובר בשני זוגות מקושרים (A>B ו-B>C), נציג את זה כשלישייה להבנה קלה!
            if len(pattern) == 2:
                (a1, b1), (a2, b2) = pattern
                if b1 == a2:
                    print(f"           (Which means the triplet: {a1} > {b1} > {b2} is forbidden!)")
                elif b2 == a1:
                    print(f"           (Which means the triplet: {a2} > {b2} > {b1} is forbidden!)")

if __name__ == "__main__":
    # שים פה את שם קובץ הפלט שיצרת קודם 
    # (זה שמכיל את SYSTEM: R = [1, 1, 3, 3, 2] וכו')
    FILE_TO_ANALYZE = "comprehensive_delta_analysis_n5.txt"
    analyze_file(FILE_TO_ANALYZE)