"""
features_v2.py — Dual-Grid Feature Engineering for StatLot 649
Grid A (Primary):   9 cols x 6 rows  (red ticket)
Grid B (Secondary): 7 cols x 7 rows  (digital blue ticket)
~90 features per draw. All computed using PRIOR draws only.

FAST PATH: build_features_batch(candidates, history) computes features for
all candidates in one vectorized pass — O(n) instead of O(n*|history|).
"""
import numpy as np
from collections import Counter
from itertools import combinations

def rowA(n): return (n - 1) // 9 + 1
def colA(n): return (n - 1) % 9 + 1
def rowB(n): return (n - 1) // 7 + 1
def colB(n): return (n - 1) % 7 + 1

def dual_grid_score(nums):
    nums = sorted(nums)
    rA=[rowA(n) for n in nums]; cA=[colA(n) for n in nums]
    rB=[rowB(n) for n in nums]; cB=[colB(n) for n in nums]
    score=0
    rcA=len(set(rA)); ccA=len(set(cA)); spancA=max(cA)-min(cA)
    score += 15 if rcA==4 else (10 if rcA in (3,5) else (-20 if rcA<=2 else 0))
    score += 12 if ccA==5 else (8 if ccA in (4,6) else (-15 if ccA<=2 else 0))
    score += 10 if spancA>=6 else (5 if spancA>=5 else 0)
    score -= 10 if sum(1 for n in nums if n>=46)>=2 else 0
    rcB=len(set(rB)); ccB=len(set(cB)); spancB=max(cB)-min(cB)
    score += 12 if 4<=rcB<=5 else (8 if rcB==3 else (-15 if rcB<=2 else 0))
    score += 8 if 4<=ccB<=5 else (5 if ccB==3 else (-10 if ccB<=2 else 0))
    score += 5 if spancB>=5 else 0
    score -= 10 if sum(1 for n in nums if n>=43)>=5 else 0
    s=sum(nums)
    score += 8 if 123<=s<=168 else (5 if 105<=s<=186 else (-10 if s<100 or s>210 else 0))
    odd=sum(1 for n in nums if n%2!=0)
    score += 5 if 2<=odd<=4 else (-10 if odd in (0,6) else 0)
    consec=sum(1 for i in range(5) if nums[i+1]-nums[i]==1)
    score -= 8 if consec>=4 else 0
    return max(0, min(100, score))

def should_eliminate(nums):
    nums=sorted(nums)
    rA=[rowA(n) for n in nums]; cA=[colA(n) for n in nums]
    rB=[rowB(n) for n in nums]; cB=[colB(n) for n in nums]
    s=sum(nums)
    consec=sum(1 for i in range(5) if nums[i+1]-nums[i]==1)
    return (
        len(set(rA))<=2 or len(set(cA))<=2 or (max(rA)-min(rA))<=1 or (max(cA)-min(cA))<=3 or
        len(set(rB))<=2 or len(set(cB))<=2 or (max(rB)-min(rB))<=1 or (max(cB)-min(cB))<=3 or
        sum(1 for n in nums if n>=46)>=4 or sum(1 for n in nums if n>=43)>=5 or
        all(n%2!=0 for n in nums) or all(n%2==0 for n in nums) or
        s<100 or s>210 or consec>=4
    )


# --- HISTORY CONTEXT (computed once per draw, shared across all candidates) ---

def _build_history_context(history):
    ctx = {}
    past_sums = [sum(d["nums"]) for d in history[-10:]]
    ctx["last_sum"] = past_sums[-1] if past_sums else 147.0
    ctx["sum_ma3"] = float(np.mean(past_sums[-3:])) if len(past_sums)>=3 else 147.0
    ctx["sum_ma5"] = float(np.mean(past_sums[-5:])) if len(past_sums)>=5 else 147.0
    ctx["sum_ma10"] = float(np.mean(past_sums)) if past_sums else 147.0
    lookback50 = history[-50:] if len(history)>=50 else history
    freq50 = Counter(n for d in lookback50 for n in d["nums"])
    ctx["freq50"] = freq50
    if freq50:
        sorted_f = [n for n,_ in freq50.most_common()]
        ctx["hot10"] = set(sorted_f[:10])
        ctx["cold10"] = set(sorted_f[-10:])
    else:
        ctx["hot10"] = set(); ctx["cold10"] = set()
    ctx["freq_expected"] = max(len(lookback50)*6/49, 0.001)
    for w in (10, 20, 50):
        window = history[-w:] if len(history)>=w else history
        ctx[f"fq{w}"] = Counter(n for d in window for n in d["nums"])
    pair_freq = Counter()
    for d in history[-50:]:
        pn = sorted(d["nums"])
        for a, b in combinations(pn, 2): pair_freq[(a,b)] += 1
    ctx["pair_freq"] = pair_freq
    trans = Counter()
    if len(history) >= 2:
        for i in range(1, min(len(history), 30)):
            prev_set = set(history[-i]["nums"])
            for a in prev_set:
                for b in prev_set:
                    if a != b: trans[(a,b)] += 1
    ctx["total_trans"] = max(sum(trans.values()), 1)
    ctx["trans"] = trans
    all_hist = history[-200:] if len(history)>=200 else history
    ctx["total_draws_200"] = len(all_hist)
    ctx["freq200"] = Counter(n for d in all_hist for n in d["nums"])
    decay = np.zeros(50, dtype=np.float32)
    for k, d in enumerate(reversed(all_hist[-50:]), 1):
        for n in d["nums"]: decay[n] += np.exp(-0.05*k)
    ctx["decay"] = decay
    ctx["prev_sets"] = [set(history[-lag]["nums"]) if len(history)>=lag else set()
                        for lag in range(1, 11)]
    return ctx


# --- FAST BATCH FEATURE BUILDER ---

def build_features_batch(candidates, history):
    """
    Compute feature matrix for ALL candidates at once.
    Returns np.ndarray shape (N, len(FEATURE_COLS)) — dtype float32.
    """
    ctx = _build_history_context(history)
    N = len(candidates)
    C = np.array([sorted(c) for c in candidates], dtype=np.int32)  # (N,6)
    rows = np.zeros((N, len(FEATURE_COLS)), dtype=np.float32)

    s = C.sum(axis=1)
    rows[:, 0] = s
    rows[:, 1] = s / 6.0
    rows[:, 2] = C.std(axis=1)
    rows[:, 3] = C[:, 5] - C[:, 0]
    odd = (C % 2 != 0).sum(axis=1)
    rows[:, 4] = odd
    rows[:, 5] = 6 - odd
    low = (C <= 24).sum(axis=1)
    rows[:, 6] = low
    rows[:, 7] = 6 - low
    for d, (lo, hi) in enumerate([(1,10),(11,20),(21,30),(31,40),(41,49)]):
        rows[:, 8+d] = ((C >= lo) & (C <= hi)).sum(axis=1)
    rows[:, 13] = (rows[:, 8:13] == 0).sum(axis=1)
    gaps = np.diff(C, axis=1)
    rows[:, 14] = gaps.min(axis=1)
    rows[:, 15] = gaps.max(axis=1)
    rows[:, 16] = gaps.mean(axis=1)
    rows[:, 17] = gaps.std(axis=1)
    rows[:, 18] = (gaps == 1).sum(axis=1)
    rA = (C - 1) // 9 + 1
    cA = (C - 1) % 9 + 1
    rows[:, 19] = np.array([len(set(r)) for r in rA.tolist()])
    rows[:, 20] = np.array([len(set(c)) for c in cA.tolist()])
    rows[:, 21] = rA.max(axis=1) - rA.min(axis=1)
    rows[:, 22] = cA.max(axis=1) - cA.min(axis=1)
    rows[:, 23] = rA.std(axis=1)
    rows[:, 24] = cA.std(axis=1)
    for r in range(1, 7): rows[:, 25+r-1] = (rA == r).sum(axis=1)
    for c in range(1, 10): rows[:, 31+c-1] = (cA == c).sum(axis=1)
    rows[:, 40] = (C >= 46).sum(axis=1)
    rB = (C - 1) // 7 + 1
    cB = (C - 1) % 7 + 1
    rows[:, 41] = np.array([len(set(r)) for r in rB.tolist()])
    rows[:, 42] = np.array([len(set(c)) for c in cB.tolist()])
    rows[:, 43] = rB.max(axis=1) - rB.min(axis=1)
    rows[:, 44] = cB.max(axis=1) - cB.min(axis=1)
    rows[:, 45] = rB.std(axis=1)
    rows[:, 46] = cB.std(axis=1)
    for r in range(1, 8): rows[:, 47+r-1] = (rB == r).sum(axis=1)
    for c in range(1, 8): rows[:, 54+c-1] = (cB == c).sum(axis=1)
    rows[:, 61] = (C >= 43).sum(axis=1)
    rows[:, 62] = np.array([dual_grid_score(list(C[i])) for i in range(N)])
    # history-dependent
    for lag, prev_set in enumerate(ctx["prev_sets"], 1):
        if prev_set:
            rows[:, 62+lag] = np.array([len(set(C[i].tolist()) & prev_set) for i in range(N)])
    rows[:, 73] = s - ctx["last_sum"]
    rows[:, 74] = ctx["sum_ma3"]
    rows[:, 75] = ctx["sum_ma5"]
    rows[:, 76] = ctx["sum_ma10"]
    hot10 = ctx["hot10"]; cold10 = ctx["cold10"]
    rows[:, 77] = np.array([sum(1 for n in C[i] if n in hot10) for i in range(N)])
    rows[:, 78] = np.array([sum(1 for n in C[i] if n in cold10) for i in range(N)])
    freq50 = ctx["freq50"]; expected = ctx["freq_expected"]
    rows[:, 79] = np.array([float(np.mean([freq50.get(int(n),0) for n in C[i]]))/expected for i in range(N)])
    for wi, w in enumerate((10, 20, 50)):
        fq = ctx[f"fq{w}"]
        rows[:, 80+wi] = np.array([float(np.mean([fq.get(int(n),0) for n in C[i]])) for i in range(N)])
    pair_freq = ctx["pair_freq"]
    def _pair(combo):
        ps = [pair_freq.get((combo[a], combo[b]), 0) for a in range(6) for b in range(a+1, 6)]
        return float(np.mean(ps)), int(max(ps))
    pr = np.array([_pair(list(C[i])) for i in range(N)])
    rows[:, 83] = pr[:, 0]; rows[:, 84] = pr[:, 1]
    trans = ctx["trans"]; total_trans = ctx["total_trans"]
    rows[:, 85] = np.array([
        float(sum(trans.get((int(C[i,a]), int(C[i,b])), 0) for a in range(6) for b in range(6) if a!=b)/total_trans)
        for i in range(N)])
    freq200 = ctx["freq200"]; total_d = ctx["total_draws_200"]; decay = ctx["decay"]
    alpha = 1.0
    def _bayes(combo):
        sc = []
        for n in combo:
            n = int(n)
            bp = (freq200.get(n,0)+alpha)/(total_d*6/49+alpha*49)
            sc.append(bp*(1+float(decay[n])))
        return float(np.mean(sc)), float(min(sc)), float(max(sc))
    br = np.array([_bayes(list(C[i])) for i in range(N)])
    rows[:, 86] = br[:, 0]; rows[:, 87] = br[:, 1]; rows[:, 88] = br[:, 2]
    def _since(combo_set):
        for k, d in enumerate(reversed(history[-50:])):
            if combo_set & set(d["nums"]): return k
        return 50
    rows[:, 89] = np.array([_since(set(int(x) for x in C[i])) for i in range(N)])
    return rows


# --- ORIGINAL SINGLE-COMBO BUILDER (kept for training data / M9 etc.) ---

def build_features(nums, history):
    nums=sorted(nums); feat={}
    rA=[rowA(n) for n in nums]; cA=[colA(n) for n in nums]
    rB=[rowB(n) for n in nums]; cB=[colB(n) for n in nums]
    feat["sum"]=sum(nums); feat["mean"]=float(np.mean(nums)); feat["std"]=float(np.std(nums))
    feat["range"]=nums[-1]-nums[0]; feat["odd_count"]=sum(1 for n in nums if n%2!=0)
    feat["even_count"]=6-feat["odd_count"]; feat["low_count"]=sum(1 for n in nums if n<=24)
    feat["high_count"]=6-feat["low_count"]
    for d,(lo,hi) in enumerate([(1,10),(11,20),(21,30),(31,40),(41,49)],1):
        feat[f"decade_{d}"]=sum(1 for n in nums if lo<=n<=hi)
    feat["empty_decades"]=sum(1 for d in range(1,6) if feat[f"decade_{d}"]==0)
    gaps=[nums[i+1]-nums[i] for i in range(5)]
    feat["gap_min"]=min(gaps); feat["gap_max"]=max(gaps)
    feat["gap_mean"]=float(np.mean(gaps)); feat["gap_std"]=float(np.std(gaps))
    feat["consecutive_pairs"]=sum(1 for g in gaps if g==1)
    feat["gridA_rows"]=len(set(rA)); feat["gridA_cols"]=len(set(cA))
    feat["gridA_row_span"]=max(rA)-min(rA); feat["gridA_col_span"]=max(cA)-min(cA)
    feat["gridA_row_std"]=float(np.std(rA)); feat["gridA_col_std"]=float(np.std(cA))
    for r in range(1,7): feat[f"rowA_{r}"]=rA.count(r)
    for c in range(1,10): feat[f"colA_{c}"]=cA.count(c)
    feat["row6A_count"]=sum(1 for n in nums if n>=46)
    feat["gridB_rows"]=len(set(rB)); feat["gridB_cols"]=len(set(cB))
    feat["gridB_row_span"]=max(rB)-min(rB); feat["gridB_col_span"]=max(cB)-min(cB)
    feat["gridB_row_std"]=float(np.std(rB)); feat["gridB_col_std"]=float(np.std(cB))
    for r in range(1,8): feat[f"rowB_{r}"]=rB.count(r)
    for c in range(1,8): feat[f"colB_{c}"]=cB.count(c)
    feat["row7B_count"]=sum(1 for n in nums if n>=43)
    feat["dual_grid_score"]=dual_grid_score(nums)
    cur=set(nums)
    for lag in range(1,11):
        prev=set(history[-lag]["nums"]) if len(history)>=lag else set()
        feat[f"repeat_prev_{lag}"]=len(cur&prev)
    past_sums=[sum(d["nums"]) for d in history[-10:]]
    feat["sum_delta"]=feat["sum"]-past_sums[-1] if past_sums else 0
    feat["sum_ma3"]=float(np.mean(past_sums[-3:])) if len(past_sums)>=3 else feat["sum"]
    feat["sum_ma5"]=float(np.mean(past_sums[-5:])) if len(past_sums)>=5 else feat["sum"]
    feat["sum_ma10"]=float(np.mean(past_sums)) if past_sums else feat["sum"]
    lookback=history[-50:] if len(history)>=50 else history
    freq50=Counter(n for d in lookback for n in d["nums"])
    if freq50:
        sorted_by_freq=[n for n,_ in freq50.most_common()]
        hot10=set(sorted_by_freq[:10]); cold10=set(sorted_by_freq[-10:])
        feat["hot_count"]=sum(1 for n in nums if n in hot10)
        feat["cold_count"]=sum(1 for n in nums if n in cold10)
    else:
        feat["hot_count"]=feat["cold_count"]=0
    expected=max(len(lookback)*6/49,0.001)
    feat["avg_freq_ratio"]=float(np.mean([freq50.get(n,0) for n in nums]))/expected
    for w in (10,20,50):
        window=history[-w:] if len(history)>=w else history
        fq=Counter(n for d in window for n in d["nums"])
        feat[f"avg_freq_{w}"]=float(np.mean([fq.get(n,0) for n in nums]))
    pair_freq=Counter()
    for d in history[-50:]:
        pn=sorted(d["nums"])
        for a,b in combinations(pn,2): pair_freq[(a,b)]+=1
    my_pairs=[(nums[a],nums[b]) for a in range(6) for b in range(a+1,6)]
    pair_scores=[pair_freq.get(p,0) for p in my_pairs]
    feat["avg_pair_freq"]=float(np.mean(pair_scores)) if pair_scores else 0
    feat["max_pair_freq"]=int(max(pair_scores)) if pair_scores else 0
    mk=0.0
    if len(history)>=2:
        trans=Counter()
        for i in range(1,min(len(history),30)):
            prev_set=set(history[-i]["nums"])
            for a in prev_set:
                for b in prev_set:
                    if a!=b: trans[(a,b)]+=1
        total_trans=max(sum(trans.values()),1)
        for a,b in combinations(nums,2): mk+=trans.get((a,b),0)/total_trans
    feat["markov_score"]=float(mk)
    alpha=1.0; all_hist=history[-200:] if len(history)>=200 else history
    total_draws=len(all_hist); freq_all=Counter(n for d in all_hist for n in d["nums"])
    bayes_scores=[]
    for n in nums:
        decay_score=0.0
        for k,d in enumerate(reversed(all_hist[-50:]),1):
            if n in d["nums"]: decay_score+=np.exp(-0.05*k)
        bayes_p=(freq_all.get(n,0)+alpha)/(total_draws*6/49+alpha*49)
        bayes_scores.append(bayes_p*(1+decay_score))
    feat["bayes_score"]=float(np.mean(bayes_scores))
    feat["bayes_score_min"]=float(min(bayes_scores))
    feat["bayes_score_max"]=float(max(bayes_scores))
    draws_since=0
    for d in reversed(history[-50:]):
        if cur&set(d["nums"]): break
        draws_since+=1
    feat["draws_since_repeat"]=draws_since
    feat["is_eliminated"]=int(should_eliminate(nums))
    return feat

FEATURE_COLS = [
    "sum","mean","std","range","odd_count","even_count","low_count","high_count",
    "decade_1","decade_2","decade_3","decade_4","decade_5","empty_decades",
    "gap_min","gap_max","gap_mean","gap_std","consecutive_pairs",
    "gridA_rows","gridA_cols","gridA_row_span","gridA_col_span","gridA_row_std","gridA_col_std",
    "rowA_1","rowA_2","rowA_3","rowA_4","rowA_5","rowA_6",
    "colA_1","colA_2","colA_3","colA_4","colA_5","colA_6","colA_7","colA_8","colA_9","row6A_count",
    "gridB_rows","gridB_cols","gridB_row_span","gridB_col_span","gridB_row_std","gridB_col_std",
    "rowB_1","rowB_2","rowB_3","rowB_4","rowB_5","rowB_6","rowB_7",
    "colB_1","colB_2","colB_3","colB_4","colB_5","colB_6","colB_7","row7B_count","dual_grid_score",
    "repeat_prev_1","repeat_prev_2","repeat_prev_3","repeat_prev_4","repeat_prev_5",
    "repeat_prev_6","repeat_prev_7","repeat_prev_8","repeat_prev_9","repeat_prev_10",
    "sum_delta","sum_ma3","sum_ma5","sum_ma10",
    "hot_count","cold_count","avg_freq_ratio","avg_freq_10","avg_freq_20","avg_freq_50",
    "avg_pair_freq","max_pair_freq","markov_score",
    "bayes_score","bayes_score_min","bayes_score_max","draws_since_repeat",
]

def build_datalake(draws):
    lake=[]
    for i,d in enumerate(draws):
        history=draws[:i]; nums=sorted(d["nums"])
        feats=build_features(nums,history)
        row={"draw_number":d["draw_number"],"n1":nums[0],"n2":nums[1],"n3":nums[2],
             "n4":nums[3],"n5":nums[4],"n6":nums[5],"additional":d.get("additional")}
        row.update(feats); lake.append(row)
    return lake
