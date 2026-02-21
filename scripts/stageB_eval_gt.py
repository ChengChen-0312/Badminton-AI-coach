import json, math, sys
DATA = {
  "corners_order": ["LB","RB","RT","LT"],
  "corners_px": {"LB":[832,1770], "RB":[1650,790], "RT":[670,630], "LT":[226,655]},
}

def l2(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

p = sys.argv[1] if len(sys.argv)>1 else "reports/demo_friend/demo_friend_report.json"
with open(p, "r") as f:
    j=json.load(f)

pred=j["court_detection"]["corners"]  # LB,RB,RT,LT
if pred is None:
    print("report:", p)
    print("no corners")
    raise SystemExit(1)

# Ensure list of lists
pred=[list(x) for x in pred]

gt=[DATA["corners_px"][k] for k in DATA["corners_order"]]
errs=[l2(pred[i], gt[i]) for i in range(4)]
print("report:", p)
print("err_px_LB,RB,RT,LT:", [round(e,2) for e in errs])
print("mean_err:", round(sum(errs)/4,2), "max_err:", round(max(errs),2))
