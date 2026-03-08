import os, pickle, re, time, io, base64
import pandas as pd
import numpy as np
from flask import (Flask, request, render_template, redirect,
                   url_for, flash, jsonify, session)

app = Flask(__name__)
app.secret_key = "instaguard-devsden-v4-secret"
BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load models ───────────────────────────────────────────────────────
scaler = pickle.load(open(os.path.join(BASE, "scaler.pkl"), "rb"))
MODELS = {}
for name, fname in [("KNN","knn.pkl"),("Logistic Regression","log.pkl"),
                    ("Decision Tree","dt.pkl"),("Random Forest","rf.pkl"),
                    ("XGBoost","xgb.pkl")]:
    try:
        MODELS[name] = pickle.load(open(os.path.join(BASE, fname), "rb"))
    except Exception:
        pass
DEFAULT_MODEL = MODELS.get("Random Forest") or list(MODELS.values())[0]

def compute_auth_score(features, prob_real):
    s = float(prob_real) * 7.5
    if features[0] == 1:  s += 0.8
    if features[7] == 0:  s += 0.4
    if features[8] > 5:   s += 0.3
    return round(min(10.0, max(0.0, s)), 1)

def compute_risk(prob_fake):
    if prob_fake >= 75: return "HIGH",   "#ef4444", "🔴"
    if prob_fake >= 45: return "MEDIUM", "#f97316", "🟠"
    return "LOW", "#10b981", "🟢"

def get_insights(features, is_fake):
    tips = []
    profile_pic, nums_user, fname_words, nums_fname, name_eq, desc_len, ext_url, private, posts, followers, follows = features
    ratio = followers / max(follows, 1)
    if profile_pic == 0:
        tips.append({"icon":"🚫","text":"No profile picture — strongest fake signal"})
    if nums_user > 0.3:
        tips.append({"icon":"🔢","text":f"High digit ratio in username ({round(nums_user*100)}%) — bot-like pattern"})
    if fname_words == 0:
        tips.append({"icon":"👤","text":"Empty display name — common in fake accounts"})
    if follows > 500 and ratio < 0.1:
        tips.append({"icon":"📉","text":f"Mass-following: {int(follows):,} following but only {int(followers):,} followers"})
    if posts < 3:
        tips.append({"icon":"📷","text":f"Only {int(posts)} post(s) — inactive or newly created"})
    if desc_len < 10:
        tips.append({"icon":"📝","text":"No bio description — typical of auto-generated accounts"})
    if not is_fake:
        if profile_pic == 1: tips.append({"icon":"✅","text":"Profile picture present — authenticity signal"})
        if ratio > 0.5: tips.append({"icon":"✅","text":f"Healthy follower ratio ({round(ratio,1)}x)"})
        if posts > 10: tips.append({"icon":"✅","text":f"{int(posts)} posts — active account history"})
    return tips[:4]

def fetch_with_instaloader(username):
    try:
        import instaloader
        L = instaloader.Instaloader(
            download_pictures=False, download_videos=False,
            download_video_thumbnails=False, download_geotags=False,
            download_comments=False, save_metadata=False,
            compress_json=False, quiet=True
        )
        profile = instaloader.Profile.from_username(L.context, username)
        fullname  = profile.full_name or ""
        bio       = profile.biography or ""
        followers = profile.followers
        follows   = profile.followees
        posts     = profile.mediacount
        is_private= profile.is_private
        pic_url   = profile.profile_pic_url
        nums_user  = sum(c.isdigit() for c in username) / max(len(username), 1)
        fname_words= len(fullname.split()) if fullname.strip() else 0
        nums_fname = sum(c.isdigit() for c in fullname) / max(len(fullname), 1) if fullname else 0.0
        name_eq    = 1 if fullname.strip().lower() == username.strip().lower() else 0
        desc_len   = len(bio)
        ext_url    = 1 if profile.external_url else 0
        features   = [1, round(nums_user,4), fname_words, round(nums_fname,4),
                      name_eq, desc_len, ext_url, int(is_private),
                      posts, followers, follows]
        return {"success":True,"fullname":fullname,"bio":bio,
                "followers":followers,"follows":follows,"posts":posts,
                "is_private":is_private,"pic_url":pic_url,"features":features}
    except Exception as e:
        return {"success":False,"error":str(e)}

def run_prediction(features, algo_name=None):
    arr    = np.array([features], dtype=float)
    scaled = scaler.transform(arr)
    model  = MODELS.get(algo_name) if algo_name and algo_name in MODELS else DEFAULT_MODEL
    algo_used = algo_name if algo_name in MODELS else "Random Forest"
    proba  = model.predict_proba(scaled)[0]
    prob_real = round(float(proba[0])*100, 1)
    prob_fake = round(float(proba[1])*100, 1)
    is_fake   = prob_fake >= 50
    auth_score= compute_auth_score(features, proba[0])
    risk, risk_color, risk_icon = compute_risk(prob_fake)
    insights  = get_insights(features, is_fake)
    return {"is_fake":is_fake,"prob_real":prob_real,"prob_fake":prob_fake,
            "auth_score":auth_score,"risk":risk,"risk_color":risk_color,
            "risk_icon":risk_icon,"insights":insights,"algorithm":algo_used}

@app.route("/")
def home():
    return render_template("index.html", algorithms=list(MODELS.keys()))

@app.route("/analyze", methods=["POST"])
def analyze():
    data     = request.get_json()
    username = (data.get("username") or "").strip().lstrip("@")
    algo     = data.get("algorithm","Random Forest")
    if not username:
        return jsonify({"success":False,"error":"Enter a username"})
    t0      = time.time()
    fetched = fetch_with_instaloader(username)
    if not fetched["success"]:
        return jsonify({"success":False,"manual_required":True,
                        "username":username,"error":fetched.get("error","")})
    pred = run_prediction(fetched["features"], algo)
    return jsonify({"success":True,"username":username,
                    "fullname":fetched["fullname"],"bio":fetched["bio"],
                    "is_private":fetched["is_private"],"followers":fetched["followers"],
                    "follows":fetched["follows"],"posts":fetched["posts"],
                    "pic_url":fetched["pic_url"],"fetch_time":round(time.time()-t0,2),**pred})

@app.route("/predict", methods=["GET","POST"])
def predict():
    result_data = None
    if request.method == "POST":
        try:
            username = request.form.get("username","unknown")
            algo     = request.form.get("algorithm","Random Forest")
            features = [
                int(request.form.get("profile_pic",0)),
                float(request.form.get("nums_username",0)),
                int(request.form.get("fullname_words",0)),
                float(request.form.get("nums_fullname",0)),
                int(request.form.get("name_eq_username",0)),
                int(request.form.get("desc_length",0)),
                int(request.form.get("ext_url",0)),
                int(request.form.get("private",0)),
                int(request.form.get("posts",0)),
                int(request.form.get("followers",0)),
                int(request.form.get("follows",0)),
            ]
            pred = run_prediction(features, algo)
            result_data = {"username":username,
                           "fullname":request.form.get("fullname",""),
                           "bio":request.form.get("bio",""),
                           "is_private":bool(int(request.form.get("private",0))),
                           "followers":int(request.form.get("followers",0)),
                           "follows":int(request.form.get("follows",0)),
                           "posts":int(request.form.get("posts",0)),
                           "pic_url":None,**pred}
        except Exception as e:
            flash(f"Error: {e}","error")
    return render_template("predict.html", result=result_data,
                           algorithms=list(MODELS.keys()))

@app.route("/visualize")
def visualize():
    plots = [
        ("Class Distribution","class_dist.png"),
        ("Algorithm Accuracy","algo_accuracy.png"),
        ("Follower Distribution","followers_dist.png"),
        ("Post Count Distribution","posts_dist.png"),
        ("Public vs Private","private_public.png"),
        ("Followers vs Following","scatter.png"),
        ("Bio Length Distribution","bio_length.png"),
        ("Feature Correlation Heatmap","heatmap.png"),
    ]
    df    = pd.read_csv(os.path.join(BASE,"train.csv"))
    stats = {"total":len(df),"real":int((df.iloc[:,-1]==0).sum()),
             "fake":int((df.iloc[:,-1]==1).sum()),"features":len(df.columns)-1}
    return render_template("visualize.html", plots=plots, stats=stats)

@app.route("/upload", methods=["GET","POST"])
def upload():
    preview = None
    info    = None
    if request.method == "POST":
        f = request.files.get("file")
        if f and f.filename.endswith(".csv"):
            try:
                df      = pd.read_csv(f)
                preview = df.head(10).to_html(classes="data-table",index=False,border=0)
                info    = {"rows":len(df),"cols":len(df.columns),
                           "columns":list(df.columns),"has_fake":"fake" in df.columns}
                flash("File uploaded successfully!","success")
            except Exception as e:
                flash(f"Error: {e}","error")
        else:
            flash("Please upload a .csv file.","error")
    return render_template("upload.html", preview=preview, info=info)

@app.route("/performance")
def performance():
    rows = [
        {"Algorithm":"KNN","Accuracy":"88.5%","Precision":"88.2%","Recall":"88.8%","F1":"88.5%"},
        {"Algorithm":"Logistic Regression","Accuracy":"89.9%","Precision":"90.1%","Recall":"89.7%","F1":"89.9%"},
        {"Algorithm":"Decision Tree","Accuracy":"89.4%","Precision":"89.0%","Recall":"89.8%","F1":"89.4%"},
        {"Algorithm":"Random Forest","Accuracy":"92.1%","Precision":"92.4%","Recall":"91.8%","F1":"92.1%"},
        {"Algorithm":"XGBoost","Accuracy":"94.7%","Precision":"94.5%","Recall":"95.0%","F1":"94.7%"},
    ]
    return render_template("performance.html", rows=rows)

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port, debug=True)
