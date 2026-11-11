"""FitSmart engine: nutrition maths, workout recommender, and the ML models."""
from datetime import date, timedelta
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import db

ACTIVITY = {"Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55, "Active": 1.725}
GOALS = ["Muscle Gain", "Fat Loss", "Weight Maintenance"]
ENERGY_NAME = {1: "Low", 2: "Medium", 3: "High"}

EXERCISES = {
    "Chest": ["Barbell Bench Press", "Incline Dumbbell Press", "Cable Fly", "Push-ups", "Chest Dips"],
    "Back": ["Lat Pulldown", "Seated Cable Row", "Barbell Row", "Pull-ups", "Single-arm Dumbbell Row"],
    "Shoulders": ["Overhead Press", "Lateral Raise", "Rear Delt Fly", "Arnold Press", "Face Pull"],
    "Biceps": ["Barbell Curl", "Hammer Curl", "Incline Dumbbell Curl", "Preacher Curl"],
    "Triceps": ["Triceps Pushdown", "Skull Crusher", "Overhead Triceps Extension", "Close-grip Bench Press"],
    "Legs": ["Barbell Squat", "Leg Press", "Romanian Deadlift", "Walking Lunges", "Leg Curl", "Calf Raise"],
    "Glutes": ["Hip Thrust", "Glute Bridge", "Bulgarian Split Squat", "Cable Kickback"],
    "Abs": ["Plank", "Hanging Leg Raise", "Cable Crunch", "Russian Twist", "Ab Wheel Rollout"],
}
# goal -> (sets, reps, rest_seconds)
SCHEME = {"Muscle Gain": (4, "8-12", 90), "Fat Loss": (3, "12-15", 45), "Weight Maintenance": (3, "10-12", 60)}


# ---------------------------------------------------------------- users / goals
def get_user(uid):
    return db.query("SELECT * FROM Users WHERE user_id=%s", (uid,)).iloc[0].to_dict()


def get_goal(uid):
    d = db.query("SELECT goal FROM Fitness_Goals WHERE user_id=%s ORDER BY goal_id DESC", (uid,))
    return d.iloc[0, 0] if len(d) else "Weight Maintenance"


# ---------------------------------------------------------------- nutrition
def bmr(weight, height, age, gender):
    """Mifflin-St Jeor equation."""
    return 10 * weight + 6.25 * height - 5 * age + (5 if gender == "Male" else -161)


def targets(user, goal):
    tdee = bmr(user["weight_kg"], user["height_cm"], user["age"], user["gender"]) * ACTIVITY[user["activity_level"]]
    cal = tdee + {"Muscle Gain": 300, "Fat Loss": -400, "Weight Maintenance": 0}[goal]
    protein = user["weight_kg"] * {"Muscle Gain": 2.0, "Fat Loss": 2.0, "Weight Maintenance": 1.6}[goal]
    fats = cal * 0.25 / 9
    carbs = (cal - protein * 4 - fats * 9) / 4
    return {"calories": round(cal), "protein": round(protein), "carbs": round(carbs), "fats": round(fats)}


def food_log(uid, start, end):
    return db.query(
        """SELECT l.log_date, l.meal, f.food_name, l.servings, f.serving_desc,
                  f.protein_g*l.servings AS protein, f.calories*l.servings AS calories,
                  f.carbs_g*l.servings AS carbs, f.fats_g*l.servings AS fats
           FROM Daily_Food_Log l JOIN Food_Database f ON f.food_id=l.food_id
           WHERE l.user_id=%s AND l.log_date>=%s AND l.log_date<=%s""",
        (uid, str(start), str(end)),
    )


def daily_totals(uid, days=14):
    end = date.today()
    d = food_log(uid, end - timedelta(days=days - 1), end)
    if d.empty:
        return pd.DataFrame(columns=["log_date", "protein", "calories", "carbs", "fats"])
    d["log_date"] = pd.to_datetime(d["log_date"])
    return d.groupby("log_date")[["protein", "calories", "carbs", "fats"]].sum().reset_index()


def recommend_foods(protein_gap, calorie_gap, veg_only, n=6):
    """Rank foods by protein per 100 kcal, filtered by diet preference and remaining calories."""
    f = db.query("SELECT * FROM Food_Database")
    if veg_only:
        f = f[f.is_veg == 1]
    f = f[f.protein_g >= 5].copy()
    f["protein_per_100kcal"] = f.protein_g / f.calories * 100
    f = f.sort_values("protein_per_100kcal", ascending=False)
    if protein_gap > 0:
        f["servings_to_fill_gap"] = (protein_gap / f.protein_g).clip(upper=3).round(1)
    return f.head(n)[["food_name", "serving_desc", "protein_g", "calories", "protein_per_100kcal"]
                     + (["servings_to_fill_gap"] if protein_gap > 0 else [])]


# ---------------------------------------------------------------- workouts
def workout_log(uid, days=30):
    d = db.query("SELECT * FROM Workout_Log WHERE user_id=%s AND log_date>=%s",
                 (uid, str(date.today() - timedelta(days=days))))
    if not d.empty:
        d["log_date"] = pd.to_datetime(d["log_date"])
    return d


def recovery_table(uid):
    w = workout_log(uid, 30)
    last = w.groupby("muscle_group")["log_date"].max().to_dict() if not w.empty else {}
    today = pd.Timestamp(date.today())
    return pd.DataFrame([{"muscle": m, "days_since": (today - last[m]).days if m in last else 30}
                         for m in EXERCISES])


def recommend_workout(uid, energy, goal, muscle=None):
    rec = recovery_table(uid)
    if energy == 1:
        return {"title": "Recovery day", "muscle": "Full body", "warning": None, "exercises": [
            {"exercise": "Brisk walking", "sets": 1, "reps": "20-30 min", "rest_sec": 0, "weight_kg": 0},
            {"exercise": "Full-body stretching", "sets": 1, "reps": "10 min", "rest_sec": 0, "weight_kg": 0},
            {"exercise": "Foam rolling", "sets": 1, "reps": "5-10 min", "rest_sec": 0, "weight_kg": 0}]}
    if not muscle or muscle.startswith("Auto"):
        muscle = rec.sort_values("days_since", ascending=False).iloc[0]["muscle"]
    since = int(rec.loc[rec.muscle == muscle, "days_since"].iloc[0])
    warning = (f"{muscle} was trained {since} day(s) ago. Muscles usually need ~48h; "
               "consider another muscle group.") if since < 2 else None
    sets, reps, rest = SCHEME[goal]
    n = 4
    if energy == 2:
        sets, n = max(2, sets - 1), 3
    w = workout_log(uid, 14)
    recent = set(w[w.muscle_group == muscle]["exercise"]) if not w.empty else set()
    chosen = sorted(EXERCISES[muscle], key=lambda e: e in recent)[:n]  # rotate: fresh exercises first
    hist = workout_log(uid, 60)
    plan = []
    for ex in chosen:
        h = hist[(hist.exercise == ex) & (hist.weight_kg > 0)] if not hist.empty else hist
        last_w = float(h.sort_values("log_date").iloc[-1]["weight_kg"]) if len(h) else 0.0
        plan.append({"exercise": ex, "sets": sets, "reps": reps, "rest_sec": rest,
                     "weight_kg": last_w + (2.5 if energy == 3 and last_w else 0)})  # progressive overload
    return {"title": f"{ENERGY_NAME[energy]}-energy {muscle} session", "muscle": muscle,
            "warning": warning, "exercises": plan}


# ---------------------------------------------------------------- ML models
def _energy_frame(uid):
    e = db.query("SELECT log_date, energy_level, sleep_hours FROM Energy_Log WHERE user_id=%s ORDER BY log_date", (uid,))
    w = db.query("SELECT log_date, SUM(duration_min) AS prev_mins FROM Workout_Log WHERE user_id=%s GROUP BY log_date", (uid,))
    if e.empty:
        return e
    e["log_date"] = pd.to_datetime(e["log_date"])
    if w.empty:
        e["prev_mins"] = 0.0
    else:
        w["log_date"] = pd.to_datetime(w["log_date"]) + pd.Timedelta(days=1)
        e = e.merge(w, on="log_date", how="left")
    e["prev_mins"] = e["prev_mins"].fillna(0)
    e["prev_energy"] = e["energy_level"].shift(1)
    return e.dropna()


def predict_energy(uid, sleep_hours):
    """Random Forest: sleep + yesterday's workout load + yesterday's energy -> today's energy."""
    e = _energy_frame(uid)
    if len(e) < 14 or e.energy_level.nunique() < 2:
        return None
    X, y = e[["sleep_hours", "prev_mins", "prev_energy"]], e["energy_level"]
    model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=0).fit(X, y)
    cv = cross_val_score(model, X, y, cv=3).mean() if len(e) >= 20 else float("nan")
    w = workout_log(uid, 2)
    yday = pd.Timestamp(date.today() - timedelta(days=1))
    mins = float(w[w.log_date == yday].duration_min.sum()) if not w.empty else 0.0
    prev = db.query("SELECT energy_level FROM Energy_Log WHERE user_id=%s ORDER BY log_date DESC", (uid,))
    prev_e = float(prev.iloc[0, 0]) if len(prev) else 2.0
    level = int(model.predict(pd.DataFrame([[sleep_hours, mins, prev_e]], columns=X.columns))[0])
    return {"level": level, "cv": cv}


def forecast_weight(uid, horizon=14):
    """Linear regression on weight vs. time -> weekly trend + forecast."""
    p = db.query("SELECT log_date, weight_kg FROM Progress WHERE user_id=%s ORDER BY log_date", (uid,))
    if len(p) < 4:
        return None
    p["log_date"] = pd.to_datetime(p["log_date"])
    x = (p.log_date - p.log_date.min()).dt.days.values.reshape(-1, 1)
    m = LinearRegression().fit(x, p.weight_kg)
    fx = np.arange(x.max() + 1, x.max() + 1 + horizon).reshape(-1, 1)
    fut = pd.DataFrame({"log_date": p.log_date.min() + pd.to_timedelta(fx.ravel(), unit="D"),
                        "weight_kg": m.predict(fx)})
    return p, fut, float(m.coef_[0] * 7)


def cluster_users(k=3):
    """KMeans over behaviour features of all users (last 30 days)."""
    users = db.query("SELECT user_id, name FROM Users")
    if len(users) < k + 1:
        return None
    rows = []
    for uid, name in zip(users.user_id, users.name):
        t, w = daily_totals(uid, 30), workout_log(uid, 30)
        e = db.query("SELECT AVG(sleep_hours) s, AVG(energy_level) en FROM Energy_Log WHERE user_id=%s AND log_date>=%s",
                     (uid, str(date.today() - timedelta(days=30))))
        rows.append({"user_id": uid, "name": name,
                     "avg_protein": t.protein.mean() if len(t) else 0,
                     "avg_calories": t.calories.mean() if len(t) else 0,
                     "workouts_per_week": (w.log_date.nunique() / 30 * 7) if not w.empty else 0,
                     "avg_sleep": e.iloc[0, 0] or 0, "avg_energy": e.iloc[0, 1] or 0})
    df = pd.DataFrame(rows)
    feats = ["avg_protein", "avg_calories", "workouts_per_week", "avg_sleep", "avg_energy"]
    df["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(df[feats]))
    order = df.groupby("cluster")["workouts_per_week"].mean().sort_values().index.tolist()
    names = ["Low-activity", "Moderately active", "Highly consistent"][-k:]
    df["group"] = df["cluster"].map(dict(zip(order, names)))
    return df


# ---------------------------------------------------------------- insights
def insights(uid):
    u, goal = get_user(uid), get_goal(uid)
    tg, msgs = targets(u, goal), []
    t = daily_totals(uid, 7)
    if len(t):
        ap, ac = t.protein.mean(), t.calories.mean()
        if ap < 0.9 * tg["protein"]:
            msgs.append(f"⚠️ Your 7-day average protein is {ap:.0f}g, below your estimated target of {tg['protein']}g.")
        elif ap > 1.3 * tg["protein"]:
            msgs.append(f"ℹ️ Protein ({ap:.0f}g) is well above your target ({tg['protein']}g). Extra protein is not adding much.")
        else:
            msgs.append(f"✅ Protein intake ({ap:.0f}g) is on target.")
        if ac < 0.9 * tg["calories"]:
            msgs.append(f"⚠️ You consumed fewer calories ({ac:.0f}) than your estimated requirement ({tg['calories']}).")
        elif ac > 1.1 * tg["calories"] and goal != "Muscle Gain":
            msgs.append(f"⚠️ Calories ({ac:.0f}) are above your {goal} target ({tg['calories']}).")
    else:
        msgs.append("Log your meals to get nutrition insights.")
    rec = recovery_table(uid)
    for m in rec[rec.days_since < 2].muscle:
        msgs.append(f"💤 You trained {m} recently. Allow recovery before another intense {m.lower()} session.")
    w = workout_log(uid, 14)
    if not w.empty:
        cut = pd.Timestamp(date.today() - timedelta(days=7))
        this_wk, last_wk = w[w.log_date > cut].log_date.nunique(), w[w.log_date <= cut].log_date.nunique()
        if this_wk > last_wk:
            msgs.append(f"🔥 Workout consistency improved: {this_wk} sessions this week vs {last_wk} last week.")
        elif this_wk < last_wk:
            msgs.append(f"📉 Workout consistency dipped: {this_wk} sessions vs {last_wk} last week.")
    fw = forecast_weight(uid)
    if fw:
        slope = fw[2]
        good = (goal == "Fat Loss" and slope < 0) or (goal == "Muscle Gain" and slope > 0) or \
               (goal == "Weight Maintenance" and abs(slope) < 0.3)
        msgs.append(f"{'✅' if good else '⚠️'} Weight trend: {slope:+.2f} kg/week "
                    f"({'matches' if good else 'does not match'} your {goal} goal).")
    return msgs
