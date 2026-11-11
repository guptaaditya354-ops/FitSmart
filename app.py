"""FitSmart: Eat Smart, Train Smart, Stay Fit.   Run:  streamlit run app.py"""
from datetime import date, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import db
import ml

st.set_page_config(page_title="FitSmart", page_icon="💪", layout="wide")
db.init_schema()
st.title("💪 FitSmart")
st.caption("Eat Smart, Train Smart, Stay Fit.")

# ------------------------------------------------------------------ sidebar
users = db.query("SELECT user_id, name FROM Users")
if users.empty:
    st.warning("Database empty. Run `python seed_data.py` first, then refresh.")
    st.stop()

with st.sidebar:
    name = st.selectbox("User", users.name)
    uid = int(users.loc[users.name == name, "user_id"].iloc[0])
    user, goal = ml.get_user(uid), ml.get_goal(uid)
    new_goal = st.selectbox("Fitness goal", ml.GOALS, index=ml.GOALS.index(goal))
    if new_goal != goal:
        db.run("INSERT INTO Fitness_Goals (user_id, goal, start_date) VALUES (%s,%s,%s)", (uid, new_goal, str(date.today())))
        goal = new_goal
    tg = ml.targets(user, goal)
    st.subheader("Daily targets")
    st.metric("Calories", f"{tg['calories']} kcal")
    st.metric("Protein", f"{tg['protein']} g")
    st.caption(f"Carbs {tg['carbs']} g · Fats {tg['fats']} g")
    st.caption(f"{user['age']} yrs · {user['height_cm']:.0f} cm · {user['weight_kg']:.1f} kg · {user['diet_pref']}")
    with st.expander("➕ New user"):
        with st.form("new_user"):
            n = st.text_input("Name")
            a = st.number_input("Age", 14, 80, 22)
            g = st.selectbox("Gender", ["Male", "Female"])
            h = st.number_input("Height (cm)", 120.0, 220.0, 170.0)
            w = st.number_input("Weight (kg)", 30.0, 200.0, 65.0)
            act = st.selectbox("Activity", list(ml.ACTIVITY))
            dp = st.selectbox("Diet", ["Veg", "Non-Veg"])
            gl = st.selectbox("Goal", ml.GOALS)
            if st.form_submit_button("Create") and n:
                new_id = db.run("INSERT INTO Users (name, age, gender, height_cm, weight_kg, activity_level, diet_pref)"
                                " VALUES (%s,%s,%s,%s,%s,%s,%s)", (n, a, g, h, w, act, dp))
                db.run("INSERT INTO Fitness_Goals (user_id, goal, start_date) VALUES (%s,%s,%s)", (new_id, gl, str(date.today())))
                db.run("INSERT INTO Progress (user_id, log_date, weight_kg) VALUES (%s,%s,%s)", (new_id, str(date.today()), w))
                st.rerun()

tab_food, tab_diet, tab_work, tab_dash, tab_ai = st.tabs(
    ["🍽️ Log Food", "🥗 Diet Plan", "🏋️ Workout", "📊 Dashboard", "🤖 AI Insights"])

today_df = ml.food_log(uid, date.today(), date.today())
tot = today_df[["protein", "calories", "carbs", "fats"]].sum() if len(today_df) else pd.Series(
    {"protein": 0, "calories": 0, "carbs": 0, "fats": 0})

# ------------------------------------------------------------------ 1. food log
with tab_food:
    foods = db.query("SELECT * FROM Food_Database ORDER BY food_name")
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    d = c1.date_input("Date", date.today())
    meal = c2.selectbox("Meal", ["Breakfast", "Lunch", "Snacks", "Dinner"])
    food = c3.selectbox("Food", foods.food_name,
                        format_func=lambda x: f"{x} ({foods.loc[foods.food_name == x, 'serving_desc'].iloc[0]})")
    qty = c4.number_input("Servings", 0.25, 20.0, 1.0, 0.25)
    if st.button("Add to log", type="primary"):
        db.run("INSERT INTO Daily_Food_Log (user_id, log_date, meal, food_id, servings) VALUES (%s,%s,%s,%s,%s)",
               (uid, str(d), meal, int(foods.loc[foods.food_name == food, "food_id"].iloc[0]), qty))
        st.rerun()
    sel = ml.food_log(uid, d, d)
    if sel.empty:
        st.info("Nothing logged for this date.")
    else:
        st.dataframe(sel[["meal", "food_name", "servings", "protein", "calories", "carbs", "fats"]].round(1),
                     width="stretch", hide_index=True)
        s = sel[["protein", "calories", "carbs", "fats"]].sum()
        m = st.columns(4)
        m[0].metric("Protein", f"{s.protein:.0f} g", f"{s.protein - tg['protein']:+.0f} vs target")
        m[1].metric("Calories", f"{s.calories:.0f}", f"{s.calories - tg['calories']:+.0f} vs target")
        m[2].metric("Carbs", f"{s.carbs:.0f} g")
        m[3].metric("Fats", f"{s.fats:.0f} g")

# ------------------------------------------------------------------ 2. diet plan
with tab_diet:
    gap_p, gap_c = tg["protein"] - tot.protein, tg["calories"] - tot.calories
    st.subheader("Today's gap")
    a, b = st.columns(2)
    a.metric("Protein remaining", f"{max(gap_p, 0):.0f} g", "deficit" if gap_p > 0 else "target met")
    b.metric("Calories remaining", f"{gap_c:.0f} kcal")
    veg = st.toggle("Vegetarian only", value=user["diet_pref"] == "Veg")
    st.markdown("**High-protein foods to add (best protein per calorie):**")
    st.dataframe(ml.recommend_foods(gap_p, gap_c, veg).round(1), width="stretch", hide_index=True)
    st.caption("Targets use the Mifflin-St Jeor equation × activity factor, adjusted for your goal "
               "(+300 kcal gain, −400 kcal loss; 2 g/kg protein for gain/loss, 1.6 g/kg maintenance).")

# ------------------------------------------------------------------ 3. workout
with tab_work:
    c1, c2 = st.columns(2)
    sleep = c1.slider("Last night's sleep (hours)", 3.0, 10.0, 7.0, 0.5)
    pred = ml.predict_energy(uid, sleep)
    default = pred["level"] if pred else 2
    if pred:
        acc = "" if pred["cv"] != pred["cv"] else f" (model CV accuracy {pred['cv']:.0%})"
        c1.caption(f"🤖 Random Forest predicts your energy today: **{ml.ENERGY_NAME[default]}**{acc}")
    energy = c2.radio("Your energy level", [1, 2, 3], index=default - 1, horizontal=True,
                      format_func=lambda x: ml.ENERGY_NAME[x])
    muscle = c2.selectbox("Target muscle", ["Auto (best recovered)"] + list(ml.EXERCISES))
    rec = ml.recovery_table(uid)
    st.plotly_chart(px.bar(rec, x="muscle", y="days_since", title="Days since each muscle was trained",
                           labels={"days_since": "days"}).add_hline(y=2, line_dash="dash", annotation_text="48h recovery"),
                    width="stretch")
    if st.button("Generate workout", type="primary"):
        db.run("DELETE FROM Energy_Log WHERE user_id=%s AND log_date=%s", (uid, str(date.today())))
        db.run("INSERT INTO Energy_Log (user_id, log_date, energy_level, sleep_hours) VALUES (%s,%s,%s,%s)",
               (uid, str(date.today()), energy, sleep))
        st.session_state.plan = ml.recommend_workout(uid, energy, goal, muscle)
    plan = st.session_state.get("plan")
    if plan:
        st.subheader(plan["title"])
        if plan["warning"]:
            st.warning(plan["warning"])
        st.dataframe(pd.DataFrame(plan["exercises"]).rename(columns={"rest_sec": "rest (s)", "weight_kg": "suggested kg"}),
                     width="stretch", hide_index=True)
        total = st.number_input("Total session minutes", 5, 180, 45)
        if st.button("✅ Log this workout"):
            per = total / len(plan["exercises"])
            db.run("INSERT INTO Workout_Log (user_id, log_date, muscle_group, exercise, sets, reps, weight_kg, duration_min)"
                   " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                   [(uid, str(date.today()), plan["muscle"], e["exercise"], e["sets"], str(e["reps"]), e["weight_kg"], per)
                    for e in plan["exercises"]], many=True)
            st.session_state.plan = None
            st.rerun()

# ------------------------------------------------------------------ 4. dashboard
with tab_dash:
    t = ml.daily_totals(uid, 14)
    w = ml.workout_log(uid, 30)
    c1, c2 = st.columns(2)
    if len(t):
        f1 = px.bar(t, x="log_date", y="protein", title="Daily protein vs target (g)").add_hline(
            y=tg["protein"], line_dash="dash", line_color="red")
        f2 = px.bar(t, x="log_date", y="calories", title="Calories vs target").add_hline(
            y=tg["calories"], line_dash="dash", line_color="red")
        c1.plotly_chart(f1, width="stretch")
        c2.plotly_chart(f2, width="stretch")
    if not w.empty:
        dur = w.groupby("log_date")["duration_min"].sum().reset_index()
        c1.plotly_chart(px.bar(dur, x="log_date", y="duration_min", title="Workout duration (min)"), width="stretch")
        w["week"] = w.log_date.dt.to_period("W").dt.start_time
        wk = w.groupby("week")["log_date"].nunique().reset_index(name="sessions")
        c2.plotly_chart(px.bar(wk, x="week", y="sessions", title="Weekly workout consistency (sessions)"), width="stretch")
    fw = ml.forecast_weight(uid)
    if fw:
        hist, fut, slope = fw
        fig = go.Figure([go.Scatter(x=hist.log_date, y=hist.weight_kg, name="Actual", mode="lines+markers"),
                         go.Scatter(x=fut.log_date, y=fut.weight_kg, name="Forecast (LinearRegression)", line=dict(dash="dash"))])
        fig.update_layout(title=f"Weight progress · trend {slope:+.2f} kg/week", yaxis_title="kg")
        st.plotly_chart(fig, width="stretch")
    e = db.query("SELECT log_date, energy_level, sleep_hours FROM Energy_Log WHERE user_id=%s ORDER BY log_date", (uid,))
    if len(e):
        e["log_date"] = pd.to_datetime(e.log_date)
        st.plotly_chart(px.line(e.tail(30), x="log_date", y=["energy_level", "sleep_hours"], markers=True,
                                title="Energy level (1-3) & sleep (hrs)"), width="stretch")

# ------------------------------------------------------------------ 5. AI insights
with tab_ai:
    st.subheader("Smart suggestions")
    for m in ml.insights(uid):
        st.markdown(f"- {m}")
    st.subheader("Users with similar fitness patterns (KMeans)")
    cl = ml.cluster_users()
    if cl is None:
        st.info("Need at least 4 users for clustering.")
    else:
        mine = cl[cl.user_id == uid].iloc[0]
        st.success(f"You belong to the **{mine.group}** group.")
        st.dataframe(cl[cl.group == mine.group][["name", "avg_protein", "avg_calories", "workouts_per_week",
                                                  "avg_sleep", "avg_energy"]].round(1),
                     width="stretch", hide_index=True)
