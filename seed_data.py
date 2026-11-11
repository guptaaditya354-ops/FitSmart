"""Run once:  python seed_data.py
Creates tables, loads the food database, and generates 30 days of synthetic history
for 8 users so the dashboard, forecasts, clustering and energy model have data to work on."""
import random
from datetime import date, timedelta
import numpy as np
import db
from ml import EXERCISES

# name, serving, grams, protein, calories, carbs, fats, is_veg
FOODS = [
    ("Egg", "1 piece", 50, 6, 72, 0.4, 4.8, 0), ("Milk", "250 ml", 250, 8, 150, 12, 8, 1),
    ("Paneer", "100 g", 100, 18, 265, 1.2, 20, 1), ("Dal", "1 bowl", 200, 7, 150, 22, 3, 1),
    ("Curd", "100 g", 100, 3.5, 60, 4.7, 3.3, 1), ("Roti", "1 piece", 40, 3, 100, 18, 2, 1),
    ("Rice", "1 cup cooked", 150, 4, 200, 44, 0.4, 1), ("Chicken Breast", "100 g", 100, 31, 165, 0, 3.6, 0),
    ("Fish (Rohu)", "100 g", 100, 17, 97, 0, 1.4, 0), ("Soya Chunks", "50 g dry", 50, 26, 170, 17, 0.5, 1),
    ("Boiled Chana", "1 cup", 160, 15, 270, 45, 4, 1), ("Rajma", "1 bowl", 200, 9, 210, 35, 1, 1),
    ("Peanuts", "30 g", 30, 7.5, 170, 5, 14, 1), ("Almonds", "20 g", 20, 4, 115, 4, 10, 1),
    ("Banana", "1 medium", 118, 1.3, 105, 27, 0.4, 1), ("Oats", "40 g", 40, 5, 150, 27, 2.5, 1),
    ("Whey Protein", "1 scoop", 30, 24, 120, 3, 1.5, 1), ("Greek Yogurt", "150 g", 150, 15, 100, 6, 0.7, 1),
    ("Poha", "1 plate", 150, 4, 250, 45, 6, 1), ("Tofu", "100 g", 100, 8, 76, 1.9, 4.8, 1),
    ("Mixed Veg Sabzi", "1 bowl", 150, 3, 90, 12, 4, 1), ("Chicken Curry", "1 bowl", 250, 24, 280, 8, 16, 0),
]
MEAL_POOLS = {
    "Breakfast": ["Egg", "Oats", "Milk", "Poha", "Banana", "Curd", "Whey Protein"],
    "Lunch": ["Roti", "Rice", "Dal", "Paneer", "Chicken Breast", "Mixed Veg Sabzi", "Rajma", "Curd"],
    "Snacks": ["Peanuts", "Almonds", "Banana", "Boiled Chana", "Greek Yogurt", "Milk"],
    "Dinner": ["Roti", "Dal", "Soya Chunks", "Paneer", "Fish (Rohu)", "Chicken Curry", "Mixed Veg Sabzi", "Tofu"],
}
# name, age, gender, h, w, activity, diet, goal, protein_factor, workout_prob, sleep_mean, weight_slope/day
PROFILES = [
    ("Aditya (Demo)", 21, "Male", 172, 70, "Moderate", "Veg", "Muscle Gain", 0.9, 0.65, 7.0, 0.03),
    ("Riya", 22, "Female", 160, 62, "Light", "Veg", "Fat Loss", 0.8, 0.5, 7.5, -0.04),
    ("Karan", 25, "Male", 178, 82, "Active", "Non-Veg", "Muscle Gain", 1.3, 0.8, 7.8, 0.04),
    ("Neha", 24, "Female", 165, 58, "Moderate", "Non-Veg", "Weight Maintenance", 1.0, 0.55, 7.2, 0.0),
    ("Vikram", 28, "Male", 175, 90, "Sedentary", "Non-Veg", "Fat Loss", 0.7, 0.2, 6.2, -0.01),
    ("Sana", 20, "Female", 158, 52, "Light", "Veg", "Weight Maintenance", 0.6, 0.3, 6.8, 0.0),
    ("Rohit", 23, "Male", 180, 75, "Active", "Veg", "Muscle Gain", 1.1, 0.75, 7.6, 0.035),
    ("Pooja", 26, "Female", 163, 68, "Sedentary", "Veg", "Fat Loss", 0.7, 0.25, 6.5, -0.015),
]
DAYS = 30


def main():
    random.seed(7)
    rng = np.random.default_rng(7)
    db.init_schema()
    if len(db.query("SELECT * FROM Food_Database")):
        print("Already seeded. Delete fitsmart.db (or truncate tables) to re-seed.")
        return
    db.run("INSERT INTO Food_Database (food_name, serving_desc, serving_g, protein_g, calories, carbs_g, fats_g, is_veg)"
           " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", FOODS, many=True)
    fid = {r.food_name: r.food_id for r in db.query("SELECT food_id, food_name FROM Food_Database").itertuples()}
    veg = {f[0]: f[7] for f in FOODS}
    today = date.today()
    muscles = list(EXERCISES)

    for (name, age, gen, h, w0, act, diet, goal, pf, wp, sleep_mu, slope) in PROFILES:
        uid = db.run("INSERT INTO Users (name, age, gender, height_cm, weight_kg, activity_level, diet_pref)"
                     " VALUES (%s,%s,%s,%s,%s,%s,%s)", (name, age, gen, h, w0, act, diet))
        db.run("INSERT INTO Fitness_Goals (user_id, goal, start_date) VALUES (%s,%s,%s)",
               (uid, goal, str(today - timedelta(days=DAYS))))
        food_rows, work_rows, energy_rows, prog_rows = [], [], [], []
        cycle, prev_mins = random.randrange(len(muscles)), 0.0
        for d in range(DAYS, -1, -1):
            day = str(today - timedelta(days=d))
            for meal, pool in MEAL_POOLS.items():
                pool = [p for p in pool if diet == "Non-Veg" or veg[p]]
                for item in random.sample(pool, random.randint(2, 3)):
                    food_rows.append((uid, day, meal, fid[item], round(random.uniform(0.6, 1.6) * pf, 1)))
            sleep = float(np.clip(rng.normal(sleep_mu, 0.9), 4, 9.5))
            score = (sleep - 6.5) * 0.7 - prev_mins / 90 + rng.normal(0, 0.4)
            energy = 3 if score > 0.6 else (1 if score < -0.5 else 2)
            energy_rows.append((uid, day, energy, round(sleep, 1)))
            mins = 0.0
            if random.random() < wp and energy > 1:
                m = muscles[cycle % len(muscles)]
                cycle += 1
                for ex in random.sample(EXERCISES[m], min(4, len(EXERCISES[m]))):
                    dur = random.randint(9, 14)
                    mins += dur
                    work_rows.append((uid, day, m, ex, random.choice([3, 4]), random.choice(["8-12", "10-12"]),
                                      round(random.uniform(10, 60) / 2.5) * 2.5, dur))
            prev_mins = mins
            if d % 3 == 0:
                prog_rows.append((uid, day, round(w0 + slope * (DAYS - d) + rng.normal(0, 0.25), 1)))
        db.run("INSERT INTO Daily_Food_Log (user_id, log_date, meal, food_id, servings) VALUES (%s,%s,%s,%s,%s)", food_rows, many=True)
        db.run("INSERT INTO Workout_Log (user_id, log_date, muscle_group, exercise, sets, reps, weight_kg, duration_min)"
               " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", work_rows, many=True)
        db.run("INSERT INTO Energy_Log (user_id, log_date, energy_level, sleep_hours) VALUES (%s,%s,%s,%s)", energy_rows, many=True)
        db.run("INSERT INTO Progress (user_id, log_date, weight_kg) VALUES (%s,%s,%s)", prog_rows, many=True)
        db.run("UPDATE Users SET weight_kg=%s WHERE user_id=%s", (prog_rows[-1][2], uid))
    print("Seeded:", len(FOODS), "foods,", len(PROFILES), "users,", DAYS, "days of history.")


if __name__ == "__main__":
    main()
