"""Database layer. Default = SQLite (zero setup). Set FITSMART_DB=mysql to use MySQL."""
import os
import sqlite3
import pandas as pd

BACKEND = os.getenv("FITSMART_DB", "sqlite")
PK = "INT AUTO_INCREMENT PRIMARY KEY" if BACKEND == "mysql" else "INTEGER PRIMARY KEY AUTOINCREMENT"

SCHEMA = [
    f"""CREATE TABLE IF NOT EXISTS Users (
        user_id {PK}, name VARCHAR(100), age INT, gender VARCHAR(10),
        height_cm FLOAT, weight_kg FLOAT, activity_level VARCHAR(20), diet_pref VARCHAR(10))""",
    f"""CREATE TABLE IF NOT EXISTS Food_Database (
        food_id {PK}, food_name VARCHAR(100), serving_desc VARCHAR(50), serving_g FLOAT,
        protein_g FLOAT, calories FLOAT, carbs_g FLOAT, fats_g FLOAT, is_veg INT)""",
    f"""CREATE TABLE IF NOT EXISTS Daily_Food_Log (
        log_id {PK}, user_id INT, log_date DATE, meal VARCHAR(20), food_id INT, servings FLOAT)""",
    f"""CREATE TABLE IF NOT EXISTS Fitness_Goals (
        goal_id {PK}, user_id INT, goal VARCHAR(30), start_date DATE)""",
    f"""CREATE TABLE IF NOT EXISTS Workout_Log (
        workout_id {PK}, user_id INT, log_date DATE, muscle_group VARCHAR(20),
        exercise VARCHAR(80), sets INT, reps VARCHAR(20), weight_kg FLOAT, duration_min FLOAT)""",
    f"""CREATE TABLE IF NOT EXISTS Energy_Log (
        energy_id {PK}, user_id INT, log_date DATE, energy_level INT, sleep_hours FLOAT)""",
    f"""CREATE TABLE IF NOT EXISTS Progress (
        progress_id {PK}, user_id INT, log_date DATE, weight_kg FLOAT)""",
]


def conn():
    if BACKEND == "mysql":
        import mysql.connector
        return mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DB", "fitsmart"),
        )
    return sqlite3.connect(os.getenv("SQLITE_PATH", "fitsmart.db"))


def _fix(sql):
    return sql if BACKEND == "mysql" else sql.replace("%s", "?")


def run(sql, params=(), many=False):
    c = conn()
    cur = c.cursor()
    try:
        (cur.executemany if many else cur.execute)(_fix(sql), params)
        c.commit()
        return cur.lastrowid
    finally:
        c.close()


def query(sql, params=()):
    c = conn()
    cur = c.cursor()
    try:
        cur.execute(_fix(sql), params)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        c.close()


def init_schema():
    for stmt in SCHEMA:
        run(stmt)
