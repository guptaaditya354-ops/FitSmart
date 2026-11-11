# FitSmart

Eat Smart, Train Smart, Stay Fit.

FitSmart is a Streamlit fitness dashboard for tracking meals, nutrition targets, workouts, sleep, and weight progress. It also provides food and workout recommendations, trend forecasts, and fitness-pattern clustering.

## Features

- Log meals and review daily calories and macronutrients.
- Get nutrition targets based on the Mifflin-St Jeor equation, activity level, and fitness goal.
- Generate workouts using energy level and muscle recovery history.
- View workout, nutrition, and weight trends.
- Explore Random Forest energy predictions, linear-regression weight forecasts, and KMeans user clusters.

## Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`

## Run locally

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app uses SQLite by default and creates its schema automatically. The included `fitsmart.db` contains generated demo users, food data, and sample activity history. To generate demo data in a new database, run:

```powershell
python seed_data.py
```

## Optional MySQL backend

Create a MySQL database named `fitsmart`, then set the connection variables before starting the app:

```powershell
$env:FITSMART_DB = "mysql"
$env:MYSQL_HOST = "localhost"
$env:MYSQL_USER = "root"
$env:MYSQL_PASSWORD = "your-password"
$env:MYSQL_DB = "fitsmart"
streamlit run app.py
```

Keep database credentials in environment variables; do not commit them to the repository.

## Project files

- `app.py` — Streamlit user interface
- `db.py` — SQLite/MySQL database layer and schema
- `ml.py` — Nutrition calculations, recommendations, and ML models
- `seed_data.py` — Food catalog and synthetic demo-data generator
- `fitsmart.db` — Seeded SQLite demo database
- `requirements.txt` — Python dependencies
