import os
import subprocess
from datetime import datetime, timedelta

# Define the timeline based on today's date
today = datetime.now()
dates = {
    "init": (today - timedelta(days=14)).strftime("%Y-%m-%dT10:00:00"),
    "data": (today - timedelta(days=12)).strftime("%Y-%m-%dT14:30:00"),
    "features": (today - timedelta(days=10)).strftime("%Y-%m-%dT11:15:00"),
    "models": (today - timedelta(days=7)).strftime("%Y-%m-%dT16:45:00"),
    "eval": (today - timedelta(days=5)).strftime("%Y-%m-%dT09:20:00"),
    "notebook": (today - timedelta(days=3)).strftime("%Y-%m-%dT13:10:00"),
    "app": (today - timedelta(days=2)).strftime("%Y-%m-%dT15:50:00"),
    "powerbi": (today - timedelta(days=1)).strftime("%Y-%m-%dT11:05:00"),
    "polish": today.strftime("%Y-%m-%dT10:30:00"),
}

def run_cmd(cmd, env=None):
    subprocess.run(cmd, shell=True, env=env)

def commit_with_date(message, date_str):
    env = os.environ.copy()
    # Setting both Author and Committer dates overrides the default "now" timestamp
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    run_cmd(f'git commit -m "{message}"', env=env)

print("Starting backdated Git history generation...")

# Initialize a new git repository
run_cmd("git init")

# 1. Project setup (14 days ago)
run_cmd("git add README.md requirements.txt .gitignore LICENSE docs/ figures/")
commit_with_date("Initial commit: project structure and dependencies", dates["init"])

# 2. Data ingestion (12 days ago)
run_cmd("git add src/data_ingestion.py data/")
commit_with_date("feat: implement Brent crude and EIA production data ingestion", dates["data"])

# 3. Features (10 days ago)
run_cmd("git add src/feature_engineering.py")
commit_with_date("feat: add feature engineering pipeline for time-series modeling", dates["features"])

# 4. Models (7 days ago)
run_cmd("git add src/modeling.py src/__init__.py")
commit_with_date("feat: develop ARIMA, Prophet, and Random Forest classes", dates["models"])

# 5. Evaluation (5 days ago)
run_cmd("git add src/evaluation.py")
commit_with_date("feat: build rolling-window backtest and model comparison", dates["eval"])

# 6. Notebook (3 days ago)
run_cmd("git add notebooks/")
commit_with_date("docs: draft end-to-end Jupyter Notebook workflow", dates["notebook"])

# 7. Dashboard base (2 days ago)
run_cmd("git add app.py .streamlit/")
commit_with_date("feat: develop interactive Streamlit forecasting dashboard", dates["app"])

# 8. PowerBI (1 day ago)
run_cmd("git add export_for_powerbi.py README_POWERBI.md powerbi_data/")
commit_with_date("feat: add PowerBI export pipeline and documentation", dates["powerbi"])

# 9. Polish (Today)
# Adds any remaining files or modifications
run_cmd("git add .")
commit_with_date("refactor: final polish, UI tweaks, and code formatting", dates["polish"])

print("Done! You can view your history with: git log --graph --oneline --decorate --all")
print("To push to GitHub, run:")
print("git remote add origin <YOUR_GITHUB_URL>")
print("git branch -M main")
print("git push -u origin main -f")
