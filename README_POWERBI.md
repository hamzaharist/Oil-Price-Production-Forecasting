# Power BI Dashboard Guide — Oil Price Forecasting

This guide walks you through building a professional Power BI dashboard
from the exported CSVs. Designed to showcase skills relevant to
**data analyst** roles.

---

## Step 1: Import Data

1. Open **Power BI Desktop**
2. **Get Data** > **Text/CSV**
3. Import all 5 files from `powerbi_data/`:

| File | Table Name (rename to) | Purpose |
|------|----------------------|---------|
| `01_raw_data.csv` | `RawData` | Monthly prices + production |
| `02_features.csv` | `Features` | Engineered feature matrix |
| `03_backtest_results.csv` | `BacktestResults` | Actual vs predicted per model |
| `04_model_summary.csv` | `ModelSummary` | RMSE, MAPE per model |
| `05_feature_importances.csv` | `FeatureImportances` | RF feature rankings |

4. In **Power Query Editor**, verify:
   - `date` columns are typed as **Date**
   - Numeric columns are typed as **Decimal Number**
   - Click **Close & Apply**

---

## Step 2: Create a Date Table (Best Practice)

In the **Modeling** tab, click **New Table** and paste:

```dax
DateTable = 
ADDCOLUMNS(
    CALENDARAUTO(),
    "Year", YEAR([Date]),
    "Month", MONTH([Date]),
    "MonthName", FORMAT([Date], "MMM"),
    "Quarter", "Q" & QUARTER([Date]),
    "YearMonth", FORMAT([Date], "YYYY-MM")
)
```

Then create relationships:
- `DateTable[Date]` → `RawData[date]` (1:many)
- `DateTable[Date]` → `BacktestResults[date]` (1:many)

---

## Step 3: Create DAX Measures

Click **New Measure** in the Modeling tab for each:

### KPI Measures
```dax
Latest Price = 
CALCULATE(
    LASTNONBLANK(RawData[price_brent], 1),
    LASTDATE(RawData[date])
)

Price YoY Change % = 
VAR CurrentPrice = [Latest Price]
VAR PriceLastYear = 
    CALCULATE(
        AVERAGE(RawData[price_brent]),
        DATEADD(DateTable[Date], -12, MONTH)
    )
RETURN
    DIVIDE(CurrentPrice - PriceLastYear, PriceLastYear) * 100

Latest Production = 
CALCULATE(
    LASTNONBLANK(RawData[production_kbpd], 1),
    LASTDATE(RawData[date])
)

Average Price = AVERAGE(RawData[price_brent])

Price Volatility = STDEV.P(RawData[price_brent])
```

### Model Performance Measures
```dax
Forecast Error = 
AVERAGE(BacktestResults[residual])

Avg Abs Pct Error = 
AVERAGE(BacktestResults[abs_pct_error])

Best Model = 
VAR MinRMSE = MIN(ModelSummary[RMSE])
RETURN
    CALCULATE(
        FIRSTNONBLANK(ModelSummary[model], 1),
        ModelSummary[RMSE] = MinRMSE
    )
```

---

## Step 4: Build Dashboard Pages

### Page 1: Executive Overview

**Layout (recommended):**

```
+--------------------------------------------------+
|  BRENT CRUDE OIL FORECASTING DASHBOARD           |
+----------+----------+----------+---------+-------+
| Card:    | Card:    | Card:    | Card:   | Card: |
| Latest   | YoY %   | Latest   | Best    | Best  |
| Price    | Change   | Prod.    | Model   | MAPE  |
+----------+----------+----------+---------+-------+
|                                |                  |
|  LINE CHART                    |  CLUSTERED BAR   |
|  Brent Price over Time         |  RMSE by Model   |
|  (RawData: date vs price)      |                  |
|                                |                  |
+--------------------------------+------------------+
|                                |                  |
|  LINE CHART                    |  CLUSTERED BAR   |
|  Production over Time          |  MAPE by Model   |
|  (RawData: date vs production) |                  |
|                                |                  |
+--------------------------------+------------------+
```

**How to build:**
1. **Cards** (top row): Drag each KPI measure into a Card visual
2. **Line Chart** (left): X = `date`, Y = `price_brent` from `RawData`
3. **Bar Charts** (right): X = `model`, Y = `RMSE` / `MAPE_pct` from `ModelSummary`
4. Add a **Date Slicer** at the top for filtering

### Page 2: Model Comparison

```
+--------------------------------------------------+
|  MODEL BENCHMARKING                               |
+--------------------------------------------------+
|  Slicer: [Model] dropdown                        |
+------------------------+-------------------------+
|                        |                         |
|  LINE CHART            |  SCATTER PLOT           |
|  Actual vs Predicted   |  Actual (X) vs          |
|  over Time             |  Predicted (Y)          |
|  (BacktestResults)     |  color by Model         |
|                        |                         |
+------------------------+-------------------------+
|                        |                         |
|  TABLE                 |  HISTOGRAM / AREA       |
|  Model Summary         |  Residual Distribution  |
|  (ModelSummary table)  |  (BacktestResults)      |
|                        |                         |
+------------------------+-------------------------+
```

**How to build:**
1. **Slicer**: Field = `BacktestResults[model]`, style = Dropdown
2. **Line Chart**: X = `date`, Y = `actual` + `predicted` (both from `BacktestResults`)
3. **Scatter**: X = `actual`, Y = `predicted`, Legend = `model`
4. **Table**: Drag all columns from `ModelSummary`

### Page 3: Feature Analysis

```
+--------------------------------------------------+
|  FEATURE ENGINEERING & IMPORTANCE                 |
+--------------------------------------------------+
|                                                  |
|  BAR CHART (horizontal)                          |
|  Feature Importances                             |
|  Y = feature, X = importance                    |
|  Sorted descending                               |
|                                                  |
+------------------------+-------------------------+
|                        |                         |
|  MATRIX / HEATMAP      |  LINE CHART             |
|  Correlation table     |  Lag Features           |
|  (from Features table) |  over Time              |
|                        |                         |
+------------------------+-------------------------+
```

---

## Step 5: Formatting Tips for DA Portfolio

### Color Theme
Use a professional oil/energy palette:
```
Primary:    #1B2838  (dark navy)
Secondary:  #2E86AB  (steel blue)
Accent 1:   #F18F01  (amber/oil)
Accent 2:   #C73E1D  (alert red)
Accent 3:   #3B9C65  (success green)
Background: #F5F5F5  (light gray)
```

Apply via **View** > **Themes** > **Customize Current Theme**

### Professional Touches
- [ ] Add a **company-style title bar** at the top of each page
- [ ] Use **conditional formatting** on the model comparison table (green = best, red = worst)
- [ ] Add **tooltips** on the backtest chart showing MAPE at each point
- [ ] Include **bookmarks** for ARIMA / Prophet / RF quick-views
- [ ] Add an **info button** (i) with methodology text overlay

### Conditional Formatting (Table)
1. Select the `ModelSummary` table visual
2. Click on `RMSE` column > **Conditional Formatting** > **Background Color**
3. Set: Rules-based, If value < 10 then Green, If value > 20 then Red

---

## Step 6: Publish (Optional but Impressive)

1. **File** > **Publish** > Select your Power BI workspace
2. Set up **Scheduled Refresh** (if using a gateway):
   - Settings > Datasets > Schedule Refresh
   - Re-run `python export_for_powerbi.py` periodically to update CSVs
3. Share the public link on your GitHub README

---

## Step 7: Python Visual in Power BI (Advanced)

You can embed Python-generated plots directly in Power BI:

1. Add a **Python Visual** from the Visualizations pane
2. Drag `date`, `actual`, `predicted`, `model` from `BacktestResults` into the Values well
3. In the Python script editor, paste:

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 5))

for model_name, group in dataset.groupby("model"):
    ax.plot(group["date"], group["actual"], color="black", alpha=0.3)
    ax.plot(group["date"], group["predicted"], label=model_name, alpha=0.8)

ax.set_title("Backtest: Actual vs Predicted")
ax.set_ylabel("USD / bbl")
ax.legend()
plt.tight_layout()
plt.show()
```

This shows recruiters you can bridge Python and Power BI.

---

## Resume Bullet Point

After completing the dashboard, you can write:

> **Oil Price & Production Forecasting Dashboard** (2026)  
> Built a Power BI dashboard on Brent crude prices and EIA production data;
> benchmarked ARIMA, Prophet, and Random Forest using rolling-window backtests
> (MAPE 6%) with Python data pipeline and interactive Power BI reporting.
