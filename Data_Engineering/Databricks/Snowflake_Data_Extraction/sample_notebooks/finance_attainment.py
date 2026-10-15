# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Forecast attainment: Imfinzi lung, year to date
# MAGIC
# MAGIC For the finance team. Prints year-to-date net sales against forecast.
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Reads** | `az_semantic_sf_db.gold_sales.fct_demand_monthly`, `az_semantic_sf_db.gold_finance.fct_forecast_monthly` |
# MAGIC | **Writes** | Nothing; prints the number |

# COMMAND ----------

from pyspark.sql import functions as F

sales = spark.table("az_semantic_sf_db.gold_sales.fct_demand_monthly")
forecast = spark.table("az_semantic_sf_db.gold_finance.fct_forecast_monthly")

month_key = F.substring(F.col("month").cast("string"), 1, 7)

# COMMAND ----------

imfinzi_sales = (sales
  .filter((F.col("brand_id") == "B01") & (F.col("tumour") == "Lung"))
  .withColumn("month_key", month_key)
  .filter(F.col("month_key").startswith("2026")))

first_month, last_month = imfinzi_sales.agg(F.min("month_key"), F.max("month_key")).first()
net_sales = imfinzi_sales.agg(F.sum("net_sales")).first()[0]

imfinzi_forecast = (forecast
  .filter((F.col("cycle") == "BUDGET_2026") & (F.col("brand_id") == "B01") & (F.col("tumour") == "Lung"))
  .withColumn("month_key", month_key)
  .filter(F.col("month_key").between(first_month, last_month)))

forecast_net_sales = imfinzi_forecast.agg(F.sum("fcst_net_sales")).first()[0]

# COMMAND ----------

attainment = float(net_sales) / float(forecast_net_sales)

print(f"Imfinzi lung, {first_month} to {last_month}")
print(f"  Net sales:          ${float(net_sales) / 1e6:,.1f}M")
print(f"  Forecast net sales: ${float(forecast_net_sales) / 1e6:,.1f}M")
print(f"  Forecast attainment: {attainment:.1%}")