# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Next best action: lung doctors
# MAGIC
# MAGIC Simple rules on the last 3 months of engagement, first match wins:
# MAGIC 1. Large doctor, not visited: visit.
# MAGIC 2. Visited, but not opening emails: email.
# MAGIC 3. Medium doctor, not visited, not opening emails: email.
# MAGIC 4. Medium doctor, not visited, but opening emails: remote call.
# MAGIC 5. Otherwise: no action.
# MAGIC
# MAGIC Size tiers come from lung patients in the last 3 months: top 20% Large, next 30% Medium.
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Reads** | `az_mock_derived.engagement.hcp_engagement_monthly` |
# MAGIC | **Writes** | `az_mock_derived.engagement.nba_recommendation` |
# MAGIC | **Grain** | One row per doctor (`hcp_id`) for the latest month |

# COMMAND ----------

from pyspark.sql import functions as F, Window as W

eng = spark.table("az_mock_derived.engagement.hcp_engagement_monthly")

last_3_months = [r["month"] for r in eng.select("month").distinct().orderBy(F.desc("month")).limit(3).collect()]
as_of_month = max(last_3_months)
print(f"Recommendations as of {as_of_month}, using {sorted(last_3_months)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Doctor size and reach over the last 3 months

# COMMAND ----------

recent = (eng
  .filter(F.col("month").isin(last_3_months))
  .groupBy("hcp_id", "npi", "crm_id", "territory", "specialty")
  .agg(F.sum("lung_patients").alias("lung_patients_3m"),
       F.max(F.col("visited").cast("int")).alias("visited_3m"),
       F.sum("emails_sent").alias("emails_sent_3m"),
       F.max(F.col("email_opened").cast("int")).alias("email_opened_3m")))

size_rank = F.percent_rank().over(W.orderBy("lung_patients_3m"))
recent = recent.withColumn(
    "size_tier",
    F.when(F.col("lung_patients_3m") == 0, "None")
     .when(size_rank >= 0.8, "Large")
     .when(size_rank >= 0.5, "Medium")
     .otherwise("Small"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rules

# COMMAND ----------

large, medium = F.col("size_tier") == "Large", F.col("size_tier") == "Medium"
treats = F.col("lung_patients_3m") > 0
visited, opened = F.col("visited_3m") == 1, F.col("email_opened_3m") ==1

nba =(recent
  .withColumn("recommended_action",
      F.when(large & ~visited, "VISIT")
       .when(treats & visited & ~opened, "EMAIL")
       .when(medium & ~visited & ~opened, "EMAIL")
       .when(medium & ~visited & opened, "REMOTE_CALL")
       .otherwise("NO_ACTION"))
  .withColumn("channel",
      F.when(F.col("recommended_action").isin("VISIT", "REMOTE_CALL"), "Field")
       .when(F.col("recommended_action") == "EMAIL", "Digital"))
  .withColumn("priority",
      F.when(large & ~visited, 1)
       .when(large & visited & ~opened, 2)
       .when(medium & ~visited, 3)
       .when(treats & visited & ~opened, 4)
       .otherwise(9))
  .withColumn("reason",
      F.when(large & ~visited, "Large lung doctor, no visit in the last 3 months")
       .when(treats & visited & ~opened, "Visited, but not opening emails")
       .when(medium & ~visited & ~opened, "No visit and no email opened in the last 3 months")
       .when(medium & ~visited & opened, "Opens emails, but no visit in the last 3 months")
       .otherwise("Reach on track"))
  .withColumn("as_of_month", F.lit(as_of_month))
  .withColumn("created_at", F.current_timestamp())
  .select("as_of_month", "hcp_id", "npi", "crm_id", "territory", "specialty",
          "lung_patients_3m", "size_tier", "visited_3m", "emails_sent_3m", "email_opened_3m",
          "recommended_action", "channel", "priority", "reason", "created_at"))

nba.write.mode("overwrite").saveAsTable("az_mock_derived.engagement.nba_recommendation")

# COMMAND ----------

display(
    spark.table("az_mock_derived.engagement.nba_recommendation")
    .groupBy("recommended_action", "channel")
    .count()
    .orderBy("recommended_action")
)