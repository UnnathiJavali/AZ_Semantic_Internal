# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # HCP engagement: monthly
# MAGIC
# MAGIC Per doctor per month: lung patients treated, whether a rep visited, and whether the doctor opened an email.
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Reads** | `az_semantic_sf_db.gold_mdm.dim_hcp`, `az_semantic_sf_db.gold_oncology.fct_patient_claims`, `az_semantic_sf_db.gold_crm.fct_calls`, `az_semantic_sf_db.gold_omni.fct_email_events` |
# MAGIC | **Writes** | `az_mock_derived.engagement.hcp_engagement_monthly` |
# MAGIC | **Grain** | One row per doctor (`hcp_id`) per month |

# COMMAND ----------

from pyspark.sql import functions as F

spark.sql("CREATE SCHEMA IF NOT EXISTS az_mock_derived.engagement")

hcp = spark.table("az_semantic_sf_db.gold_mdm.dim_hcp")
claims = spark.table("az_semantic_sf_db.gold_oncology.fct_patient_claims").filter("tumour = 'Lung'")
calls = spark.table("az_semantic_sf_db.gold_crm.fct_calls")
emails = spark.table("az_semantic_sf_db.gold_omni.fct_email_events")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Activity per doctor per month

# COMMAND ----------

patients = (claims
  .withColumn("month", F.date_format("service_date", "yyyy-MM"))
  .groupBy(F.col("hcp_npi").alias("npi"), "month")
  .agg(F.countDistinct("patient_token").alias("lung_patients")))

visits = (calls
  .withColumn("month", F.date_format("call_date", "yyyy-MM"))
  .groupBy("crm_id", "month")
  .agg(F.count("*").alias("calls"),
       F.sum(F.when(F.col("channel") == "in-person", 1).otherwise(0)).alias("in_person_calls")))

email_activity = (emails
  .withColumn("month", F.date_format("send_date", "yyyy-MM"))
  .groupBy("npi", "month")
  .agg(F.count("*").alias("emails_sent"),
       F.max("opened").alias("emails_opened_flag")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## One row per doctor per month, including months with no activity

# COMMAND ----------

months = (patients.select("month")
  .union(visits.select("month"))
  .union(email_activity.select("month"))
  .distinct())

base = (hcp
  .select("hcp_id", "npi", "crm_id", "TERRITORY", "specialty")
  .crossJoin(months))

engagement = (base
  .join(patients, ["npi", "month"], "left")
  .join(visits, ["crm_id", "month"], "left")
  .join(email_activity, ["npi", "month"], "left")
  .fillna(0, subset=["lung_patients", "calls", "in_person_calls", "emails_sent"])
  .withColumn("visited", F.col("calls") > 0)
  .withColumn("email_opened", F.coalesce(F.col("emails_opened_flag"), F.lit(False)))
  .select("month", "hcp_id", "npi", "crm_id", "TERRITORY", "specialty",
          "lung_patients", "calls", "in_person_calls", "visited",
          "emails_sent", "email_opened"))

engagement.write.mode("overwrite").saveAsTable("az_mock_derived.engagement.hcp_engagement_monthly")

# COMMAND ----------

display(
    spark.table("az_mock_derived.engagement.hcp_engagement_monthly")
    .groupBy("month")
    .agg(F.sum("lung_patients").alias("lung_patients"),
         F.sum(F.col("visited").cast("int")).alias("doctors_visited"),
         F.sum(F.col("email_opened").cast("int")).alias("doctors_opened_email"))
    .orderBy("month")
)