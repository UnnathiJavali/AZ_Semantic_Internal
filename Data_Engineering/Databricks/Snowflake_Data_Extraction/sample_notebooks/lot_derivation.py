# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Line of therapy derivation: lung
# MAGIC
# MAGIC Works out the line of therapy for each lung patient from treatment claims.
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Reads** | `az_semantic_sf_db.gold_oncology.fct_patient_claims` |
# MAGIC | **Writes** | `az_mock_derived.oncology.patient_lot` |
# MAGIC | **Grain** | One row per claim, with the patient's line of therapy at that claim |

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS az_mock_derived.oncology")

# COMMAND ----------

from pyspark.sql import functions as F, Window as W
claims = spark.table("az_semantic_sf_db.gold_oncology.fct_patient_claims").filter("tumour = 'Lung'")
w = W.partitionBy("patient_token").orderBy("service_date")
lot = (claims
  .withColumn("prev_date", F.lag("service_date").over(w))
  .withColumn("prev_regimen", F.lag("regimen").over(w))
  .withColumn("gap_days", F.datediff("service_date", "prev_date"))
  .withColumn("new_line", F.when(F.col("prev_date").isNull(), 1)
                           .when(F.col("regimen") != F.col("prev_regimen"), 1)
                           .when(F.col("gap_days") > 90, 1).otherwise(0))
  .withColumn("line_of_therapy", F.sum("new_line").over(w)))
lot.write.mode("overwrite").saveAsTable("az_mock_derived.oncology.patient_lot")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Patients by sub-type and line of therapy

# COMMAND ----------

display(
    spark.table("az_mock_derived.oncology.patient_lot")
    .groupBy("sub_type", "line_of_therapy")
    .agg(F.countDistinct("patient_token").alias("patients"))
    .orderBy("sub_type", "line_of_therapy")
)