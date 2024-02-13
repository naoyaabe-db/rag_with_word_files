# Databricks notebook source
# MAGIC %md
# MAGIC ## コンフィグ
# MAGIC #### 以下のコンフィグは、ご自身の環境に合わせて書き換えてください

# COMMAND ----------

# DBTITLE 1,任意の名称を付与すれば良いコンフィグ
# 作成するテーブルとVector Search Indexを格納するカタログ
catalog_name = "nabe_rag_demo_catalog"
# 作成するテーブルとVector Search Indexを格納するスキーマ (上記のカタログ配下にスキーマが作成されます)
schema_name = "rag_word"

# 作成するVector Search Endpointに付ける名前
vector_search_endpoint_name = "one-env-shared-prod"
# 作成するVector Search Indexに付ける名前
index_name = "word_rag_vector_index"

# 埋め込みモデルを動作させるモデルサービングエンドポイントに付ける名前
embedding_endpoint_name = "openai-embedding-endpoint"

# OpenAIのチャットモデルをデプロイするエンドポイントに付ける名前
chat_model_endpoint_name = "openai-chat-endpoint-nabe"
# RAGチェーンに付ける名前
rag_model_name = "rag_nabe"
# MLFlowエクスペリメントを格納するディレクトリ名
experiment_dir_name = "nabe"
# MLFlowエクスペリメントに付ける名前
experiment_name = "rag_expeiment_nabe"
# RAGCチェーンをデプロイするエンドポイントに付ける名前
rag_endpoint_name = "rag_endpoint_nabe"

# COMMAND ----------

# DBTITLE 1,予め行った設定等を確認して指定すべき情報
# 生のWordファイルが格納されているVolumeのパス
raw_file_path = "/Volumes/rag_word_jp_catalog/rag_word_jp_schema/rag_word_jp_volume"

# Notebook「01_Wordファイル取り込み・加工」で加工済みデータを書き込んだテーブル名
source_table_name = "chunked_contents"

# 予め作成したService PrincipalのApplication ID
sp_name = "naoya.abe@databricks.com"
# 予め作成したService Principalに紐づくPATが保存されたSecret Scopeの名前
scope_name = "fieldeng"
# 予め作成したService Principalに紐づくPATが保存されたSecret Keyの名前
secret_name = "nabe-field-eng-ws"

openai_scope_name = "fieldeng"
openai_secret_name = "nabe_openai"

# COMMAND ----------

# DBTITLE 1,デフォルトのカタログ・スキーマを指定
# デフォルトで使用するカタログとスキーマを上記で指定したものに変えておく
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
spark.sql(f"USE SCHEMA {schema_name}")
