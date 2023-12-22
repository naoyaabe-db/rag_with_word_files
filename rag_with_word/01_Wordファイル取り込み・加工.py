# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Wordファイル取り込み・加工
# MAGIC
# MAGIC ### このNotebookの流れ
# MAGIC このNotebookでは、Volumeまたは外部ロケーションに配置したWordファイルを取り込み、<br/>
# MAGIC 加工・チャンク化した上でテーブルに書き出すところまでを行います。以下の流れです。<br/><br/>
# MAGIC 1. 必要なライブラリのインストール
# MAGIC 2. コンフィグ（自身の環境に合わせた各種パラメータの指定）
# MAGIC 3. AutoLoaderを使用して取り込み対象のファイルをリストアップ
# MAGIC 4. Wordファイルのチャンク化処理
# MAGIC
# MAGIC ### このNotebookで出てくる主な機能・技術
# MAGIC このNotebookでは、以下の解説されている機能・技術を使用しています。より詳細を調べたい場合は各リンク先のドキュメントをご覧ください。<br/><br/>
# MAGIC
# MAGIC - Spark(Python版)の基本 [(Databricks公式ドキュメント)](https://docs.databricks.com/ja/getting-started/dataframes-python.html)
# MAGIC - AutoLoader [(Databricks公式ドキュメント)](https://docs.databricks.com/ja/ingestion/auto-loader/index.html)
# MAGIC - Wordファイルローダー [(LangChain APIリファレンス)](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.word_document.Docx2txtLoader.html?highlight=docx2txtloader#langchain_community.document_loaders.word_document.Docx2txtLoader)
# MAGIC - RecursiveCharacterTextSplitter [(LangChain APIリファレンス)](https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html?highlight=recursive%20text#langchain.text_splitter.RecursiveCharacterTextSplitter)
# MAGIC - SparkのPandas UDF [(Databricks公式ドキュメント)](https://docs.databricks.com/ja/udf/pandas.html)
# MAGIC
# MAGIC ### このNotebookの動作環境
# MAGIC Databricks Runtime Version 14.2 ML のクラスター

# COMMAND ----------

# MAGIC %md
# MAGIC ## 必要なライブラリのインストール

# COMMAND ----------

# MAGIC %pip install docx2txt

# COMMAND ----------

# MAGIC %md
# MAGIC ## コンフィグ
# MAGIC #### 以下のコンフィグは、ご自身の環境に合わせて書き換えてください

# COMMAND ----------

# 作成するテーブルとVector Search Indexを格納するカタログ
catalog_name = "nabe_rag_demo_catalog"
# 作成するテーブルとVector Search Indexを格納するスキーマ (上記のカタログ配下にスキーマが作成されます)
schema_name = "rag_word"
# 生のWordファイルが格納されているVolumeのパス（外部ロケーションに直接格納している場合は、そのパス(s3://xxxxx....)でも良い）
raw_file_path = "/Volumes/nabe_rag_demo_catalog/rag_word/raw_word_doc"
# AutoLoaderがどのファイルまで取り込んだかを記録するための、チェックポイント情報を保存するパス。
# Volumeまたは外部ロケーション上で任意のディレクトリを決めておき、それを指定する。
checkpoint_path = "/Volumes/nabe_rag_demo_catalog/rag_word/autoloader_checkpoint"

# デフォルトで使用するカタログとスキーマを上記で指定したものに変えておく
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
spark.sql(f"USE SCHEMA {schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## AutoLoaderを使用して取り込み対象のファイルをリストアップ

# COMMAND ----------

# DBTITLE 1,ディレクトリに存在するファイルを確認
# ディレクトリにあるファイル一覧を表示
display(dbutils.fs.ls(raw_file_path))

# COMMAND ----------

# DBTITLE 1,AutoLoader Checkpointのリフレッシュ
# 2回目以降に本Notebookを実行する場合で、最初のファイルから全て取り込み直したい場合は
# 下記のチェックポイント外して実行し、チェックポイントの情報を削除する。

# dbutils.fs.rm(f"{checkpoint_path}/metadata")
# dbutils.fs.rm(f"{checkpoint_path}/commits", recurse=True)
# dbutils.fs.rm(f"{checkpoint_path}/offsets", recurse=True)
# dbutils.fs.rm(f"{checkpoint_path}/sources", recurse=True)

# COMMAND ----------

# DBTITLE 1,生ファイルのリストをテーブルに取り込む
from pyspark.sql.functions import regexp_replace

# raw_file_path内に格納されたWordファイルのリストを取得する
df = (spark.readStream
        .format('cloudFiles')
        .option('cloudFiles.format', 'BINARYFILE')
        .option("pathGlobfilter", "*.docx")
        .load(raw_file_path)
        # Wordファイルを Volume に置いている場合はregexp_replaceを使用してパスを加工
        .withColumn("file_path", regexp_replace("path", "dbfs:", ""))
        # Wordファイルを 外部ロケーション に置いている場合はそのまま
        #.withColumn("file_path", "path")
        .select("file_path", "modificationTime", "length"))

# 取得したファイルリストをテーブルに書き込む
(df.writeStream
  .trigger(once=True)
  .option("checkpointLocation", checkpoint_path)
  .table('raw_word_files').awaitTermination())

# COMMAND ----------

# DBTITLE 1,取り込んだテーブルの確認
# MAGIC %sql
# MAGIC SELECT * FROM raw_word_files

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wordファイルのチャンク化処理

# COMMAND ----------

# DBTITLE 1,　単一のWordファイルを複数のチャンクに分割する関数
def split_wordfile_into_chunks(word_file, chunk_overlap=20, max_chunk_size=300):

  # ファイルパスからWordファイルをロード
  from langchain.document_loaders import Docx2txtLoader
  word_loader = Docx2txtLoader(word_file)
  entire_document = word_loader.load()

  # Wordファイルからロードしたドキュメントを、max_chunk_size以下になるように分割
  # chunk_overlap では隣接するチャンク間での重複範囲を指定
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=max_chunk_size,
    chunk_overlap=chunk_overlap
  )
  documents = text_splitter.split_text(entire_document[0].page_content)

  # 連続する改行、タブ、スペースを削除
  def remove_blanks(txt):
    import re
    txt = re.sub(r'\n+', '\n', txt)
    txt = re.sub(r'\t+', '\t', txt)
    return re.sub(r' +', ' ', txt)

  return [remove_blanks(doc) for doc in documents]

# COMMAND ----------

# DBTITLE 1,上で作成した関数の動作確認
spark.table("raw_word_files").limit(1).collect()[0]['file_path']

sample_file = spark.table("raw_word_files").limit(1).collect()[0]['file_path']

result = split_wordfile_into_chunks(sample_file)
result

# COMMAND ----------

# DBTITLE 1,上で作成した関数をSparkのPandas UDFにする　
import pandas as pd
from pyspark.sql.functions import pandas_udf

@pandas_udf("array<string>")
def parse_and_split(docs: pd.Series) -> pd.Series:
    return docs.apply(split_wordfile_into_chunks)

# COMMAND ----------

# DBTITLE 1,チャンク化したコンテンツを格納するテーブルを事前に作成
# MAGIC %sql
# MAGIC
# MAGIC -- 後でVector Search を使用するためにPrimary Keyが必要となるため、idカラムを作って整数の連番を自動で割り当てる
# MAGIC -- TBLPROPERTIES (delta.enableChangeDataFeed = true) は、後で追加されたコンテンツもVector Searchへ自動同期するために必要な設定
# MAGIC CREATE TABLE IF NOT EXISTS chunked_contents (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   file_path STRING,
# MAGIC   content STRING
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

# DBTITLE 1,作成した関数を使って、Wordファイルのロードとチャンク化を実行
import pyspark.sql.functions as F

# 上で定義したparse_and_split関数を使用して、ドキュメントのチャンク化をSparkで実行
# 処理した結果は上で作成したテーブル "chunked_contents" に書き出す
(spark.table("raw_word_files")
      .withColumn('content', F.explode(parse_and_split('file_path')))
      .select("file_path", "content")
      .write.mode('overwrite').saveAsTable("chunked_contents"))

# テーブルの中身を確認
display(spark.table("chunked_contents"))
