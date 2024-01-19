# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Vector Search Index の構築
# MAGIC
# MAGIC <br/>
# MAGIC <img src="https://github.com/naoyaabe-db/public_demo_images/blob/3380b6d73937cd95efae845799c37de910b7394c/rag_demo_images/diagram_notebook2.png?raw=true" style="float: right" width="1000px">
# MAGIC <br/>
# MAGIC
# MAGIC ### このNotebookの流れ
# MAGIC このNotebookでは、前のNotebookで加工・チャンク化したコンテンツをベクトル化し、Vector Search Indexで類似検索が行えるようにします。<br/><br/>
# MAGIC
# MAGIC 1. 必要なライブラリのインストール
# MAGIC 2. コンフィグ（自身の環境に合わせた各種パラメータの指定）
# MAGIC 3. 埋め込みモデルの準備
# MAGIC 4. Vector Search Endpointの作成
# MAGIC 5. Vector Search Indexの作成
# MAGIC
# MAGIC ### このNotebookで出てくる主な機能・技術
# MAGIC このNotebookでは、以下の解説されている機能・技術を使用しています。より詳細を調べたい場合は各リンク先のドキュメントをご覧ください。<br/><br/>
# MAGIC
# MAGIC - 外部モデルのモデルサービングエンドポイント [(Databricks公式ドキュメント)](https://docs.databricks.com/ja/generative-ai/external-models/external-models-tutorial.html)
# MAGIC - Databricks Vector Search [(Databricks公式ドキュメント)](https://docs.databricks.com/ja/generative-ai/vector-search.html)
# MAGIC
# MAGIC ### このNotebookの動作環境
# MAGIC Databricks Runtime Version 14.2 ML のクラスター

# COMMAND ----------

# MAGIC %md
# MAGIC ## 必要なライブラリのインストール

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch mlflow[genai]>=2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## コンフィグのロード
# MAGIC #### 別のNotebook `config` の中の変数名を自身の環境用に書き換えてから下記を実行してください。

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 埋め込みモデルの準備
# MAGIC
# MAGIC #### このサンプルではOpenAIを使用していますが、Azure OpenAIを使用する場合は以下のドキュメントに従ってコードを変更してください
# MAGIC https://docs.databricks.com/ja/generative-ai/external-models/external-models-tutorial.html
# MAGIC #### 以下のコード内の`openai_api_key` は 管理者から提供されたService Principalの情報を使って、`{{secrets/スコープ名/シークレット名}}`に書き換えてください

# COMMAND ----------

# DBTITLE 1,埋め込みモデルのサービングエンドポイント作成
import mlflow.deployments

# MLFLow Deployments の各種機能を操作するためのクライアントを用意
mlflow_deploy_client = mlflow.deployments.get_deploy_client("databricks")

# MLFLow Deployments のクライアントを使い、OpenAIの埋め込みモデル(text-embedding-ada-002)への
# Proxyとなるモデルサービングエンドポイントを作成する
mlflow_deploy_client.create_endpoint(
    name="openai-embedding-endpoint",
    config={
        "served_entities": [{
            "external_model": {
                "name": "text-embedding-ada-002",
                "provider": "openai",
                "task": "llm/v1/embeddings",
                "openai_config": {
                    # 下記は管理者から提供されたService Principalのシークレット情報で書き換える
                    "openai_api_key": "{{secrets/fieldeng/nabe_openai}}"
                }
            }
    }]
    }
)

# COMMAND ----------

# DBTITLE 1,埋め込みモデル単体でのテスト
test_response = mlflow_deploy_client.predict(
    endpoint="openai-embedding-endpoint",
    inputs={"input": "パーソナル情報とは何ですか"}
)
test_response

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Search Endpointの作成

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# Vector Searchの各種機能を使用するためのクライアントを作成
vector_search_client = VectorSearchClient()

# # Vector Search Endpointを作成
# vector_search_client.create_endpoint(
#     name=vector_search_endpoint_name,
#     endpoint_type="STANDARD"
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Search Indexの作成

# COMMAND ----------

# DBTITLE 1,Managed Embedding方式でVector Search Indexを作成
# Vector Search Indexの作成
index = vector_search_client.create_delta_sync_index(
  # 上で作成したVector Search Endpointの名前
  endpoint_name=vector_search_endpoint_name,
  # チャンク化したコンテンツが入ったテーブル
  source_table_name=f"{catalog_name}.{schema_name}.{source_table_name}",
  # 作成するVector Search Indexの名前
  index_name=f"{catalog_name}.{schema_name}.{index_name}",
  # ソーステーブルへのコンテンツ追加が自動で同期されるよう設定する場合はCONTINUOUSにする
  # デモ環境の都合でここではTRIGGEREDにしている
  pipeline_type='TRIGGERED',
  # PKとして使用するカラム名
  primary_key="id",
  # チャンク化したコンテンツが入ったカラム名
  embedding_source_column="content",
  # 埋め込みモデルのサービングエンドポイント名
  embedding_model_endpoint_name=embedding_endpoint_name
)

# COMMAND ----------

# DBTITLE 1,作成したVector Search Indexをテスト
index = vector_search_client.get_index(
  endpoint_name=vector_search_endpoint_name,
  index_name=f"{catalog_name}.{schema_name}.{index_name}"
)

results = index.similarity_search(
  query_text="サブプロセッサーの定義を教えてください",
  columns=["id", "content"],
  num_results=3
)

results
