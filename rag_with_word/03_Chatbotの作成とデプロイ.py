# Databricks notebook source
# MAGIC %md
# MAGIC # 3. Chatbotの作成とデプロイ
# MAGIC
# MAGIC <br/>
# MAGIC <img src="https://github.com/naoyaabe-db/public_demo_images/blob/3380b6d73937cd95efae845799c37de910b7394c/rag_demo_images/diagram_notebook3.png?raw=true" style="float: right" width="1000px">
# MAGIC <br/>
# MAGIC
# MAGIC ### このNotebookの流れ
# MAGIC このNotebookでは、OpenAIのチャットモデルと前のNotebookで作成したVector Search Indexを組み合わせて、RAGを使用したChatbotを完成させます。<br/><br/>
# MAGIC
# MAGIC 1. 必要なライブラリのインストール
# MAGIC 2. コンフィグ（自身の環境に合わせた各種パラメータの指定）
# MAGIC 3. カタログとスキーマのアクセス権限付与
# MAGIC 4. Vector Search Indexへの権限付与
# MAGIC 5. LangChain Retriever の作成
# MAGIC 6. Chatで使用するOpenAIモデルをエンドポイントとしてデプロイ
# MAGIC 7. RAG Chainを作成する
# MAGIC 8. 作成したRAGチェーンをMLFlowモデルレジストリへ登録する
# MAGIC 9. RAGチェーンをモデルサービングエンドポイントにデプロイする
# MAGIC
# MAGIC ### このNotebookで出てくる主な機能・技術
# MAGIC このNotebookでは、以下の解説されている機能・技術を使用しています。より詳細を調べたい場合は各リンク先のドキュメントをご覧ください。<br/><br/>
# MAGIC
# MAGIC - Unity Catalogによる各オブジェクトの権限管理 [(Databricks公式ドキュメント)](https://docs.databricks.com/ja/data-governance/unity-catalog/manage-privileges/privileges.html)
# MAGIC - ノートブック上でのシークレットの使用 [(Databricks公式ドキュメント)](https://docs.databricks.com/ja/security/secrets/example-secret-workflow.html#use-the-secrets-in-a-notebook)
# MAGIC - 外部モデルのモデルサービングエンドポイント [(Databricks公式ドキュメント)](https://docs.databricks.com/ja/generative-ai/external-models/external-models-tutorial.html)
# MAGIC - LangChainのDatabricks Vector Searchインテグレーション [(LangChain APIリファレンス)](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.databricks_vector_search.DatabricksVectorSearch.html)
# MAGIC - LangChain RetrievalQAチェーン [(LangChain APIリファレンス)](https://api.python.langchain.com/en/stable/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html)
# MAGIC - MLFlowのLangChainインテグレーション [(MLFlow公式ドキュメント)](https://mlflow.org/docs/latest/python_api/mlflow.langchain.html)
# MAGIC
# MAGIC
# MAGIC ### このNotebookの動作環境
# MAGIC Databricks Runtime Version 14.2 ML のクラスター

# COMMAND ----------

# MAGIC %md
# MAGIC ## 必要なライブラリのインストール

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch mlflow[genai]>=2.9.0 langchain==0.0.344 databricks-sdk==0.12.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## コンフィグのロード
# MAGIC #### 別のNotebook `config` の中の変数名を自身の環境用に書き換えてから下記を実行してください。

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC ## カタログとスキーマのアクセス権限の付与

# COMMAND ----------

spark.sql(f'GRANT USAGE ON CATALOG {catalog_name} TO `{sp_name}`');
spark.sql(f'GRANT USAGE ON DATABASE {catalog_name}.{schema_name} TO `{sp_name}`');

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Search Indexへの権限付与
# MAGIC 以下の手順で、GUIからVector Search Indexへの権限付与を行う。<br/><br/>
# MAGIC 1. 左のメインメニューから`Catalog`を選択
# MAGIC 2. Catalog Explorer内で自身が作成したVector Search Indexの画面まで行く
# MAGIC 3. `Permission`タブを開き、`GRANT`ボタンを押す
# MAGIC <br/>
# MAGIC <img src="https://github.com/naoyaabe-db/public_demo_images/blob/f0bfafc1d892a93c6397bc279c1c0779f7bf4275/rag_demo_images/find_vs_index.png?raw=true" style="float: right" width="600px">
# MAGIC <br/>
# MAGIC 4. 管理者から提供されるService Principalに対して、`SELECT`権限を付与する
# MAGIC <br/>
# MAGIC <img src="https://github.com/naoyaabe-db/public_demo_images/blob/c59e134528fb56d7bcd762d73fe5167a3cf2ff82/rag_demo_images/vector_index_permission.png?raw=true" style="float: right" width="600px">

# COMMAND ----------

import os

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(f"{scope_name}", f"{secret_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 埋め込みモデルのサービングエンドポイントに対する権限付与
# MAGIC 以下の手順で、GUIから埋め込みモデルのサービングエンドポイントへの権限付与を行う。<br/><br/>
# MAGIC 1. 左のメインメニューから`Serving`を選択
# MAGIC 2. エンドポイントの一覧から、前のNotebook「02_Vector Search Indexの構築」で自身が作成した埋め込みモデルのサービングエンドポイントを開く
# MAGIC <br/>
# MAGIC <img src="https://github.com/naoyaabe-db/public_demo_images/blob/9a0bb53524ec7009fe6167287b11431f223f17fb/rag_demo_images/model_serving_endpoint_list.png?raw=true" style="float: right" width="600px">
# MAGIC <br/>
# MAGIC 3. 画面右上の`Permission`ボタンから権限設定の画面を開き、Service Principalに対して`Can Manage`権限を付与する
# MAGIC <br/>
# MAGIC <img src="https://github.com/naoyaabe-db/public_demo_images/blob/9a0bb53524ec7009fe6167287b11431f223f17fb/rag_demo_images/model_serving_endpoint_permission.png?raw=true" style="float: right" width="600px">

# COMMAND ----------

# MAGIC %md
# MAGIC ## LangChain Retriever の作成

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch

def get_retriever(persist_dir: str = None):
    # Vector Search Client の作成
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])

    # 作成したVector Search Indexをロード
    vs_index = vsc.get_index(
        endpoint_name=vector_search_endpoint_name,
        index_name=f"{catalog_name}.{schema_name}.{index_name}"
        )

    # LangChainのvectorstoresオブジェクトにする
    vectorstore = DatabricksVectorSearch(vs_index)
    return vectorstore.as_retriever()

vectorstore_ret = get_retriever()

# COMMAND ----------

# Retrieverのテスト
similar_documents = vectorstore_ret.get_relevant_documents("顧客情報の用途は？")
print(f"Relevant documents: {similar_documents[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chatで使用するOpenAIモデルをエンドポイントとしてデプロイ
# MAGIC
# MAGIC #### このサンプルではOpenAIを使用していますが、Azure OpenAIを使用する場合は以下のドキュメントに従ってコードを変更してください
# MAGIC https://docs.databricks.com/ja/generative-ai/external-models/external-models-tutorial.html
# MAGIC #### 以下のコード内の`openai_api_key` は 予め管理者が作成したService Principalの情報を使って、`{{secrets/スコープ名/シークレット名}}`に書き換えてください

# COMMAND ----------

# DBTITLE 1,チャットモデルのサービングエンドポイント作成
import mlflow.deployments

# MLFLow Deployments の各種機能を操作するためのクライアントを用意
mlflow_deploy_client = mlflow.deployments.get_deploy_client("databricks")

try:
    # MLFLow Deployments のクライアントを使い、OpenAIのチャットモデル(gpt-3.5-turbo)への
    # Proxyとなるモデルサービングエンドポイントを作成する
    mlflow_deploy_client.create_endpoint(
        name=chat_model_endpoint_name,
        config={
            "served_entities": [{
                "external_model": {
                    "name": "gpt-3.5-turbo",
                    "provider": "openai",
                    "task": "llm/v1/chat",
                    "openai_config": {
                        # 下記は管理者から提供されたService Principalのシークレット情報で書き換える
                        "openai_api_key": "{{secrets/fieldeng/nabe_openai}}"
                    }
                }
        }]
        }
    )
except Exception as e:
    print(e)

# COMMAND ----------

# DBTITLE 1,OpenAIのチャットモデル単体でテスト（文脈と異なる、期待していない答え）
from langchain.chat_models import ChatDatabricks

# 今回の文脈では「サブプロセッサー」はサードパーティの事業者といった意味だが、
# OpenAIのチャットモデル単体では全く違う意味の答えが返ってくる
chat_model = ChatDatabricks(endpoint=chat_model_endpoint_name, max_tokens = 200)
print(f"Test chat model: {chat_model.predict('サブプロセッサーとは何ですか？')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAG Chainを作成する

# COMMAND ----------

# DBTITLE 1,RAGチェーンの作成
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

TEMPLATE = """あなたはDatabricksの顧客情報の取り扱い規定を熟知しています。次に与えられるコンテキストを使用して、その後の質問に答えなさい:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

# ここまで用意したチャットモデル、Vector Search Index、プロンプトテンプレートを1つのチェーンにまとめる
chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=vectorstore_ret,
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

# DBTITLE 1,RAGチェーンでテスト（ドキュメントの内容を踏まえた期待通りの答え）
# 先ほどのOpenAIのチャットモデル単体では全く違う意味の答えが返ってきたが、
# 今度はVector Search内の情報を使って適切な答えを返してくれる
question = {"query": "サブプロセッサーとは何ですか？"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 作成したRAGチェーンをMLFlowモデルレジストリへ登録する

# COMMAND ----------

# DBTITLE 1,MLFlowエクスペリメントを事前に作成・指定
import requests

api_key = dbutils.secrets.get(f"{scope_name}", f"{secret_name}")

xp_root_path = f"/Shared/rag_demo/experiments/{experiment_dir_name}"
r = requests.post(f"{host}/api/2.0/workspace/mkdirs", headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}, json={ "path": xp_root_path})
mlflow.set_experiment(f"{xp_root_path}/{experiment_name}")

# COMMAND ----------

from mlflow import MlflowClient

# モデルレジストリから最新バージョンの番号を取得する
def get_latest_model_version(model_name):
    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

# DBTITLE 1,RAGチェーンをMLFlowにロギングし、モデルレジストリに登録
from mlflow.models import infer_signature
import mlflow
import langchain

# Databricks Unity Catalog上にモデルを保存するよう指定
mlflow.set_registry_uri("databricks-uc")
# モデルレジストリに登録する際のモデル名
model_name = f"{catalog_name}.{schema_name}.{rag_model_name}"

with mlflow.start_run(run_name="dbdemos_chatbot_rag") as run:
    # 上のセルでサンプルとして試した質問と回答を、入出力のシグネチャとして登録
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow[genai]>=2.9.0",
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

    import mlflow.models.utils
    mlflow.models.utils.add_libraries_to_model(
        f"models:/{model_name}/{get_latest_model_version(model_name)}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAGチェーンをモデルサービングエンドポイントにデプロイする
# MAGIC
# MAGIC #### 以下のコード内の`DATABRICKS_TOKEN`と`DATABRICKS_HOST`は 予め管理者が作成したService Principalの情報を使って、`{{secrets/スコープ名/シークレット名}}`に書き換えてください

# COMMAND ----------

# DBTITLE 1,RAGチェーンのデプロイ
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

latest_model_version = get_latest_model_version(model_name)

#　モデルサービングエンドポイントへのデプロイ内容を指定
w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=rag_endpoint_name,
    served_models=[
        ServedModelInput(
            # デプロイするモデルのモデルレジストリ上での名前
            model_name=model_name,
            # デプロイするモデルのバージョン（上で取得した最新バージョンを指定）
            model_version=latest_model_version,
            # モデルサービングエンドポイントのキャパシティ
            workload_size="Small",
            # リクエストが無い時はリソースをゼロまでスケールダウンする
            scale_to_zero_enabled=True,
            # モデルサービングエンドポイントに対して渡す環境変数 (SPのシークレット)
            environment_vars={
                # 下記は、SPのワークスペース接続用のシークレットに書き換え
                "DATABRICKS_TOKEN": "{{secrets/fieldeng/nabe-field-eng-ws}}",
                # 下記は、SPがシークレットに保存しているワークスペースのホスト名に書き換え
                "DATABRICKS_HOST": "{{secrets/fieldeng/nabe-field-eng-host}}"
            }
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == rag_endpoint_name), None
)

# 同じ名前のモデルサービングエンドポイントがまだデプロイされていない場合は新規にデプロイ
if existing_endpoint == None:
    print(f"Creating the endpoint {rag_endpoint_name}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=rag_endpoint_name, config=endpoint_config)
# 同じ名前のモデルサービングエンドポイントがすでにプロイされている場合はアップデート
else:
    print(f"Updating the endpoint {rag_endpoint_name} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=rag_endpoint_name)

# COMMAND ----------

# DBTITLE 1,デプロイしたRAGチェーンのテスト
question = "サブプロセッサーとは何ですか？"

answer = w.serving_endpoints.query(rag_endpoint_name, inputs=[{"query": question}])
print(answer.predictions[0])
