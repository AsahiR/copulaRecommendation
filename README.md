# できること

* データセット+推薦手法を指定すると四分交差で推薦テスト+  
テスト結果(評価指標はiP,precision,nDCG)をcsvで保存. 
* csvのテスト結果からmatplotlibによる分析用のプロット画像を出力 
* texのテーブルフォーマット形式で全比較手法の結果比較と有意差検定の結果を出力．

# ディレクトリ構成

    .
    ├── abstract/
    ├── asahi_bthesis/
    ├── cont/
    ├── disc/
    ├── ids.txt
    ├── lib_inf
    ├── readme
    └── src/

* lib_inf  
Rのライブラリリスト.環境構築で参照されたし.

* ids.txt  
割り振るクラスタリングIDを100個保持.  
このファイルからクラスタリングIDを指定して実験を行う．    
defaultでは1行目のIDが使用されるがコマンドで行番号を指定可能(\_\_init\_\_.py参照)  
指定したクラスタリングIDが割り振り済みであった場合はクラスタリング結果が再利用される.  
未使用であった場合は,クラスタリング後にその結果が対象ディレクトリに記録される．

* abstract/  
300字研究概要+概要図

* asahi_bthesis/  
卒論のtexファイル+pdf

* src/  
ソースコード. 
ユーザは\_\_init\_\_.pyへコマンドを与える．    
\_\_init\_\_.pyはパースしたコマンドを基に
他のmoduleを呼び出す.各moduleの詳細はmoduleの冒頭のコメント参照.

* disc/ or cont/   
  - disc=>discontinue.離散値特徴量を含むデータセット+結果.  
  - cont=>continue.連続値のみのデータセット+結果.  

# disc/下の構成. 
    .
    ├── all_items.json
    ├── cluster_id_depend/
    │   └── cluster_id_201803072213_0/
    │       ├── cluster_data/
    │       ├── cluster_data.bak/
    │       ├── label/
    │       ├── param/
    │       ├── plot/
    │       ├── ranking/
    │       ├── result/
    │       └── tex/
    ├── false_data/
    ├── ppl/
    ├── questionnaire/
    ├── train_data/
    └── true_data/  
clusteringの結果はclustering_idと対応しているため 実験済みのclustering_idを指定すればそのidに対応するclustering結果が再利用される.  

# 出力結果を見る． 
* plot    
/disc/cluster_id_depend/cluster_id_201803072213_0/plot/以下最下層のファイル  
cluster_id_201803072213_0の部分はcluster_idに相当する.
* tex  
/disc/cluster_id_depend/cluster_id_201803072213_0/tex/以下最下層のファイル
# 実行方法
\_\_init\_\_.pyへ与えるコマンドはdocoptというモジュールでパースされる.  
\_\_init\_\_.pyの冒頭コメントアウトされてる箇所でdocoptのパースするコマンドの設定を記述.  
以下*argsは各パラメータ値を指定する引数を表す.詳しくは\_\_init\_\_.pyを参照.

* recommendataion test  
    `python __init.py__ method_name *args`  
    でmethod_nameによる推薦テストを開始.  
    例.`python __init__.py kl kde_cv`(提案手法)
* plot  
`python __init__.py plot *args`  
で推薦結果のプロットをする．結果はdisc/深層部のplot/最下層に出力される． 
* for tex  
`python __init__.py get_result_table *args`  
でtexの表形式の全比較手法の結果を出力.

# 環境構築
## python
1. python3をOSに合わせてダウンロード.https://www.continuum.io/downloads
1. 基本的に殆どのモジュールがデフォルトでanacondaに同梱されているがいくつかのモジュールは手動で  
`pip install`する.   
    pip install pyper
    pip install docopt  
## R
1. rのダウンロード https://www.r-project.org/  
1. CRANからコピュラのライブラリをダウンロード　https://cran.r-project.org/web/packages/copula/index.html  


## svmRankのビルド
src/svmRank下で  
`make`  
を実行

