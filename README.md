# b_thesis_top
#できること
1.データセット+推薦手法を指定すると四分交差で推薦テストをする．提案手法はコピュラという確率モデルを用いた協調フィルタリング.
テスト結果(評価指標はiP,precision,nDCG)をcsvで保存する．
2.csvのテスト結果からmatplotlibによる比較用のプロット画像やtexのテーブルフォーマットの出力をする．

#ディレクトリ構成
abstract/,src/,disc/,cont/
@abstract/
研究概要
@src/
ソースコード.
python __init__.py method_name *optional_listでmethod_nameによる推薦テストを開始.
  例.python __init__.py kl kde_cv(提案手法).*optional_listは__init__.pyを参照.
    python __init__.py plot *optionla_listで推薦結果のプロットをする．結果はdisc/以降のplot/以下に出力される．
    python __init__.py get_result_tableでtexの表形式の比較結果を出力.

@disc/ or cont/
disc=>discontinue.離散値特徴量を含むデータセット+推薦結果.
cont=>continue.連続値のみのデータセット+推薦結果.
disc/下の構成.
../disc/{ppl/,questionnaire/,true_data/,false_data/,all_items.json,train_data/,cluster_depend/}
/cluster_depend/cluster_id_x/{result/,param/,tex/,plot/,cluster_data/,ranking/}
/cluster_data/cluster_num/user_x_train_id.txt
/result/model_name/user_id.txt
/param/{kl-profile,pickle}
/pickle/{all_items,weight_and_score_model_list}
all_items/model_name/mapping_id.txt
weight_and_score_model_list/model_name/user_id_train_id.txt

#出力結果を見る．
plot=>/disc/cluster_id_depend/cluster_id_201803072213_0/plot/以下最下層のファイル
tex=>/disc/cluster_id_depend/cluster_id_201803072213_0/tex/以下最下層のファイル
cluster_id_201803072213_0の部分はcluster_idに相当する.clusteringの結果はclustering_idと対応しているため
実験済みのclustering_idを指定すればそのidに対応するclustering結果が再利用される.




