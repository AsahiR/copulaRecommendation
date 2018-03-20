# 論文中には載せてないけど研究で色々試したモデル群

import pandas as pd
import numpy as np
import statsmodels.api as sm
from marginal import marginal
from copula import copula
from typing import List
from scoring import models
from utils import util


class NestedCopulaKLParametricModel(models.ScoreModel):
    param_a = 10

    def __init__(self):
        self.score_type_list = util.DEFAULT_SCORE_TYPE_LIST
        self.nested_copula_list = []
        self.marg_dict = {}
        self.main_axis = ''

    def get_modelname(self) -> str:
        return "nested_copula_kl_only_one_0.05kl_log1p" + str(self.param_a)

    def get_dirname(self) -> str:
        return "nested"

    def train(self, training_data_t: pd.DataFrame, training_data_f: pd.DataFrame, user_id):
        marg_dict = {}
        kl_dict = {}
        max_kl_value = 0
        for score_type in util.DEFAULT_SCORE_TYPE_LIST:
            marg = marginal.Norm(training_data_t[score_type])
            marg_dict[score_type] = marg
            kl = util.kl_divergence_between_population_and_users(marg, score_type)
            kl_dict[score_type] = kl
            if kl > max_kl_value:
                max_kl_value = kl
                self.main_axis = score_type

        self.marg_dict = marg_dict
        print(kl_dict.values())
        new_kl_dict = {k: v for k, v in kl_dict.items() if v > 0.05}
        self.score_type_list = [k for k, v in sorted(new_kl_dict.items(), key=lambda x: x[1])]
        if len(self.score_type_list) == 1:
            self.score_type_list = util.DEFAULT_SCORE_TYPE_LIST
        else:
            kl_dict = new_kl_dict
        #todo remove
        for score_type in self.score_type_list:
            print(score_type, kl_dict[score_type])
        #todo remove
        nested_copula_list = []
        for i in range(0, len(self.score_type_list) - 1):
            former_kl = kl_dict[self.score_type_list[i]]
            current_kl = kl_dict[self.score_type_list[i + 1]]
            nested_copula_list.append(copula.Copula(np.matrix([]), 'gumbel', param=1 + self.param_a * np.log1p(current_kl), dim=2))
        self.nested_copula_list = nested_copula_list

    def calc_ranking(self, all_items: pd.DataFrame) -> dict:
        dict_list = []
        dict_list_main_prod = []
        for index, row in all_items.iterrows():
            hotel_id = row['id']
            nested_cdf = None
            for i in range(0, len(self.score_type_list) - 1):
                former = self.score_type_list[i]
                current = self.score_type_list[i + 1]
                copula_model = self.nested_copula_list[i]
                if nested_cdf is None:
                    former_cdf = self.marg_dict[former].cdf(row[former])
                    current_cdf = self.marg_dict[current].cdf(row[current])
                    nested_cdf = copula_model.cdf(np.matrix([current_cdf, former_cdf]))
                else:
                    current_cdf = self.marg_dict[current].cdf(row[current])
                    nested_cdf = copula_model.cdf(np.matrix([current_cdf, nested_cdf]))
            dict_list.append({"id": hotel_id, "score": nested_cdf})
            dict_list_main_prod.append({"id": hotel_id, "score": nested_cdf * current_cdf})
        dr1 = pd.DataFrame.from_records(dict_list, index='id')
        dr2 = pd.DataFrame.from_records(dict_list_main_prod, index='id')
        return {
            'normal': dr1.sort_values(by='score', ascending=False),
            'main_prod': dr2.sort_values(by='score', ascending=False),
        }


class NewProposedScoreModel(models.ScoreModel):
    def __init__(self, marg: str, cop: str, n_clusters: int, indep_copulaed=False):
        self.marg = marg
        self.cop = cop
        self.n_clusters = n_clusters
        self.score_type_list = util.DEFAULT_SCORE_TYPE_LIST
        self.main_axis = ''
        self.copula_dict = None
        self.marg_dict = None
        self.kl_dict = None
        self.top_copula1 = None
        self.top_copula2 = None
        self.top_copula3 = None
        self.top_copula4 = None
        self.indep_copulaed = indep_copulaed

    def get_dirname(self) -> str:
        return 'proposed-properly-reduced'

    def get_modelname(self) -> str:
        if self.indep_copulaed:
            return 'indep_copuled_cluster' + str(self.n_clusters) + "_" + self.marg + "_" + self.cop
        return 'copuled_cluster' + str(self.n_clusters) + "_" + self.marg + "_" + self.cop

    def train(self, training_data_t: pd.DataFrame, training_data_f: pd.DataFrame, user_id=None):
        self.score_type_list = util.list_of_users_axis_has_weight(user_id)# todo remove
        self.copula_dict = {}
        self.marg_dict = {}
        self.kl_dict = {}
        max_kl_value = -1000
        self.kl_dict = {'sum': 0}
        score_type_list = []
        for score_type in self.score_type_list:
            marg = marginal.Norm(training_data_t[score_type])
            self.marg_dict[score_type] = marg
            kl = util.kl_divergence_between_population_and_users(marg, score_type)
            self.kl_dict['sum'] += kl
            self.kl_dict[score_type] = kl
            print(score_type, kl)
            if kl > max_kl_value:
                max_kl_value = kl
                self.main_axis = score_type
        #todo remove
        test = [x for x in util.DEFAULT_SCORE_TYPE_LIST if x not in self.score_type_list]
        for score_type in test:
            print('test kl 0 weight')
            marg = marginal.Norm(training_data_t[score_type])
            kl = util.kl_divergence_between_population_and_users(marg, score_type)
            print(score_type, kl)
            print('test kl 0 weight')
        #todo remove
        print(self.main_axis)
        self.kl_dict['sum'] -= max_kl_value

        main_marg = marginal.Norm(training_data_t[self.main_axis])
        main_cdf_list = [main_marg.cdf(x) for x in training_data_t[self.main_axis]]
        for score_type in self.score_type_list:
            if score_type == self.main_axis:
                continue
            target = [self.main_axis, score_type]
            clust = models.create_cluster(training_data_t, self.n_clusters, target)
            self.copula_dict[score_type] = models.create_weight_and_scoring_model_list(clust, self.marg, self.cop, target, [])

        each_copula_cdf_list_list =[]
        each_copula_cdf_list_list2 =[]
        each_copula_cdf_list_list3 =[]
        each_copula_cdf_list_list4 =[]
        for score_type in self.score_type_list:
            if score_type == self.main_axis:
                continue
            each_copula_cdf_list_list.append([])
            each_copula_cdf_list_list2.append([])
            each_copula_cdf_list_list3.append([])
            each_copula_cdf_list_list4.append([])

        for index, row in training_data_t.iterrows():
            #main_cdf = self.marg_dict[self.main_axis].cdf(row[self.main_axis])
            cnt = 0
            for score_type in self.score_type_list:
                if score_type == self.main_axis:
                    continue
                main_cdf = 0
                sub_cdf = 0
                cop_cdf = 0
                for weight_and_scoring_model in self.copula_dict[score_type]:
                    weight = weight_and_scoring_model[0]
                    score_model = weight_and_scoring_model[1]
                    marginal_cdf_list = []
                    for axis in [self.main_axis, score_type]:
                        marginal_score_model = score_model[axis]
                        marg_cdf = marginal_score_model.cdf(row[axis])
                        marginal_cdf_list.append(marg_cdf)
                        if axis != self.main_axis:
                            sub_cdf += weight * marg_cdf
                        else:
                            main_cdf += weight * marg_cdf
                    cop_cdf += score_model['copula'].cdf(np.matrix(marginal_cdf_list)) * weight

                each_copula_cdf_list_list[cnt].append(cop_cdf)
                each_copula_cdf_list_list2[cnt].append(cop_cdf * sub_cdf)
                each_copula_cdf_list_list3[cnt].append(cop_cdf * main_cdf * sub_cdf)
                each_copula_cdf_list_list4[cnt].append(cop_cdf * main_cdf)
                cnt += 1

        cop_mat1 = np.matrix(each_copula_cdf_list_list).T
        cop_mat2 = np.matrix(each_copula_cdf_list_list2).T
        cop_mat3 = np.matrix(each_copula_cdf_list_list3).T
        cop_mat4 = np.matrix(each_copula_cdf_list_list4).T
        if self.indep_copulaed:
            self.top_copula1 = copula.Copula(cop_mat1, 'indep')
            self.top_copula2 = copula.Copula(cop_mat2, 'indep')
            self.top_copula3 = copula.Copula(cop_mat3, 'indep')
            self.top_copula4 = copula.Copula(cop_mat4, 'indep')
        else:
            self.top_copula1 = copula.Copula(cop_mat1, self.cop)
            self.top_copula2 = copula.Copula(cop_mat2, self.cop)
            self.top_copula3 = copula.Copula(cop_mat3, self.cop)
            self.top_copula4 = copula.Copula(cop_mat4, self.cop)

    def calc_ranking(self, all_items: pd.DataFrame) -> dict:
        dict_list = []
        dict_list_half_prod = []
        dict_list_full_prod = []
        dict_list_main_half_prod = []
        for index, row in all_items.iterrows():
            hotel_id = row['id']
            #main_cdf = self.marg_dict[self.main_axis].cdf(row[self.main_axis])
            cops1 = []
            cops2 = []
            cops3 = []
            cops4 = []
            for score_type in self.score_type_list:
                if score_type == self.main_axis:
                    continue
                main_cdf = 0
                sub_cdf = 0
                cop_cdf = 0
                for weight_and_scoring_model in self.copula_dict[score_type]:
                    weight = weight_and_scoring_model[0]
                    score_model = weight_and_scoring_model[1]
                    marginal_cdf_list = []
                    for axis in [self.main_axis, score_type]:
                        marginal_score_model = score_model[axis]
                        marg_cdf = marginal_score_model.cdf(row[axis])
                        marginal_cdf_list.append(marg_cdf)
                        if axis != self.main_axis:
                            sub_cdf += weight * marg_cdf
                        else:
                            main_cdf += weight * marg_cdf
                    cop_cdf += score_model['copula'].cdf(np.matrix(marginal_cdf_list)) * weight
                cops1.append(cop_cdf)
                cops2.append(cop_cdf * sub_cdf)
                cops3.append(cop_cdf * main_cdf * sub_cdf)
                cops4.append(cop_cdf * main_cdf)
            if len(self.score_type_list) <= 2:
                c_mix_score = cops1[0]
                c_mix_half_score = cops2[0]
                c_mix_full_score = cops3[0]
                c_mix_main_half_score = cops3[0]
            else:
                c_mix_score = self.top_copula1.cdf(np.matrix(cops1))
                c_mix_half_score = self.top_copula2.cdf(np.matrix(cops2))
                c_mix_full_score = self.top_copula3.cdf(np.matrix(cops3))
                c_mix_main_half_score = self.top_copula4.cdf(np.matrix(cops4))

            dict_list.append({"id": hotel_id, "score": c_mix_score})
            dict_list_half_prod.append({"id": hotel_id, "score": c_mix_half_score})
            dict_list_full_prod.append({"id": hotel_id, "score": c_mix_full_score})
            dict_list_main_half_prod.append({"id": hotel_id, "score": c_mix_main_half_score})
        dr1 = pd.DataFrame.from_records(dict_list, index='id')
        dr2 = pd.DataFrame.from_records(dict_list_half_prod, index='id')
        dr3 = pd.DataFrame.from_records(dict_list_full_prod, index='id')
        dr4 = pd.DataFrame.from_records(dict_list_main_half_prod, index='id')
        return {
            'mix': dr1.sort_values(by='score', ascending=False),
            'mix-half-prod': dr2.sort_values(by='score', ascending=False),
            'mix-full-prod': dr3.sort_values(by='score', ascending=False),
            'mix-main-half-prod': dr4.sort_values(by='score', ascending=False)
        }


class NewProposedScoreModelQuantize(models.ScoreModel):
    def __init__(self, marg: str, cop: str, n_clusters: int, indep_copulaed=False):
        self.marg = marg
        self.cop = cop
        self.n_clusters = n_clusters
        self.score_type_list = util.DEFAULT_SCORE_TYPE_LIST
        self.main_axis = ''
        self.copula_dict = None
        self.marg_dict = None
        self.kl_dict = None
        self.top_copula1 = None
        self.top_copula2 = None
        self.top_copula3 = None
        self.indep_copulaed = indep_copulaed

    def get_dirname(self) -> str:
        return 'proposed'

    def get_modelname(self) -> str:
        if self.indep_copulaed:
            return 'indeped_new_simple0.9_sub_quantized' + 'cluster' + str(self.n_clusters) + "_" + self.marg + "_" + self.cop
        return 'new_quantized_' + 'cluster' + str(self.n_clusters) + "_" + self.marg + "_" + self.cop

    def train(self, training_data_t: pd.DataFrame, training_data_f: pd.DataFrame, user_id=None):
        self.copula_dict = {}
        self.marg_dict = {}
        self.kl_dict = {}
        max_kl_value = -1000
        self.kl_dict = {'sum': 0}
        score_type_list = []
        for score_type in util.DEFAULT_SCORE_TYPE_LIST:
            marg = marginal.Norm(training_data_t[score_type])
            self.marg_dict[score_type] = marg
            kl = util.kl_divergence_between_population_and_users(marg, score_type)
            print(score_type, kl)
            self.kl_dict['sum'] += kl
            self.kl_dict[score_type] = kl
            if kl > max_kl_value:
                max_kl_value = kl
                self.main_axis = score_type
        print(self.main_axis)
        self.kl_dict['sum'] -= max_kl_value
        main_marg = marginal.Norm(training_data_t[self.main_axis])
        main_cdf_list = [main_marg.cdf(x) for x in training_data_t[self.main_axis]]
        for score_type in self.score_type_list:
            if score_type == self.main_axis:
                continue
            target = [self.main_axis, score_type]
            clust = models.create_cluster(training_data_t, self.n_clusters, target)
            self.copula_dict[score_type] = models.create_weight_and_scoring_model_list(clust, self.marg, self.cop,
                                                                                target, [score_type], [])
        each_copula_cdf_list_list =[]
        each_copula_cdf_list_list2 =[]
        each_copula_cdf_list_list3 =[]
        for score_type in self.score_type_list:
            if score_type == self.main_axis:
                continue
            each_copula_cdf_list_list.append([])
            each_copula_cdf_list_list2.append([])
            each_copula_cdf_list_list3.append([])

        for index, row in training_data_t.iterrows():
            #main_cdf = self.marg_dict[self.main_axis].cdf(row[self.main_axis])
            cnt = 0
            for score_type in self.score_type_list:
                if score_type == self.main_axis:
                    continue
                main_cdf = 0
                sub_cdf = 0
                cop_cdf = 0
                for weight_and_scoring_model in self.copula_dict[score_type]:
                    weight = weight_and_scoring_model[0]
                    score_model = weight_and_scoring_model[1]
                    marginal_cdf_list = []
                    for axis in [self.main_axis, score_type]:
                        marginal_score_model = score_model[axis]
                        marg_cdf = marginal_score_model.cdf(row[axis])
                        marginal_cdf_list.append(marg_cdf)
                        if axis != self.main_axis:
                            sub_cdf += weight * marg_cdf
                        else:
                            main_cdf += weight * marg_cdf
                    cop_cdf += score_model['copula'].cdf(np.matrix(marginal_cdf_list)) * weight

                each_copula_cdf_list_list[cnt].append(cop_cdf)
                each_copula_cdf_list_list2[cnt].append(cop_cdf * sub_cdf)
                each_copula_cdf_list_list3[cnt].append(cop_cdf * main_cdf * sub_cdf)
                cnt += 1

        cop_mat1 = np.matrix(each_copula_cdf_list_list).T
        cop_mat2 = np.matrix(each_copula_cdf_list_list2).T
        cop_mat3 = np.matrix(each_copula_cdf_list_list3).T
        if self.indep_copulaed:
            pass
        else:
            self.top_copula1 = copula.Copula(cop_mat1, self.cop)
            self.top_copula2 = copula.Copula(cop_mat2, self.cop)
            self.top_copula3 = copula.Copula(cop_mat3, self.cop)

    def calc_ranking(self, all_items: pd.DataFrame) -> dict:
        dict_list = []
        dict_list_half_prod = []
        dict_list_full_prod = []
        for index, row in all_items.iterrows():
            hotel_id = row['id']
            #main_cdf = self.marg_dict[self.main_axis].cdf(row[self.main_axis])
            cops1 = []
            cops2 = []
            cops3 = []
            for score_type in self.score_type_list:
                if score_type == self.main_axis:
                    continue
                main_cdf = 0
                sub_cdf = 0
                cop_cdf = 0
                for weight_and_scoring_model in self.copula_dict[score_type]:
                    weight = weight_and_scoring_model[0]
                    score_model = weight_and_scoring_model[1]
                    marginal_cdf_list = []
                    for axis in [self.main_axis, score_type]:
                        marginal_score_model = score_model[axis]
                        marg_cdf = marginal_score_model.cdf(row[axis])
                        marginal_cdf_list.append(marg_cdf)
                        if axis != self.main_axis:
                            sub_cdf += weight * marg_cdf
                        else:
                            main_cdf += weight * marg_cdf
                    cop_cdf += score_model['copula'].cdf(np.matrix(marginal_cdf_list)) * weight
                cops1.append(cop_cdf)
                cops2.append(cop_cdf * sub_cdf)
                cops3.append(cop_cdf * main_cdf * sub_cdf)
            if len(self.score_type_list) <= 2:
                c_mix_score = cops1[0]
                c_mix_half_score = cops2[0]
                c_mix_full_score = cops3[0]
            else:
                if self.indep_copulaed:
                    c_mix_score = 1
                    c_mix_half_score = 1
                    c_mix_full_score = 1
                    for x in cops1:
                        c_mix_score *= x
                    for x in cops2:
                        c_mix_half_score *= x
                    for x in cops3:
                        c_mix_full_score *= x
                else:
                    c_mix_score = self.top_copula1.cdf(np.matrix(cops1))
                    c_mix_half_score = self.top_copula2.cdf(np.matrix(cops2))
                    c_mix_full_score = self.top_copula3.cdf(np.matrix(cops3))

            dict_list.append({"id": hotel_id, "score": c_mix_score})
            dict_list_half_prod.append({"id": hotel_id, "score": c_mix_half_score})
            dict_list_full_prod.append({"id": hotel_id, "score": c_mix_full_score})
        dr1 = pd.DataFrame.from_records(dict_list, index='id')
        dr2 = pd.DataFrame.from_records(dict_list_half_prod, index='id')
        dr3 = pd.DataFrame.from_records(dict_list_full_prod, index='id')
        return {
            'mix': dr1.sort_values(by='score', ascending=False),
            'mix-half-prod': dr2.sort_values(by='score', ascending=False),
            'mix-full-prod': dr3.sort_values(by='score', ascending=False)
        }

    # 訓練データを回帰して、影響度の大きい軸のみをコピュラで統合するクラス
    class CopulaScoreModelAxisReducedRegression(models.CopulaScoreModel):
        def __init__(self, marg: str, cop: str, n_clusters: int, reg_p: float):
            models.CopulaScoreModel.__init__(self, marg, cop, n_clusters)
            self.reg_p = reg_p

        def get_dirname(self) -> str:
            return 'axis-regressionp' + str(self.reg_p)

        def reduce_axis(self, training_data_t, training_data_f) -> List[str]:
            positive_df = pd.DataFrame.from_records(training_data_t)[util.DEFAULT_SCORE_TYPE_LIST]
            negative_df = pd.DataFrame.from_records(training_data_f)[util.DEFAULT_SCORE_TYPE_LIST]
            positive_df['suit'] = True
            negative_df['suit'] = False
            df = positive_df.append(negative_df)
            train_cols = df.columns[:-1]
            logit = sm.Logit(df['suit'], df[train_cols])
            res = logit.fit()
            labels = res.pvalues.index.tolist()
            p_values = res.pvalues.values.tolist()

            score_type_list = []
            for label, p in zip(labels, p_values):
                if p < self.reg_p:
                    score_type_list.append(label)
            if len(score_type_list) < 2:
                score_type_list = util.DEFAULT_SCORE_TYPE_LIST
            print(score_type_list)
            return score_type_list

        def train(self, training_data_t: pd.DataFrame, training_data_f: pd.DataFrame, user_id=None):
            #軸の絞り込み
            self.score_type_list = self.reduce_axis(training_data_t, training_data_f)
            hotel_cluster = models.create_cluster(training_data_t, self.n_clusters, self.score_type_list)
            # 混合コピュラの構築
            self.weight_and_score_model_list = models.create_weight_and_scoring_model_list(hotel_cluster, self.marg, self.cop, self.score_type_list, [])


    class ProposedScoreModel(models.ScoreModel):
        def __init__(self, marg: str, cop: str):
            self.marg = marg
            self.cop = cop
            self.weight_and_score_model_list = None
            self.score_type_list = util.DEFAULT_SCORE_TYPE_LIST
            self.main_axis = ''
            self.copula_dict = None
            self.marg_dict = None
            self.kl_dict = None

        def get_dirname(self) -> str:
            return 'proposed'

        def get_modelname(self) -> str:
            return 'cluster' + str(1) + "_" + self.marg + "_" + self.cop

        def train(self, training_data_t: pd.DataFrame, training_data_f: pd.DataFrame, user_id=None):
            self.copula_dict = {}
            self.marg_dict = {}
            self.kl_dict = {}
            max_kl_value = -1000
            self.kl_dict = {'sum': 0}
            for score_type in util.DEFAULT_SCORE_TYPE_LIST:
                marg = marginal.Norm(training_data_t[score_type])
                self.marg_dict[score_type] = marg
                kl = util.kl_divergence_between_population_and_users(marg, score_type)
                self.kl_dict['sum'] += kl
                self.kl_dict[score_type] = kl
                if kl > max_kl_value:
                    max_kl_value = kl
                    self.main_axis = score_type
            self.kl_dict['sum'] -= max_kl_value

            main_marg = marginal.Norm(training_data_t[self.main_axis])
            main_cdf_list = [main_marg.cdf(x) for x in training_data_t[self.main_axis]]
            for score_type in util.DEFAULT_SCORE_TYPE_LIST:
                if score_type == self.main_axis:
                    continue
                marginal_cdf_list_list = [main_cdf_list, [marg.cdf(x) for x in training_data_t[score_type]]]
                cdf_matrix = np.matrix(marginal_cdf_list_list).T
                self.copula_dict[score_type] = copula.Copula(cdf_matrix, self.cop)

        def calc_ranking(self, all_items: pd.DataFrame) -> dict:
            dict_list = []
            dict_list_half_prod = []
            dict_list_weighted = []
            dict_list_half_prod_weighted = []
            for index, row in all_items.iterrows():
                hotel_id = row['id']
                c_mix_score = 0
                c_mix_half_score = 0
                w_c_mix_score = 0
                w_c_mix_half_score = 0
                main_cdf = self.marg_dict[self.main_axis].cdf(row[self.main_axis])
                for score_type in self.score_type_list:
                    if score_type == self.main_axis:
                        continue
                    sub_cdf = self.marg_dict[score_type].cdf(row[score_type])
                    cop_cdf = self.copula_dict[score_type].cdf(np.matrix([main_cdf, sub_cdf]))
                    c_mix_score += cop_cdf
                    c_mix_half_score += cop_cdf * sub_cdf
                    w_c_mix_score += cop_cdf * (self.kl_dict[score_type]/self.kl_dict['sum'])
                    w_c_mix_half_score += (cop_cdf * sub_cdf) * (self.kl_dict[score_type]/self.kl_dict['sum'])

                dict_list.append({"id": hotel_id, "score": c_mix_score})
                dict_list_half_prod.append({"id": hotel_id, "score": c_mix_half_score})
                dict_list_weighted.append({"id": hotel_id, "score": w_c_mix_score})
                dict_list_half_prod_weighted.append({"id": hotel_id, "score": w_c_mix_half_score})
            dr1 = pd.DataFrame.from_records(dict_list, index='id')
            dr2 = pd.DataFrame.from_records(dict_list_half_prod, index='id')
            dr3 = pd.DataFrame.from_records(dict_list_weighted, index='id')
            dr4 = pd.DataFrame.from_records(dict_list_half_prod_weighted, index='id')
            return {
                'mix': dr1.sort_values(by='score', ascending=False),
                'mix-half-prod': dr2.sort_values(by='score', ascending=False),
                'weighted-mix': dr3.sort_values(by='score', ascending=False),
                'weighted-mix-half-prod': dr4.sort_values(by='score', ascending=False)
            }


    class ProposedScoreModelQuantize(models.ScoreModel):
        def __init__(self, marg: str, cop: str):
            self.marg = marg
            self.cop = cop
            self.weight_and_score_model_list = None
            self.score_type_list = util.DEFAULT_SCORE_TYPE_LIST
            self.main_axis = ''
            self.copula_dict = None
            self.marg_dict = None
            self.kl_dict = None

        def get_dirname(self) -> str:
            return 'proposed'

        def get_modelname(self) -> str:
            return 'quantized_' + 'cluster' + str(1) + "_" + self.marg + "_" + self.cop

        def train(self, training_data_t: pd.DataFrame, training_data_f: pd.DataFrame, user_id=None):
            self.copula_dict = {}
            self.marg_dict = {}
            self.kl_dict = {}
            max_kl_value = -1000
            self.kl_dict = {'sum': 0}
            for score_type in util.DEFAULT_SCORE_TYPE_LIST:
                marg = marginal.Norm(training_data_t[score_type])
                self.marg_dict[score_type] = marg
                kl = util.kl_divergence_between_population_and_users(marg, score_type)
                self.kl_dict['sum'] += kl
                self.kl_dict[score_type] = kl
                if kl > max_kl_value:
                    max_kl_value = kl
                    self.main_axis = score_type
            self.kl_dict['sum'] -= max_kl_value

            main_marg = marginal.Norm(training_data_t[self.main_axis])
            main_cdf_list = [main_marg.cdf(x) for x in training_data_t[self.main_axis]]
            for score_type in util.DEFAULT_SCORE_TYPE_LIST:
                if score_type == self.main_axis:
                    continue
                marginal_cdf_list_list = [main_cdf_list, [marg.cdf(x) for x in training_data_t[score_type]]]
                cdf_matrix = np.matrix(marginal_cdf_list_list).T
                self.copula_dict[score_type] = copula.Copula(cdf_matrix, self.cop)
            self.marg_dict[self.main_axis] = marginal.QuantizedNorm(training_data_t[self.main_axis])

        def calc_ranking(self, all_items: pd.DataFrame) -> dict:
            dict_list = []
            dict_list_half_prod = []
            dict_list_weighted = []
            dict_list_half_prod_weighted = []
            for index, row in all_items.iterrows():
                hotel_id = row['id']
                c_mix_score = 0
                c_mix_half_score = 0
                w_c_mix_score = 0
                w_c_mix_half_score = 0
                main_cdf = self.marg_dict[self.main_axis].cdf(row[self.main_axis])
                for score_type in self.score_type_list:
                    if score_type == self.main_axis:
                        continue
                    sub_cdf = self.marg_dict[score_type].cdf(row[score_type])
                    cop_cdf = self.copula_dict[score_type].cdf(np.matrix([main_cdf, sub_cdf]))
                    c_mix_score += cop_cdf
                    c_mix_half_score += cop_cdf * sub_cdf
                    w_c_mix_score += cop_cdf * (self.kl_dict[score_type]/self.kl_dict['sum'])
                    w_c_mix_half_score += (cop_cdf * sub_cdf) * (self.kl_dict[score_type]/self.kl_dict['sum'])

                dict_list.append({"id": hotel_id, "score": c_mix_score})
                dict_list_half_prod.append({"id": hotel_id, "score": c_mix_half_score})
                dict_list_weighted.append({"id": hotel_id, "score": w_c_mix_score})
                dict_list_half_prod_weighted.append({"id": hotel_id, "score": w_c_mix_half_score})
            dr1 = pd.DataFrame.from_records(dict_list, index='id')
            dr2 = pd.DataFrame.from_records(dict_list_half_prod, index='id')
            dr3 = pd.DataFrame.from_records(dict_list_weighted, index='id')
            dr4 = pd.DataFrame.from_records(dict_list_half_prod_weighted, index='id')
            return {
                'mix': dr1.sort_values(by='score', ascending=False),
                'mix-half-prod': dr2.sort_values(by='score', ascending=False),
                'weighted-mix': dr3.sort_values(by='score', ascending=False),
                'weighted-mix-half-prod': dr4.sort_values(by='score', ascending=False)
            }


    # 特定のp値以上を統合軸として採用し、均等重みで統合していく方法
    class LinearScoreModel(models.ScoreModel):
        def __init__(self, n_axis: int):
            self.score_type_list = util.DEFAULT_SCORE_TYPE_LIST
            self.n_axis = n_axis

        def get_dirname(self):
            return 'regression_linear'

        def get_modelname(self):
            return ''

        def reduce_axis(self, training_data_t, training_data_f):
            positive_df = pd.DataFrame.from_records(training_data_t)[util.DEFAULT_SCORE_TYPE_LIST]
            negative_df = pd.DataFrame.from_records(training_data_f)[util.DEFAULT_SCORE_TYPE_LIST]
            positive_df['suit'] = True
            negative_df['suit'] = False
            df = positive_df.append(negative_df)
            train_cols = df.columns[:-1]
            logit = sm.Logit(df['suit'], df[train_cols])
            res = logit.fit()
            labels = res.pvalues.index.tolist()
            p_values = res.pvalues.values.tolist()

            score_type_list = []
            dic = {}
            for label, p in zip(labels, p_values):
                dic[p] = label
            cnt = 0
            for key, value in sorted(dic.items()):
                cnt += 1
                if cnt > self.n_axis:
                    break
                score_type_list.append(value)
            print(score_type_list)
            return score_type_list

        def train(self, training_data_t: pd.DataFrame, training_data_f: pd.DataFrame, user_id=None):
            self.score_type_list = self.reduce_axis(training_data_t, training_data_f)

        def calc_ranking(self, all_items: pd.DataFrame) -> dict:
            dict_list = []
            for index, row in all_items.iterrows():
                hotel_id = row['id']
                size = len(self.score_type_list)
                score = 0
                for score_type in self.score_type_list:
                    score += row[score_type]/size
                dict_list.append({"id": hotel_id, "score": score})
            df_for_ranking = pd.DataFrame.from_records(dict_list, index='id')
            method_name = 'reg_' + str(self.n_axis)
            return {method_name: df_for_ranking.sort_values(by='score', ascending=False),
                    }


