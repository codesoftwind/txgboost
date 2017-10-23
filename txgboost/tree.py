# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool
import copy_reg
import types


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

class TreeNode(object):

    def __init__(self,is_leaf=False,split_feature=None,threshold=None,nan_direction=None,lchild=None,rchild=None,leaf_score=None):
        self.is_leaf = is_leaf
        self.split_feature = split_feature
        self.threshold = threshold
        self.nan_direction = nan_direction
        self.lchild = lchild
        self.rchild = rchild
        self.leaf_score = leaf_score

class Tree(object):

    def __init__(self):
        self.root = None
        self.thread_num = -1
        self.feature_importance = {}

        self.max_depth = None
        self.min_sample_split = None
        self.min_child_weight = None

        self.reg_lambda = None
        self.gamma = None
        self.colsample_bylevel = None

    def fit(self,X,Y,max_depth = 3,min_sample_split = 10,min_child_weight = 1.0,reg_lambda = 1.0,gamma = 0.0,colsample_bylevel = 1.0,thread_num = -1):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.colsample_bylevel = colsample_bylevel
        self.thread_num = thread_num

        self.root = self.build(X,Y,0)

    def predict_one(self,treeNode,X):
        if treeNode.is_leaf == True:
            return treeNode.leaf_score

        if pd.isnull(X[1][treeNode.split_feature]):
            if treeNode.nan_direction == 0:
                return self.predict_one(treeNode.lchild,X)
            else:
                return self.predict_one(treeNode.rchild,X)

        if X[1][treeNode.split_feature] < treeNode.threshold:
            return self.predict_one(treeNode.lchild,X)
        else:
            return self.predict_one(treeNode.rchild,X)

    def predict(self,X):

        x_iter = X.iterrows()

        predict_ = partial(self.predict_one,self.root)

        if self.thread_num == -1:
            pool = Pool()
            res = pool.map(predict_,x_iter)
            pool.close()
            pool.join()
        else:
            pool = Pool(self.thread_num)
            res = pool.map(predict_,x_iter)
            pool.close()
            pool.join()
        return  np.array(res)

    def build(self,X,Y,depth):

        score = self.cal_leaf_score(Y)
        if depth == self.max_depth or X.shape[0]<self.min_sample_split or Y.hess.sum()<self.min_child_weight:
            return TreeNode(is_leaf=True,leaf_score=score)

        X_colsample = X.sample(frac=self.colsample_bylevel,axis=1)

        split_feature, feature_threshold,nan_direction,gain = self.find_best_feature(X_colsample,Y)

        if gain <0:
            return TreeNode(is_leaf=True, leaf_score=score)

        left_X,left_Y,right_X,right_Y = self.split_data(X,Y,split_feature,feature_threshold,nan_direction)

        lchild = self.build(left_X,left_Y,depth+1)
        rchild = self.build(right_X,right_Y,depth+1)

        if self.feature_importance.has_key(split_feature):

            self.feature_importance[split_feature] += 1
        else:

            self.feature_importance[split_feature] = 0

        node = TreeNode(is_leaf=False,split_feature=split_feature,threshold=feature_threshold,nan_direction=nan_direction,lchild=lchild,rchild=rchild)

        return node

    def cal_leaf_score(self,Y):
        # -G/(H+reg_lambda)
        return -1*Y.grad.sum()/(Y.hess.sum()+self.reg_lambda)

    def find_best_feature(self, X, Y):
        cols = list(X.columns)
        best_feature = None
        threshold = None
        nan_direction = None
        gain = -np.inf
        data = pd.concat([X,Y],axis=1)

        threshold_find = partial(self.find_feature_threshold,data)

        if self.thread_num == -1:
            pool = Pool()
            ress = pool.map(threshold_find,cols)
            pool.close()
            pool.join()
        else:
            pool = Pool(self.thread_num)
            ress = pool.map(threshold_find,cols)
            pool.close()
            pool.join()

        for res in ress:
            if res[0] > gain:
                gain = res[0]
                best_feature = res[1]
                threshold = res[2]
                nan_direction = res[3]

        return best_feature,threshold,nan_direction,gain

    def find_feature_threshold(self,data,col):

        best_gain = -np.inf
        best_threshold = None
        best_nan_direction = None

        data_use = data[[col,'grad','hess']]

        mask = data_use[col].isnull()

        data_nan = data_use[mask]
        G_nan = data_nan.grad.sum()
        H_nan = data_nan.hess.sum()

        data_not_nan = data_use[~mask]
        data_not_nan.reset_index(drop=True)
        data_not_nan = data_not_nan[data_not_nan[col].argsort()]

        for i in range(data_not_nan.shape[0]-1):
            value , next_value = data_not_nan.iloc[i][col],data_not_nan.iloc[i+1][col]
            temp_shold = (value+next_value)/2
            if value == next_value:
                continue

            left_data = data_not_nan.iloc[0:i+1]
            right_data = data_not_nan.iloc[i+1:]

            gain_nan_left = self.cal_gain(left_data,right_data,G_nan,H_nan,0)
            gain_nan_right = self.cal_gain(left_data,right_data,G_nan,H_nan,1)

            if gain_nan_left>best_gain:
                best_gain = gain_nan_left
                best_threshold = temp_shold
                best_nan_direction = 0

            if gain_nan_right>best_gain:
                best_gain = gain_nan_right
                best_threshold = temp_shold
                best_nan_direction = 1

        return best_gain,col,best_threshold,best_nan_direction

    def cal_gain(self,left,right,G_nan,H_nan,nan_direction):

        G_left = left.grad.sum() + (1-nan_direction)*G_nan
        G_right = right.grad.sum() + nan_direction*G_nan
        H_left = left.hess.sum() + (1-nan_direction)*H_nan
        H_right = right.hess.sum() + nan_direction*H_nan

        gain = 0.5*( G_left**2/(H_left+self.reg_lambda) + G_right**2/(H_right+self.reg_lambda)  - (G_left+G_right)**2/(H_left+H_right+self.reg_lambda) )-self.gamma

        return gain

    def split_data(self,X,Y,feature,threshold,nan_direction):

        if nan_direction==0:

            mask = X[feature]>=threshold
            right_X = X[mask]
            right_Y = Y[mask]
            left_X = X[~mask]
            left_Y = Y[~mask]

        else:

            mask = X[feature]<threshold
            right_X = X[~mask]
            right_Y = Y[~mask]
            left_X = X[mask]
            left_Y = Y[mask]

        return left_X,left_Y,right_X,right_Y
