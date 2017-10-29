from loss import CustomizeLoss,LogisticLoss,SquareLoss
from metric import get_metric
from tree import Tree
import numpy as np
import pandas as pd


class TXgboost(object):
    def __init__(self):

        self.num_boost_round = None
        self.first_round_pred = None
        self.loss = None

        self.max_depth = None
        self.min_sample_split = None
        self.min_child_weight = None
        self.scale_pos_weight = None

        self.eta = None
        self.reg_lambda = None
        self.gamma = None
        self.colsample_bylevel = None
        self.colsample_bytree = None
        self.row_sample = None

        self.thread_num = -1
        self.feature_importance={}

        self.trees=[]


    def fit(self,X,Y,eval_set=(None,None),boost_round=1000,early_stopping_rounds=np.inf,loss_func='logisticloss',
            metric='error',maximize=False,thread_num=-1,eta=0.3,row_sample=1.0,colsample_bytree=0.8,
            colsample_bylevel=0.8,max_depth=3,min_sample_split=10,min_child_weight=1.0,reg_lambda=1.0,
            gamma=0,scale_pos_weight=1.0):

        self.num_boost_round=boost_round
        self.first_round_pred = 0.0
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self.eta=eta
        self.reg_lambda=reg_lambda
        self.gamma=gamma
        self.colsample_bylevel=colsample_bylevel
        self.colsample_bytree=colsample_bytree
        self.row_sample=row_sample
        self.thread_num=thread_num


        X.reset_index(inplace=True,drop=True)
        Y.reset_index(inplace=True,drop=True)

        if eval_set[0]==None or eval_set[1]==None:
            is_valid = False
        else:
            is_valid = True
            valid_x = eval_set[0].reset_index(drop=True)
            valid_y = eval_set[1].reset_index(drop=True)
            valid_y = pd.DataFrame(valid_y.values, columns=['labels'])
            valid_y['pred'] = self.first_round_pred

        if loss_func == 'logisticloss':
            self.loss = LogisticLoss
        elif loss_func == 'squareloss':
            self.loss = SquareLoss
        else:
            self.loss = CustomizeLoss(loss_func)

        best_round = 0
        become_worse_round = 0
        if maximize:
            best_metric = -np.inf
        else:
            best_metric = np.inf
        Y = pd.DataFrame(Y.values, columns=['labels'])
        Y['pred'] = self.first_round_pred
        Y['grad'] = self.loss(Y.pred.values,Y.lables.values)
        Y['hess'] = self.loss(Y.pred.values,Y.lables.values)
        Y['sample_weight'] = 1.0
        Y[Y['labels']==1.0]['sample_weight'] = self.scale_pos_weight

        for i in range(boost_round):

            Y['hess'] = Y['hess'] * Y['sample_weight']
            Y['grad'] = Y['grad'] * Y['sample_weight']

            X_colsample = X.sample(frac=self.colsample_bytree,axis=1)
            data = pd.concat([X_colsample,Y],axis=1)
            data = data.sample(frac=self.row_sample,axis=0)

            X_selected = data.drop(['pred','grad','hess','sample_weight','lables'],axis=1)
            Y = data[['pred','grad','hess','lables']]


            tree = Tree()
            tree.fit(X_selected,Y,self.max_depth,self.min_sample_split,self.min_child_weight,self.reg_lambda,self.gamma,self.colsample_bylevel,self.thread_num)

            Y['pred'] += tree.predict(X_selected)
            Y['pred'] = Y['pred'] * self.eta
            Y['grad'] = self.loss(Y.pred.values, Y.lables.values)
            Y['hess'] = self.loss(Y.pred.values, Y.lables.values)

            for feature in tree.feature_importance:
                self.feature_importance[feature] += tree.feature_importance[feature]
            self.trees.append(tree)

            train_loss = get_metric(metric)(Y['pred'],Y['lables'])

            if not is_valid:
                print 'Txboost %s metrics loss of %s round on train is: %s'%(metric,i+1,train_loss)
            else:
                print 'Txboost %s metrics loss of %s round on train is: %s' % (metric, i + 1, train_loss)

                valid_y['pred'] += tree.predict(valid_x)*self.eta
                valid_loss = get_metric(metric)(valid_y['pred'],valid_y)

                print 'Txboost %s metrics loss of %s round on valid is: %s' % (metric, i + 1, valid_loss)

                if maximize:
                    if valid_loss>best_metric:
                        best_metric = valid_loss
                        best_round = i
                        become_worse_round = 0
                    else:
                        become_worse_round += 1
                else:
                    if valid_loss<best_metric:
                        best_metric = valid_loss
                        best_round = i
                        become_worse_round = 0
                    else:
                        become_worse_round += 1

                if become_worse_round >= early_stopping_rounds:
                    print 'stop early'
                    print 'best round is %s , best %s metric is %s'%(best_round+1,metric,best_metric)
                    break

    def predict(self,X):
        preds = np.zeros([X.shape[0],0])
        preds += self.first_round_pred
        for tree in self.trees:
            preds += tree.predict(X)*self.eta

        return preds