from autograd import elementwise_grad
import numpy as np

elementwise_hess = lambda func : elementwise_grad(elementwise_grad(func))

class BaseLoss(object):
    def __init__(self):
        pass

    def grad(self,pred,y):
        raise NotImplementedError()

    def hess(self,pred,y):
        raise NotImplementedError()

class SquareLoss(BaseLoss):
    #0.5*(y-pred)**2
    def grad(self,pred,y):
        return (pred - y)

    def hess(self,pred,y):
        return np.ones_like(pred)

class LogisticLoss(BaseLoss):
    #-[y*log(p)+(1-y)*(log(1-p))]

    def transform(self,pred):
        return 1/(1+np.exp(-pred))

    def grad(self,pred,y):
        pred = self.transform(pred)
        return (1-y)/(1-pred) -y/pred

    def hess(self,pred,y):
        pred = self.transform(pred)
        return (1-y)/(1-pred)**2 + y/pred**2

class CustomizeLoss(BaseLoss):

    def __init__(self,loss_func):
        self.loss_func = loss_func

    def grad(self,pred,y):
        pred = self.transfrom(pred)
        return elementwise_grad(self.loss_func)(pred,y)

    def hess(self,pred,y):
        pred = self.transfrom(pred)
        return elementwise_hess(self.loss_func)(pred,y)

    def transfrom(self,pred):
        return 1/(1+np.exp(-pred))