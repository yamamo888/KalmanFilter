# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 13:12:17 2019

@author: yu
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pdb

class ParticlerFilter():
    
    def __init__(self,y,nP,sigma,alpha):
        
        self.y =  y
        self.nP = nP
        self.sigma = sigma
        self.alpha = alpha
        self.logLF = -np.inf
        
    
    def norm_likelihood(self,y,x,s2):
        # 尤度
        return np.power(np.sqrt(2*np.pi*s2),-1) * np.exp(-(y-x)**2/(2*s2))
    
    def InvF(self,WC,idex,u):
        if np.any(WC<u) == False:
            return 0
        k = np.max(idex[WC<u])
        return k+1
    
    def resampling(self,weights):
        wc = np.cumsum(weights)
        idx = np.asanyarray(range(self.nP))
        # サンプリングしたkのリスト
        Klist = np.zeros(self.nP,dtype=np.int32)
        
        # 一様分布から重みに応じでリサンプリングする添え字を取得
        for i,u in enumerate(np.random.uniform(0,1,size=self.nP)):
            k = self.InvF(wc,idx,u)
            Klist[i] = k
        return Klist
    
    def resampling2(self,weights):
        """
        計算量の少ない層化サンプリング
        """
        idx = np.asanyarray(range(self.nP))
        initU = np.random.uniform(0,1/self.nP)
        u = [1/self.nP*1+initU for i in range(self.nP)]
        wc = np.cumsum(weights)
        k = np.asanyarray([self.InvF(wc,idx,val) for val in u])
        return k
    
    def simulate(self,seed=71):
        # 毎回同じ予測をするため?
        # 一様分布(0.0以上,1.0未満)
        np.random.seed(seed)
        
        # 時系列データ数
        time = len(self.y)
        # 潜在変数
        self.x = np.zeros((time+1,self.nP))
        self.xResample = np.zeros((time+1,self.nP))
        
        # 潜在変数(直接観測できない)の初期値
        initX = np.random.normal(0,1,size=self.nP)
        self.x[0] = initX
        self.xResample[0] = initX
        
        # 重み
        self.w = np.zeros((time,self.nP))
        self.wNorm = np.zeros((time,self.nP))
        
        #　時刻毎の尤度
        self.lh = np.zeros(time)
        
        for t in range(time):
            print("time{}".format(t))
            for i in range(self.nP):
                """
                1階差分トレンド
                """                
                # システムノイズ
                v = np.random.normal(0,np.sqrt(self.alpha*self.sigma))
                # システムノイズの付加
                self.x[t+1,i] = self.xResample[t,i] + v
                # y[t]に対する各粒子の尤度
                self.w[t,i] = self.norm_likelihood(self.y[t],self.x[t+1,i],self.sigma)
            # 規格化
            self.wNorm[t] = self.w[t]/np.sum(self.w[t])
            # 各時刻対数尤度
            self.lh[t] = np.log(np.sum(self.w[t]))
        
        
            # リサンプリングで取得した粒子の添え字
            #k = self.resampling(self.wNorm[t])
            # リサンプリングで取得した粒子の添え字(層化サンプリング)
            k = self.resampling2(self.wNorm[t])
            self.xResample[t+1] = self.x[t+1,k]
            
        # 全体の対数尤度
        self.logLF = np.sum(self.lh)-time*np.log(nP)
        
    def FilterValue(self):
        """
        尤度の重みで加重平均した値でフィルタリングされた値を算出
        """
        return np.diag(np.dot(self.wNorm,self.x[1:].T))
    
    def Plot(self):
        """
        赤い粒：パーティクル
        パーティクルに尤度をベースにした重みを割り当て、加重平均をとることで
        緑の線のような潜在状態を推定できる
        青線：観測モデル、潜在状態から正規分布に従うようなノイズが発生、潜在状態の値＋ノイズ＝実現値としての観測値
        緑線の潜在状態の値を推定するのが目的
        """
        
        time = len(self.y)
        
        plt.figure(figsize=(16,8))
        plt.plot(range(time),self.y)
        plt.plot(self.FilterValue(),'g')
        
        for t in range(time):
            plt.scatter(np.ones(self.nP)*t,self.x[t],color='r',s=2,alpha=0.1)
            
        plt.savefig('PF.png')
            
        
if __name__ == '__main__':
    
    # データダウンロード
    myData = pd.read_csv("http://daweb.ism.ac.jp/yosoku/materials/PF-example-data.txt",header=-1)
    myData.columns = ['data']
        
    # ハイパーパラメータ
    ##################
    beta = -2
    gamma = -1
    
    # 粒子数
    nP = 1000
    sigma = 2**beta
    alpha = 10**gamma
    ##################
    
    
    # 粒子フィルタ
    PF = ParticlerFilter(myData.data.values,nP,sigma,alpha)
    PF.simulate()
    PF.Plot()
    
    
    