from pickletools import optimize
import pandas as pd
import numpy as np
from icecream import ic
import os
import sys

from requests import Session
from sklearn.model_selection import learning_curve
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
import tensorflow.compat.v1 as tf


class Cabbage():
    def __init__(self):
        self.basedir= os.path.join(basedir, 'model')
        self.data = None
        self.df = None
        self.x_data = None
        self.y_data = None


    def cabbage_input(self,avgTemp,minTemp,maxTemp,rainFall):
        print(f'훅에 전달된 avgTemp : {avgTemp}, minTemp : {minTemp}, maxTemp : {maxTemp}, rainFall : {rainFall} ')

    def preprocessing(self):
        self.data = pd.read_csv('../data/price_data.csv', encoding='UTF-8', thousands=',')
        # year,avgTemp, minTemp, maxTemp, rainFall, avgPrice
        xy = np.array(self.data, dtype=np.float32) # csv파일을 배열로변환
        ic(type(xy)) # <class 'numpy.ndarray'>
        ic(xy.ndim) # xy.ndim: 2 
        ic(xy.shape) # xy.shape: (2922, 6)
        self.x_data = xy[:, 1:-1] # 해당 날짜의 날씨에 해당하는 기후요소 4개를 변인으로 받는다.
        # 슬라이싱 기준은 순서대로 행 ,렬 [ : ] 처음부터 끝까지 행을 가져와 그중 [ 1: -1 ] 열값을 가져온다. 
        self.y_data = xy[:,[-1]]  # 해당날짜의 가격을 입력한다.
        # [-1] 인덱싱 문법 

 
    def cerate_model(self):
        # 텐서모델 초기화 (모델템플릿 생성, (깡통모델))
        model = tf.global_variables_initializer()
        # 확률변수 데이터
        self.preprocessing()
        df = self.data
        # 선형식(y = wx + b) 제작
        # placeholder를 사용하여 그릇을 만들고 필요할 때 마다 feed 값을 던진다.
        X = tf.placeholder(tf.float32, shape=[None, 4] ) # shape 투입되는 값 
        Y = tf.placeholder(tf.float32, shape = [None,1])
        # 텐서플로우에서는 가중치와 바이어스에 해당하는 변수를 Variable 을 사용하여 나타낸다. 
        # name은 반드시 있어야한다.
        W = tf.Variable(tf.random_normal([4,1]), name="weight") # ([4,1]) 4개가 투입돼서 1개가 나온다
        b = tf.Variable(tf.random_normal([1]), name="bias")
        hypothesis = tf.matmul(X,W) + b # Wx +b (가설), matmul 상호곱셈
 
        # 손실함수
        cost = tf.reduce_mean(tf.square(hypothesis - Y))  # reduce_mean 평균을 최소화 시킴
        # 최적화 알고리즘
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00005)
        train = optimizer.minimize(cost)

        # 세션 생성
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # 트레이닝
        for step in range(100000):
            cost_, hypo_, _= sess.run([cost,hypothesis,train],
                                     feed_dict={X:self.x_data, Y:self.y_data})
            if step % 500 == 0:
                print('# %d 손실비용: %d'%(step,cost_))
                print('-배추가격:%d'%(hypo_[0]))

        # 모델저장
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.basedir, 'cabbage', 'cabbage.ckpt'), global_step=1000)
        
    def load_model(self, avgTemp,minTemp,maxTemp,rainFall): # 모델로드
        # 선형식(가설)제작 y = Wx+b
        X = tf.placeholder(tf.float32, shape=[None, 4])
        W = tf.Variable(tf.random_normal([4, 1]), name="weight")
        b = tf.Variable(tf.random_normal([1]), name="bias")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.path.join(self.basedir, 'cabbage', 'cabbage.ckpt'))
            data = [[avgTemp,minTemp,maxTemp,rainFall],]
            arr = np.array(data, dtype=np.float32)
            dict = sess.run(tf.matmul(X, W) + b, {X: arr[0:4]})
            print(dict)
        return int(dict[0])
        
if __name__=='__main__':
    tf.disable_v2_behavior()
    Cabbage().preprocessing()
    Cabbage().cerate_model()