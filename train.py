# _*_ coding: utf-8 _*_
#@Time  : 2021/04/28
#@Author: Eric
#@File  : train.py

import pandas as pd
import numpy as np
import kashgari
from kashgari.embeddings import BertEmbedding
# 載入 tensorflow 訓練時，需要用到的模組
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
# 載入 kashgari 需要用到的分類模組
from kashgari.tasks.classification import BiLSTM_Model
import logging
def main():
    # 請輸入訓練檔案名稱
    result = pd.read_csv('train.csv',encoding='utf_8_sig')
    # 將缺值補成N
    result = result.fillna(axis=0,value="N")
    # 將問題整理成訓練格式
    qs= []
    for items in result["問題"]:
      only =[]
      for item in items:
        only.append(item)
      qs.append(only)
    # 將答案整理成訓練格式
    raws = []
    for i in result["答案"]:
        raws.append(i)
    index = [i for i in range(len(raws))]
    np.random.shuffle(index)
    # 因為訓練的格式需要再包再一個list中，所以需要再做一次這個操作
    qs2 =[]
    aws2=[]
    for i in index:
      qs2.append(qs[i])
      aws2.append(raws[i])
    # 印出問題及答案總長度
    print(qs2)
    print( "問題總長度 : " +str(len(qs2)))
    print(aws2)
    print( "答案總長度 : " +str(len(aws2)))
    # 調整訓練的參數
    train_x =  qs2[0:950]
    train_y =  aws2[0:950]
    valid_x =  qs2[951:1030]
    valid_y =  aws2[951:1030]
    # test_x =  qs2[1202:1280]
    # test_y =  aws2[1202:1280]
    # 選擇預訓練bert模型，rbt3
    bert_embed = BertEmbedding('rbt3')
    # log
    logging.basicConfig(level='DEBUG')
    # 以下為訓練時的參數調整
    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.1, patience=2, verbose=1, min_lr=1e-6)
    stop_callback = EarlyStopping(patience=3, restore_best_weights=True)
    model = BiLSTM_Model(bert_embed)
    model.fit(train_x,
              train_y,
              valid_x,
              valid_y,
              callbacks=[stop_callback,reduce_lr],
              batch_size=3,
              epochs=50)
    # 將模型儲存
    model.save('result_model')

# 模型評估
# loaded_model = kashgari.utils.load_model('result_model')
# loaded_model.evaluate(test_x, test_y)

