# -*- coding: utf-8 -*-
import pandas as pd
import torch
from deepctr_torch.inputs import DenseFeat, SparseFeat, get_feature_names
from deepctr_torch.models import *
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

if __name__ == "__main__":
    # The Criteo Display Ads dataset 是kaggle上的一个CTR预估竞赛数据集。里面包含13个数值特征I1-I13和26个类别特征C1-C26
    # 使用pandas读取上面介绍的数据，并进行简单的缺失值填充
    data = pd.read_csv('../data/criteo_sample.txt')
    # print(data)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # Step1. 对稀疏特征进行标签编码，对稠密特征进行0-1标准化缩放
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # 对于数值特征使用MinMaxScaler压缩到0~1之间。
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    # Step2. 计算每个稀疏特征的不同值个数，并记录字段名
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]
    # print(fixlen_feature_columns)
    # Show like this: [SparseFeat(name='C1', vocabulary_size=27, embedding_dim=4, use_hash=False, dtype='int32', embedding_name='C1', group_name='default_group'), ..., DenseFeat(name='I1', dimension=1, dtype='float32'), ...]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)
    # print(feature_names)
    # Show like this: ['C1', ..., 'C26', 'I1', ..., 'I13']

    # 3.generate input data for model
    # Step3. 为模型生成输入数据
    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    # for name, model_input in train_model_input.items():
    #     print(name)
    #     print(model_input)
    #     print(type(model_input))
    # Show like this: <class 'pandas.core.series.Series'>

    # 4.Define Model,train,predict and evaluate
    # Step4. 定义模型，训练，预测，评估
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    # print("strart to train...")
    history = model.fit(train_model_input, train[target].values, batch_size=32, epochs=10, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
