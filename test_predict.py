import kashgari

# 測試模型答案
loaded_model = kashgari.utils.load_model('result_model')
print(loaded_model.predict([["預測值"]]))


