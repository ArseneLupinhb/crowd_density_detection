import json
import pandas as pa
from sklearn.model_selection import train_test_split

def read_json(read_path):
	with open(read_path, "r", encoding='utf-8') as f:
		data_json = json.load(f)
	return data_json

json_data = read_json("data/train.json")
print(type(json_data))
len(json_data['annotations'][0]['annotation'])
data_str = json.dumps(json_data['annotations'])
print(data_str)
data_df = pa.read_json(data_str, orient='records', encoding='utf-8')

data_df

data_df.info()


use_clo = ['id', 'num']
result_df = data_df[use_clo]

train_data_df,test_data_df = train_test_split(result_df, test_size=0.2)



# train_data_df = result_df.sample(frac=0.8)
# test_data_df = result_df.sample(frac=0.2)

# train_data_df = result_df.loc[: result_df.shape[0] * 0.8]
# test_data_df = result_df.loc[result_df.shape[0] * 0.8:]

train_data_df.to_csv('data/train_data.csv')
test_data_df.to_csv('data/test_data.csv')

'''
参数配置
'''
train_parameters = {
	"input_size": [3, 224, 224],  # 输入图片的shape
	"class_dim": -1,  # 分类数
	"train_list_path": "data/train.txt",  # train.txt路径
	"eval_list_path": "data/eval.txt",  # eval.txt路径
	"readme_path": "data/readme.json",  # readme.json路径
	"label_dict": {},  # 标签字典
	"num_epochs": 3,  # 训练轮数
	"train_batch_size": 16,  # 训练时每个批次的大小
	"learning_strategy": {  # 优化函数相关的配置
		"lr": 0.001  # 超参数学习率
	}
}




