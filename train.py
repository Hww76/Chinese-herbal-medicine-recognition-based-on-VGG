import paddle
from base_fun import unzip_data,get_data_list,dataset,draw_process,read_json_file,write_json_file
from model import VGGNet

train_parameters = read_json_file("work/parameters.json")
src_path=train_parameters['src_path']
target_path=train_parameters['target_path']
train_list_path=train_parameters['train_list_path']
eval_list_path=train_parameters['eval_list_path']

# 调用解压函数解压数据集
unzip_data(src_path,target_path)

# 划分训练集与验证集，乱序，生成数据列表
#每次生成数据列表前，首先清空train.txt和eval.txt
with open(train_list_path, 'w') as f: 
    f.seek(0)
    f.truncate() 
with open(eval_list_path, 'w') as f: 
    f.seek(0)
    f.truncate() 
#生成数据列表   
get_data_list(target_path,train_list_path,eval_list_path)

#训练数据加载
train_dataset = dataset('C:/Users/Somls/Desktop/code/data',mode='train')
train_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 参数配置，要保留之前数据集准备阶段配置的参数，所以使用update更新字典
train_parameters.update({
    "input_size": [3, 224, 224],                              #输入图片的shape
    "num_epochs": 35,                                         #训练轮数
    "skip_steps": 10,                                         #训练时输出日志的间隔
    "save_steps": 100,                                         #训练时保存模型参数的间隔
    "learning_strategy": {                                    #优化函数相关的配置
        "lr": 0.0001                                          #超参数学习率
    },
    "checkpoints": "C:/Users/Somls/Desktop/code/work/checkpoints"          #保存的路径
})
write_json_file("work/parameters.json", train_parameters) # 将字典写入json文件

model = VGGNet()
model.train()
# 配置loss函数
cross_entropy = paddle.nn.CrossEntropyLoss()
# 配置参数优化器
optimizer = paddle.optimizer.Adam(learning_rate=train_parameters['learning_strategy']['lr'],
                                  parameters=model.parameters()) 

steps = 0
Iters, total_loss, total_acc = [], [], []

for epo in range(train_parameters['num_epochs']):
    for _, data in enumerate(train_loader()):
        steps += 1
        x_data = data[0]
        y_data = data[1]
        predicts, acc = model(x_data, y_data)
        loss = cross_entropy(predicts, y_data)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        if steps % train_parameters["skip_steps"] == 0:
            Iters.append(steps)
            # total_loss.append(loss.numpy()[0])
            total_loss.append(loss.numpy())
            # total_acc.append(acc.numpy()[0])
            total_acc.append(acc.numpy())
            #打印中间过程
            print('epo: {}, step: {}, loss is: {}, acc is: {}'\
                  .format(epo, steps, loss.numpy(), acc.numpy()))
        #保存模型参数
        if steps % train_parameters["save_steps"] == 0:
            save_path = train_parameters["checkpoints"]+"/"+"save_dir_" + str(steps) + '.pdparams'
            print('save model to: ' + save_path)
            paddle.save(model.state_dict(),save_path)
paddle.save(model.state_dict(),train_parameters["checkpoints"]+"/"+"save_dir_final.pdparams")
draw_process("trainning loss","red",Iters,total_loss,"trainning loss")
draw_process("trainning acc","green",Iters,total_acc,"trainning acc")
