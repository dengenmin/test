# author:komorebi time:2020/3/28
import numpy
import scipy.special
#定义神经网络类
class Neuralnetwork:
    #初始化函数
    def __init__(self,inputnodes,hidddennodes,outputnodes,learningrate):
        #设置输出输入中间层节点个数
        self.inodes = inputnodes
        self.hnodes = hidddennodes
        self.onodes = outputnodes

        #设置学习率
        self.lr = learningrate

        #设置两个权重矩阵wih who
        self.wih = (numpy.random.normal( 0.0 , pow(self.hnodes, -0.5), (self.hnodes,self.inodes)))
        self.who = (numpy.random.normal( 0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        #设置sigmoid函数
        self.activation_function = lambda x: scipy.special.expit(x)

    #训练神经网络
    def train(self，inputs_list,targets_list):
        # 将输入列表和目标列表转化为二维数组 不加不知道会怎样
        inputs = numpy.array(inputs_lists, ndmin=2).t
        targets = numpy.array(targets_lists, ndmin=2).t


        # 计算输出结果
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #计算误差 误差 = 目标 — 实际输出
        output_errors = targets -final_outputs

        #计算隐藏节点误差
        hidden_errors = numpy.dot(self.who.T,output_errors)
        #更新隐藏层与输出层的权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs *(1.0 - final_outputs )),numpy.transpose(hidden_outputs))
        #更新输入层与隐藏层的权重
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs )),numpy.transpose(inputs))

        pass

    #查询神经网络
    def query(self,inputs_lists):
        #将输入列表转化为二维数组 不加不知道会怎样
        inputs = numpy.array(inputs_lists,ndmin=2).t

        #计算输出结果
        hidden_inputs = numpy.dot(self.wih , inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #打印输出结果
        #print(final_outputs)
        pass

if __name__ == '__main__':
    n = Neuralnetwork(3,3,3,0.25)
    print(n.who,'\n',n.wih)
    n.query([1,2,3])