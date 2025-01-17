# Autofocus-RNN

## 介绍
《Autofocus of whole slide imaging based on convolution and recurrent neural
networks》自动聚焦算法的实现。

## 程序依赖

- Pytorch 1.1.0
- PIL
- opencv2

## 数据集
百度网盘链接：[https://pan.baidu.com/s/1w8P_1iloZrqw-XeeuTUooQ](https://pan.baidu.com/s/1w8P_1iloZrqw-XeeuTUooQ) 提取码：nn2u 

## 模型参数
百度网盘链接：[https://pan.baidu.com/s/1bZfugCtaq83EkUlpwp1QEA](https://pan.baidu.com/s/1bZfugCtaq83EkUlpwp1QEA) 提取码：bqf8 

## 使用引导
### 数据集处理
下载并解压数据集后，通过 `dataset/tools` 目录下的工具处理数据集，把数据集转化成训练程序需要的数据结构。

1. 构建 `dataset/tools/focus_measures` 目录里面计算 `focus_measures` 的工具，需要依赖 opencv2，CMake 构建工具。
2. 通过 `dataset/tools` 目录下的 python 脚本生成记录数据组信息的 json 文件。`calc_focus_measures.py` 脚本使用步骤 1 生成的工具计算 focus_measures，并把数据保存在 json 文件中，方面训练模型时使用。

### 训练/测试模型
1. 配置 `config.py`，主要设置数据集路径，设置训练集、验证值和测试集。
2. 执行 `train.py`/`evaluate.py` 进行训练或测试。

## 题外话
该项目暂停维护。时隔多年，作者对代码的实现细节记不清楚了。🐶
