### T5

编写一个函数，该函数接受一个 NumPy 数组（ndarray）作为输入，要求返回一个新的数组，新数组只包含原数组中所有偶数行（索引为0, 2, 4...）的数据。

> 考察知识点，numpy array的创建，切片操作练习。

### T6 

编写一个函数，该函数接受一个 Pandas DataFrame 和一个年龄阈值（整数）作为输入。函数应返回一个新的 DataFrame，其中只包含 'Age' 列大于该阈值的行。
这道题基于第三题，请先完成第三题！

> 两种实现手段，mask的生成和使用，还有用query的

### T7

编写一个函数，创建两个 PyTorch 张量（Tensor）：
A 的形状为 (2, 3)
B 的形状为 (3, 4)
然后，计算这两个张量的矩阵乘法（matrix multiplication），并返回结果。
> Pytorch中的矩阵乘法

### T8
编写一个函数，执行以下操作：
创建一个值为 3.0 的浮点数张量 x，并设置 requires_grad=True。
根据 x 计算 y，公式为 y = 2*x**2 + 5。
使用 PyTorch 自动计算 y 相对于 x 的梯度（也就是导数 dy/dx）。
返回这个梯度值。
> 了解pytorch中的自动微分
### T9

编写一个函数，该函数：
接受一个 NumPy 数组作为输入。
将这个 NumPy 数组转换成一个 PyTorch 张量。
将张量中的每一个元素乘以 2。
将结果张量转换回 NumPy 数组，并返回。

> 练习数据结构的转换


### T10

探究Python中的dataloader，dataset。研究如何自己创建数据集。
研究如何用dataloader快速遍历数据集。

> Dataset的构建逻辑，以及如何使用loader


### T11
手写一个MLP，拟合一个复杂函数，给出拟合效果。

### T12
现在，请你把我们目前学到的所有东西整合起来：
- 加载你的彗星数据集 CSV 文件 (near-earth-comets.csv)。
- 创建一个为该数据集服务的 CometDataset 类（就像你在第10题做的那样）。
- 创建一个 DataLoader 来批量加载数据。
- 创建一个继承自 nn.Module 的线性回归模型 LinearRegressionModel（就像我刚刚上面展示的那样），确保输入和输出维度正确。
写一个完整的训练循环，使用 MSELoss 和 Adam 优化器，对你的模型进行至少100个 epoch 的训练。在训练过程中，每10个 epoch 打印一次当前的 loss。
这将是你第一个完整的、端到端（从原始文件到训练好的模型）的项目。




