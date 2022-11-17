
# 计算图设计
采用静态计算图架构。
## 节点
使用`GraphNode`表示节点。
节点有名称、ID、shape、输入节点列表、输出节点列表、是否需要计算梯度的标记、节点类型等信息。
其中节点类型包括了 `GRAPH_NODE_TYPE_PARAM`/`GRAPH_NODE_TYPE_INSTANCE`/`GRAPH_NODE_TYPE_HIDDEN` 3种类型。

// TODO .input_fork_  .is_target_

以`GraphNode`为基类，衍生出了一系列的子类。
* 基础子类，用于收拢同一类节点的共用逻辑（如构造函数、输入节点的形状检查、输出节点的形状推理）
    * `GraphNodeUnaryBase`
    * `GraphNodeUnaryElementWiseBase`
    * `GraphNodeBinaryBase`
    * `GraphNodeBinaryElementWiseBase`
    * `GraphNodeBroadcastBase`
    * `GraphNodeForAxisBase`
    * `GraphNodeReduceAxisBase`
* 单输入 element-wise 节点
    * `SigmoidNode`
    * `TanhNode`
    * `AbsNode`
    * `ClipByValueNode`
    * `IdentityNode`
    * ...
* 双输入 element-wise 节点
    * `AddNode`
    * `SubNode`
    * `GreaterNode`
    * ...
* 双输入 broadcast 节点
    * `BroadcastAddNode`
    * `BroadcastSubNode`
    * `BroadcastGreaterNode`
    * ...
* Repeat类型
    * `BroadcastToNode`
    * `BroadcastToLikeNode`
* 按轴遍历的节点
    * `SoftmaxNode`
    * `Softmax2Node`
    * `LogSoftmaxNode`
    * ...
* 按轴reduce的节点
    * `ReduceMeanNode`
    * `ReduceSumNode`
    * `ReduceL1Node`
    * `ReduceL2Node`
    * `ArgMaxNode`
    * ...
* FM家族节点
    * `BatchFMInteractionNode`
    * `BatchFMInteraction2Node`
    * `BatchFMQuadraticNode`
    * ...
* 卷积节点
    * `GraphNodeConvBase`
    * `Conv1dNode`
    * `Conv2dNode`
    * ...
* 池化节点
    * `GraphNodePoolBase`
    * `MaxPool1dNode`
    * ...
* Loss节点
    * `AbsoluteErrorNode`
    * `SquareErrorNode`
    * `BCELossNode`
    * ...
* 样本节点
    * `InstanceNode`
* 参数节点
    * `VariableNode`
* 常量节点
    * `ConstantNode`
    * `ConstantLikeNode`
    * `ZerosNode`
    * `OnesNode`
    * ..
* 随机节点
    * `RandomNormalNode`
    * `RandomUniformNode`
    * ...
* 常用网络结构
    * `EmbeddingLookupNode`
    * `GroupEmbeddingLookupNode`
    * `FullyConnectNode`
    * `TensorDotNode`
    * ...
* 矩阵相关
    * `GEMMNode`
    * `BatchGEMMNode`
    * `MatmulNode`
    * ...
* ...


**从节点的设计可以看到，DX 静态图的设计和 TensorFlow 的很类似，但是抽象和封装的程度较低，基本上就是按照应用需求封装对应的能力，这种方法在设计和实现上更加容易，但是代码量可能会更多，灵活性和扩展性也更低**

为什么 loss 需要定义一种独有的节点类型？

## OP
就像节点一样，在DX中，OP的概念和 TensorFlow 的也是类似的。
不同之处是在 DX 中，OP 和 Node 是一一对应的，并且在命名上也遵循固定的规则，例如 `AbcOp` 对应 `AbcNode` ，其中 `Abc` 在代码中称为 `class_name`。
`class_name` 是一个关键的概念，它既是一个类的名称（的一部分），也是计算图序列化、可移植性的基础（通过 string-name 找到 code）。
基于这个关联关系，计算图执行引擎（`OpContext` 这是一个奇怪的名字）从 node 获得 `class_name`，然后创建对应的 op。

基类`Op`定义了几个纯虚函数，基于此可以看到一个 op 包含了哪些内容和逻辑。
```c++
class Op : public DataType {
 public:
  virtual ~Op() = default;
  virtual const char* class_name() const noexcept = 0;
  virtual void Init(const Graph* graph, const GraphNode* node, TensorMap* param,
                    Hidden* hidden, TensorMap* ptr, TensorMap* grad,
                    TensorMap* grad_ptr, TensorMap* overwritten_param,
                    TensorMap* overwritten_ptr) = 0;
  virtual void InitForward() = 0;
  virtual void InitPredict() = 0;
  virtual void InitBackward() = 0;
  virtual void Forward() = 0;
  virtual void Predict() = 0;
  virtual void Backward() = 0;
  virtual void GetPullRequest(PullRequest* pull_request) const = 0;
};
```


# 编程模型
使用 DX 进行编程的主要内容是构建计算图，这一点和 TensorFlow V1 是类似的。
构建计算图的过程，实际上就是定义一些节点，并将这些节点进行关联的过程。
DX 通过宏定义的方式，为每个类型的节点定义对应的创建节点的函数，例如`MatmulNode`节点对应的创建节点的函数为`Matmul`，这使得用户可以在一定程度上摆脱计算图的细节，使得构建计算图的过程更加自然（命令式编程）。
