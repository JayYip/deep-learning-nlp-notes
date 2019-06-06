# ICLR2019 Oral: 图网络有多猛?

前一阵子看到了这篇很有意思的文章: [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826)
, 最近在想transformer结构的时候又想起了这篇文章, 并且特意看了以下ICLR这篇文章的会议视频. 有意思的是, 在作者分享完了之后, 有一位研究者也问了作者怎么看待Self Attention, 作者并没有回答的很详细, 只是说了Self Attention不是Max power GNN(这个概念下面会解释), 但是在node feature比较丰富的时候自有它的应用.

闲话说完, 说回这篇文章.

这篇文章说了以下事情:

1. 介绍了目前GNN的一般框架和WL test(Weisfeiler-Lehman test)
2. 定义了什么是强大的GNN, 证明了什么样的GNN符合强大的标准, 并给出一个具体实现(GIN)
3. 讲了一下一些不powerful的操作

这篇文章个人感觉写的蛮好的, 浅显易懂, 证明都在附录中, 看完对图网络也有了大概的认识. 这篇文章会介绍一下这个工作的结果, 并不会深入理论证明(其实是懒得看了). 水平有限, 如有谬误, 欢迎指正.

## GNN的一般框架

符号解释:

- $V$, $E$: 图的节点和边的集合
- $h_v^k$: 节点$v$在第$k$次循环(第$k$层)的特征向量
- $N(v)$: 节点$v$的所有相邻节点

我们知道, 一般图包含有节点($V$)和边($E$), 我们用$X_v$来表示图的初始节点特征向量(比如one hot), 在GNN中, 我们希望将整个图的信息都学习到一个个节点特征向量$h_v$中, 然后对这个图进行节点分类和整个图的分类.

因此, 一般的NLP任务也可以看作是图, 节点为词, 边就是相邻的词, 对每个词做分类, 即序列标注(NER等任务), 就是对节点做分类. 而一般的分类任务, 比如话题分类, 就是对图做分类.

一般的GNN都是一个循环聚合相邻节点的过程, 也就是说, 一个节点在第k次循环的特征$h_{k}$取决于: 前一次循环的特征$h_{k-1}$和前一次循环的所有邻居的特征$\{h_{u}^{k-1}, u \in N(v) \}$, 而最终整个图的表示, 则是综合所有节点的特征向量, 具体说来就是以下三个公式:

$$
\begin{aligned}
    &a_v^{k} = AGGREGATE^k(\{h_u^{k-1}: u \in N(v) \}) \\
    &h_v^k = COMBINE^k(h_v^{k-1}, a_v^k) \\
    &h_G = READOUT(\{ h_v^k|v \in G \})
\end{aligned}
$$

上面第一条公式得到相邻节点的特征, 第二条公式结合了相邻节点特征和自身特征得到新的特征, 第三条公式结合了所有节点的最终表示得到图表示. GNN实现的不同点一般在与$AGGREGATE$, $COMBINE$和$READOUT$的选择. 比如在GraphSAGE中, $AGGREGATE$就是dense + ReLu + Max Pooling, $COMBINE$就是concat+dense.

## WL Test

WL Test是将节点不断循环聚合的过程, 每个循环里面会做以下两个事情:

1. 整合节点及其相邻节点的标签
2. 将整合后的标签hash成一个新的标签

对于两个图, 如果在WL test的循环中出现任何一个节点的标签不一致, 那么这两个图是不类似的(non-isomorphic).

和GNN不同的是, GNN将节点映射成一个低维的, dense的向量, 而WL test则映射成一个one-hot.

比如, 对"机器学习真有趣", "机器学习真无趣"这个两图的"习"节点进行WL test循环, 得到的aggregated label为:

```
iter 0: 习        
iter 1: 学习真     
iter 2: 器学习真有

iter 0: 习        
iter 1: 学习真     
iter 2: 器学习真无
```

在`iter 2`发现'习'节点标签不一样, 那么这两个图是non-isomorphic的.

## 什么是强大的GNN?

感觉有股中二的气息...

强大的GNN指的是, 对于任意不同的图, 都能够通过将它们映射到同一个空间中的不同向量来区分它们.

接着作者证明了以下两个theorem:

1. WL test是强大的
2. 如果$AGGREGATION$, $COMBINE$和$READOUT$函数都是一对一的映射的话, 那么GNN和WL test一样强大

因此, 只要满足上面的2的GNN, 就是强大的, 作者提出了一个强大并且简单的GNN: Graph Isomorphism Network(GIN).

## GIN

接着, 作者证明了下面的引理(原文并不限于图的语境下, 这里为了方便理解稍做修改, 不严谨!!!):

假设图$G$的节点是可数的, 且节点的相邻节点$N(v)$数量有上界, 那么**存在**一个函数$f: V \rightarrow R^n$, 使得有无限个$\epsilon$, 函数 

$$h(v, N(v)) = (1+\epsilon)\cdot f(v) + \Sigma_{u \in N(v)}f(u)$$

和$(v, N(v))$是一一对应的(为啥要加和? 请见Lemma 5). 并且任意函数$g$都可以拆解成$g(v, N(v)) = \phi(h(v, N(v)))$.

存在这样一个函数就好办了, 由于universal approximation theorem, MLP可以拟合任意函数, 直接一个MLP怼上去就好了, 顺带还拟合了复合函数$f\circ \phi$:

$$ h_v^k=MLP((1 + \epsilon ) \cdot h_v^{k-1} + \Sigma_{u \in N(v)}h_u^{k-1}) $$

这里$\epsilon$可以预先设定一个固定值, 也可以通过学习得到.

好了, 到这里我们有$AGGREGATION$, $COMBINE$都是一对一映射了

对于$READOUT$, 直接加和就好了, 为了让结果漂亮一点, 作者还concat了每一层的特征

$$ h_G = CONCAT(SUM(\{ h_v^k | v \in G \}) | k = 0, 1, \dots, K)$$

## 不够强大的GNN

如果用下面的结构就不够强:

- 单层Perceptron
- 用mean pooling或者max pooling代替sum
- 对于mean, 如果图的统计信息和分布信息比图结构重要的话 那么mean pooling的结果也会不错. 另外, 如果节点特征差异比较大并且很少重复的话, 那么mean和sum一样强大
- 如果对于任务来说, 最重要的是确定边缘节点, 或者说数据“骨架”而不是图结构的话, max pooling可能效果也不错

## 实验结果

这里就不详细叙述了, 个人感觉比较有意思的实验结果是Figure 4, 作者分别比较了sum, mean, max, mlp, single layer perceptron在**训练集**的效果, 看能不能拟合到WL subtree kernel的效果, 实验结果证明了作者是对的. 至于泛化能力, 理论里面并没有对泛化能力做保证, 但是还是效果还是很不错的.

原文链接: https://github.com/JayYip/deep-learning-nlp-notes/blob/master/ICLR2019%20Oral-%20%E5%9B%BE%E7%BD%91%E7%BB%9C%E6%9C%89%E5%A4%9A%E7%8C%9B%3F.md