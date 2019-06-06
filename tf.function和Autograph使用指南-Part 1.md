# tf.function和Autograph使用指南-Part 1

AutoGraph是TF提供的一个非常具有前景的工具, 它能够将一部分python语法的代码转译成高效的图表示代码. 由于从TF 2.0开始, TF将会默认使用动态图(eager execution), 因此利用AutoGraph, **在理想情况下**, 能让我们实现用动态图写(方便, 灵活), 用静态图跑(高效, 稳定).

但是! 在使用的过程中, 如无意外肯定是会有意外的, 这篇文章就是指出一些AutoGraph和tf.function的奇怪的行为, 让你更愉快地使用它们.

本文假设读者具有一定的Python和TensorFlow的使用经验.

## 会话执行

对tf1.X有经验的读者应该不会对让我们~~又爱~~又恨的计算图(`tf.Graph`)和执行会话(`tf.Session`)感到陌生, 一个常规的流程如下:
1. 初始化一个计算图并且将该计算图设置为当前scope下的默认计算图
2. 用TF API设计计算图(比如: `y=tf.matmul(a, x) + b`)
3. 提前界定好参数共享并划分相应的参数scope
4. 创建并配置好`tf.Session`
5. 将计算图传给`tf.Session`
6. 初始化参数
7. 用`tf.Session.run`来执行计算图的节点, 被执行的节点会反向追踪所有依赖的需要执行的节点并**执行计算**.

以下是上述过程的一个代码例子:
```python
g = tf.Graph() #初始化计算图
with g.as_default(): # 设置为默认计算图
    a = tf.constant([[10,10],[11.,1.]]) 
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b # 描述计算图
    init_op = tf.global_variables_initializer() # 待执行节点

with tf.Session() as sess: # 配置会话
    sess.run(init_op) # 执行节点
    print(sess.run(y)) # 输出结果
```

在TF 2.0中, 由于默认为动态图, 计算会直接被执行, 也就是说, 我们不需要

- 定义计算图
- 会话执行
- 参数初始化
- 用scope定义参数分享
- 用`tf.control_dependencies`来声明节点的非直接依赖

我们可以像写普通python代码(or pytorch)一样, 写了就执行:
```python
a = tf.constant([[10,10],[11.,1.]])
x = tf.constant([[1.,0.],[0.,1.]])
b = tf.Variable(12.)
y = tf.matmul(a, x) + b
print(y.numpy())
```
一般来说, eager代码会比执行相同操作的静态图代码的效率低, 因为很多计算图优化的方法只能用在数据流图上.

如果想在TF 2.0上构建传统的计算图, 我们就需要用到`tf.function`.

## 函数, 而非会话
TF 2.0的其中一个重要改变就是[去除`tf.Session`](https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md)(此处应有掌声). 这个改变会迫使用户用更好的方式来组织代码: 不用再用让人纠结的`tf.Session`来执行代码, 就是一个个python函数, 加上一个简单的装饰器.

在TF 2.0里面, 如果需要构建计算图, 我们只需要给python函数加上`@tf.function`的装饰器.

> 上文提到静态图的执行效率更高, 但是加速并不是一定的. 一般来说, 计算图越复杂, 加速效果越明显. 对于复杂的计算图, 比如训练深度学习模型, 获得的加速是巨大的. (译者注: 个人感觉还是要结合实际来看, 如果某一部分的计算既有复杂的计算图, 而计算图的复杂性又带来了额外的[内存消耗](https://mxnet.incubator.apache.org/versions/master/architecture/note_memory.html)
或者计算量, 那么加速会比较明显, 但是很多时候, 比如一般的CNN模型, 主要计算量并不在于图的复杂性, 而在于卷积、矩阵乘法等操作, 加速并不会很明显. 此处想法有待验证)

这个自动将python代码转成图表示代码的工具就叫做AutoGraph.

在TF 2.0中, 如果一个函数被`@tf.function`装饰了, 那么AutoGraph将会被自动调用, 从而将python函数转换成可执行的图表示.

## tf.function: 究竟发生了什么?
在第一次调用被`@tf.function`装饰的函数时, 下列事情将会发生:

- 该函数被执行并跟踪。和Tensorflow 1.x类似, Eager会在这个函数中被禁用，因此每个`tf.`API只会定义一个生成`tf.Tensor`输出的节点
- AutoGraph用于检测可以转换为等效图表示的Python操作（`while`→`tf.while`，`for`→`tf.while`，`if`→`tf.cond`，`assert`→`tf.assert`...)
- 为了保留执行顺序，在每个语句之后自动添加`tf.control_dependencies`，以便在执行第`i+1`行时确保第`i`行已经被执行. 至此计算图已经确定
- 根据函数名称和输入参数，创建唯一ID并将其与定义好的计算图相关联。计算图被缓存到一个映射表中：`map [id] = graph`
- 如果ID配对上了，之后的函数调用都会直接使用该计算图

下一节将会具体阐述如何将TF 1.X代码块分别改写到eager和计算图版本.

## 改写到eager execution

要使用`tf.function`, 第一步需要先将TF 1.X的设计计算图的代码放进python函数里面.

```python
def f():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    return y
```

应为TF 2.0默认是eager的, 我们可以直接执行该函数(不需要`tf.Session`):

```python
print(f().numpy())
```

我们就会得到输出:

```python
[[22. 22.]
 [23. 13.]]
```

## 从eager到tf.function

我们可以直接用`@tf.function`来装饰函数`f`, 我们在原来`f`的基础上加上宇宙第一的debug大法: `print`来更好地看看究竟发生了什么.
```python
@tf.function
def f():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y

f()
```

所以发生了什么呢?

- `@tf.function`将函数`f`包进了[`tensorflow.python.eager.def_function.Function`](https://github.com/tensorflow/tensorflow/blob/0596296aed520fdc81829297d01cad9f8f48da14/tensorflow/python/eager/function.py#L1268)这个对象, 函数`f`被赋予到了这个对象的`.python_function`属性.
- 当`f()`被执行的时候, 计算图会同时被构建, 但是计算不会执行, 因此我们会得到以下结果, `tf.`的操作不会被执行:
```python
PRINT:  Tensor("add:0", shape=(2, 2), dtype=float32)
```
- 最终, 你会看到代码会执行失败:
```python
ValueError: tf.function-decorated function tried to create variables on non-first call.
```
在 [RFC: Functions, not Session](https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md#functions-that-create-state)里面有个非常明确的指示:
> State (like `tf.Variable` objects) are only created the first time the function f is called. 状态(比如`tf.Variable`) 只会在函数被第一次调用时创建.

但是 [Alexandre Passos](https://github.com/tensorflow/tensorflow/issues/26812#issuecomment-474595919)指出, 在函数转换成图表示时, 我们没有办法确定`tf.function`调用了多少次函数, 因此我们在第一次调用函数`f`时, 在图构建的过程中, 可能会被执行了多次, 这就导致了上述错误.

造成这个错误的根源在于同样的命令在动态图和静态图中的不一致性. 在动态图中, `tf.Variable`时一个普通的python变量, 超出了其作用域范围就会被销毁. 而在静态图中, `tf.Variable`则是计算图中一个持续存在的节点, 不受python的作用域的影响. 因此, 这是使用`tf.function`的第一个教训:
> 将一个在动态图中可行的函数转换成静态图需要用静态图的方式思考该函数是否可行

那么我们可以怎样去规避这个错误呢? 

1. 将`tf.Variable`作为函数的参数传入
2. 将父作用域继承`tf.Variable`
3. 将`tf.Variable`作为类属性来调用

## 用改变变量作用域来处理

这里指方法2和方法3. 显然的, 我们推荐使用方法3:

```python
class F():
    def __init__(self):
        self._b = None

    @tf.function
    def __call__(self):
        a = tf.constant([[10, 10], [11., 1.]])
        x = tf.constant([[1., 0.], [0., 1.]])
        if self._b is None:
            self._b = tf.Variable(12.)
        y = tf.matmul(a, x) + self._b
        print("PRINT: ", y)
        tf.print("TF-PRINT: ", y)
        return y

f = F()
f()
```

## 将状态作为传入参数来处理

我们之后会看到, 我们并不能随意地用`tf.function`来转化eager的代码并达到加速的目的, 我们需要想象一下转化是怎么完成的, 在转python的代码到图操作的时候究竟发生了什么, 这些转化包含了什么**黑魔法**. 这里的例子比较简单, 我们会在接下来的文章中更深入的探讨.
```python
@tf.function
def f(b):
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y

b = tf.Variable(12.)
f(b)
```
上述函数会得到我们想要的结果, 另外, 作为参数被传入的变量能够在函数中直接更新, 而更新后的值会在函数外也适用. 下面的代码会打印出1,2,3
```python
a = tf.Variable(0)

@tf.function
def g(x):
    x.assign_add(1)
    return x

print(g(a))
print(g(a))
print(g(a))
```
## 总结

- 我们可以用`@tf.function`装饰器来将python代码转成图表示代码
- 我们不能在被装饰函数中初始化`tf.Variable`
- 可以用变量作用域继承(对象属性)或者参数传入的方法使用在函数外初始化的变量

在之后的部分我们会更加深入地探讨输入参数类型对效率的影响, 以及python操作的转换细节.

原文链接: https://github.com/JayYip/deep-learning-nlp-notes/blob/master/tf.function%E5%92%8CAutograph%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97-Part%201.md

声明: 本文翻译自[Paolo Galeone的博客](https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/), 已取得作者的同意, 如需转载本文请联系本人

Disclaimer: This is a translation of the article [Analyzing tf.function to discover AutoGraph strengths and subtleties](https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/) by Paolo Galeone.