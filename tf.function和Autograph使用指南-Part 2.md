# tf.function和Autograph使用指南-Part 2

在第1部分中，我们已经知道了如何将TF 1.x代码转换为其eager的代码，然后又将eager的代码通过`tf.function`转换为图表示代码，并遇到了在该函数中创建状态(`tf.Variable`)时由于eager和图表示的差异而导致的问题。

在本文中，我们将要分析用`tf.Tensor`和python对象作为被`tf.function`装饰的函数的输入时的异同. 那么是否在被装饰函数中的所有逻辑都将转换为符合期望的图表示代码呢？

## tf.function调用了AutoGraph

首先我们来看下`tf.function`的所有输入标志(`input_signature`):
```
def function(func=None,
             input_signature=None,
             autograph=True,
             experimental_autograph_options=None)
```
参数`autograph`的默认值为`True`, 因此之前我们用`@tf.function`装饰的函数是使用了`autograph`了的. 我们可以看[文档](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/function)中对该参数的描述:

- 当该值为`True`时, 所有依赖于tensor值的python操作都会被加入到TensorFlow图中
- 当该值为`False`时, 该函数的操作会被加入到Tensorflow图中, 但是python的逻辑不会被AutoGraph转化

因此, `tf.function`默认会调用`AutoGraph`. 我们接下来看看如果我们改变输入参数类型和函数结构的话会发生什么.

## 改变tf.Tensor的数据类型

我们先构思一个用来测试的函数, 函数输入参数的数据类型是很重要的, 因为输入参数会被用来构造图, 而且这个图是静态类型的, 并且会有它的一个独特的ID.(请参见第一部分)
```
@tf.function
def f(x):
    print("Python execution: ", x)
    tf.print("Graph execution: ", x)
    return x
```
注意两个print是不同的. 接着我们执行以下测试:
```python
print("##### float32 test #####")
a = tf.constant(1, dtype=tf.float32)
print("first call")
f(a)
a = tf.constant(1.1, dtype=tf.float32)
print("second call")
f(a)

print("##### uint8 test #####")

b = tf.constant(2, dtype=tf.uint8)
print("first call")
f(b)
b = tf.constant(3, dtype=tf.uint8)
print("second call")
f(b)
```
我们得到以下结果:
```
##### float32 test #####
first call
Python execution:  Tensor("x:0", shape=(), dtype=float32)
Graph execution:  1
second call
Graph execution:  1.1
##### uint8 test #####
first call
Python execution:  Tensor("x:0", shape=(), dtype=uint8)
Graph execution:  2
second call
Graph execution:  3
```
我们可以看到传入不同数据类型的tensor时, 图会重新被构建一次(注意: `print`并没有被转成`tf.print`). 我们可以用`tf.autograph.to_code(f.python_function)`来看看生成的图表示代码:
```
def tf__f(x):
  try:
    with ag__.function_scope('f'):
      do_return = False
      retval_ = None
      with ag__.utils.control_dependency_on_returns(ag__.converted_call(print, None, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=ag__.Feature.ALL, internal_convert_user_code=True), ('Python execution: ', x), {})):
        tf_1, x_1 = ag__.utils.alias_tensors(tf, x)
        with ag__.utils.control_dependency_on_returns(ag__.converted_call('print', tf_1, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=ag__.Feature.ALL, internal_convert_user_code=True), ('Graph execution: ', x_1), {})):
          x_2 = ag__.utils.alias_tensors(x_1)
          do_return = True
          retval_ = x_1
          return retval_
  except:
    ag__.rewrite_graph_construction_error(ag_source_map__)
```

其中, 有些python代码只会在构建图的时候被执行:

```
with ag__.utils.control_dependency_on_returns(
        ag__.converted_call(
            print, None, ag__.ConversionOptions(
                recursive=True,
                force_conversion=False,
                optional_features=ag__.Feature.ALL,
                internal_convert_user_code=True),
            ('Python execution: ', x), {})
        ):
```
我们可以看到`ag__.utils.control_dependency_on_returns`会在`ag__.converted_call`返回结果时建立一个`tf.control_dependencies`的依赖, 这样能确保图是按照python代码的顺序执行的.

[converted_call](https://github.com/tensorflow/tensorflow/blob/56c8527fa73f694b76963dbb28a9d011d233086f/tensorflow/python/autograph/impl/api.py#L206)会打包函数的执行. 它的输入包含了可能需要的转换和执行该函数需要的所有输入参数:

- `f`: 函数本身, 这里是`print`.
- `owner`: 函数所在模块, 由于`print`是python内置函数, 因此为`None`. `tf.print`的owner就是`tf_1`, `tf `的别名
- `options`: 转换选项, 也就是`ag__.ConversionOptions`
- `args`, `kwargs`: 函数`f`(`print`)的参数

**那么问题来了:**

为什么在追踪函数执行的时候要确保这些python代码只被执行一次呢?

**猜测:**

我的猜测是没有直接的办法知道该python代码有没有可能会让图效果改变的副作用, 因此在函数第一次执行的时候就直接追踪并加入到图中了. 

如果在第一次执行的时候监测到副作用, 那么图会被更新, 不然的话, 就像在这个例子中, python函数`print`会被`tf.no_op`取代.

由于这只是我的猜测, 我在[这里](https://groups.google.com/a/tensorflow.org/forum/#!topic/developers/SD_ijT4MuPw)提问了, 如果你也对这个问题感兴趣可以留意下.

## 使用python原生类型

我们可以不止用`tf.Tensor`作为输入, AutoGraph能够根据输入类型自动转化成新的图, 接下来我们会测试一下python原生类型和`tf.Tensor`作为输入时的异同.

由于python有三种数值类型: 整数, 浮点数, 和复数, 我们逐一对此进行测试:
```
def printinfo(x):
  print("Type: ", type(x), " value: ", x)

print("##### int test #####")
print("first call")
a = 1
printinfo(a)
f(a)
print("second call")
b = 2
printinfo(b)
f(b)

print("##### float test #####")
print("first call")
a = 1.0
printinfo(a)
f(a)
print("second call")
b = 2.0
printinfo(b)
f(b)

print("##### complex test #####")
print("first call")
a = complex(1.0, 2.0)
printinfo(a)
f(a)
print("second call")
b = complex(2.0, 1.0)
printinfo(b)
f(b)
```
输出有点不太符合预期了:
```
##### int test #####
first call
Type:  <class 'int'>  value:  1
Python execution:  1
Graph execution:  1

second call
Type:  <class 'int'>  value:  2
Python execution:  2
Graph execution:  2

##### float test #####
first call
Type:  <class 'float'>  value:  1.0
Graph execution:  1
second call
Type:  <class 'float'>  value:  2.0
Graph execution:  2

##### complex test #####
first call
Type:  <class 'complex'>  value:  (1+2j)
Python execution:  (1+2j)
Graph execution:  (1+2j)
second call
Type:  <class 'complex'>  value:  (2+1j)
Python execution:  (2+1j)
Graph execution:  (2+1j)
```
这意味着对于每一个不同的数值, 都有一个独立的图! 就是说:

- `f(1)`构造了图, `f(1.0)`重复使用了这个图
- `f(2)`构造了图, `f(2.0)`重复使用了这个图
- `f(1+2j)`和`f(2+1j)`都分别构造了图

这就很奇怪了.

我们可以通过调用函数`f(1.0)`看返回值的类型来看看其是否调用了输入整数的图:
```
ret = f(1.0)
if tf.float32 == ret.dtype:
    print("f(1.0) returns float")
else:
    print("f(1.0) return ", ret)
```
结果:
```
Graph execution:  1
f(1.0) return  tf.Tensor(1, shape=(), dtype=int32)
```
因此, 在输入参数是python原生类型的时候, 与ID相关联的是参数的**值**(1.0==1)而不是类型.

**警告**: 由于每次不同的python值都会生成一个图, 因此对于每个可能的值, 都会执行一次python代码和图构造, 从而极大地降低效率. 

(官方文档: Therefore, python numerical inputs should be restricted to arguments that will have few distinct values, such as hyperparameters like the number of layers in a neural network. This allows TensorFlow to optimize each variant of the neural network.)

## 效率测量

我们用以下代码来验证:

```
@tf.function
def g(x):
  return x

start = time.time()
for i in tf.range(1000):
  g(i)
end = time.time()

print("tf.Tensor time elapsed: ", (end-start))

start = time.time()
for i in range(1000):
  g(i)
end = time.time()

print("Native type time elapsed: ", (end-start))
```

按照上面的理论, 第一个循环只会执行一次python代码和构建一个图, 而第二次循环则会执行1000次python代码和构建1000个图.

结果是符合预期的:

```
tf.Tensor time elapsed:  0.41594886779785156
Native type time elapsed:  5.189513444900513
```
结论: 在每个地方都用`tf.Tensor`. 

(译者注: 这并不准确, 我们应该理解成: 用python原生类型来控制构造图, 用`tf.Tensor`做一切实际运算. 比如我们应该用python的整数来控制隐层数量, 用`tf.Tensor`来传入训练数据, 而不应该用  `tf.Tensor`来控制隐层数量, numpy array来传入训练数据)

## tf.function真的是在用AutoGraph吗?

这个部分暂且省略, 因为在这个[帖子](https://groups.google.com/a/tensorflow.org/forum/#!topic/developers/SD_ijT4MuPw)里面的开发者说, 他们之后会让`tf.function`和AutoGraph的行为一致.

## 结论

在本文中我们分析了`tf.function`在启用AutoGraph的情况下的行为:

- 在用`tf.Tensor`的时候, 所有东西都在预期之中
- 如果是在用python原生类型的时候, 每个不同的值都会建立一个不同的图, 相同的值的话会被重复
- python的函数代码只会在构建图的时候被执行一次

在第三部分的文章中, 我们会探寻比`print`更加复杂的函数, 来看看各种操作重载等python操作的行为.

原文链接: https://github.com/JayYip/deep-learning-nlp-notes/blob/master/tf.function%E5%92%8CAutograph%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97-Part%202.md

声明: 本文翻译自[Paolo Galeone的博客](https://links.jianshu.com/go?to=https%3A%2F%2Fpgaleone.eu%2Ftensorflow%2Ftf.function%2F2019%2F03%2F21%2Fdissecting-tf-function-part-1%2F), 已取得作者的同意, 如需转载本文请联系本人

Disclaimer: This is a translation of the article [Analyzing tf.function to discover AutoGraph strengths and subtleties](https://pgaleone.eu/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/) by Paolo Galeone.
