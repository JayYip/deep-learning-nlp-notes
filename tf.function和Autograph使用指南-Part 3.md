# tf.function和Autograph使用指南-Part 3

在第一部分的文章中我们看到了怎么将TF 1.X的代码改写到eager的代码, 再从eager的代码改写到图表示的代码. 并且我们发现了我们不能在转化成图表示的函数中创建状态.

在第二部分的文章中我们探讨了python原生类型数据和`tf.Tensor`作为被`tf.function`装饰的函数的输入时的不同, 并且发现了如果两者使用不当的话会造成运行速度极大地下降和可能不能得到我们想要的结果.

在这最后的文章中, 我们会分析一下用`tf.function`来装饰更加复杂的python函数的情况, 来看看我们是否需要在写函数的时候分析代码的转化过程.

## AutoGraph的功能和限制

我们在TF的官方repo的`python/autograph`里面可以找到[这个文档](https://github.com/tensorflow/tensorflow/blob/560e2575ecad30bedff5b192f33f6d06b19ccaeb/tensorflow/python/autograph/LIMITATIONS.md). 在这个文档中我们可以看到AutoGraph能做什么, 和有什么限制. 在里面的表中我们可以具体地知道什么python操作会被转换, 或者将来会被转换, 什么操作则不支持. 在这一节里面, 我们会分析一下函数是否按照我们预期那样被转换和我们是否需要在写函数的时候想清楚转换的过程.

### if ... else

我们分析以下的简单函数:

```python
@tf.function
def if_else(a, b):
  if a > b:
    tf.print("a > b", a, b)
  else:
    tf.print("a <= b", a, b)
```

我们先来看看转换出来的代码:

```python
print(tf.autograph.to_code(if_else.python_function))
```

得到的结果:

```python
def tf__if_else(a, b):
    cond = a > b

    def get_state():
        return ()

    def set_state(_):
        pass

    def if_true():
        ag__.converted_call(
            "print",
            tf,
            ag__.ConversionOptions(
                recursive=True,
                force_conversion=False,
                optional_features=(),
                internal_convert_user_code=True,
            ),
            ("a > b", a, b),
            None,
        )
        return ag__.match_staging_level(1, cond)

    def if_false():
        ag__.converted_call(
            "print",
            tf,
            ag__.ConversionOptions(
                recursive=True,
                force_conversion=False,
                optional_features=(),
                internal_convert_user_code=True,
            ),
            ("a <= b", a, b),
            None,
        )
        return ag__.match_staging_level(1, cond)

    ag__.if_stmt(cond, if_true, if_false, get_state, set_state)
```

我们可以看到就像我们写`tf.cond`一样, `ag__.if_stmt`接收`cond`, `true_fn`, `false_fn`, 如果`cond`为`True`则执行`true_fn`, 否则则执行`false_fn`. 先忽略掉`get_state`和`set_state`吧.

现在我们可以执行图表示代码了:

```python
x = tf.constant(1)
if_else(x, x)
```
结果如我们所料: `a <= b 1 1`

### if ... elif ... else

稍微修改一下函数:

```python
@tf.function
def if_elif(a, b):
  if a > b:
    tf.print("a > b", a, b)
  elif a == b:
    tf.print("a == b", a, b)
  else:
    tf.print("a < b", a, b)
```

先来看看图表示的代码:

```python
def tf__if_elif(a, b):
    cond_1 = a > b

    def if_true_1():
        # tf.print("a > b", a, b)
        return ag__.match_staging_level(1, cond_1)

    def if_false_1():
        cond = a == b

        def if_true():
            # tf.print(a == b, a, b)
            return ag__.match_staging_level(1, cond)

        def if_false():
            # tf.print(a < b, a,b)
            return ag__.match_staging_level(1, cond)

        ag__.if_stmt(cond, if_true, if_false, get_state, set_state)
        return ag__.match_staging_level(1, cond_1)

    ag__.if_stmt(cond_1, if_true_1, if_false_1, get_state_1, set_state_1)
```

我们可以看到其实就是嵌套的`tf.cond`, 没啥好说的.

我们开看看执行结果吧:

```python
x = tf.constant(1)
if_elif(x, x)
```

我们预期结果是`a == b 1 1`, 但是如果你会发现, 实际的结果是`a < b 1 1`! 什么鬼!

我们来看看eager模式的代码执行情况:

```python
x = tf.constant(1)
if_elif.python_function(x, x)
```

结果是正确的: `a == b 1 1`. 但是! 如果你执行下面代码:

```python
x, y = tf.constant(1), tf.constant(1)
if_elif.python_function(x, y)
```

你会看到`a < b 1 1`! 惊喜不惊喜?!!

**第一个教训: 并不是所有操作都一样地转换**

这个奇怪的现象其实是因为`tf.Tensor`的`__eq__`方法的不一样的改写. 详情可以参考[这个StackOverflow的回答](https://stackoverflow.com/questions/46785041/why-does-tensorflow-not-override-eq)和[这个Github issue](https://github.com/tensorflow/tensorflow/issues/9359). 简单的来说, 就是用`==`来比较`tf.Tensor`的时候, 检查的并不是`tf.Tensor`的**值**, 而是比较两个`tf.Tensor`的**hash**. 

私货: 为了更清楚地看看这个现象, 我们可以执行下面的代码:

```python
@tf.function
def if_elif(a, b):
  print(a.__hash__())
  print(b.__hash__())
  if a > b:
    tf.print("a > b", a, b)
  elif a == b:
    tf.print("a == b", a, b)
  else:
    tf.print("a < b", a, b)

if_elif(x, x)

if_elif.python_function(x, x)
```

我们会看到结果(hash值可能会不一样):
```python
4975557040
5209512760
a < b 1 1

5196691160
5196691160
a == b 1 1
```

**第二个教训: AutoGraph是如何(不)转化算子的**

在上面的例子中, 我们事实上假设了AutoGraph不止会转换`if`, `elif`, `else`等命令, 还会转换python内置的计算, 如`__eq__`, `__gt__`, `__lt__`等. 但是从上面图表示的代码可以看到, 这些都没有像函数那用用`ag__.converted_call`转换. 从上面例子我们也可以猜到, 实际上所有条件判断都是`False`的:

```python
@tf.function
def if_elif(a, b):
  if a > b:
    tf.print("a > b", a, b)
  elif a == b:
    tf.print("a == b", a, b)
  elif a < b:
    tf.print("a < b", a, b)
  else:
    tf.print("wat")
x = tf.constant(1)
if_elif(x,x)
```

输出: `wat`

**第三个教训: 怎么写函数**

为了让eager和图表示代码表现一直, 我们需要知道:

1. 算子的含义很重要, 有些算子被改写到了和原生python有不一样的意义
2. AutoGrah能够自动转换`if`, `elif`等语句, 但是我们在写的时候仍然需要加倍小心

因此, 在实际应用中, 我们最好**处处都使用显式的TF操作**. 我们可以用最安全的方法改写上述例子:
```python
@tf.function
def if_elif(a, b):
  if tf.math.greater(a, b):
    tf.print("a > b", a, b)
  elif tf.math.equal(a, b):
    tf.print("a == b", a, b)
  elif tf.math.less(a, b):
    tf.print("a < b", a, b)
  else:
    tf.print("wat")
```

得到的图表示代码(为了方便展示, 稍微清理了一下):

```python
def tf__if_elif(a, b):
    cond_2 = ag__.converted_call("greater", ...)  # a > b

    def if_true_2():
        ag__.converted_call("print", ...)  # tf.print a > b
        return ag__.match_staging_level(1, cond_2)

    def if_false_2():
        cond_1 = ag__.converted_call("equal", ...)  # tf.math.equal

        def if_true_1():
            ag__.converted_call("print", ...)  # tf.print a == b
            return ag__.match_staging_level(1, cond_1)

        def if_false_1():
            cond = ag__.converted_call("less", ...)  # a < b

            def if_true():
                ag__.converted_call("print", ...)  # tf.print a < b
                return ag__.match_staging_level(1, cond)

            def if_false():
                ag__.converted_call("print", ...)  # tf.print wat
                return ag__.match_staging_level(1, cond)

            ag__.if_stmt(cond, if_true, if_false, get_state, set_state)
            return ag__.match_staging_level(1, cond_1)

        ag__.if_stmt(cond_1, if_true_1, if_false_1, get_state_1, set_state_1)
        return ag__.match_staging_level(1, cond_2)

    ag__.if_stmt(cond_2, if_true_2, if_false_2, get_state_2, set_state_2)
```

可以看到, 现在每一部分都用`ag__.converted_call`转换了.

### for ... in range

根据之前的三个教训, 我们就可以顺利地用`tf.function`转换`for`循环了, 我们用一个简单的从1加到`x-1`的函数作为例子, 需要注意两点:

1. 不要在函数里面创建状态
2. 用`tf.range`, 而不是`range`

**译者注:** 在被`tf.function`装饰的函数内使用`tf.range`和`range`的时候, `tf.range`是动态展开的, `range`是静态展开的, 具体可以看[这里](https://www.tensorflow.org/alpha/tutorials/eager/tf_function#autograph_and_loops). 动态展开是指, 我们可以循环到某个地方中断循环(`break`或者`return`), 而静态展开则是把每个循环的图都事先构造好了. 如果我们改写一下官方文档里面的例子, 你会发现:
```python
@tf.function
def buggy_py_for_tf_break(upto):
  x = 0
  for i in range(upto):
    print(i)
    if tf.equal(i, 10):
      break
    x += i
  return x

@tf.function
def tf_for_tf_break(upto):
  x = 0
  for i in tf.range(upto):
    if tf.equal(i, 10):
      break
    x += i
  return x

print(buggy_py_for_tf_break(tf.constant(100))) # it's ok!
print(tf_for_tf_break(tf.constant(100))) # it's ok!
print(tf_for_tf_break(100)) # it's ok!
print(buggy_py_for_tf_break(100)) # ERROR!!
```

为什么呢? 我不告诉你.

(译者注完)

记住上面两点, 可以写成:
```python
x = tf.Variable(1)
@tf.function
def test_for(upto):
  for i in tf.range(upto):
    x.assign_add(i)

x.assign(tf.constant(0))
test_for(tf.constant(5))
print("x value: ", x.numpy())
```

就像我们想的那样, `x`的值为10.

**思考问题:** 如果我们用`x += i`替换掉`x.assign_add(i)`会怎样呢? 

## 结论

总结一下这三篇文章的要点:

- 如果函数中创建了状态的话, 需要格外小心. 该函数在转换成图表示的时候可能会有问题(Part 1)
- AutoGraph**不会**封装python原生类型, 这可能会导致严重的效率问题(Part 2). 请谨慎使用python原生类型.
- `tf.print`和`print`是不同的, 被转换的函数在第一次被调用时和之后的调用可能会有不同的结果(Part2)
- `tf.Tensor`的运算重载和我们想象中不太一样, 为了保证正常运行, 推荐在被转换函数里面使用`tf.equal`这样的tf的操作而不是`==`这样的python操作.

原文链接: https://github.com/JayYip/deep-learning-nlp-notes/blob/master/tf.function%E5%92%8CAutograph%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97-Part%203.md

声明: 本文翻译自[Paolo Galeone的博客](https://links.jianshu.com/go?to=https%3A%2F%2Fpgaleone.eu%2Ftensorflow%2Ftf.function%2F2019%2F03%2F21%2Fdissecting-tf-function-part-1%2F), 已取得作者的同意, 如需转载本文请联系本人

Disclaimer: This is a translation of the article [Analyzing tf.function to discover AutoGraph strengths and subtleties](https://pgaleone.eu/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/) by Paolo Galeone.