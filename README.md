# README
## Language
- [English](#English)
- [中文](#中文)



# <div id = "English">English <div>
## Introduction

pymatchingtools is a tools for matching methods in common causal inference.

I've used some of the common causal inference packages available, and found that almost  of them just implement the methods, ignoring the balancing checks before matching, and the refutation tests after matching. We can't judge the usability of these matches.

This python package is designed to help you complete the following in a relatively simple way: 
- 1) evaluate the balance of variables before matching; 
- 2) complete data matching; 
- 3) evaluate the robustness of the results by refutation tests.

Due to my heavy work with limited time and energy, I am only able to complete the propensity score matching method. If there is a need for other methods, please leave me a message and I will schedule updates and complete them.

## Installation
Recommend to use pip to install, the installed python version should be limited to 3.7 or above.

```bash
$ pip install pymatchingtools
```

## Example
This is an example of using the Boston house price dataset, which is divided into five steps.
- Data Preparation
- Initialising the Matching class
- Variable balance checking before matching
- Matching
- Rebuttal check after match

For more information you can see example.ipynb

### Data Preparation
We need to get the data first, only DataFrame format is supported.

```python
column_names = [‘CRIM’, ‘ZN’, ‘INDUS’, ‘CHAS’, ‘NOX’, ‘RM’, ‘AGE’, ‘DIS’, ‘RAD’, ‘TAX’, ‘PTRATIO’, ‘B’, ‘LSTAT’, ‘MEDV’]

data = pd.read_csv(‘housing.csv’, header=None, delimiter=r‘\s+’, names=column_names)
```

### Initialising the Matching class
Initialise the Matching class with data prepared

```python
from pymatchingtools.matching import PropensityScoreMatch
matcher = PropensityScoreMatch(data=data)
```

### Variable balance checking before matching
There are two ways to complete this, one is to use a patsy-formatted formula, and the other is to pass in the covariates(x) and indicator variables(y).

The way to use the formula is as follows. You can print out the result of the balance check with  ```summary_print=True```.

```python
formula = ‘CHAS ~ CRIM + ZN + INDUS + NOX + RM + AGE + DIS + RAD’

summary_df = matcher.get_match_info(formula=formula, summary_print=True)
```


The way to use covariates(x) and indicator variables(y)

```python
y = data[[‘CHAS’]] 

x = data[[‘CRIM, ZN, INDUS, NOX, RM, AGE, DIS, RAD’]]

summary_df = matcher.get_match_info(x=x, y=y, summary_print=True)
```


### Matching


Get matches via the ``match`` method, with the restriction ``is_fliter==True`` in case of no-putback sampling.

Support both GLM and LGBM methods to train propensity score models

Only support the Manhattan distance now, and I will be gradually updated more distances.

It only supports the nearest match, so there is no need to restrict it.

```python
matched_data = matcher.match(
    method='min',
    is_fliter=True,
    fit_mathod='glm
)
```

### Rebuttal check after match
Use the ```after_match_check``` method to perform a rebuttal test, currently the following rebuttal tests are supported: 
- 1) add random confusion; 
- 2) placebo test; 
- 3) data subset test.


```python
matcher.after_match_check(
    outcome_var=‘MEDV’,
    frac=0.8,
    match_method=‘min’
)
```

# <div id = "中文">中文 <div>
## 简介

pymatchingtools是一个常见的因果推断中匹配方法的工具箱

我曾经用过现在python里有的常见的因果推断相关的包, 但发现几乎所有的包只是实现了方法,而忽视了推断前的平衡性检查,以及推断后的反驳式检验. 这样的匹配结果,我们无法判断其可用性

这个python包的设计初衷是, 能够用较为简单的方式,帮助大家完成:
- 1)评估匹配前的变量平衡性;
- 2)完成一次Matching方式的推断;
- 3)评估当前Matching方式得到的结果是否具备鲁棒性

由于平时工作繁忙,时间精力有限,目前仅实现了倾向性得分匹配的方法,如果有其他方法需要,请给我留言,我会排期更新和实现
## 安装方法
建议使用pip方式安装, 安装的python版本需要限制在3.7以上

```bash
$ pip install pymatchingtools
```

## 使用示例
这里采用波士顿房价数据集进行说明,整个使用分为5个步骤
- 数据准备
- 初始化Matching类
- 匹配前的变量平衡性检查
- 匹配
- 匹配后的反驳式检验


更多信息可以看example.ipynb

### 数据准备
需要先导入相关的数据,目前仅支持DataFrame格式

```python
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)
```

### 初始化Matching类
将我们准备好的原始数据放入Matching类中进行初始化

```python
from pymatchingtools.matching import PropensityScoreMatch
matcher = PropensityScoreMatch(data=data)
```

### 匹配前的变量平衡性检查
目前支持两种方式,一种是使用patsy格式的公式,另一种是传入相应的协变量和指示变量

使用公式的方法如下, 如果需要打印出相应的检查结果,可以令```summary_print=True```

```python
formula = 'CHAS ~ CRIM + ZN + INDUS + NOX + RM + AGE + DIS + RAD'

summary_df = matcher.get_match_info(formula=formula, summary_print=True)
```


如果是传入相应的协变量和指示变量,则需要
```python
y = data[['CHAS']] 

x = data[['CRIM, ZN, INDUS, NOX, RM, AGE, DIS, RAD']]

summary_df = matcher.get_match_info(x=x, y=y, summary_print=True)
```


### 匹配


通过```match```方法获取匹配结果,如果是无放回抽样,限制```is_fliter==True```

支持GLM和LGBM两种模式去训练倾向性得分模型

距离的实现方式目前仅实现了曼哈顿距离,后续会逐渐更新和补充更多距离

这里method仅实现了最近匹配,无需限制

```python
matched_data = matcher.match(
    method='min',
    is_fliter=True,
    fit_mathod='glm
)
```

### 匹配后的反驳式检验
使用```after_match_check```方法进行反驳式检验, 目前支持的反驳式检验有: 
- 1)添加随机混淆;
- 2)安慰剂检验;
- 3)数据子集检验


```python
matcher.after_match_check(
    outcome_var='MEDV',
    frac=0.8,
    match_method='min'
)
```