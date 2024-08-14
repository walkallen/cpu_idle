

# CLIP-Adapter：通过特征适配器实现更好的视觉-语言模型

Peng Gao ${}^{*1}$，Shijie Geng ${}^{*2}$，Renrui Zhang ${}^{*1}$，Teli Ma ${}^{1}$，Rongyao Fang ${}^{3}$，

Yongfeng ${\text{Zhang}}^{2}$，Hongsheng ${\mathrm{{Li}}}^{3}$，Yu ${\text{Qiao}}^{1}$

${}^{1}$ 上海人工智能实验室 ${}^{2}$ 罗格斯大学

${}^{3}$ 香港中文大学

\{gaopeng, zhangrenrui, qiaoyu\}@pjlab.org.cn

sg1309@rutgers.edu, hsli@ee.cuhk.edu.hk

## 摘要

大规模对比视觉-语言预训练在视觉表示学习方面取得了显著进展。与传统通过固定离散标签训练的视觉系统不同，(Radford et al., 2021)引入了一种新范式，直接在开放词汇环境下学习将图像与原始文本对齐。在下游任务中，通过精心选择的文本提示进行零样本预测。为了避免非平凡的提示工程，上下文优化(Zhou et al., 2021)提出通过少量训练样本来学习连续向量作为任务特定的提示。在本文中，我们展示了除了提示调整之外，还有另一种途径可以实现更好的视觉-语言模型。虽然提示调整针对文本输入，但我们提出CLIP-Adapter，通过在视觉或语言分支上使用特征适配器进行微调。具体来说，CLIP-Adapter采用一个额外的瓶颈层来学习新特征，并通过残差风格的特征混合与原始预训练特征相结合。因此，CLIP-Adapter能够在保持简单设计的同时优于上下文优化。在各种视觉分类任务上的实验和广泛的消融研究证明了我们方法的有效性。

## 1 引言

视觉理解任务，如分类(Krizhevsky et al., 2012; He et al., 2016; Howard et al., 2017; Dosovitskiy et al., 2021; Touvron et al., 2021; Gao et al., 2021a; Mao et al., 2021)、目标检测(Ren et al., 2015; Carion et al., 2020; Gao et al., 2021b)和语义分割(Long et al., 2015)，已经基于更好的架构设计和大规模高质量数据集显著改进。不幸的是，为每个视觉任务收集大规模高质量数据集是劳动密集型的，且成本过高。为了解决这个问题，"预训练-微调"范式，即在大型数据集如ImageNet(Krizhevsky et al., 2012)上预训练，然后在各种下游任务上微调，已经在视觉领域广泛采用。然而，这种方法仍然需要在许多下游任务上进行大量标注的微调。最近，对比语言-图像预训练(CLIP)(Radford et al., 2021)被提出，通过利用大规模噪声图像-文本对进行对比学习来解决视觉任务。它在没有任何标注的情况下（即零样本迁移）通过将视觉类别放入适当的手工模板作为提示，在各种视觉分类任务上取得了令人鼓舞的性能。

尽管基于提示的零样本迁移学习显示出有前景的性能，但设计好的提示仍然是一个需要大量时间和领域知识的工程问题。为了解决这个问题，上下文优化(CoOp)(Zhou et al., 2021)进一步提出通过少量样本学习连续软提示来替代精心选择的硬提示。CoOp在少样本分类上相较于零样本CLIP和线性探针CLIP设置带来了显著改进，展示了在大规模预训练视觉-语言模型上进行提示调整的潜力。

在本文中，我们提出了一种不同的方法，通过特征适配器而不是提示调整来更好地适应视觉-语言模型。与CoOp进行软提示优化不同，我们简单地在轻量级的额外特征适配器上进行微调。由于CLIP的过度参数化和训练样本不足，朴素的微调会导致在特定数据集上过拟合，并且由于所有CLIP层的正向和反向传播，训练过程会非常缓慢。受参数高效迁移学习中适配器模块(Houlsby et al., 2019)的启发，我们提出CLIP-Adapter，它只微调少量额外权重而不是优化所有CLIP参数。CLIP-Adapter采用轻量级瓶颈架构，通过减少参数数量来防止少样本学习中的潜在过拟合问题。同时，CLIP-Adapter与Houlsby et al. (2019)在两个重要方面不同：CLIP-Adapter只在视觉或语言骨干的最后一层添加两个额外的线性层。相比之下，原始适配器模块插入到语言骨干的所有层中；此外，CLIP-Adapter通过残差连接混合原始零样本视觉或语言嵌入与相应的微调特征。通过这种"残差风格混合"，CLIP-Adapter可以同时利用原始CLIP中存储的知识和从少样本训练样本中新鲜学习的知识。总的来说，我们的贡献可以总结如下：

- 我们提出CLIP-Adapter，通过微调进行残差风格特征混合，实现高效的少样本迁移学习。

- 与CoOp相比，CLIP-Adapter在少样本分类性能上表现更好，同时设计更简单，表明CLIP-Adapter是提示调整的有前景的替代方案。

- 我们在十一个分类数据集上对CLIP-Adapter进行了广泛的消融研究，以分析其特性。代码将在https://github.com/gaopengcuhk/CLIP-Adapter发布。

## 2 相关工作

## 2.1 模型微调

深度神经网络对数据需求量大。然而，收集和标注大量高质量数据成本高昂，甚至对某些特殊领域是不可能的。"预训练-微调范式"为不同的计算机视觉(Krizhevsky et al., 2012; Simonyan and Zisserman, 2015; He et al., 2016)和自然语言处理(Kenton and Toutanova, 2019; Dong et al., 2019; Conneau et al., 2020)任务提供了一个很好的解决方案，并已被广泛采用多年。为了在下游任务上进行数据高效的微调，适配器模块(Houlsby et al., 2019)被提出冻结骨干权重并在每个Transformer层插入可学习的线性层。与适配器模块不同，提出的CLIP-Adapter在由CLIP生成的特征嵌入或分类器权重上应用了一个简单的残差变换层。得益于残差连接和瓶颈线性层，CLIP-Adapter可以在少样本学习设置下提高CLIP的性能，并实现比最近提出的CoOp更优越的性能。为了缓解分布偏移下的性能差距，WiSE-FT(Wortsman et al., 2021)提出了一种后集成方法来提高CLIP的分布外鲁棒性。虽然WiSE-FT在微调期间冻结了图像分支的权重，但我们的CLIP-Adapter可以应用于图像和文本分支，并通过可学习的门控比率动态平衡和混合来自原始特征和CLIP-Adapter输出的知识。

## 2.2 提示设计

提示设计(Liu et al., 2021a)因GPT系列(Radford et al., 2019; Brown et al., 2020)的成功而普及。GPT-3展示了在大规模数据集上训练的巨大自回归语言模型可以在零样本或少样本风格下执行任何NLP任务，无需微调基础架构。遵循全新的"预训练，提示，预测"范式，最近提出了各种提示设计方法。其中一种类型通过挖掘或生成适当的离散提示(Jiang et al., 2020; Shin et al., 2020; Gao et al., 2021c)专注于提示工程。相比之下，连续提示规避了预训练语言模型的限制，并被Li和Liang (2021); Liu et al. (2021b); Lester et al. (2021); Gu et al. (2021)应用于NLP任务。受GPT-3的启发，CLIP在4亿图像-文本对上训练了一个大型对比学习模型，并展示了基于提示的零样本视觉分类的潜力。以CLIP为骨干，CoOp(Zhou et al., 2021)和CPT(Yao et al., 2021)进一步展示了优化连续提示可以在视觉任务上大幅超越手工设计的离散提示。在本文中，我们展示了提示调整不是实现更好视觉-语言模型的唯一途径。通过微调少量参数，也可以在视觉任务上实现可比甚至更好的性能，且设计更简单。

## 2.3 视觉-语言模型

探索视觉和语言之间的交互是人工智能的核心研究课题。以前，基于注意力的方法如自下而上自上而下的注意力(Anderson et al., 2018)，BAN(Kim et al., 2018)，Intra-Inter(Gao et al., 2019)和MCAN(Yu et al., 2019)主导了视觉-语言任务。受BERT(Kenton and Toutanova, 2019)成功的启发，ViLBERT(Lu et al., 2019)，LXMERT(Tan and Bansal, 2019)，UNITER(Chen et al., 2020)和Oscar(Li et al., 2020)进一步推动了多模态推理的边界。最近，CLIP(Radford et al., 2021)和ALIGN(Jia et al., 2021)展示了视觉-语言对比表示学习的强大能力。它们在广泛的视觉任务上取得了惊人的结果，无需任何微调。为了进一步缩小CLIP与监督训练之间的差距，CoOp提出了一种连续提示优化方法，以提高视觉分类任务的性能。虽然CoOp从提示设计的角度改进了视觉-语言模型，但我们的CLIP-Adapter探索了通过轻量级特征适配器的简单微调。

## 3 我们的方法

在本节中，我们介绍提出的CLIP-Adapter。在3.1节中，我们从分类器权重生成的角度回顾了CLIP和CoOp。在3.2节中，我们详细阐述了提出的CLIP-Adapter的细节。在3.3节中，我们提供了CLIP-Adapter的几种变体。

## 3.1 少样本学习的分类器权重生成

让我们首先回顾使用深度神经网络进行图像分类的基本框架：给定一张图像$\mathbf{I} \in {\mathbb{R}}^{H \times W \times 3}$，其中$H$和$W$分别表示图像的高度和宽度，一个由基本组件（如CNN，Transformer(Vaswani et al., 2017)或两者的混合）组成的神经网络骨干将$\mathbf{I}$转换为特征流形$f \in {\mathbb{R}}^{D}$，其中$D$表示特征维度。为了进行分类，图像特征向量$f$然后与分类器权重矩阵$\mathbf{W} \in {\mathbb{R}}^{D \times K}$相乘，其中$K$表示分类的类别数。经过矩阵乘法后，我们可以得到一个$K$维的logit。一个Softmax函数用于将logit转换为$K$个类别的概率向量$p \in {\mathbb{R}}^{K}$。整个过程可以写成以下方程：

$$
f = \operatorname{Backbone}\left( \mathbf{I}\right) ,{p}_{i} = \frac{\exp \left( {{\mathbf{W}}_{i}^{T}f}\right) /\tau }{\mathop{\sum }\limits_{{j = 1}}^{N}\exp \left( {{\mathbf{W}}_{j}^{T}f}\right) /\tau }, \tag{1}
$$

其中$\tau$表示Softmax的温度，${\mathbf{W}}_{i}$表示类别$i$的原型权重向量，${p}_{i}$表示类别$i$的概率。

与监督训练不同，在本文中，我们对少样本示例的图像分类感兴趣。从头开始用少量样本训练骨干和分类器容易过拟合某些数据集，并且在测试分割上可能会遭受严重的性能下降。通常，少样本学习的代表性范式是首先在大规模数据集上预训练骨干，然后通过直接进行零样本预测或进一步在少样本示例上微调，将学习到的知识转移到下游任务。

CLIP遵循零样本迁移风格——它首先通过在大规模噪声图像-文本对上进行对比学习预训练视觉骨干和文本编码器，然后在预训练后，CLIP直接进行图像分类，无需任何微调。给定一个包含$K$个类别及其自然语言名称$\left\{ {{C}_{1},\ldots ,{C}_{k}}\right\}$的下游图像分类数据集，CLIP构造将每个类别名称${C}_{i}$放入预定义的硬提示模板$H$中。然后语言特征提取器将生成的提示编码为分类器权重${\mathbf{W}}_{i}$。我们将分类器权重生成过程表示如下：

$$
{\mathbf{W}}_{i} = \operatorname{BERT}\left( {\operatorname{Tokenizer}\left( \left\lbrack {H;{C}_{i}}\right\rbrack \right) }\right) . \tag{2}
$$

或者，CoOp采用连续提示而不是手工硬提示。CoOp创建一个随机初始化的可学习软令牌列表$S \in {\mathbb{R}}^{L \times D}$，其中$L$表示软令牌序列的长度。软令牌序列$S$然后与每个类别名称${C}_{i}$连接，从而形成一个提示。我们将整个过程表示为

$$
{\mathbf{W}}_{i} = \operatorname{BERT}\left( \left\lbrack {S;\operatorname{Tokenizer}\left( {C}_{i}\right) }\right\rbrack \right) . \tag{3}
$$

对于CLIP和CoOp，通过生成的分类器权重${\mathbf{W}}_{i}$，其中$i \in \{ 1,\cdots, K\}$，我们可以通过之前提到的方程(1)计算类别$i$的预测概率${p}_{i}$。

## 3.2 CLIP-Adapter

与CoOp的提示调整不同，我们提出了一种替代框架，通过微调额外特征适配器在少样本图像分类上实现更好的视觉-语言模型。我们声称，由于参数数量巨大和训练样本不足，先前广泛采用的"预训练-微调"范式在少样本设置下微调整个CLIP骨干会失败。因此，我们提出CLIP-Adapter，它只在CLIP的语言和图像分支上附加少量额外的可学习瓶颈线性层，同时在少样本微调期间保持原始CLIP骨干冻结。然而，朴素的微调附加层仍可能在少样本示例上过拟合。为了处理过拟合并提高CLIP-Adapter的鲁棒性，我们进一步采用残差连接，动态混合微调知识和原始CLIP骨干的知识。

具体来说，给定输入图像$\mathbf{I}$和一组类别自然语言名称${\left\{ {C}_{i}\right\} }_{i = 1}^{K}$，图像特征$f$和分类器权重$\mathbf{W}$从原始CLIP骨干通过方程(1)和(2)计算。之后，两个可学习的特征适配器，${A}_{v}\left( \cdot \right)$和${A}_{t}\left( \cdot \right)$，每个包含两层线性变换，分别集成到$f$和$\mathbf{W}$的变换中。我们采用残差连接以避免遗忘预训练CLIP编码的原始知识。两个常数值$\alpha$和$\beta$作为"残差比率"，帮助调整保持原始知识的程度以获得更好性能。总之，特征适配器可以写成

$$
{A}_{v}\left( f\right) = \operatorname{ReLU}\left( {{f}^{T}{\mathbf{W}}_{1}^{v}}\right) {\mathbf{W}}_{2}^{v}, \tag{4}
$$

$$
{A}_{t}\left( \mathbf{W}\right) = \operatorname{ReLU}\left( {{\mathbf{W}}^{T}{\mathbf{W}}_{1}^{t}}\right) {\mathbf{W}}_{2}^{t}. \tag{5}
$$

通过微调捕获的新知识通过残差连接与原始特征相加：

$$
{f}^{ \star } = \alpha {A}_{v}{\left( f\right) }^{T} + \left( {1 - \alpha }\right) f \tag{6}
$$

$$
{\mathbf{W}}^{ \star } = \beta {A}_{t}{\left( \mathbf{W}\right) }^{T} + \left( {1 - \beta }\right) \mathbf{W}. \tag{7}
$$

在获得新的图像特征${f}^{ \star }$和分类器权重${\mathbf{W}}^{ \star }$后，我们也采用方程(1)计算类别概率向量$P = {\left\{ {p}_{i}\right\} }_{i = 1}^{K}$，并通过选择具有最高概率的类别$\widehat{i}$来预测图像类别：$\widehat{i} = \arg \mathop{\max }\limits_{i}{p}_{i}$。

在少样本训练期间，${A}_{v}\left( \cdot \right)$和${A}_{t}\left( \cdot \right)$的权重通过交叉熵损失进行优化：

$$
\mathcal{L}\left( \theta \right) = - \frac{1}{N}\mathop{\sum }\limits_{{n = 1}}^{N}\mathop{\sum }\limits_{{i = 1}}^{K}{y}_{i}^{\left( n\right) }\log {\widehat{y}}_{i}^{\left( n\right) }, \tag{8}
$$

其中$N$是训练示例总数；如果$i$等于真实类别标签$\widehat{i}$，则${y}_{i}
