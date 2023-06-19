# 1.介绍

为了最大程度在规模扩张时提高模型性能，根据Chinchilla scaling law缩放定律（2022年由Deepmind提出），模型大小和数据集大小应该同时增加。而早期的KM scaling law，则认为应该首先关注模型大小，减少数据集的增大。

scaling law：以模型大小、数据规模和总计算量作为决定模型性能的关键要素，计算资源有限时三种要素的分配方式使得模型预期性能最大化。

根据Deepmind团队的描述，理想情况下训练一个175B模型需要至少35000亿token的文本。这种规模是目前最大预训练数据集的2倍，最大公开英文数据集的10倍。

为了满足这种需求，大部分数据都是通过网络爬取的方式获得，这种数据通常被认为质量低于人工审核的数据。该文章关注了数据质量对模型训练效果的影响，做出了以下贡献：

- 制作了一个**高质量的包含5万亿token的英文数据集**——**REFINEDWEB**
- 证明了**仅使用网络数据就足够使模型性能（zero-shot能力）超过用人工审核数据集训练的模型**
- **开源了RefinedWeb中的6000亿token和基于该数据集训练的1/7B模型**



## OpenLLM排名

2023.6.19时的[OpenLLMLeaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)截图，可以看到falcon的排名很高，超越了LLaMA的同时，还有使用了更小的数据集

![image-20230619115730200](./The RefinedWeb Dataset for Falcon LLM/image-20230619115730200.png)

简而言之，在目前所有开源模型中，Falcon有最好的表现和相对较低GPU内存占用。

**对于常规的任务部署来说，Falcon-40B-Instruct是最优选择。**



# 2. 数据集构建

文中提出了MDR（MacroData Refinement），这是一个用于大规模过滤和消除CommonCrawl中web数据重复的管道。

**MDR设计准则**

- **Scale first.** 规模优先。为了能够训练40-200B的模型，优先满足数据集规模达到万亿级别的需要。数据来源于CommonCrawl来避免领域知识单一 
- **Strict deduplication.** 严格去重。结合精准和模糊去重，实现一个严格的去重流水线
- **Neutral filtering.** 中立过滤。为了避免在模型引入歧视信息，文中没有使用机器学习来过滤数据，而是使用规则和启发式方法，且仅用URL过滤成人内容

