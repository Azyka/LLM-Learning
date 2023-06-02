# GPT-4 Technical Report

Created by: Azyka
Created time: May 31, 2023 11:41 AM
Last edited by: Azyka
Last edited time: May 31, 2023 12:26 PM
Tags: gpt

这篇技术报告重点介绍GPT-4的功能、限制和安全特性，而不包含模型的架构信息和训练细节。但重点在于GPT-4实现了在训练前准确预测其性能，这对未来关于模型安全性的研究至关重要

# GPT-4 能力

GPT-4是一个大型多模态模型，接收图像和文本输入生成文本输出。尽管GPT-4在许多真实场景中表现逊于真人，但它在各种专业和学术基准上已经达到人类水平的表现。

## 基本性能

### 考试得分

![Untitled](GPT-4%20Technical%20Report%20519e3cf3cf6649479b2898d909efe525/Untitled.png)

### Benchmark

![Untitled](GPT-4%20Technical%20Report%20519e3cf3cf6649479b2898d909efe525/Untitled%201.png)

### 跨语言能力

将MMLU（包含57个学科的14000个多选题）翻译成不同的语言

![Untitled](GPT-4%20Technical%20Report%20519e3cf3cf6649479b2898d909efe525/Untitled%202.png)

### 视觉Benchmark

![Untitled](GPT-4%20Technical%20Report%20519e3cf3cf6649479b2898d909efe525/Untitled%203.png)

### 语气

GPT-4可以根据用户定义变更行为风格，但这也是绕过道德限制的最快方法。

# GPT-4缺点

长期存在的问题

- 输出结果不完全可靠
- 上下文窗口大小受限
- 无法从经验中学习

尽管仍然存在问题，GPT-4在对抗测试中的得分比GPT-3.5高了40%。GPT-4无法获知2021年9约以后的事件，也不会从经验中学习新内容。它可能会犯简单的逻辑错误，且容易接受用户编造的虚假内容。同样，在部分人类难以完成的任务中，GPT-4也会产生错误，如生成不安全的代码。

有趣的是，GPT-4基础预训练模型是高度校准的（其对答案的预测置信度通常与正确的概率相匹配）。然而，经过RLHF之后，其准确率反而降低了。

# 训练