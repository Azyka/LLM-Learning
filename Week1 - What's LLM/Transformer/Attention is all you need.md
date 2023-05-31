# Attention is all you need

# Transformer

### **Self-Attention Mechanism**

> 为每个元素分配一个权重，这个权重反映了其他元素对当前元素的重要性。
> 
- 使用权重矩阵对每个元素生成query和key
- 对单个元素，算出该元素的query和其他元素的key的点积，运用softmax函数算出注意力分数
- 对注意力分数进行加权求和

### Multi-Head Self-Attention

> 通过多个“头”进行处理，每个“头”都有自己的权重矩阵。然后，将所有头的输出拼接，通过线性变换产生最终的输出。
> 
- 对于每个头，使用不同的权重矩阵对每个元素生成query和key
- 对单个元素，在每个头下，算出该元素的query和其他元素的key的点积，运用softmax函数算出注意力分数
- 将所有头的输出拼接在一起，对拼接的输出进行线性变换，得到最终的输出。

> 在自然语言处理任务中，一个头可能专注于捕捉句法依赖，另一个头可能专注于捕捉语义依赖。这种机制使模型能够更好地理解和生成复杂的文本。
> 

---

### **Encoder-Decoder Structure**

### Encoder

> Encoder将输入转换为“上下文向量”
> 
- Encoder由六层identical layers组成，每一层都由多头自注意力机制和前馈神经网络组成。

### Decoder

> Decoder接收编码器产生的上下文向量，将其转换为目标输出
> 
- Decoder同样由六层identical layers组成，但每一层都在encoder的layer中针对encoder的输出添加了一个注意力机制。
- 这个额外的注意力机制使得decoder在生成每个输出词的时候，能够关注到输入的所有部分

---

# **Why Self-Attention**

RNN和LSTM都是通过递归的方式处理序列，即当前的隐藏状态是基于前一时间步的隐藏状态和当前的输入来计算。

### Recurrent Neural Network

通过将隐藏状态从一个时间步传递到下一个时间步，以此捕获序列中的信息

### Long Short-Term Memory

引入了一种叫做“门”的机制来控制信息的流动，这使得LSTM解决RNN的梯度消失问题，能够捕获更长距离依赖。

**计算效率**

RNN和LSTM的计算无法并行化。相比之下，Transformer的自注意力机制可以同时处理所有时间步，因此其计算可以高度并行化，提高计算效率。

**捕捉长距离依赖**

Transformer通过自注意力机制可以直接捕获序列中任何两个位置之间的依赖关系，无论距离多远。

**全局上下文理解**

Transformer中的每个元素的表示都由整个序列中的所有元素共同决定，这使得Transformer能更好地理解全局上下文。

**更易于优化**

由于RNN和LSTM的递归性质，它们在优化上可能面临梯度消失和梯度爆炸等问题。

同时RNN和LSTM需要一个一个处理时间步，而Transformer可以并行处理所有时间步，这使得Transformer的计算更小，可以使用标准的反向传播算法进行优化。