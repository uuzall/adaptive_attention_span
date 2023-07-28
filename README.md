# Self-Written Code for the Paper "Adaptive Attention Span in Transformers" 
Paper link: https://arxiv.org/pdf/1905.07799.pdf

The paper explores using Attention spans that can adapt depending on the input to a transformer. In this repository, I used the Friends dataset and implemented the paper.

## What does the paper contribute? 
The basic idea of the paper is that adding learnable parameters in the self-attention mechanism will cause the model to learn the ideal length of the context depending on the input given to the model.

For this experiment, I used a GPT model with 8 heads and 8 layers. The sequence length used in 4096 (The paper uses span lengths of up to 8192 but I did not have the memory to do that). The results I got from my experiments were similar to that the authors achieved. The figure below shows the adaptive attention masks for each of the head and layer of the transformer (Horizontally across: Heads from 0-7, Vertically down: Layers 0-7).

![adaptive_attention_span](https://github.com/uuzall/adaptive_attention_span/blob/main/photos/attention_span_no_log.png)

As you can see, the first layer of heads has more variance in weights. After the first layer almost all heads share a negative linear relationship with respect to the sequence length. There are some notable differences between the layers and heads (like some heads have no attention given to the nearest characters, whereas other heads give a lot of attention to the nearest characters). This seems like a division of labour to focus on different features of the input by each head. 

In the end, I the implementation was a success because I got to see how the attention spans changed while different inputs were given to the model.

## This Repo
* The "data.py" and "gpt.py" includes the dataloaders, the neural network models required by the repo. 
* The "main.ipynb" contains the training of the model using adaptive span. 
* The "making_span.ipynb" contains the original code written by Facebook on their official implementation of the algorithm. I commented all my notes in that folder, and after understanding how everything worked, rewrote the algorithm in my own code in "gpt.py". 