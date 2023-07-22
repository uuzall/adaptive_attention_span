# Self-Written Code for the Paper "Adaptive Attention Span in Transformers" 

The paper explores using Attention spans that can adapt depending on the input to a transformer. In this repository, I used the Friends dataset and implemented the paper.
* The "data.py" and "gpt.py" includes the dataloaders, the neural network models required by the repo. 
* The "main.ipynb" contains the training of the model using adaptive span. 
* The "making_span.ipynb" contains the original code written by Facebook on their official implementation of the algorithm. I commented all my notes in that folder, and after understanding how everything worked, rewrote the algorithm in my own code in "gpt.py". 

In the end, I the implementation was a success because I got to see how the attention spans changed while different inputs were given to the model. Although, I think that using the Friends dataset was a mistake because in this particular dataset, attention spans greater than ~100+ were not required that much. 