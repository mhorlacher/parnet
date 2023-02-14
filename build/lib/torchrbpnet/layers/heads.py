# %%
import torch
import torch.nn as nn

# %%
class IndexEmbeddingOutputHead(nn.Module):
    def __init__(self, n_tasks, dims):
        super(IndexEmbeddingOutputHead, self).__init__()

        # protein/experiment embedding of shape (p, d)
        self.embedding = torch.nn.Embedding(n_tasks, dims)
    
    def forward(self, bottleneck, **kwargs):
        # bottleneck of shape (batch, d, n) --> (batch, n, d)
        bottleneck = torch.transpose(bottleneck, -1, -2)
        
        # embedding of (batch, p, d) --> (batch, d, p)
        embedding = torch.transpose(self.embedding.weight, 0, 1)

        logits = torch.matmul(bottleneck, embedding) # torch.transpose(self.embedding.weight, 0, 1)  
        return logits