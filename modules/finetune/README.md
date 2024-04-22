FINETUNER

Finetune any HF model using Qlora, BNB, Lora


## 🚀 Quick Start 🚀

```python
import commune as c
c.module('finetune')(model="togethercomputer/LLaMA-2-7B-32K", dataset='glue.cola')
```
or 
```bash
c finetune model="togethercomputer/LLaMA-2-7B-32K" dataset="glue.cola"
```

To see the config 
```bash
c finetune config
```
or 
```python
c.module('finetune').config()
```
```



