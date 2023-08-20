



## Step 1: Register a Module
 To register one on the commune network, you need to specify the module path.
The following example involves deploying model.openai



c register model.openai tag=sup api_key=sk-...


Now your module will serve on a port within your **port_range** (c port_range). 

## Step 2: Use the Module
```python
import commune a c
# get the module
module = c.connect('model.openai', network='subspace')
output = module.forward('sup dawg')
```
