

## Vali.Text Module

This module performs tests on truthful_qa. 


To serve the module

```bash
c vali.text serve tag=10
```
```python
c.serve('vali.text', tag=10)
```

To register the module

```bash
c vali.text register tag=10
```

```python
c.register('vali.text', tag=10)
```


## Staking from another module

The key is based on the name of the module you serve. If you serve a new module, you will need to stake it if you want the votes to count. To do that you may need to stake it from another module.


```python
c.stake('key_with_stake', module_key='vali.text::2', amount=1000)
```

If you do not have any stake then unstake it
```python
c.unstake('key_with_stake', amount=100)
c.stake('key_with_stake', module_key='vali.text::2', amount=100)
```


