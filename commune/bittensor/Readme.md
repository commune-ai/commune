## Bittensor


Bittensor works on a coldkey and hotkey abstraction. We have a wallet abstraction called {coldkey}.{hotkey}
```python

wallet = f"{coldkey}.{hotkey}"

module = commune.get_commune('bittensor')
moduel(wallet=wallet).register(dev_ids = [0,1,2])

```

1. Enter commune
    
    ```bash
    make enter; 
    #or 
    source env/bin/activate; 
    ```

2. Run registration
    ```
    python3 

    ```