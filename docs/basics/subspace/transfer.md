to transfer a function

```python
c.transfer(dest='5Gc8pi8h9Tg8XE31U8zxJU5dtzqkQJjbG7ZzjAoBjKAWXUq8', amount=1000)
```
```bash
c transfer 5Gc8pi8h9Tg8XE31U8zxJU5dtzqkQJjbG7ZzjAoBjKAWXUq8 1000
```

Multiple transfers can be done at once by providing a list of destinations and amounts:
We recommend to only do this in python as it is easier to manage the list of destinations and amounts.

```python
c.transfer_multiple(dest=['5Gc8pi8h9Tg8XE31U8zxJU5dtzqkQJjbG7ZzjAoBjKAWXUq8', '5Gc8pi8h9Tg8XE31U8zxJU5dtzqkQJjbG7ZzjAoBjKAWXUq8'], amount=[1000, 2000])
```
