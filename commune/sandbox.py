import commune
secret = commune.encrypt('whadup')
print(commune.decrypt(secret))