

This repo is for managing commune on multiple hosts through Ansible.


Step 1:

Install Ansible

- Regular Enviornment

    ```
    pip install ansible

    ```
- Virtual Env

    - Enter Virtual Environment and install ansbile

        ```
        python3 -m venv env
        source env/bin/activate
        pip install ansible

        ```



- Install Commune

    ```
    make sync

    ```


- Generate Keys

    ```
    make gen_keys

    ```


- start registratin
    ```
    make register
    ```
