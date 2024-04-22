
# **GPTs**

## Setting a OpenAI Key

To set a [OpenAI](https://platform.openai.com/api-keys) key, use the following command:

```bash
c model.gpts add_api_key YOUR_API_KEY
```

Once this is done, you won't need to specify the key every time you start your module. The key will be saved in the module's folder at `.commune/model.gpts/api_keys.json`.

To remove an API key, use:

```bash
c model.gpts rm_api_key YOUR_API_KEY
```

To list all API keys:

```bash
c model.gpts api_keys
```

## Serving the Model

To serve the model with a specific tag and API key, use the following command:

```bash
c model.gpts serve tag=10 api_key=....
```

## Registering the Model

To register the model with a specific tag and name, use:

```bash
c model.gpts register api_key=.... tag=10
```

This will set the name to `model.gpts::10`.

## Testing the Model

To test the model, use:

### math-gpt

```bash
c model.gpts call type="math" content="i need to solve the equation '3x + 11 = 14'. can you help me?"
```

**Response: (You can directly open it in the browser by pasting this data into the URL box)**
```
data=[
    ThreadMessage(
        id='msg_DKupInqks8C0hXyeig8aHTCT',
        assistant_id='asst_wjQVBPBRSvkYgn0WFvxKUpQ5',
        content=[MessageContentText(text=Text(annotations=[], value='The solution to the equation `3x + 11 = 14` is `x = 1`.'), type='text')],
        created_at=1700078252,
        file_ids=[],
        metadata={},
        object='thread.message',
        role='assistant',
        run_id='run_yeokk0kSDR1mvPTxE7T5WD68',
        thread_id='thread_VmPVRIZjDAF7d7M4e0Ltb6cf'
    ),
    ThreadMessage(
        id='msg_CEjAuuwhC7y5wvKHkzm5Jrj7',
        assistant_id='asst_wjQVBPBRSvkYgn0WFvxKUpQ5',
        content=[
            MessageContentText(
                text=Text(
                    annotations=[],
                    value="Sure, John Doe. To solve for `x` in the equation `3x + 11 = 14`, you will need to isolate the variable `x` on one side of the 
equation. Here's how to do it step by step:\n\n1. Subtract 11 from both sides of the equation to get rid of the constant term on the left side. This will 
give you `3x = 14 - 11`.\n2. Calculate `14 - 11` to find out what `3x` is equal to.\n3. Divide both sides of the equation by 3 to solve for `x`.\n\nLet's 
calculate that."
                ),
                type='text'
            )
        ],
        created_at=1700078243,
        file_ids=[],
        metadata={},
        object='thread.message',
        role='assistant',
        run_id='run_yeokk0kSDR1mvPTxE7T5WD68',
        thread_id='thread_VmPVRIZjDAF7d7M4e0Ltb6cf'
    ),
    ThreadMessage(
        id='msg_HohAg9kKr7lpG0UY7PjnSmOB',
        assistant_id=None,
        content=[MessageContentText(text=Text(annotations=[], value="i need to solve the equation '3x + 11 = 14'. can you help me?"), type='text')],
        created_at=1700078242,
        file_ids=[],
        metadata={},
        object='thread.message',
        role='user',
        run_id=None,
        thread_id='thread_VmPVRIZjDAF7d7M4e0Ltb6cf'
    )
]
```
