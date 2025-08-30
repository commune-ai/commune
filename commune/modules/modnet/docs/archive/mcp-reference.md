# Erasmus Development Workflow

## Model Context Protocol (MCP)

MCP is a protocol that allows the model to access tools and resources through a RPC interface. It is implemented as a server that will provide their service to the mcp client through RPC Commands.

There is a `rmcp` MCP library that can be used if we like in `/home/bakobi/repos/mcp/rust-sdk` but there are a lot of sharp edges with their implementation and so far I have preferred to just implement my own RPC client and server for the tools preferring StdIO transports when possible but SSE and WebSockets are also available.

### Tools:

MCP Tools are tools that are provided to the model through the model context protocol(MCP) and are implemented as servers that will provide their service to the mcp client through RPC Commands.


```json
# tool
{
  "name": "string",          // Unique identifier for the tool
  "description": "string",  // Human-readable description
  "inputSchema": {         // JSON Schema for the tool's parameters
    "type": "object",
    "properties": { ... }  // Tool-specific parameters
  },
  "annotations": {        // Optional hints about tool behavior
    "title": "string",      // Human-readable title for the tool
    "readOnlyHint": false,    // If true, the tool does not modify its environment
    "destructiveHint": false, // If true, the tool may perform destructive updates
    "idempotentHint": false,  // If true, repeated calls with same args have no additional effect
    "openWorldHint": false,   // If true, tool interacts with external entities
  }
}
```

```json
# tools/call - Call a tool
{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "tool_name": "<TOOL_NAME>",
        "arguments": {
            ["key": "string"]: "value"
        }
    },
    "id": "<MSG_ID>"
}
```

```json
# tools/list - List available tools
{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "params": {},
    "id": "<MSG_ID>"
}
```

```json
# result - Result of a tool call
{
    "jsonrpc": "2.0",
    "result": {
        ["key": "string"]: "value"
    },
    "id": "<MSG_ID>"
}
```

```json
# error - Error response
{
    "jsonrpc": "2.0",
    "error": {
        "code": "number",
        "message": "string",
        "data": "value"
    },
    "id": "<MSG_ID>"
}
```
```json
# notification - One way message that expects not response
{
    "method": "initialize",
    "params": {
        "protocol_version": "1.0",
        "capabilities": {
            "tools": true,
            "notifications": true,
        }
    }
}
```

### Connection Lifecycle

1. Initialize connection request with protocol version and capabilities
2. Server responds with protocol version and capabilities
3. Client sends a initilialized notification.
4. Normal communication exchanging requests and responses.
5. Client sends a closed notification.
6. Server closes connection.

### Resources

Resources are files and data provided to the model through a MCP server like databases or the project filesystem.

Resources are selected and injected into the model context by the orchestrator protocol writing them into  FSM Protocol `Available Resources` based on the `context` they will be working in. Almost all protocols will get the file system as a resource for the workspace root.

**Types of resources**:
- Text Resources
    - Source code
    - Configuration files
    - Log files
    - Plain text
- Binary Resources
    - Images
    - Videos
    - Audio
    - PDFs
    - Other non-text formats

```json
# resource
{
    "uri": "file:///path/to/resource",
    "name": "resource_name",
    "description": "Optional description of the resource",
    "mimeType": "type of data in the resource",
}

```json
# resources/template - Template to modify dynamic resources
{
    "uriTemplate": "<RFC_6578_Template>",
    "name": "resource_name",
    "description": "Optional description of the resource",
    "mimeType": "type of data in the resource",
}

```json
# resources/list
{
    "jsonrpc": "2.0",
    "method": "resources/list",
    "params": {},
    "id": "<MSG_ID>"
}
```

```json
# resources/list/result
{
    "jsonrpc": "2.0",
    "result": [
        {
            "uri": "file:///path/to/resource",
            "name": "resource_name",
            "description": "Optional description of the resource",
            "mimeType": "type of data in the resource",
        }
    ],
    "id": "<MSG_ID>"
}

```json
# resources/read
{
    "jsonrpc": "2.0",
    "method": "resources/read",
    "params": {
        "uri": "file:///path/to/resource",
        "name": "resource_name",
    },
    "id": "<MSG_ID>"
}
```

```json
# resources/read/result
{
    "jsonrpc": "2.0",
    "result": {
        "uri": "file:///path/to/resource",
        "mimeType": "type of data in the resource",
        "text": "optional text content of the resource",
        "blob": "optional binary content of the resource",
    },
    "id": "<MSG_ID>"
}
```

```json
# resources/write
{
    "jsonrpc": "2.0",
    "method": "resources/write",
    "params": {
        "uri": "file:///path/to/resource",
        "name": "resource_name",
        "mimeType": "type of data in the resource",
        "text": "optional text content of the resource",
        "blob": "optional binary content of the resource",
    },
    "id": "<MSG_ID>"
}
```

```json
# resources/write/result
{
    "jsonrpc": "2.0",
    "result": {
        "uri": "file:///path/to/resource",
        "name": "resource_name",
        "code": 200,
        "message": "Resource written successfully",
    },
    "id": "<MSG_ID>"
}
```

```json
# resources/delete
{
    "jsonrpc": "2.0",
    "method": "resources/delete",
    "params": {
        "uri": "file:///path/to/resource",
        "name": "resource_name",
    },
    "id": "<MSG_ID>"
}
```

```json
# resources/delete/result
{
    "jsonrpc": "2.0",
    "result": {
        "uri": "file:///path/to/resource",
        "name": "resource_name",
        "code": 200,
        "message": "Resource deleted successfully",
    },
    "id": "<MSG_ID>"
}
```

### Subscriptions and Notifications

Subscriptions allow the model to subscribe to resources and receive notifications when they are updated.

```json
# resources/subscribe
{
    "jsonrpc": "2.0",
    "method": "resources/subscribe",
    "params": {
        "uri": "file:///path/to/resource",
        "name": "resource_name",
    },
    "id": "<MSG_ID>"
}
```

```json
# resources/subscribe/result
{
    "jsonrpc": "2.0",
    "result": {
        "uri": "file:///path/to/resource",
        "name": "resource_name",
        "code": 200,
        "message": "Resource subscribed successfully",
    },
    "id": "<MSG_ID>"
}
```

```json
# notification/resources/updated
{
    "jsonrpc": "2.0",
    "result": {
        "uri": "file:///path/to/resource",
        "mimeType": "type of data in the resource",
        "text": "optional text content of the resource",
        "blob": "optional binary content of the resource",
    },
    "id": "<MSG_ID>"
}
```

```json
# notification/resources/unsubscribed
{
    "jsonrpc": "2.0",
    "result": {
        "uri": "file:///path/to/resource",
        "name": "resource_name",
    },
    "id": "<MSG_ID>"
}
```

```json
# notification/resources/unsubscribed/result
{
    "jsonrpc": "2.0",
    "result": {
        "uri": "file:///path/to/resource",
        "name": "resource_name",
        "code": 200,
        "message": "Resource unsubscribed successfully",
    },
    "id": "<MSG_ID>"
}
```

### Prompts

Prompts are preconfigured prompt template with optional dynamic variables that the model can use to generate specific requests with custom context.

```json
# prompt
{
    "name": "prompt name",
    "description": "description of the prompt",
    "arguments": [
        {
            "name": "argument name",
            "description": "description of the dynamic arguement",
            "required": "bool indicating if the argument is required",
        }
    ]
}
```

When the orchestrator requests a prompt it should be assumed that it is a request for inference. The server should fill the prompt arguments with values passed and then generate inference with the cusomized prompt.

We will want to save and collect these for future use.

```json
# prompt/list
{

    "jsonrpc": "2.0",
    "method": "prompt/list",
    "params": {},
    "id": "<MSG_ID>"
}
```

```json
# prompt/list/result
{
    "jsonrpc": "2.0",
    "result": [
        {
            "name": "prompt name",
            "description": "description of the prompt",
            "arguments": [
                {
                    "name": "argument name",
                    "description": "description of the dynamic arguement",
                    "required": "bool indicating if the argument is required",
                }
            ]
        }
    ],
    "id": "<MSG_ID>"
}
```

```json
# prompt/call - this is marked as prompts/get in the model context protocol library but i dont like that it differs from the previous conventions so I'm modifying it for our uses. We almost never will request prompts from third party servers regardless.
{
    "jsonrpc": "2.0",
    "method": "prompt/call",
    "params": {
        "prompt_name": "<PROMPT_NAME>",
        "arguments": {
            "<ARGUMENT_NAME>": "<ARGUMENT_VALUE>"
        }
    },
    "id": "<MSG_ID>"
}
```

```json
# prompt/call/result
{
    "jsonrpc": "2.0",
    "result": {
        "content": "<INFERENCED_CONTENT>",
        "model": "<MODEL_NAME>",
        "usage": {
            "prompt_tokens": <PROMPT_TOKENS>,
            "completion_tokens": <COMPLETION_TOKENS>,
            "total_tokens": <TOTAL_TOKENS>
        }
    },
    "id": "<MSG_ID>"
}
```

We should craft a context prompt that we can dynamically fill with the compiled context document as well as a conversation context prompt that will structure the entire available context including conversation history with a token limit dropping the oldest messages as we go. We should archive those messages rather than just delete them for future training.


### OpenAI API integration

We can attach these tools to the OpenAI API requests and response objects to provide a more complete context to the model. Using the MCP servers to attach the tool results as messages to the conversation history.
```json
{
    "messages": [
        {
            "role": "system",
            "content": "compiled context document"
        },
        {
            "role": "user",
            "content": "development request"
        },
        {
            "role": "assistant",
            "content": "assistance response"
        }
    ]
}
```

Ultimately we will want to convert all response and request types to these messages as a context history for the model to track its work. Every time we swap contexts we will swap out this conversation history and save it. Context that exceed the context limit will be archived for future training and retrival if needed"

```json
{
    "messages": [
        {
            "role": "system",
            "content": "compiled context document"
        },
    ],
    "tool_calls": [
        {
            "resource_call": {
                "jsonrpc": "2.0",
                "method": "resources/read",
                "params": {
                    "uri": "file:///path/to/resource",
                    "name": "resource_name",
                },
                "id": "ResourceCallID"
            }
        },
        {
            "tool_call": {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "tool_name": "text_summarizer",
                    "arguments": {
                        "depends": "ResourceCallID",
                        "text": "<ResourceCallID response>"
                    }
                },
                "id": "ToolCallID"
            }
        }
    ]
}
```
Would turn into
```json
{
    "messages": [
        {
            "role": "system",
            "content": "compiled context document"
        },
        {
            "role": "assistant",
            "content": "resource request"
        },
        {
            "role": "system",
            "content": "<ResourceCallID response>"
        },
        {
            "role": "assistant",
            "content": " tool request using <ResourceCallID> response"
        },
        {
            "role": "tool",
            "content": "<ToolCallID response>"
        }
    ]
}
```


### Server Configuration

MCP Servers are configured in $HOME/.erasmus/mcp/config.json
```json
{
    "mcpServers": {
        "github": {
            "command": "./erasmus/mcp/github/server",
            "args": [
                "studio",
            ],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}",
            }
        }
    }
}
```

We will have to dynamically parse the contents of the message from the tool calls. The model has a tendancy to respond in a verbose manner or with json objects or markdown code blocks. We can modfy syn-agent which has a comprehensive parser for these types of responses and edge cases. We can use it to parse the tool call responses and extract the relevant information.

### Tool Commands

+We need a convention for running commands from the model output. We can use `!` prepended to a string to indicate that this h how it is a command to run. We can use the same syntax as shell commands to run commands.


## ToolSets

A group of tools that work within a specific scope.
    - ToolSets are provided to the model through the model context protocol(MCP) and are implemented as servers that will provide their service to the mcp client.
    - They are selected and injected into the model context by the orchestrator protocol writing them into  FSM Protocol `Available Tools` based on the `context` they will be working in.
+    - ToolSets should be grouped into categorically related groups and the context manager should detect available protocols making sure only their name and brief description appear in the available toolset section of the protocol.
    - When a toolset is called then it should return a list of available tools, their description and arguments through the `tools/list` rpc method. The toolset should be structured as an RPC call:


#### FileSystem

Read/Write/Append/Move/Rename/Remove directories and files.
https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem

- Configurable scope, default to project root
- Prompt for confirmation on any destructive actions
- Hard restriction on removing files/directories directly in / or /home or /root. Provide an override option with a warning for advanced users.

#### SQLite

CRUD operations on SQLite databases.
https://github.com/modelcontextprotocol/servers/tree/main/src/sqlite

#### GoogleDrive

CRUD operations on Google Drive files.
https://github.com/modelcontextprotocol/servers/tree/main/src/gdrive

#### BraveSearch

Web search using Brave Search API.
https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search


#### Puppeteer

Allows the agent to pilot a headless browser to scrape websites and interact with them.
https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer


#### Memory

Knowledge graph persistent memory for the agent
https://github.com/modelcontextprotocol/servers/tree/main/src/memory


#### Sequential Thinking

A toolset that enables reflective thinking for complex problem solving
https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking


#### Runners

Application runner commands. These can be grouped together or broken into seperate tools base on complexity of the commands available.

**1. UV**:
    - Package manager for Python.
    - Provides runtime compiled environments on single file scripts using ```uv add --script path/to/script.py <PACKAGE>```. We can make rust bindings for the ```uv run``` collecting our written single file scripts into a dynamically loaded tool set with a description for future use.
    - If the model requests to use python we will need to swap it for uv during command parsing.
        - python/python3/python -m --> uv run
        - `pip/pip3 install` --> `uv add` or `uv pip install` checking if its a single file and if it is add the --script flag and collect the description and the tool for future use.
        - `pip install -r requirements.txt` --> `uv add -r requirements.txt` or `uv sync` if pyproject.toml in root with optional flags [`all-extras`, `--dev`, `--freeze`] for additional depdendency options.
        - `python -m venv .venv` --> `uv venv`. Source remains the same.
**2. Cargo**:
    - Package manager for Rust.
    - Just provide a generic cargo command and accept all cargo flags.

**3. PNPM**:
    - Package manager for Typescript.
    - Convert
    - `npm` --> `pnpm`
    - `npx` --> `pnpx`


**4. Bash**:
    Scripting commands and writing bash scripts.
    - Automated review request sent to the inference model to confirm the functionality of the script. We will need to get the model to state its purpose parse that and the output script and return it to the inference endpoint with a fresh context and a review protocol.
    - Confirmation on destructive actions
    -
**5. Docker**:
    - Containerization commands and writing dockerfiles.
    - Automated review request sent to the inference model to confirm the functionality of the dockerfile. We will need to get the model to state its purpose parse that and the output dockerfile and return it to the inference endpoint with a fresh context and a review protocol.
    - Confirmation on destructive actions
    -
**6. Git**:
    We already have a stand alone git mcp server in $HOME/.erasmus/mcp/github/server
    - Version control commands and writing git commands.
    - Automated review request sent to the inference model to confirm the functionality of the git command. We will need to get the model to state its purpose parse that and the output git command and return it to the inference endpoint with a fresh context and a review protocol.
    - Confirmation on destructive actions

lets model the tools after the tools in your ide context. it should be somewhere before or after the rules files that i populate your context with project details for you to be able to track. effectively we are replicating the use of it with my own inference. the workflow should be


### Detailed Workflow
1. **Design Protocol Loaded**:
    - Design agent is loaded to plan development

2. **Project Context Created/Loaded**:
    - if empty prompt the user for a development project and user response is used to create an architecture context file in $HOME/.erasmus/context/path_to_context/ctx/orchestration/architecture.md
    - if not empty load the context

3. **Progress Schedule Created/Loaded**:
    - If empty prompt inference with the architecture document to break it down into a progress schedule context file in $HOME/.erasmus/context/path_to_context/ctx/orchestration/progress_schedule.md
    - If not empty load the context

4. **Task List Created/Loaded**:
    - If empty prompt inference with the progress schedule document to break it down into a task list context file in $HOME/.erasmus/context/path_to_context/ctx/orchestration/task_list.md
    - If not empty load the context

5. **Orchestration Protocol is loaded**:
    - The orchestration protocol constructs the FSM with approrpriate protocols using the process and task list creating new contexts for individual components and loading the appropriate protocols and tools into them and organzing them into a finite state machine(FSM).

6. **FSM is loaded**:
    - The FSM is loaded and the user is prompted to review the development plan and confirm it.

7. **Development begins**:
    - The FSM dynamically swaps contexts as it works through its scheduled tasks iteratively until completion.
    - The development workflow broadly looks like this
        - A component is selected for development
        - The development protocol is loaded with approrpiate component context
        - The developer implements the code updating the component context as they work
        - The testing protocol is loaded retaining the same context
        - Tests are developed and implemented for the component updating the component context as they work
        - The debugging protocol is loaded retaining the same context
        - The debugging protocol is run to debug the component and ensure all tests pass updating the component context as they work
        - The review protocol is loaded retaining the same context
        - The review protocol is run to do a comprehensive review of the component and ensure it meets the requirements generating a report that is saved in the docs folder.
        - The development protocol is loaded and run to address any issues identified in the review report updating the component context as they work.
        - The documentation protocol is loaded and run to document the component updating the component context as they work.
        - The component is tested again.
        - The component is marked as complete.
        - The changes are recorded in git
    - A new component is selected and the process repeats until the application is complete.
    - In between components the orchrestration protocol is run to update its main progress schedule and task lists ensuring that development is sticking to is on track adjust contexts as needed.

8. **Development ends**:
    - The testing protocol is loaded and the product is tested  end to end ensuring it meets requirements and functionality was not missed.

9. **Final documentation**
    - The documentation protocol is loaded and run to document the product ensuring comprehensive doc strings, detailed documentation and easy to understand readme is generated.

10. **Final review**
    - The review protocol is loaded and run to do a comprehensive review of the product ensuring it meets the requirements and functionality was not missed.

11. **Complete**
    - The project is complete and the process can begin again.

### Additional Third Party MCP Servers

These could be useful tools but must be audited be fore being added.


#### Atlassian

CRUD operations on Atlassian Jira and Confluence.
https://github.com/sooperset/mcp-atlassian


#### BigQuery

CRUD operations on Google BigQuery.
https://github.com/LucasHild/mcp-server-bigquery


#### Calculator

Basic calculator
https://github.com/githejie/mcp-server-calculator


#### Code Executor

Python code executor
https://github.com/bazinga012/mcp_code_executor

#### Chroma

Chroma vector database
https://github.com/privetin/chroma

#### ClaudeCode(Gmail integration)

Claude Gmail integration
https://github.com/ZilongXue/claude-post


#### Code Assistant

Framework for repo data ingestion
https://github.com/stippi/code-assistant

#### CoinMarketCap

Interact with coin market cap data
https://github.com/shinzo-labs/coinmarketcap-mcp


#### DaVinci Resolve

Interact with the davinci resolve video editor
https://github.com/samuelgursky/davinci-resolve-mcp


#### Deep Research

Deep research style research toolset
https://github.com/reading-plus-ai/mcp-server-deep-research

#### Dataset Viewer

Interact with the dataset viewer
https://github.com/privetin/dataset-viewer

#### Discord

Interact with discord
https://github.com/SaseQ/discord-mcp

#### Docker

Interact with docker
https://github.com/ckreiling/mcp-server-docker

#### Gmail

Interact with gmail
https://github.com/GongRzhe/Gmail-MCP-Server


#### Google Sheets

Interact with google sheets
https://github.com/rohans2/mcp-google-sheets

#### Google Calendar

Interact with google calendar
https://github.com/nspady/google-calendar-mcp

#### Heurist Mesh

Interact with heurist mesh
https://github.com/heurist-network/heurist-mesh-mcp-server

#### Jupyter Notebook

Interact with jupyter notebook
https://github.com/jjsantos01/jupyter-notebook-mcp

#### LSP

Interact with Language Service Provider
https://github.com/Tritlo/lsp-mcp

#### MCP create

Create MCP servers
https://github.com/tesla0225/mcp-create

#### MCP Proxy Server

Aggregate multiple MCP servers into a single endpoint
https://github.com/TBXark/mcp-proxy


#### Markdownify

Convert anything to markdown
https://github.com/zcaceres/mcp-markdownify-server

#### Nasdaq Data Link

Interact with nasdaq data link
https://github.com/stefanoamorelli/nasdaq-data-link-mcp

#### Notion

Interact with notion
https://github.com/suekou/mcp-notion-server

#### OpenAI

Interact with openai
https://github.com/sooperset/mcp-openai


#### Obsidian

Interact with obsidian
https://github.com/calclavia/mcp-obsidian

#### OpenRPC

Interact with openrpc
https://github.com/shanejonas/openrpc-mpc-server


#### Riot

Get riot games data
https://github.com/jifrozen0110/mcp-riot


#### Solana kit

Interact with solana
https://github.com/sendaifun/solana-agent-kit/tree/main/examples/agent-kit-mcp-server

#### Telegram

Interact with telegram
https://github.com/chigwell/telegram-mcp


#### TFT

Interact with tft
https://github.com/GeLi2001/tft-mcp-server


#### Terminal Command

Interact with terminal
https://github.com/GongRzhe/terminal-controller-mcp


#### Todos

Interact with todos
https://github.com/abhiz123/todoist-mcp-server


#### wikidata

Interact with wikidata
https://github.com/zzaebok/mcp-wikidata
