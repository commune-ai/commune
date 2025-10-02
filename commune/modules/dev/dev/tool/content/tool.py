
from typing import List
import commune as c

class ContentTool:

    def forward(self, 
                path: str = './', 
                query=None, 
                max_size=100000, 
                threads = 4,
                
                timeout=20) -> List[str]:
        """
        Find files in a directory matching a specific pattern.
        """
        result = c.fn('dev.tool.select.files/')(path=path, query=query, trials=4)
        content = str(result)
        size = len(content)
        c.print(f"path={path} max_size={max_size} size={size}", color='cyan')
        if size > max_size:
            if threads > 1:
                c.print(f"Using {threads} threads for summarization", color='yellow')
                from concurrent.futures import ThreadPoolExecutor
                summarize = c.fn('dev.tool.sum.file/')
                future2name = {}
                for k, v in result.items():
                    params = {'content': v, "query": query}
                    future =  c.submit(summarize, **params, timeout=timeout)
                    future2name[future] = k
                
                for future in c.as_completed(future2name, timeout=timeout):
                    k = future2name[future]
                    result[k] = future.result()

            else:
                c.print(f"Using single thread for summarization", color='yellow')
                for k, v in result.items():
                    result[k] = c.mod('dev.tool.sum.file').forward({'content': v, "query": query}, timeout=timeout)
        else:
            result = content
        c.print(f"Content found: {len(result)} items", color='green')
        return result
