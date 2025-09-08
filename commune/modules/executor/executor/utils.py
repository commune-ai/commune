
import traceback
import asyncio
import os
import nest_asyncio

def new_event_loop() -> 'asyncio.AbstractEventLoop':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    nest_asyncio.apply()
    return loop

def detailed_error(e) -> dict:
    tb = traceback.extract_tb(e.__traceback__)
    filename = tb[-1].filename
    line_no = tb[-1].lineno
    line_text = tb[-1].line
    response = {
        'success': False,
        'error': str(e),
        'filename': filename.replace(os.path.expanduser('~'), '~'),
        'line_no': line_no,
        'line_text': line_text
    }   
    return response


def wait(futures:list, timeout:int = None, generator:bool=False, return_dict:bool = True) -> list:
    is_singleton = bool(not isinstance(futures, list))

    futures = [futures] if is_singleton else futures

    if len(futures) == 0:
        return []
    future2idx = {future:i for i,future in enumerate(futures)}
    if timeout == None:
        if hasattr(futures[0], 'timeout'):
            timeout = futures[0].timeout
        else:
            timeout = 30
    if generator:
        def get_results(futures):
            import concurrent 
            try: 
                for future in concurrent.futures.as_completed(futures, timeout=timeout):
                    if return_dict:
                        idx = future2idx[future]
                        yield {'idx': idx, 'result': future.result()}
                    else:
                        yield future.result()
            except Exception as e:
                yield None
    else:
        def get_results(futures):
            import concurrent
            results = [None]*len(futures)
            try:
                for future in concurrent.futures.as_completed(futures, timeout=timeout):
                    idx = future2idx[future]
                    results[idx] = future.result()
                    del future2idx[future]
                if is_singleton: 
                    results = results[0]
            except Exception as e:
                unfinished_futures = [future for future in futures if future in future2idx]
                print(f'Error: {e}, {len(unfinished_futures)} unfinished futures with timeout {timeout} seconds')
            return results

    return get_results(futures)

def as_completed( futures:list, timeout:int=10, **kwargs):
    import concurrent
    return concurrent.futures.as_completed(futures, timeout=timeout)

