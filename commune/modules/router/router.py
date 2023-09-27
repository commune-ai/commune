import commune as c
import concurrent
class Router(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)
    def square_number(x):
        return x ** 2

        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(do_the_work, item) for item in work_list.items()]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                status = future.result()
                print('DONE: count:{} result:{}'.format(i, status))
            print(list(executor.map(square_number, range(10))))


