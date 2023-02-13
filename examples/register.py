
import commune
# commune.new_event_loop()
# commune.get_module('block.bittensor').register_loop()
commune.launch(module='block.bittensor', fn='register_loop', name='registration', mode='local')
    # self.register(wait_for_inclusion: bool = False,
    #         wait_for_finalization: bool = True,
    #         prompt: bool = False,
    #         max_allowed_attempts: int = 3,
    #         cuda: bool = True,
    #         dev_id: int = commune.gpus(),
    #         TPB: int = 256,
    #         num_processes: Optional[int] = None,
    #         update_interval: Optional[int] = None,
    #         output_in_place: bool = True,
    #         log_verbose: bool = False,
    #     )