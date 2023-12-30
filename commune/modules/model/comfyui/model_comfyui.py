import commune as c

class ModelComfyui(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y

    def install(self):
        c.cmd(f"pip install -e ./{self.dirpath()}")

    def run(self):
        c.cmd(f"python3 {self.dirpath()}/main.py --directm")
    def gradio(): 
        if args.temp_directory:
            temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
            print(f"Setting temp directory to: {temp_dir}")
            set_temp_directory(temp_dir)
        cleanup_temp()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server = server.PromptServer(loop)
        q = execution.PromptQueue(server)

        extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
        if os.path.isfile(extra_model_paths_config_path):
            load_extra_path_config(extra_model_paths_config_path)

        if args.extra_model_paths_config:
            for config_path in itertools.chain(*args.extra_model_paths_config):
                load_extra_path_config(config_path)

        init_custom_nodes()

        cuda_malloc_warning()

        server.add_routes()
        hijack_progress(server)

        threading.Thread(target=prompt_worker, daemon=True, args=(q, server,)).start()

        if args.output_directory:
            output_dir = os.path.abspath(args.output_directory)
            print(f"Setting output directory to: {output_dir}")
            set_output_directory(output_dir)

        #These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
        add_model_folder_path("checkpoints", os.path.join(get_output_directory(), "checkpoints"))
        add_model_folder_path("clip", os.path.join(get_output_directory(), "clip"))
        add_model_folder_path("vae", os.path.join(get_output_directory(), "vae"))

        if args.input_directory:
            input_dir = os.path.abspath(args.input_directory)
            print(f"Setting input directory to: {input_dir}")
            set_input_directory(input_dir)

        if args.quick_test_for_ci:
            exit(0)

        call_on_start = None
        if args.auto_launch:
            def startup_server(address, port):
                import webbrowser
                if os.name == 'nt' and address == '0.0.0.0':
                    address = '127.0.0.1'
                webbrowser.open(f"http://{address}:{port}")
            call_on_start = startup_server

        try:
            loop.run_until_complete(run(server, address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=call_on_start))
        except KeyboardInterrupt:
            print("\nStopped server")

        cleanup_temp()
    