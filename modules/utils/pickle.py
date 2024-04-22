
def load_pickle(file_path, verbose=True):
    import pickle
    with open(file_path, 'rb') as f:
        object = pickle.load(f)
    if verbose:
        print("Loaded: ", file_path)
    return object


def dump_pickle(object, file_path, verbose=True):
    ensure_dir(file_path=file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)
    if verbose:
        print("Saved: ", file_path)

