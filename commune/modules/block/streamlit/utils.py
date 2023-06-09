import streamlit as st

def describe(module =None, sidebar = True, detail=False, expand=True):
    
    _st = st.sidebar if sidebar else st
    st.sidebar.markdown('# '+str(module))
    fn_list = list(filter(lambda fn: callable(getattr(module,fn)) and '__' not in fn,  dir(module)))
    
    def content_fn(fn_list=fn_list):
        fn_list = _st.multiselect('fns', fn_list)
        for fn_key in fn_list:
            fn = getattr(module,fn_key)
            if callable(fn):
                _st.markdown('#### '+fn_key)
                _st.write(fn)
                _st.write(type(fn))
    if expand:
        with st.sidebar.expander(str(module)):
            content_fn()
    else:
        content_fn()


def row_column_bundles(fn_list, fn_args_list,cols_per_row=3):     
    
    cols = cols_per_row
    item_count = len(fn_list)
    rows = item_count // cols
    row2cols = []

    for row_idx in range(rows+1):
        row2cols.append(st.columns(cols))

    for fn_idx, fn in enumerate(fn_list):
        row_idx = fn_idx // cols
        col_idx = fn_idx % cols
        with row2cols[row_idx][col_idx]:
            fn(*fn_args_list[fn_idx])


def streamlit_thread(thread):
    try:
        # Streamlit >= 1.12.0
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
    except:
        # Streamlit <= 1.11.0
        from streamlit.scriptrunner import add_script_run_ctx
        from streamlit.scriptrunner.script_run_context import get_script_run_ctx
    
    return get_script_run_ctx(t)

