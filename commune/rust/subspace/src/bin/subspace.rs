use pyo3::prelude::*;
use subxt::{ClientBuilder, DefaultNodeRuntime};
use std::collections::HashMap;


#[pyfunction]
async fn get_modules(
    keys: Option<Vec<String>>,
    network: Option<String>,
    timeout: Option<u64>,
    netuid: Option<u64>,
    fmt: Option<String>,
    include_uids: Option<bool>,
    kwargs: Option<HashMap<String, PyObject>>,
) -> PyResult<Vec<HashMap<String, PyObject>>> {
    // Default values
    let keys = keys.unwrap_or_else(|| vec![]);
    let network = network.unwrap_or_else(|| "main".to_string());
    let timeout = timeout.unwrap_or_else(|| 20);
    let netuid = netuid.unwrap_or_else(|| 0);
    let fmt = fmt.unwrap_or_else(|| "j".to_string());
    let include_uids = include_uids.unwrap_or(true);
    let kwargs = kwargs.unwrap_or_else(HashMap::new);

    let mut key2module = HashMap::new();
    let mut key2future = HashMap::new();
    let mut progress_bar = ProgressBar::new(keys.len() as u64);
    println!("Querying {} keys for modules", keys.len());
    let future_keys: Vec<_> = keys
        .iter()
        .filter(|&k| !key2module.contains_key(k) && !key2future.contains_key(k))
        .cloned()
        .collect();

    let mut futures = Vec::new();
    let mut results = Vec::new();

    if include_uids {
        let name2uid = key2uid(netuid);
    }

    for key in &future_keys {
        let module_args: HashMap<String, PyObject> = [
            ("module".to_string(), key.into()),
            ("netuid".to_string(), netuid.into()),
            ("network".to_string(), network.clone().into()),
            ("fmt".to_string(), fmt.clone().into()),
        ]
        .iter()
        .cloned()
        .collect();

        let future = get_module(module_args);
        key2future.insert(key.clone(), future.clone());
        futures.push(future);
    }

    for (future, key) in futures.iter().zip(&future_keys) {
        let module = future.await?;
        if let (Ok(module), Ok(key)) = (module.extract::<HashMap<String, PyObject>>(), key.extract::<String>()) {
            if module.contains_key("name") {
                results.push(module);
            } else {
                println!("Error querying module for key {}", key);
            }
        }
    }

    Ok(results)
}



#[pyfunction]
fn create_substrate_interface(params: HashMap<String, PyObject>) -> PyResult<()> {
    let url = params.get("url").ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'url' parameter"))?.extract::<String>()?;
    let websocket = params.get("websocket").map(|v| v.extract::<String>().ok()).flatten();
    let ss58_format = params.get("ss58_format").map(|v| v.extract::<u32>().ok()).flatten();
    let type_registry = params.get("type_registry").map(|v| v.extract::<String>().ok()).flatten();
    let type_registry_preset = params.get("type_registry_preset").map(|v| v.extract::<String>().ok()).flatten();
    let cache_region = params.get("cache_region").map(|v| v.extract::<String>().ok()).flatten();
    let runtime_config = params.get("runtime_config").map(|v| v.extract::<String>().ok()).flatten();
    let ws_options = params.get("ws_options").map(|v| v.extract::<String>().ok()).flatten();
    let auto_discover = params.get("auto_discover").map(|v| v.extract::<bool>().ok()).flatten();
    let auto_reconnect = params.get("auto_reconnect").map(|v| v.extract::<bool>().ok()).flatten();

    let substrate = ClientBuilder::<DefaultNodeRuntime>::new()
        .set_url(url)
        .set_websocket(websocket)
        .set_ss58_format(ss58_format)
        .set_type_registry(type_registry)
        .set_type_registry_preset(type_registry_preset)
        .set_cache_region(cache_region)
        .set_runtime_config(runtime_config)
        .set_ws_options(ws_options)
        .set_auto_discover(auto_discover)
        .set_auto_reconnect(auto_reconnect)
        .build();
        

    println!("{:?}", substrate);

    Ok(())
}

#[pymodule]
fn substrate_subxt_module(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_substrate_interface, m)?)?;
    m.add_function(wrap_pyfunction!(get_modules, m)?)?;
    Ok(())
}