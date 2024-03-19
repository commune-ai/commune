use pyo3::prelude::*;
use reqwest::Client;
use serde_json::Value;
use std::error::Error;
use tokio::sync::mpsc::{self, Sender};

#[pyfunction]
async fn fetch_data_worker(url: String, request: String, headers: Vec<(&str, &str)>, timeout: u64, sender: Sender<Result<Value, Box<dyn Error>>>) {
    let client = Client::new();
    let response = client.post(&url)
        .json(&serde_json::from_str::<Value>(&request).unwrap())
        .headers(headers.iter().map(|&(k, v)| (k, v).into()))
        .send()
        .await; 
 
    let result = match response {
        Ok(response) => {
            match response.headers().get(reqwest::header::CONTENT_TYPE) {
                Some(content_type) if content_type == "text/event-stream" => {
                    const STREAM_PREFIX: &str = "data: ";
                    const BYTES_PER_MB: f64 = 1e6;
                    
                    // Assuming `response.content` is a Tokio stream (e.g., `tokio::io::AsyncRead`).
                    async fn process_response_content<R>(response_content: R, debug: bool) -> Vec<String>
                    where
                        R: tokio::io::AsyncRead + Unpin,
                    {
                        if(debug){
                            let gil = Python::acquire_gil();
                            let py = gil.python();

                            // Import the tqdm module and get the tqdm function
                            let tqdm = py.import("tqdm")?.getattr("tqdm")?;
                            // Call tqdm function with desired arguments
                            let progress_bar = tqdm.call1(("MB per Second",0))?; 
                        }
                        let mut result = Vec::new();
                        

                        // Create a buffer reader to read lines from the stream.
                        let mut reader = BufReader::new(response_content);

                        while let Ok(mut line) = reader.read_line().await {
                            if line.is_empty() {
                                continue;
                            }

                            let event_data = line.trim().to_string();

                            if debug {
                                let event_bytes = event_data.len() as f64;
                                // Update progress bar
                                progress_bar.call_method1("update", (event_bytes / BYTES_PER_MB,))?;
                            }

                            // Remove the "data: " prefix
                            if event_data.starts_with(STREAM_PREFIX) {
                                line.replace_range(..STREAM_PREFIX.len(), "");
                            }

                            // If the data is formatted as a JSON string, load it
                            if let Ok(event_data) = serde_json::from_str::<serde_json::Value>(&event_data) {
                                if let Some(data) = event_data.get("data") {
                                    if let Some(data_str) = data.as_str() {
                                        result.push(data_str.to_string());
                                    } else {
                                        result.push(data.to_string());
                                    }
                                }
                            } else {
                                // Otherwise, add the event data as is
                                result.push(event_data);
                            }
                        }

                        result
                    }
                    
                }
                Some(content_type) if content_type == "application/json" => {
                    response.json().await.map_err(|e| e.into())
                }
                Some(content_type) if content_type == "text/plain" => {
                    response.text().await.map(|text| text.into())
                }                
                _ => Err(format!("Invalid response content type: {:?}", content_type).into())
            }
        }
        Err(e) => Err(e.into())
    };

    sender.send(result).await.expect("Failed to send result");
}

async fn forward_worker(obj: &PyAny, fn_name: &str, input: &PyDict,sender: Sender<Result<Value, Box<dyn Error>>>){
    let mut user_info: Option<PyObject> = None;

    // Wrap the code inside Python's with_gil to safely interact with Python objects
    Python::with_gil(|py| {
        let isPublic = obj.getattr(public)?.extract::<bool>(py)?;
        // Verify the input with the server key class
        if !isPublic {
            // Assume `key.verify()` and `serializer.deserialize()` are Python methods
            let result = obj.getattr("key")?.call_method1("verify", (input,))?;
    
            // Extract the result as a boolean
            let is_verified: bool = result.extract(py)?;
        
            // Check if the verification passed
            if !is_verified {
                return Err(PyErr::new::<PyAssertionError, _>("Data not signed with correct key"));
            }
        }
        let args_exist = input.contains_key("args");
        let kwargs_exist = input.contains_key("kwargs");
        if args_exist && kwargs_exist {
            let mut data = PyDict::new(py);
            data.set_item("args", input.get_item("args"))?;
            data.set_item("kwargs", input.get_item("kwargs"))?;
            data.set_item("timestamp", input.get_item("timestamp"))?;
            data.set_item("address", input.get_item("address"))?;
            
            // Assign the 'data' dictionary to the 'data' key of the input dictionary
            input.set_item("data", data)?;
        }
        let deserialized = obj.getattr("serializer")?.call_method1("deserialize", (input.get_item("data"),))?;
        input.set_item("data", deserialized)?;
        let c = PyModule::import(py, "c")?;
        let timestamp_method = c.getattr("timestamp")?;

        // Execute the method and get the result
        let result: i64 = timestamp_method.call(())?.extract()?;



        let mut data: PyDict = input.get_item("data").unwrap().extract(py)?;

        // Verify the request is not too old
        let request_staleness: i64 = c.timestamp() - data.get_item("timestamp").unwrap().extract(py)?;
        if request_staleness >= obj.max_request_staleness {
            return Err(PyErr::new::<PyAssertionError, _>(format!("Request is too old, {} > MAX_STALENESS ({}) seconds old", request_staleness, obj.max_request_staleness)));
        }

        // Verify the access module
        user_info = obj.access_module.call_method1("verify", (input,))?;
        if user_info.get_item("passed").unwrap().extract::<bool>(py)? {
            return Ok(user_info);
        }

        let data_args: PyObject = data.get_item("args").unwrap().extract(py)?;
        let data_kwargs: PyObject = data.get_item("kwargs").unwrap().extract(py)?;
        let args: Vec<PyObject> = data_args.extract(py)?;
        let kwargs: HashMap<String, PyObject> = data_kwargs.extract(py)?;

        let fn_obj: PyObject = obj.module.getattr(fn_name)?;
        let fn_obj_callable: bool = fn_obj.hasattr("__call__")?;

        let result: PyObject;
        if fn_obj_callable {
            result = fn_obj.call(py, args, kwargs)?;
        } else {
            result = fn_obj;
        }

        sender.send(result).await.expect("Failed to send result");
    })
}
#[tokio::main]
fn forward(py: Python, obj: &PyAny, fn_name: &str, input: &PyDict) -> PyResult<PyObject> {
    let (tx, rx) = mpsc::channel(1); // Channel for sending result
    
    // Spawn a new thread for each request
    tokio::spawn(async move {
        forward_worker(obj, fn_name, input, tx).await;
    });

    // Receive and return the result
    rx.recv().await.expect("Failed to receive result")
}
#[pymodule]
fn My_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forward, m)?)?;
    Ok(())
}
