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
#[tokio::main]
async fn fetch_data(url: String, request: String, headers: Vec<(&str, &str)>, timeout: u64) -> Result<Value, Box<dyn Error>> {
    let (tx, rx) = mpsc::channel(1); // Channel for sending result
    
    // Spawn a new thread for each request
    tokio::spawn(async move {
        fetch_data_worker(url, request, headers, timeout, tx).await;
    });

    // Receive and return the result
    rx.recv().await.expect("Failed to receive result")
}

#[pymodule]
fn My_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fetch_data, m)?)?;
    Ok(())
}
