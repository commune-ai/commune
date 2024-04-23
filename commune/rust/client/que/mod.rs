use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use reqwest::header::CONTENT_TYPE;


// Some 
#[derive(Serialize, Deserialize, Debug)]
struct GETAPIResponse {
    origin: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct JSONResponse {
    json: HashMap<String, String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // - Create a new client which is re-used between requests
    let client = reqwest::Client::new();
 

    // - Doing a GET request
    // - Parse the response to the "GETAPIResponse" struct
    let resp200 = client.get("https://httpbin.org/ip")
        .header(CONTENT_TYPE, "application/json")
        .send()
        .await?
        .json::<GETAPIResponse>()
        .await?;

    println!("{:#?}", resp200);
    // Output:
    /*
    GETAPIResponse {
        origin: "182.190.14.159",
    }
    */


    // Create a Map of string key-value pairs 
    // to represent the body payload
    let mut map = HashMap::new();
    map.insert("lang", "rust");
    map.insert("body", "json");


    // - Doing a POST request
    // - Parse the response to the "JSONResponse" struct
    let resp_json = client.post("0.0.0.0")
        .json(&map)
        .send()
        .await?
        .json::<JSONResponse>()
        .await?;

    println!("{:#?}", resp_json);
    // Output:
    /*
    JSONResponse {
        json: {
            "body": "json",
            "lang": "rust",
        },
    }
    */


    // - Doing a GET request
    let resp404 = client.get("https://httpbin.org/status/404")
        .send()
        .await?;

    // - Matching the HTTP status code of the request
    match resp404.status() {
        // - "OK - 200" everything was good
        reqwest::StatusCode::OK => {
            println!("Success!");
            // ...
        },
        // - "NOT_FOUND - 404" the resource wasn't found
        reqwest::StatusCode::NOT_FOUND => {
            println!("Got 404! Haven't found resource!");
            // Output: 
            /*
            Got 404! Haven't found resource!
            */
        },
        // - Every other status code which doesn't match the above ones
        _ => {
            panic!("Okay... this shouldn't happen...");
        },
    };


   

    Ok(())
}