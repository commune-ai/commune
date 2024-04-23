use reqwest::{Error}; 

async fn post_it() -> Result<(), Error> {
    let url = "https://0.0.0.0/post";
    let json_data = r#"{<coroutine object Client.async_forward at 0x7293caca57e0>}"#;

    let client = reqwest::Client::new();

    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .body(json_data.to_owned())
        .send()
        .await?;

    println!("Status: {}", response.status());

    let response_body = response.text().await?;
    println!("Response body:\n{}", response_body);
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    post_it().await?;
    Ok(())
}