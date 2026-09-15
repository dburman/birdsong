//! `birdsong healthcheck` against throwaway TCP servers.
#![forbid(unsafe_code)]

use std::time::{Duration, Instant};

use birdsong_server::healthcheck::check;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Serve `response` (or nothing, when `None`) to every connection; returns the base URL.
async fn server(response: Option<&'static str>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut buf = [0u8; 1024];
                let _ = socket.read(&mut buf).await;
                match response {
                    Some(r) => {
                        let _ = socket.write_all(r.as_bytes()).await;
                    }
                    None => tokio::time::sleep(Duration::from_secs(30)).await,
                }
            });
        }
    });
    format!("http://{addr}/api/v1/health")
}

const OK: &str = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";
const UNAVAILABLE: &str =
    "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

#[tokio::test]
async fn success_and_failures() {
    assert_eq!(
        check(&server(Some(OK)).await, Duration::from_secs(2))
            .await
            .unwrap(),
        200
    );

    let err = check(&server(Some(UNAVAILABLE)).await, Duration::from_secs(2))
        .await
        .unwrap_err();
    assert!(format!("{err:#}").contains("503"), "{err:#}");

    let err = check(
        &server(Some("garbage\r\n\r\n")).await,
        Duration::from_secs(2),
    )
    .await
    .unwrap_err();
    assert!(
        format!("{err:#}").contains("not an HTTP response"),
        "{err:#}"
    );

    let started = Instant::now();
    let err = check(&server(None).await, Duration::from_millis(300))
        .await
        .unwrap_err();
    assert!(format!("{err:#}").contains("no response"), "{err:#}");
    assert!(started.elapsed() < Duration::from_secs(2));

    let closed = {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        l.local_addr().unwrap()
    };
    assert!(check(&format!("http://{closed}/"), Duration::from_secs(2))
        .await
        .is_err());
}

#[tokio::test]
async fn command_exit_codes() {
    let run = |url: String| async move {
        tokio::process::Command::new(env!("CARGO_BIN_EXE_birdsong"))
            .args(["healthcheck", "--url", &url, "--timeout-secs", "2"])
            .output()
            .await
            .unwrap()
    };
    let ok = run(server(Some(OK)).await).await;
    assert!(
        ok.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&ok.stderr)
    );
    assert!(String::from_utf8_lossy(&ok.stdout).contains("HTTP 200"));

    let bad = run(server(Some(UNAVAILABLE)).await).await;
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("503"));
}
