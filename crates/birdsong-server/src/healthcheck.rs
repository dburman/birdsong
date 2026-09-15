//! `birdsong healthcheck`: a tiny HTTP GET so the Docker image needs no curl.

use std::time::Duration;

use anyhow::{bail, Context};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Split `http://host[:port][/path]` into host, port (default 80) and path (default `/`).
/// Only plain `http://` is supported: the check always targets the local container.
pub fn parse_http_url(url: &str) -> anyhow::Result<(String, u16, String)> {
    let rest = url
        .strip_prefix("http://")
        .with_context(|| format!("only http:// URLs are supported, got {url:?}"))?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = if let Some(bracketed) = authority.strip_prefix('[') {
        // [::1]:8080
        let (host, after) = bracketed
            .split_once(']')
            .with_context(|| format!("bad IPv6 host in {url:?}"))?;
        let port = match after.strip_prefix(':') {
            Some(p) => p.parse().with_context(|| format!("bad port in {url:?}"))?,
            None => 80,
        };
        (host.to_string(), port)
    } else {
        match authority.rsplit_once(':') {
            Some((host, port)) => (
                host.to_string(),
                port.parse()
                    .with_context(|| format!("bad port in {url:?}"))?,
            ),
            None => (authority.to_string(), 80),
        }
    };
    if host.is_empty() {
        bail!("missing host in {url:?}");
    }
    Ok((host, port, path.to_string()))
}

/// GET `url` and return the status code if it is 2xx; otherwise (or on timeout) an error.
pub async fn check(url: &str, timeout: Duration) -> anyhow::Result<u16> {
    let (host, port, path) = parse_http_url(url)?;
    let request = async {
        let mut stream = TcpStream::connect((host.as_str(), port))
            .await
            .with_context(|| format!("connecting to {host}:{port}"))?;
        let request = format!(
            "GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: birdsong-healthcheck\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
        );
        stream.write_all(request.as_bytes()).await?;
        let mut head = Vec::with_capacity(256);
        let mut buf = [0u8; 256];
        while !head.windows(2).any(|w| w == b"\r\n") && head.len() < 8192 {
            let n = stream.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            head.extend_from_slice(&buf[..n]);
        }
        let text = String::from_utf8_lossy(&head);
        let first = text.lines().next().unwrap_or_default().to_string();
        let mut parts = first.split_whitespace();
        match (
            parts.next(),
            parts.next().and_then(|code| code.parse::<u16>().ok()),
        ) {
            (Some(version), Some(code)) if version.starts_with("HTTP/") => {
                Ok::<u16, anyhow::Error>(code)
            }
            _ => bail!("not an HTTP response from {url}: {first:?}"),
        }
    };
    let status = tokio::time::timeout(timeout, request)
        .await
        .with_context(|| format!("no response from {url} within {timeout:?}"))??;
    if !(200..300).contains(&status) {
        bail!("{url} returned HTTP {status}");
    }
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_urls() {
        assert_eq!(
            parse_http_url("http://127.0.0.1:8080/api/v1/health").unwrap(),
            ("127.0.0.1".into(), 8080, "/api/v1/health".into())
        );
        assert_eq!(
            parse_http_url("http://pi.local").unwrap(),
            ("pi.local".into(), 80, "/".into())
        );
        assert_eq!(
            parse_http_url("http://[::1]:9000/x").unwrap(),
            ("::1".into(), 9000, "/x".into())
        );
        assert!(parse_http_url("https://pi.local/").is_err());
        assert!(parse_http_url("http://:8080/").is_err());
        assert!(parse_http_url("http://host:port/").is_err());
    }
}
