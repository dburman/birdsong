//! The built-in web dashboard. Files under `static/` are compiled into the binary, so the Docker
//! image needs nothing besides the executable. Assets are sent with `Cache-Control: no-store` so a
//! browser never keeps an old dashboard after an upgrade; they are small and gzip-compressed.

use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::response::IntoResponse;

macro_rules! asset {
    ($name:ident, $file:literal, $mime:literal) => {
        pub async fn $name() -> impl IntoResponse {
            (
                [(CONTENT_TYPE, $mime), (CACHE_CONTROL, "no-store")],
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../static/", $file)),
            )
        }
    };
}

asset!(index, "index.html", "text/html; charset=utf-8");
asset!(app_js, "app.js", "text/javascript; charset=utf-8");
asset!(style_css, "style.css", "text/css; charset=utf-8");
asset!(favicon, "favicon.svg", "image/svg+xml");
