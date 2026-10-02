//! The VP8 comprehensive test vectors (`vp80-00-comprehensive-001` to
//! `-018`): each `.ivf` stream must decode to frames whose MD5s match its
//! `.ivf.md5` list, frame for frame. See `tests/data/README.md` for where
//! the files come from.

use std::path::Path;

/// Decodes one vector; returns (frames matching, frames expected) and a
/// description of the first mismatch.
fn check(name: &str) -> (usize, usize, Option<String>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let ivf = std::fs::read(dir.join(format!("{name}.ivf"))).unwrap();
    let md5s = std::fs::read_to_string(dir.join(format!("{name}.ivf.md5"))).unwrap();
    let expected: Vec<&str> = md5s.lines().filter_map(|l| l.split_whitespace().next()).collect();

    let mut reader = vp8::ivf::IvfReader::new(&ivf[..]).unwrap();
    let mut dec = vp8::Decoder::new();
    let mut shown = 0;
    let mut matched = 0;
    let mut first_bad = None;
    let mut index = 0;
    while let Some(frame) = reader.next_frame().unwrap() {
        index += 1;
        match dec.decode(&frame.data) {
            Ok(Some(picture)) => {
                let digest = format!("{:x}", md5::compute(picture.packed()));
                if expected.get(shown) == Some(&digest.as_str()) {
                    matched += 1;
                } else if first_bad.is_none() {
                    first_bad = Some(format!("frame {index} (shown #{shown})"));
                } else if std::env::var_os("VP8_ALL_MISMATCHES").is_some() {
                    println!("  {name}: frame {index} differs");
                }
                shown += 1;
            }
            Ok(None) => {}
            Err(e) => {
                if first_bad.is_none() {
                    first_bad = Some(format!("frame {index}: {e}"));
                }
            }
        }
    }
    if shown != expected.len() && first_bad.is_none() {
        first_bad = Some(format!("{shown} frames shown, {} expected", expected.len()));
    }
    (matched, expected.len(), first_bad)
}

#[test]
fn comprehensive_vectors() {
    let mut failures = Vec::new();
    let mut total = (0, 0);
    for i in 1..=18 {
        let name = format!("vp80-00-comprehensive-{i:03}");
        let (ok, n, bad) = check(&name);
        total.0 += ok;
        total.1 += n;
        println!("{name}: {ok}/{n} frames{}", bad.as_ref().map(|b| format!(", first mismatch at {b}")).unwrap_or_default());
        if ok != n || bad.is_some() {
            failures.push(name);
        }
    }
    println!("total: {}/{} frames; {} of 18 vectors bit-exact", total.0, total.1, 18 - failures.len());
    assert!(failures.is_empty(), "vectors not bit-exact: {failures:?}");
}
