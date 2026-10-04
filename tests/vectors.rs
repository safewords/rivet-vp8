//! The VP8 test vectors: each `.ivf` stream must decode to frames whose
//! MD5s match its `.ivf.md5` list, frame for frame. The eighteen
//! comprehensive vectors (`vp80-00-comprehensive-001` to `-018`) are
//! committed under `tests/data/` (see its README); the other 44 public
//! vectors (`vp80-01` to `vp80-06`) are downloaded into `tests/vectors/` by
//! `tools/fetch-vectors.sh` and checked when present — `VP8_REQUIRE_VECTORS`
//! makes a missing set a failure.

use std::path::Path;

/// Decodes one vector on 1 and on 3 threads; returns (frames matching,
/// frames expected) and a description of the first mismatch.
fn check(dir: &Path, name: &str) -> (usize, usize, Option<String>) {
    let one = check_threads(dir, name, 1);
    let three = check_threads(dir, name, 3);
    if three.2.is_some() && one.2.is_none() {
        return (
            three.0,
            three.1,
            three.2.map(|b| format!("{b} (3 threads)")),
        );
    }
    one
}

fn check_threads(dir: &Path, name: &str, threads: usize) -> (usize, usize, Option<String>) {
    let ivf = std::fs::read(dir.join(format!("{name}.ivf"))).unwrap();
    let md5s = std::fs::read_to_string(dir.join(format!("{name}.ivf.md5"))).unwrap();
    let expected: Vec<&str> = md5s
        .lines()
        .filter_map(|l| l.split_whitespace().next())
        .collect();

    let mut reader = vp8::ivf::IvfReader::new(&ivf[..]).unwrap();
    let mut dec = vp8::Decoder::with_threads(threads);
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
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let (ok, n, bad) = check(&dir, &name);
        total.0 += ok;
        total.1 += n;
        println!(
            "{name}: {ok}/{n} frames{}",
            bad.as_ref()
                .map(|b| format!(", first mismatch at {b}"))
                .unwrap_or_default()
        );
        if ok != n || bad.is_some() {
            failures.push(name);
        }
    }
    println!(
        "total: {}/{} frames; {} of 18 vectors bit-exact",
        total.0,
        total.1,
        18 - failures.len()
    );
    assert!(failures.is_empty(), "vectors not bit-exact: {failures:?}");
}

#[test]
fn downloaded_vectors() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors");
    let list = include_str!("../tools/vectors.txt");
    let names: Vec<&str> = list
        .lines()
        .map(|l| l.trim().trim_end_matches(".ivf"))
        .filter(|l| !l.is_empty())
        .collect();
    if !dir.join(format!("{}.ivf", names[0])).exists() {
        assert!(
            std::env::var_os("VP8_REQUIRE_VECTORS").is_none(),
            "tests/vectors is empty: run tools/fetch-vectors.sh"
        );
        println!("tests/vectors is empty (tools/fetch-vectors.sh downloads it): skipped");
        return;
    }
    let mut failures = Vec::new();
    let mut total = (0, 0);
    for name in &names {
        let (ok, n, bad) = check(&dir, name);
        total.0 += ok;
        total.1 += n;
        if ok != n || bad.is_some() {
            println!(
                "{name}: {ok}/{n} frames{}",
                bad.map(|b| format!(", first mismatch at {b}"))
                    .unwrap_or_default()
            );
            failures.push(name.to_string());
        }
    }
    println!(
        "total: {}/{} frames; {} of {} vectors bit-exact",
        total.0,
        total.1,
        names.len() - failures.len(),
        names.len()
    );
    assert!(failures.is_empty(), "vectors not bit-exact: {failures:?}");
}
