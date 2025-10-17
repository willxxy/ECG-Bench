use pyo3::prelude::*;
use std::collections::HashMap;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use fxhash::FxHashMap;


#[inline(always)]
fn merge(ids: &mut Vec<u32>, pair: (u32, u32), new_id: u32) {
    let mut i = 0;
    let mut write = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
            ids[write] = new_id;
            write += 1;
            i += 2;
        } else {
            ids[write] = ids[i];
            write += 1;
            i += 1;
        }
    }
    ids.truncate(write);
}

fn get_stats(ids: &[u32]) -> FxHashMap<(u32, u32), u32> {
    if ids.len() < 1000 {
        let mut acc = FxHashMap::default();
        for window in ids.windows(2) {
            *acc.entry((window[0], window[1])).or_insert(0) += 1;
        }
        acc
    } else {
        ids.par_windows(2)
            .fold(FxHashMap::default, |mut acc, window| {
                *acc.entry((window[0], window[1])).or_insert(0) += 1;
                acc
            })
            .reduce(FxHashMap::default, |mut acc1, acc2| {
                for (k, v) in acc2 {
                    *acc1.entry(k).or_insert(0) += v;
                }
                acc1
            })
    }
}

fn byte_to_string(b: u8) -> String {
    if b <= 127 {
        String::from_utf8(vec![b]).unwrap()
    } else {
        format!("<{}>", b)
    }
}

#[pyfunction]
fn byte_pair_encoding(
    text: &str,
    num_merges: usize,
    num_threads: usize,
) -> PyResult<(Vec<u32>, HashMap<u32, String>, Vec<(Vec<u32>, u32)>)> {
    let start_total = Instant::now();
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();
    let pool = Arc::new(pool);

    let mut ids: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32).collect();
    let mut vocab: HashMap<u32, String> =
        (0..256).map(|idx| (idx, byte_to_string(idx as u8))).collect();
    let mut vocab_tokens: HashMap<u32, Vec<u32>> = (0..256).map(|idx| (idx, vec![idx])).collect();
    let mut merges = Vec::new();

    let pb = ProgressBar::new(num_merges as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    for i in 0..num_merges {
        let pairs = pool.install(|| get_stats(&ids));

        if pairs.is_empty() {
            break;
        }

        let best = pool
            .install(|| pairs.par_iter().max_by_key(|&(_, count)| count))
            .and_then(|(&pair, _)| Some(pair));

        if let Some(best_pair) = best {
            let new_id = 256 + i as u32;

            merge(&mut ids, best_pair, new_id);

            vocab.insert(
                new_id,
                vocab[&best_pair.0].clone() + &vocab[&best_pair.1],
            );

            let mut new_token = vocab_tokens.get(&best_pair.0).unwrap().clone();
            new_token.extend(vocab_tokens.get(&best_pair.1).unwrap());
            vocab_tokens.insert(new_id, new_token.clone());

            merges.push((new_token, new_id));

            pb.set_message(format!("Merge {}", i + 1));
            pb.inc(1);
        } else {
            break;
        }
    }

    pb.finish_with_message("BPE completed");

    let total_duration = start_total.elapsed();
    println!("Total time for byte_pair_encoding: {:?}", total_duration);

    Ok((ids, vocab, merges))
}


struct TrieNode {
    children: HashMap<u32, TrieNode>,
    token_id: Option<u32>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            token_id: None,
        }
    }

    fn insert(&mut self, token: &[u32], token_id: u32) {
        let mut node = self;
        for &id in token {
            node = node.children.entry(id).or_insert_with(TrieNode::new);
        }
        node.token_id = Some(token_id);
    }
}

#[pyfunction]
fn encode_symbol(text: &str, merges: Vec<(Vec<u32>, u32)>) -> PyResult<Vec<u32>> {
    let ids: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32).collect();

    let mut trie_root = TrieNode::new();

    for b in 0..=255u32 {
        trie_root.insert(&[b], b);
    }

    for (token_sequence, token_id) in &merges {
        trie_root.insert(&token_sequence, *token_id);
    }

    let mut output_ids = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        let mut node = &trie_root;
        let mut match_len = 0;
        let mut match_id = None;

        for j in i..ids.len() {
            let id = ids[j];
            if let Some(child) = node.children.get(&id) {
                node = child;
                if let Some(token_id) = node.token_id {
                    match_len = j - i + 1;
                    match_id = Some(token_id);
                }
            } else {
                break;
            }
        }

        if let Some(token_id) = match_id {
            output_ids.push(token_id);
            i += match_len;
        } else {
            output_ids.push(ids[i]);
            i += 1;
        }
    }

    Ok(output_ids)
}

#[pymodule]
fn ecg_byte(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(byte_pair_encoding, m)?)?;
    m.add_function(wrap_pyfunction!(encode_symbol, m)?)?;
    Ok(())
}