use pyo3::prelude::*;
use std::collections::HashMap;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;

fn merge(ids: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    // Merges the pair in the ids and returns a new vector
    let mut new_ids = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i < ids.len() - 1 && (ids[i], ids[i + 1]) == pair {
            new_ids.push(new_id);
            i += 2;
        } else {
            new_ids.push(ids[i]);
            i += 1;
        }
    }
    new_ids
}

fn get_stats(ids: &[u32]) -> HashMap<(u32, u32), u32> {
    // Identifies the most frequent pair of bytes in the ids
    ids.par_windows(2)
        .fold(HashMap::new, |mut acc, window| {
            *acc.entry((window[0], window[1])).or_insert(0) += 1;
            acc
        })
        .reduce(HashMap::new, |mut acc1, acc2| {
            for (k, v) in acc2 {
                *acc1.entry(k).or_insert(0) += v;
            }
            acc1
        })
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

            ids = merge(&ids, best_pair, new_id);

            vocab.insert(
                new_id,
                vocab[&best_pair.0].clone() + &vocab[&best_pair.1],
            );

            // Update vocab_tokens
            let new_token = [
                vocab_tokens[&best_pair.0].clone(),
                vocab_tokens[&best_pair.1].clone(),
            ]
            .concat();
            vocab_tokens.insert(new_id, new_token.clone());

            // Store the new token sequence and its ID
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

    // Add merged tokens to the trie
    for (token_sequence, token_id) in &merges {
        trie_root.insert(&token_sequence, *token_id);
    }

    // Tokenize the input IDs using the trie
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
            // No match found, use the original byte
            output_ids.push(ids[i]);
            i += 1;
        }
    }

    Ok(output_ids)
}

#[pymodule]
fn bpe(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(byte_pair_encoding, m)?)?;
    m.add_function(wrap_pyfunction!(encode_symbol, m)?)?;
    Ok(())
}
