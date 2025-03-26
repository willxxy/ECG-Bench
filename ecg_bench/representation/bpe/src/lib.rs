use pyo3::prelude::*;
use std::collections::HashMap;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use fxhash::FxHashMap;
use regex::Regex;


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

#[pyfunction]
fn super_byte_pair_encoding(
    text: &str,
    num_merges: usize,
    transition_point: usize,
    num_threads: usize,
) -> PyResult<(Vec<u32>, HashMap<u32, String>, Vec<(Vec<u32>, u32)>)> {
    let start_total = Instant::now();
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();
    let pool = Arc::new(pool);

    // Define regex patterns
    let whitespace_regex = match Regex::new(r"\p{L}+| ?[^\s\p{L}\p{N}]+|\s+") {
        Ok(regex) => regex,
        Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to compile whitespace regex: {}", e))),
    };
    
    // Step 1: Pretokenize the text (only for stage 1)
    let mut pretokenized_text = String::new();
    let mut in_digit_sequence = false;
    let mut digit_buffer = String::new();

    for c in text.chars() {
        if c.is_digit(10) {
            in_digit_sequence = true;
            digit_buffer.push(c);
        } else {
            if in_digit_sequence {
                // Process digit buffer - group digits into threes from the right
                let mut processed_digits = String::new();
                let digits: Vec<char> = digit_buffer.chars().collect();
                
                let mut i = 0;
                while i < digits.len() {
                    let pos_from_right = digits.len() - i;
                    if i > 0 && pos_from_right % 3 == 0 {
                        processed_digits.push(' ');
                    }
                    processed_digits.push(digits[i]);
                    i += 1;
                }
                
                pretokenized_text.push_str(&processed_digits);
                digit_buffer.clear();
                in_digit_sequence = false;
            }
            pretokenized_text.push(c);
        }
    }

    if in_digit_sequence {
        let mut processed_digits = String::new();
        let digits: Vec<char> = digit_buffer.chars().collect();
        
        let mut i = 0;
        while i < digits.len() {
            let pos_from_right = digits.len() - i;
            if i > 0 && pos_from_right % 3 == 0 {
                processed_digits.push(' ');
            }
            processed_digits.push(digits[i]);
            i += 1;
        }
        
        pretokenized_text.push_str(&processed_digits);
    }

    // Create chunk boundaries for stage 1
    let mut chunks = Vec::new();
    if transition_point > 0 {
        let chunk_boundaries: Vec<_> = whitespace_regex.find_iter(&pretokenized_text)
            .map(|m| (m.start(), m.end()))
            .collect();
        
        let mut start = 0;
        for (s, e) in chunk_boundaries {
            if s > start {
                chunks.push(pretokenized_text[start..s].to_string());
            }
            chunks.push(pretokenized_text[s..e].to_string());
            start = e;
        }
        if start < pretokenized_text.len() {
            chunks.push(pretokenized_text[start..].to_string());
        }
    } else {
        // If transition_point is 0, skip stage 1 entirely
        chunks.push(pretokenized_text);
    }

    let mut vocab: FxHashMap<u32, String> = 
        (0..256).map(|idx| (idx, byte_to_string(idx as u8))).collect();
    let mut vocab_tokens: FxHashMap<u32, Vec<u32>> = (0..256).map(|idx| (idx, vec![idx])).collect();
    let mut merges = Vec::with_capacity(num_merges);
    
    let pb = ProgressBar::new(num_merges as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    // Stage 1: Standard BPE with pretokenization
    let mut current_vocab_size = 256;
    let mut stage = 1;
    
    let mut chunk_ids: Vec<Vec<u32>> = chunks.iter()
        .map(|chunk| chunk.as_bytes().iter().map(|&b| b as u32).collect())
        .collect();

    for i in 0..num_merges {
        if current_vocab_size - 256 >= transition_point && stage == 1 {
            // Transition to stage 2: merge all chunks
            pb.set_message("Transitioning to stage 2: merging all chunks");
            
            let total_capacity: usize = chunk_ids.iter().map(|chunk| chunk.len()).sum();
            let mut merged_ids = Vec::with_capacity(total_capacity);
            for chunk in &chunk_ids {
                merged_ids.extend_from_slice(chunk);
            }
            chunk_ids = vec![merged_ids];
            
            stage = 2;
        }
        
        let pairs = if chunk_ids.len() == 1 {
            pool.install(|| get_stats(&chunk_ids[0]))
        } else {
            let mut pairs = FxHashMap::default();
            let chunk_pairs: Vec<_> = pool.install(|| {
                chunk_ids.par_iter()
                    .map(|chunk| get_stats(chunk))
                    .collect()
            });
            
            for chunk_pair in chunk_pairs {
                for (pair, count) in chunk_pair {
                    *pairs.entry(pair).or_insert(0) += count;
                }
            }
            pairs
        };
        
        if pairs.is_empty() {
            break;
        }
        
        let best = pool
            .install(|| pairs.par_iter().max_by_key(|&(_, count)| count))
            .and_then(|(&pair, _)| Some(pair));
            
        if let Some(best_pair) = best {
            // Skip pairs containing ": " to avoid prompt boundary issues
            let pair_str = vocab[&best_pair.0].clone() + &vocab[&best_pair.1];
            if pair_str.contains(": ") {
                continue;
            }
            
            let word_count = pair_str.split_whitespace().count();
            
            // Skip if token contains more than 4 words
            if word_count > 4 {
                continue;
            }
            
            let new_id = 256 + i as u32;
            
            for chunk in &mut chunk_ids {
                merge(chunk, best_pair, new_id);
            }
            
            vocab.insert(new_id, pair_str);
            
            let mut new_token = vocab_tokens.get(&best_pair.0).unwrap().clone();
            new_token.extend(vocab_tokens.get(&best_pair.1).unwrap());
            vocab_tokens.insert(new_id, new_token.clone());
            
            merges.push((new_token, new_id));
            current_vocab_size += 1;
            
            pb.set_message(format!("Stage {}: Merge {}", stage, i + 1));
            pb.inc(1);
        } else {
            break;
        }
    }
    
    pb.finish_with_message("SuperBPE completed");
    
    let ids = chunk_ids.into_iter().flatten().collect();
    
    let total_duration = start_total.elapsed();
    println!("Total time for super_byte_pair_encoding: {:?}", total_duration);
    
    let vocab_hashmap: HashMap<u32, String> = vocab.into_iter().collect();
    
    Ok((ids, vocab_hashmap, merges))
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
fn bpe(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(byte_pair_encoding, m)?)?;
    m.add_function(wrap_pyfunction!(encode_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(super_byte_pair_encoding, m)?)?;
    Ok(())
}
