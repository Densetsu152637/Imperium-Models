//! End to end on a tiny archive written to a temporary directory. The real archive is never
//! needed by tests or CI.

use ncm_sft::pipeline::{Model, build, policy_for};
use serde_json::{Value, json};
use std::fs;
use std::path::Path;

fn write(root: &Path, relative: &str, lines: &[Value]) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
    fs::write(path, text).unwrap();
}

fn hosted(id: &str, messages: Value) -> Value {
    json!({
        "id": id, "task_class": "coding", "specificity": "vague", "context_budget_k": 64,
        "tools": ["web_search", "await_tasks"],
        "messages": messages,
        "meta": {"teacher": "deepseek-v4-pro", "licence": "DeepSeek ToS", "pilot": 4,
                 "judge": {"model": "kimi-k3.1-pro", "keep": true, "pass": true, "mean": 4.8}}
    })
}

fn good(id: &str) -> Value {
    hosted(
        id,
        json!([
            {"role": "system", "content": "You are Emperor."},
            {"role": "user", "content": "Compare two queues."},
            {"role": "assistant", "content": "", "reasoning_content": "Search first.",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "web_search", "arguments": "{\"query\": \"queues\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "assistant", "content": "Use the first.", "reasoning_content": "Enough."}
        ]),
    )
}

fn plain(id: &str) -> Value {
    hosted(
        id,
        json!([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."}
        ]),
    )
}

fn broken(id: &str) -> Value {
    hosted(
        id,
        json!([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "web_search", "arguments": "{}"}}]}
        ]),
    )
}

fn archive(root: &Path) {
    let data = "emperor/data";
    write(
        root,
        &format!("{data}/hosted-1/final/final.jsonl"),
        &[good("a"), plain("b"), broken("c")],
    );
    write(root, &format!("{data}/hosted-1/excluded-pilots.json"), &[]);
    fs::write(root.join(format!("{data}/hosted-1/excluded-pilots.json")), "{\"3\": \"old\"}")
        .unwrap();
    let mut pilot3 = plain("d");
    pilot3["meta"]["pilot"] = json!(3);
    write(root, &format!("{data}/hosted-1/final/final2.jsonl"), &[pilot3]);
    let graft = json!({"id": "g1", "tools": [], "messages": plain("x")["messages"],
        "meta": {"generator": "QwQ-32B", "revision": "r", "judged": false}});
    write(root, &format!("{data}/mix-emperor-v2/final.jsonl"), &[good("a"), graft]);
    let mut local = plain("l");
    local["meta"]["teacher"] = json!("gpt-oss-20b");
    write(root, &format!("{data}/final/final.jsonl"), &[local]);
    write(
        root,
        &format!("{data}/hosted-1/rejects/rejects-00001.jsonl"),
        &[json!({"id": "r1", "reason": "too_long"}), json!({"id": "r2", "reason": "too_long"})],
    );
    write(root, &format!("{data}/quarantine/old/graft.jsonl"), &[json!({"id": "q"})]);
    write(
        root,
        &format!("{data}/hosted-1/judged/judged-00001.jsonl"),
        &[
            json!({"id": "a", "meta": {"judge": {"keep": true}}}),
            json!({"id": "z", "meta": {"judge": {"keep": false}}}),
        ],
    );
}

#[test]
fn emperor_build_counts_and_outputs() {
    let source = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    archive(source.path());
    let policy = policy_for(source.path(), Model::Emperor, &[], &[]).unwrap();
    let stats = build(source.path(), out.path(), Model::Emperor, &policy).unwrap();

    let hosted = &stats.files["emperor/data/hosted-1/final/final.jsonl"];
    assert_eq!((hosted.records_in, hosted.accepted), (3, 2));
    assert_eq!(hosted.needs_regeneration["unanswered_tool_call"], 1);
    assert_eq!((hosted.with_think, hosted.with_tool_calls), (1, 1));

    let mix = &stats.files["emperor/data/mix-emperor-v2/final.jsonl"];
    assert_eq!(mix.excluded["duplicate_id"], 1);
    assert_eq!(mix.accepted, 1);
    let local = &stats.files["emperor/data/final/final.jsonl"];
    assert_eq!(local.accepted, 1);

    assert_eq!(stats.totals.accepted, 4);
    assert_eq!(stats.totals.train + stats.totals.val, 4);
    assert_eq!(stats.judged_not_accepted, 1);
    assert_eq!(
        stats.not_training_data["emperor/data/hosted-1/rejects/rejects-00001.jsonl"].reasons["too_long"],
        2
    );
    assert_eq!(stats.not_training_data_by_kind["quarantine"], 1);
    assert!(stats.not_training_data.contains_key("emperor/data/hosted-1/final/final2.jsonl"));

    let dataset = fs::read_to_string(out.path().join("emperor/sft.jsonl")).unwrap();
    let first: Value = serde_json::from_str(dataset.lines().next().unwrap()).unwrap();
    assert_eq!(first["id"], "a");
    assert_eq!(first["source_file"], "emperor/data/hosted-1/final/final.jsonl");
    let text = first["text"].as_str().unwrap();
    assert!(text.starts_with("<|im_start|>system\nYou are Emperor.\n\n<tools>\n"));
    assert!(text.ends_with(
        "<|im_start|>assistant\n<think>\nEnough.\n</think>\nUse the first.\n<|im_end|>\n"
    ));
    assert_eq!(first["messages"].as_array().unwrap().len(), 5);
    assert_eq!(first["tools"].as_array().unwrap().len(), 2);
    let regen = fs::read_to_string(out.path().join("emperor/needs-regeneration.jsonl")).unwrap();
    assert!(regen.contains("\"id\":\"c\"") && regen.contains("unanswered_tool_call"));
    assert!(out.path().join("emperor/stats.json").is_file());
}

#[test]
fn the_build_is_deterministic() {
    let source = tempfile::tempdir().unwrap();
    archive(source.path());
    let policy = policy_for(source.path(), Model::Emperor, &[], &[]).unwrap();
    let one = tempfile::tempdir().unwrap();
    let two = tempfile::tempdir().unwrap();
    build(source.path(), one.path(), Model::Emperor, &policy).unwrap();
    build(source.path(), two.path(), Model::Emperor, &policy).unwrap();
    for file in ["sft.jsonl", "needs-regeneration.jsonl", "stats.json"] {
        let a = fs::read(one.path().join("emperor").join(file)).unwrap();
        let b = fs::read(two.path().join("emperor").join(file)).unwrap();
        assert_eq!(a, b, "{file}");
    }
}

#[test]
fn a_denied_generator_is_excluded() {
    let source = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    archive(source.path());
    let policy = policy_for(source.path(), Model::Emperor, &[], &["QwQ-32B".to_owned()]).unwrap();
    let stats = build(source.path(), out.path(), Model::Emperor, &policy).unwrap();
    assert!(!stats.accepted_by_generator.contains_key("QwQ-32B"));
    assert_eq!(
        stats.files["emperor/data/mix-emperor-v2/final.jsonl"].excluded["other_generator:QwQ-32B"],
        1
    );
}
