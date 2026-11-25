//! The real manifest and the rules the loader enforces.

mod common;

use ncm_data::manifest::{Category, Kind, Mix, Phase};
use ncm_data::{model_vocab_size, workspace_root};

fn real_mix() -> Mix {
    Mix::load(&workspace_root().join("configs").join("data-mix.toml")).unwrap()
}

#[test]
fn every_source_has_a_licence_pinned_revision_and_is_ungated() {
    let mix = real_mix();
    assert!(mix.sources.len() >= 8);
    for s in &mix.sources {
        assert!(!s.license.trim().is_empty(), "{} has no licence", s.name);
        assert_eq!(s.revision.len(), 40, "{} revision is not a commit", s.name);
        assert!(s.revision.bytes().all(|b| b.is_ascii_hexdigit()));
        assert!(!s.gated, "{} is gated", s.name);
        assert!(!s.claude_check.trim().is_empty(), "{} has no Claude check", s.name);
    }
}

#[test]
fn shares_cover_every_category_and_the_planned_proportions() {
    let mix = real_mix();
    let share = |c: Category| -> f64 {
        mix.sources.iter().filter(|s| s.category == c).map(|s| s.share).sum()
    };
    assert!((share(Category::Web) - 0.45).abs() < 1e-9);
    assert!((share(Category::Code) - 0.25).abs() < 1e-9);
    assert!((share(Category::Research) - 0.10).abs() < 1e-9);
    assert!((share(Category::Math) - 0.10).abs() < 1e-9);
    assert!((share(Category::Reasoning) + share(Category::Instruct) - 0.10).abs() < 1e-9);
    assert!(mix.sources.iter().any(|s| s.kind == Kind::Trace));
    assert!(mix.sources.iter().any(|s| s.phase == Phase::Anneal));
    assert!(mix.val_fraction > 0.0 && mix.val_fraction <= 0.01);
}

#[test]
fn model_configs_agree_on_the_vocabulary_the_tokenizer_targets() {
    assert_eq!(model_vocab_size(&workspace_root()).unwrap(), 32000);
}

#[test]
fn rejects_a_gated_source() {
    let text = common::mix_text(0.75, 0.25).replacen("gated = false", "gated = true", 1);
    let err = Mix::parse(&text).unwrap_err().to_string();
    assert!(err.contains("gated"), "{err}");
}

#[test]
fn rejects_a_floating_revision() {
    let text = common::mix_text(0.75, 0.25).replacen(
        "0123456789abcdef0123456789abcdef01234567",
        "main",
        1,
    );
    assert!(Mix::parse(&text).unwrap_err().to_string().contains("revision"));
}

#[test]
fn rejects_shares_that_do_not_sum_to_one() {
    let err = Mix::parse(&common::mix_text(0.7, 0.2)).unwrap_err().to_string();
    assert!(err.contains("sum"), "{err}");
}

#[test]
fn rejects_an_anneal_source_larger_than_the_anneal_phase() {
    // b is anneal-only with 60% of all tokens, but the anneal phase is half the run at 50%.
    let err = Mix::parse(&common::mix_text(0.4, 0.6)).unwrap_err().to_string();
    assert!(err.contains("anneal"), "{err}");
}

#[test]
fn accepts_the_fixture_mix() {
    Mix::parse(&common::mix_text(0.75, 0.25)).unwrap();
}

#[test]
fn estimated_available_tokens_cover_every_target() {
    let mix = real_mix();
    for s in &mix.sources {
        let target = s.target_tokens(mix.budget_tokens);
        let est = s.est_tokens.unwrap_or(0);
        assert!(est >= target, "{}: estimated {est} tokens, target {target}", s.name);
    }
}

#[test]
fn reasoning_and_anneal_sources_are_recorded() {
    let mix = real_mix();
    let anneal: Vec<_> =
        mix.sources.iter().filter(|s| s.phase == Phase::Anneal).map(|s| s.name.as_str()).collect();
    assert_eq!(anneal, ["openmathreasoning", "openr1-math"]);
    assert!(mix.anneal.fraction > 0.0 && mix.anneal.shares.len() >= 5);
}
