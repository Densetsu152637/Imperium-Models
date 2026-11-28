//! The no-refusals rule for every Princeps dataset (user decision, 2026-11-28): a sample whose
//! assistant refuses, or moralises about, a topic the user asked for is not training data.
//! Princeps learns its limits from Imperium's harness and permission system, not from a chat
//! model's topic refusals.
//!
//! What is dropped: a refusal phrase ("I cannot assist", "I'm not able to help with", "as an
//! AI I can't", ...) given for a harm, legality, ethics, explicit-content or guidelines reason,
//! and outright lectures ("violates safety", "against my guidelines") wherever they appear,
//! together with any unrequested "safer alternatives" offered in their place.
//!
//! What is kept, because it is honest and not a topic refusal: "I don't know", "I couldn't
//! verify that", reports of tool, permission or sandbox failures, capability limits ("I can't
//! order a pizza: I only have the BMI tool"), and questions asking the user for missing
//! information.
//!
//! Only the assistant's final content is checked, not its `<think>` reasoning: reasoning that
//! weighs a request and then fulfils it is fine. Matching is on lowercased text with curly
//! apostrophes folded, so the patterns below are written in that form.

use crate::chat::{Message, Role};

/// Phrases that refuse a requested topic outright.
const REFUSALS: [&str; 22] = [
    "i cannot assist",
    "i can't assist",
    "i cannot help with",
    "i can't help with",
    "i am unable to help with",
    "i'm unable to help with",
    "i am not able to help with",
    "i'm not able to help with",
    "i cannot provide",
    "i can't provide",
    "i cannot fulfill",
    "i can't fulfill",
    "i cannot comply",
    "i can't comply",
    "i won't be able to help with",
    "i must decline",
    "i have to decline",
    "i'm sorry, but i can't",
    "i'm sorry, but i cannot",
    "i apologize, but i cannot",
    "as an ai language model, i",
    "as an ai, i cannot",
];

/// Phrases that are a topic refusal wherever they appear.
const ALWAYS: [&str; 4] = [
    "violates safety",
    "against my guidelines",
    "against my ethical guidelines",
    "i'd encourage you to reconsider",
];

/// Reasons a chat-tuned model gives for refusing a topic: harm, legality, ethics, explicit
/// content, guidelines. A refusal phrase only counts beside one of these; a bare "I can't" is
/// almost always an honest scope, tool, capability or operator-policy limit, which stays.
const MORALISING: [&str; 21] = [
    "harmful",
    "unethical",
    "illegal",
    "inappropriate",
    "not appropriate",
    "explicit content",
    "adult content",
    "sexual",
    "nsfw",
    "offensive",
    "ethical, legal",
    "ethical restrictions",
    "ethical concerns",
    "ethical guidelines",
    "safety guidelines",
    "my guidelines",
    "usage guidelines",
    "content policy",
    "promote or",
    "condone",
    "morally",
];

fn fold(text: &str) -> String {
    text.to_lowercase().replace(['\u{2019}', '\u{2018}'], "'")
}

/// The first refusal pattern found in one assistant reply, if any.
pub fn refusal_in(content: &str) -> Option<&'static str> {
    let text = fold(content);
    let hit = |list: &[&'static str]| list.iter().copied().find(|p| text.contains(p));
    if let Some(p) = hit(&ALWAYS) {
        return Some(p);
    }
    // An unrequested "safer alternative" is only dropped as part of such a refusal.
    hit(&REFUSALS).filter(|_| hit(&MORALISING).is_some())
}

/// The first refusal pattern in any assistant turn of a conversation, if any.
pub fn refusal(messages: &[Message]) -> Option<&'static str> {
    messages.iter().filter(|m| m.role == Role::Assistant).find_map(|m| refusal_in(&m.content))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_refusals_are_found() {
        for reply in [
            "I cannot assist with finding explicit content.",
            "I'm sorry, but I can't help with NSFW mods.",
            "This request violates safety and ethical guidelines.",
            "I\u{2019}m not able to help with that request; it would be inappropriate.",
            "As an AI language model, I must decline: it is unethical.",
            "I can't provide that because it could be harmful. Instead, I can suggest safer alternatives.",
            "I can't help with that; it goes against my ethical guidelines and our content policy.",
        ] {
            assert!(refusal_in(reply).is_some(), "{reply}");
        }
    }

    #[test]
    fn honest_limits_are_kept() {
        for reply in [
            "I don't know which release added that flag.",
            "I couldn't verify the claim: the page returned 404.",
            "The web_search tool failed with a timeout, so I have no results yet.",
            "Permission to write outside the workspace was denied.",
            "Which branch should I base the fix on?",
            "Instead, I can run the narrower test first if you'd like.",
            "Here are three Cyberpunk 2077 mod collections on Nexus Mods.",
            "I'm sorry, but I can't assist with ordering a pizza. I only have the capability to              help you create todo items using the `create_todo` tool.",
            "I cannot fulfill this request with the available tools.",
            "I cannot assist with this request as I do not have access to the required tools.",
            "I'm sorry, but I can't approve a refund: our policy only allows refunds within 30 days.",
            "I'm sorry, but I can't access your booking without first verifying your identity.",
            "Health advisory: follow local safety guidelines when the AQI is above 100.",
            "I'm a food delivery agent, so I can't assist with job applications.",
            "There's no tool for ice hockey schedules, so I cannot provide them.",
        ] {
            assert_eq!(refusal_in(reply), None, "{reply}");
        }
    }

    #[test]
    fn only_assistant_content_is_checked() {
        let mut thinking = Message::new(Role::Assistant, "Here is the list you asked for.");
        thinking.reasoning = Some("Should I decline? No: it violates no rule.".into());
        let user = Message::new(Role::User, "I cannot assist myself, please help.");
        assert_eq!(refusal(&[user, thinking]), None);
    }
}
