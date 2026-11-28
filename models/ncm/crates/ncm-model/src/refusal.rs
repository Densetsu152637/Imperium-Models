//! The no-refusals rule for every Princeps dataset (user decision, 2026-11-28): a sample whose
//! assistant refuses, or moralises about, a topic the user asked for is not training data.
//! Princeps learns its limits from Imperium's harness and permission system, not from a chat
//! model's topic refusals.
//!
//! What is dropped: "I cannot assist with that", "I'm not able to help with …", "violates
//! safety and ethical guidelines", "as an AI I can't …", unrequested safer alternatives offered
//! in place of the request, and lectures about why the request is inappropriate.
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

/// Phrases that moralise about the request or cite guidelines as the reason not to do it.
const MORALISING: [&str; 10] = [
    "violates safety",
    "against my guidelines",
    "against my ethical guidelines",
    "ethical guidelines",
    "safety guidelines",
    "content policy",
    "it would be inappropriate",
    "it is not appropriate for me",
    "it's not appropriate for me",
    "i'd encourage you to reconsider",
];

/// Phrases that mark a "can't" as an honest capability limit (the tools offered, access, what
/// the assistant can actually do) rather than a refusal of the topic. A refusal phrase next to
/// one of these is kept; moralising never is.
const CAPABILITY: [&str; 17] = [
    "capabilit",
    "i only have",
    "i can only",
    "only help with",
    "available tool",
    "provided tool",
    "tools available",
    "the tool",
    "the provided \"",
    "function only",
    "don't have access",
    "do not have access",
    "not available",
    "functionality",
    "live updates",
    "isn't covered by the tools",
    "only current",
];

fn fold(text: &str) -> String {
    text.to_lowercase().replace(['\u{2019}', '\u{2018}'], "'")
}

/// The first refusal pattern found in one assistant reply, if any.
pub fn refusal_in(content: &str) -> Option<&'static str> {
    let text = fold(content);
    let hit = |list: &[&'static str]| list.iter().copied().find(|p| text.contains(p));
    // An unrequested "safer alternative" is only dropped with a refusal or lecture beside it;
    // offering alternatives on its own is fine.
    if let Some(p) = hit(&MORALISING) {
        return Some(p);
    }
    hit(&REFUSALS).filter(|_| hit(&CAPABILITY).is_none())
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
            "I cannot assist with finding that content.",
            "I'm sorry, but I can't help with NSFW mods.",
            "This request violates safety and ethical guidelines.",
            "I\u{2019}m not able to help with that request.",
            "As an AI language model, I must decline.",
            "I can't provide that. Instead, I can suggest some safer alternatives.",
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
