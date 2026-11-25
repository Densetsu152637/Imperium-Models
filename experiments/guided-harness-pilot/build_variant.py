"""Write a harness-variant TOML from a prompt-body text file.

The variant rewrites Imperium's base system prompt (identity sentence changed, the remaining base sentences
blanked, the other paragraphs removed), appends the body (the evidence policy), drops the tools the task does not
need and sets loop flags.  Usage: python build_variant.py <body.txt> <out.toml> [name] [max_steps]
"""
import json
import re
import subprocess
import sys

EXE = r"C:\Users\Nicholas\AppData\Local\Imperium\imperium.exe"
KEEP = {"read_file", "grep", "list_dir", "run_command"}

body, out = open(sys.argv[1], encoding="utf-8").read().strip(), sys.argv[2]
name = sys.argv[3] if len(sys.argv) > 3 else "refined"
steps = sys.argv[4] if len(sys.argv) > 4 else "20"
j = json.loads(subprocess.run([EXE, "run", "--headless", "--json", "--print-prompt", "--local", "--thinking", "off"],
                              capture_output=True, text=True, encoding="utf-8").stdout)
paras = j["system_prompt"].split("\n\n")
base = paras[0]
drop = [t["function"]["name"] for t in j["tools"] if t["function"]["name"] not in KEEP]
sents = re.split(r"(?<=[.])\s+(?=[A-Z<])", base)
assert "'''" not in base and "'''" not in body


def lit(t):
    return "'''" + t + "'''"


reps = [("You are Imperium, a coding agent", "You are a careful question-answering assistant")]
reps += [(x, "") for x in sents[1:]]
rep_toml = ",\n  ".join("{ find = " + lit(f) + ", with = " + lit(w) + " }" for f, w in reps)
toml = ('name = "' + name + '"\n'
        'description = "evidence policy, lean tool set, arithmetic through the shell"\n\n'
        "[prompt]\nreplace = [\n  " + rep_toml + "\n]\n"
        'remove = ["Replies: lead with the result", "When you use information from web_search"]\n'
        "append = " + lit(body) + "\n\n"
        "[tools]\ndrop = " + json.dumps(drop) + "\n\n"
        "[flags]\nmax_steps = " + steps + "\n")
open(out, "w", encoding="utf-8").write(toml)
print("dropped", len(drop), "tools; base sentences", len(sents))
