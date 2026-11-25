"""Generate the held-out hard suite: suite/hard_kb.jsonl (extra facts) and suite/hard_tasks.jsonl (72 items).

Authored by Claude AFTER the refined harness variant was frozen; nothing is tuned on it. It reuses the dev
world (importing build_suite regenerates the dev files, identically, from the fixed seed) and adds new facts
in hard_kb.jsonl; the dev kb.jsonl is untouched. Hard runs use kb.jsonl + hard_kb.jsonl (hard_merged_kb.jsonl).
Eight categories of nine items. Gold answers are computed from the structured world.
"""
import datetime as dt
import json
import pathlib
import random

import build_suite as W

OUT = pathlib.Path(__file__).parent
R = random.Random(20261125)

hard = []          # (text, topic) in creation order; support placeholders are "@<index>"


def hf(text, topic):
    hard.append((text, topic))
    return f"@{len(hard) - 1}"


tasks = []
counter = {}


def task(cat, q, ans, atype, support=(), tol=0.0, **extra):
    counter[cat] = counter.get(cat, 0) + 1
    sup = [x if isinstance(x, str) and x.startswith("@") else W.FID[x] for x in support]
    t = {"id": f"{cat}-{counter[cat]:02d}", "category": cat, "question": q, "answer": ans,
         "answer_type": atype, "tol": tol, "support": sup}
    t.update(extra)
    tasks.append(t)


people, COMP = W.people, W.COMPANIES
ok_people = [p for p in people if p not in W.w_birth]
ok_cos = [c for c in COMP if c not in W.w_founder]
rev_cos = [c for c in COMP if c not in W.w_rev]
used_people = set()


# ---------------------------------------------------------------- 1 chains (3 and 4 hops over the dev KB)
for p in R.sample(people, 3):
    c = W.hq[W.employer[p]]
    task("hard_chain", f"Who is the mayor of the city where the company that employs {p} has its headquarters?",
         W.mayor[c], "string", [("works", p), ("hq", W.employer[p]), ("mayor", c)])
for co in R.sample(ok_cos, 2):
    f = W.founder[co]
    task("hard_chain", f"In which country does the founder of {co} live?", W.country_of_city[W.lives[f]], "string",
         [("founder", co), ("lives", f), ("ccountry", W.lives[f])])
for co in R.sample(ok_cos, 2):
    f = W.founder[co]
    task("hard_chain", f"Who is the mayor of the city where the founder of {co} lives?", W.mayor[W.lives[f]],
         "string", [("founder", co), ("lives", f), ("mayor", W.lives[f])])
for p in R.sample(people, 2):
    c = W.lives[p]
    m = W.mayor[c]
    task("hard_chain", f"In which city is the headquarters of the company that employs the mayor of the city "
         f"where {p} lives?", W.hq[W.employer[m]], "string",
         [("lives", p), ("mayor", c), ("works", m), ("hq", W.employer[m])])

# ---------------------------------------------------------------- 2 distractor-heavy lookups
firsts = [n.split()[0] for n in people]
lasts = [n.split()[1] for n in people]
FIRST_EXTRA = [x for x in W.FIRST if x not in firsts]
LAST_EXTRA = [x for x in W.LAST if x not in lasts]
taken = set(people)


def decoy_name(p):
    f, l = p.split()
    for _ in range(100):
        a = f"{R.choice(FIRST_EXTRA + firsts)} {l}"
        b = f"{f} {R.choice(LAST_EXTRA + lasts)}"
        if a not in taken and b not in taken and a != b:
            taken.update([a, b])
            return a, b
    raise RuntimeError("no decoy names")


picks = R.sample(ok_people, 6)
for i, p in enumerate(picks):
    a, b = decoy_name(p)
    if i < 3:
        others = R.sample([c for c in COMP if c != W.employer[p]], 2)
        s = [hf(f"{a} works at {others[0]} as a {R.choice(W.ROLES)}.", "extra-people.md"),
             hf(f"{b} works at {others[1]} as a {R.choice(W.ROLES)}.", "extra-people.md")]
        task("hard_distractor", f"Which company employs {p}?", W.employer[p], "string", [("works", p)],
             distractors=others)
    else:
        others = R.sample([c for c in W.CITIES if c != W.lives[p]], 2)
        s = [hf(f"{a} lives in {others[0]}.", "extra-people.md"), hf(f"{b} lives in {others[1]}.", "extra-people.md")]
        task("hard_distractor", f"In which city does {p} live?", W.lives[p], "string", [("lives", p)],
             distractors=others)
for co in R.sample(COMP, 3):
    first = co.split()[0]
    suffix = R.choice([x for x in ["Foundry", "Holdings", "Trading", "Mills", "Partners"]])
    decoy = f"{first} {suffix}"
    city = R.choice([c for c in W.CITIES if c != W.hq[co]])
    hf(f"{decoy} has its headquarters in {city}.", "extra-companies.md")
    hf(f"{decoy} was founded in {R.randrange(1900, 2010)}.", "extra-companies.md")
    task("hard_distractor", f"In which city does {co} have its headquarters?", W.hq[co], "string", [("hq", co)],
         distractors=[city])

# ---------------------------------------------------------------- 3 temporal / versioned facts
cities_pool = R.sample(W.CITIES, 4)
for k, c in enumerate(cities_pool):
    a, b = R.sample([p for p in people if p != W.mayor[c]], 2)
    y0, y1, y2 = 1990 + k, 2002 + k, 2014 + k
    s1 = hf(f"From {y0} until {y1}, {a} served as mayor of {c}.", "history.md")
    s2 = hf(f"From {y1} until {y2}, {b} served as mayor of {c}.", "history.md")
    s3 = hf(f"Since {y2}, {W.mayor[c]} has been the mayor of {c}.", "history.md")
    if k % 2 == 0:
        yr, who, ds = y1 + 3, b, [a, W.mayor[c]]
        sup = [s2]
    else:
        yr, who, ds = y0 + 4, a, [b, W.mayor[c]]
        sup = [s1]
    task("hard_temporal", f"Who was the mayor of {c} in {yr}?", who, "string", sup, distractors=ds)
for c in R.sample([c for c in W.CITIES if c not in W.w_pop], 2):
    v = [R.randrange(300, 9000) * 100 for _ in range(3)]
    ss = [hf(f"The population of {c} in {y} was {W.money(x)}.", "history.md") for y, x in zip((1985, 2000, 2015), v)]
    task("hard_temporal", f"Using the most recent population figure recorded for {c} on or before 2010, what was its "
         f"population?", v[1], "number", [ss[1]], 0.001)
co = R.choice(rev_cos)
rv = [R.randrange(10, 300) for _ in range(3)]
ss = [hf(f"In {y}, {co} reported revenue of {x} million ducats.", "history.md") for y, x in zip((2008, 2014, 2020), rv)]
task("hard_temporal", f"Using the most recent revenue figure recorded for {co} on or before 2018, what was its "
     f"revenue in millions of ducats?", rv[1], "number", [ss[1]], 0.001)
for co in R.sample(COMP, 2):
    old = R.choice([c for c in W.CITIES if c != W.hq[co]])
    yr = R.randrange(2003, 2015)
    s = hf(f"In {yr}, {co} moved its headquarters from {old} to {W.hq[co]}.", "history.md")
    task("hard_temporal", f"In which city was the headquarters of {co} in {yr - 3}?", old, "string", [s],
         distractors=[W.hq[co]])

# ---------------------------------------------------------------- 4 multi-constraint filters over a table
rows = [(c, W.c_year[c], W.c_rev[c], W.country_of_city[W.hq[c]]) for c in rev_cos]
countries = sorted({r[3] for r in rows})
sup_all = []
for c, _, _, _ in rows:
    sup_all += [("hq", c), ("cyear", c), ("rev", c), ("ccountry", W.hq[c])]
sup_all = list(dict.fromkeys(sup_all))
made = 0
tries = 0
while made < 9 and tries < 5000:
    tries += 1
    y, rvt, k = R.choice(range(1900, 2000, 10)), R.choice(range(50, 400, 25)), R.choice(countries)
    kind = R.choice(["after", "before"])
    match = [r for r in rows if (r[1] > y if kind == "after" else r[1] < y) and r[2] > rvt and r[3] == k]
    near = [r[0] for r in rows if r not in match and r[3] == k and ((r[1] > y if kind == "after" else r[1] < y)
                                                                    or r[2] > rvt)][:3]
    txt = (f"{{}} founded {kind} {y} with revenue above {rvt} million ducats and a headquarters in {k}")
    pre = "Considering only companies with a recorded revenue, "
    if made < 5 and len(match) == 1:
        task("hard_filter", pre + "which company " + txt.format("was") + "?", match[0][0], "string", sup_all, distractors=near)
        made += 1
    elif made >= 5 and 2 <= len(match) <= 4:
        task("hard_filter", pre + "how many companies " + txt.format("were") + "?", len(match), "number", sup_all, 0.001)
        made += 1

# ---------------------------------------------------------------- 5 multi-step compute with unit changes
VEH = [("barge Moorhen", 12, 3.5), ("truck Kestrel", 60, 14), ("train Dunlin", 90, 40)]
veh_ids = [hf(f"The {n} travels at {sp} km/h and burns {bu} litres of diesel per hour.", "vehicles-units.md")
           for n, sp, bu in VEH]
fuel = hf("Diesel costs 2.4 mar per litre at the Grain Exchange.", "vehicles-units.md")
stone = hf("One Veloran stone is 6.5 kg.", "vehicles-units.md")
ton = hf("One Marenthian ton is 1,250 kg.", "vehicles-units.md")
conv = {"mar": 1.0, "dun": 0.8, "verin": 1 / 1.5}
conv_support = {"mar": [], "dun": [("rate", "mar")], "verin": [("rate", "verin")]}
pairs = sorted(W.dist.items())
for i, tgt in enumerate(["mar", "dun", "dun", "verin", "mar"]):
    vi = i % 3
    (a, b), km = pairs[i]
    n, sp, bu = VEH[vi]
    cost = km / sp * bu * 2.4 * conv[tgt]
    task("hard_compute", f"What is the diesel cost of one trip of the {n} from {a} to {b}, in {tgt}? Give a decimal "
         f"number.", round(cost, 2), "number", [veh_ids[vi], fuel, ("dist", f"{a}|{b}")] + conv_support[tgt], 0.02)
for pr in R.sample(W.PRODUCTS, 2):
    n = R.randrange(40, 120)
    task("hard_compute", f"How many Veloran stones does a load of {n} of the {pr} weigh? Give a decimal number.",
         round(W.weight[pr] * n / 6.5, 2), "number", [stone, ("weight", pr)], 0.02)
for pr in R.sample(W.PRODUCTS, 2):
    n = R.randrange(500, 1500)
    task("hard_compute", f"How many Marenthian tons does a load of {n} of the {pr} weigh? Give a decimal number.",
         round(W.weight[pr] * n / 1250, 2), "number", [ton, ("weight", pr)], 0.02)

# ---------------------------------------------------------------- 6 partially answerable (stated rule)
RULE = " If any figure needed is not recorded in the knowledge base, answer Unknown."
a_rev = R.sample(rev_cos, 4)
task("hard_partial", f"What is the combined revenue of {a_rev[0]} and {W.w_rev[0]}, in millions of ducats?" + RULE,
     "unknown", "abstain", absent={"entity": W.w_rev[0], "keywords": ["revenue"]})
task("hard_partial", f"What is the combined revenue of {a_rev[1]} and {a_rev[2]}, in millions of ducats?" + RULE,
     W.c_rev[a_rev[1]] + W.c_rev[a_rev[2]], "number", [("rev", a_rev[1]), ("rev", a_rev[2])], 0.001)
pc = R.sample([c for c in W.CITIES if c not in W.w_pop], 3)
task("hard_partial", f"How many more people live in {pc[0]} than in {W.w_pop[1]}?" + RULE, "unknown", "abstain",
     absent={"entity": W.w_pop[1], "keywords": ["population"]})
big, small = sorted(pc[1:], key=lambda c: -W.pop[c])
task("hard_partial", f"How many more people live in {big} than in {small}?" + RULE, W.pop[big] - W.pop[small],
     "number", [("pop", big), ("pop", small)], 0.001)
pp = R.sample(ok_people, 3)
task("hard_partial", f"How many days older is {pp[0]} than {W.w_birth[1]}? If it is negative, answer with its absolute "
     f"value." + RULE, "unknown", "abstain", absent={"entity": W.w_birth[1], "keywords": ["born", "birth"]})
o, y = sorted(pp[1:], key=lambda p: W.birth[p])
task("hard_partial", f"How many days older is {o} than {y}?" + RULE, (W.birth[y] - W.birth[o]).days, "number",
     [("born", o), ("born", y)], 0.001)
for co in W.w_founder:
    task("hard_partial", f"In which city does the founder of {co} live?" + RULE, "unknown", "abstain",
         absent={"entity": co, "keywords": ["founded by", "founder", "set up by"]})
fc = R.choice(ok_cos)
task("hard_partial", f"In which city does the founder of {fc} live?" + RULE, W.lives[W.founder[fc]], "string",
     [("founder", fc), ("lives", W.founder[fc])])

# ---------------------------------------------------------------- 7 counterfactual rules needing two facts
ovr = lambda i: ("override", str(i))
new = {
    "year": hf("A Veloran year has 40 weeks.", "rules.md"),
    "match": hf("A match of Orlund football has 3 periods of 25 minutes each.", "rules.md"),
    "kettle": hf("A kettle on the Kaldris plateau heats water by 3 degrees Celsius per minute, starting from 20 "
                 "degrees Celsius.", "rules.md"),
    "vote": hf("In Dunmoor the voting age is five years above the age of adulthood.", "rules.md"),
    "sack": hf("A Veloran sack holds 12 Veloran pounds.", "rules.md"),
    "chord": hf("A Veloran chord uses 4 of the notes of the octave.", "rules.md"),
    "loaf": hf("A loaf of bread costs 175 pips in Marenthia.", "rules.md"),
    "tri": hf("A Tessarine geometry class draws 6 triangles.", "rules.md"),
    "market": hf("The Kaldrian market opens 3 hours after a new day begins.", "rules.md"),
}
task("hard_counterfactual", "How many days are in 2 Veloran years?", 720, "number", [ovr(0), new["year"]], 0.001)
task("hard_counterfactual", "How many player-minutes does one Orlund football match have if every player of both "
     "teams plays the whole match?", 1350, "number", [ovr(10), new["match"]], 0.001)
task("hard_counterfactual", "How many minutes does the kettle on the Kaldris plateau need to bring water to the boil?",
     20, "number", [ovr(3), new["kettle"]], 0.001)
task("hard_counterfactual", "What is the voting age in Dunmoor?", 21, "number", [ovr(4), new["vote"]], 0.001)
task("hard_counterfactual", "How many grams are in 3 Veloran sacks?", 14400, "number", [ovr(13), new["sack"]], 0.001)
task("hard_counterfactual", "How many notes of the Veloran octave are not part of a Veloran chord?", 6, "number",
     [ovr(12), new["chord"]], 0.001)
task("hard_counterfactual", "How many mar do 4 loaves of bread cost in Marenthia?", 14, "number",
     [ovr(7), new["loaf"]], 0.001)
task("hard_counterfactual", "How many sides do the triangles drawn in a Tessarine geometry class have altogether?", 24,
     "number", [ovr(14), new["tri"]], 0.001)
task("hard_counterfactual", "At what hour of the day, as a number from 0 to 23 on a 24-hour clock, does the Kaldrian "
     "market open?", 15, "number", [ovr(9), new["market"]], 0.001)

# ---------------------------------------------------------------- 8 mixed real-world knowledge and the knowledge base
co = R.choice(rev_cos)
task("hard_mixed", f"If the revenue of {co} last year is spread evenly over the months of a calendar year, how many "
     f"million ducats is that per month? Give a decimal number.", round(W.c_rev[co] / 12, 2), "number",
     [("rev", co)], 0.02)
task("hard_mixed", "How many more days are in 4 Veloran weeks than in 4 weeks of the real-world calendar?", 8,
     "number", [ovr(0)], 0.001)
task("hard_mixed", "How many degrees Celsius lower is the boiling point of water on the Kaldris plateau than at sea "
     "level in the real world?", 20, "number", [ovr(3)], 0.001)
task("hard_mixed", "How many months fewer are there in 5 years of the Orlund Republic than in 5 years of the "
     "real-world calendar?", 10, "number", [ovr(5)], 0.001)
task("hard_mixed", "A real-world pound weighs about 453.6 grams. How many grams heavier are 5 real-world pounds than "
     "5 Veloran pounds?", 268.0, "number", [ovr(13)], 0.1)
late = [c for c in COMP if W.c_year[c] > 1945]
c1 = R.choice(late)
task("hard_mixed", f"How many years after the end of the Second World War was {c1} founded?", W.c_year[c1] - 1945,
     "number", [("cyear", c1)], 0.001)
cy = R.choice(W.CITIES)
task("hard_mixed", f"How many years before the first Moon landing was {cy} established?", 1969 - W.city_year[cy],
     "number", [("cyear", cy)], 0.001)
early = [c for c in COMP if W.c_year[c] < 1989]
c2 = R.choice(early)
task("hard_mixed", f"How many years before the fall of the Berlin Wall was {c2} founded?", 1989 - W.c_year[c2],
     "number", [("cyear", c2)], 0.001)
task("hard_mixed", "How many more sides do 3 triangles of a Tessarine geometry class have than 3 real-world "
     "triangles?", 3, "number", [ovr(14)], 0.001)

# ---------------------------------------------------------------- finalise: ids, aliases, output
order = list(range(len(hard)))
R.shuffle(order)
hid = {i: f"H{n + 1:03d}" for n, i in enumerate(order)}
hard_rows = [{"id": hid[i], "text": hard[i][0], "topic": hard[i][1]} for i in sorted(range(len(hard)),
                                                                                       key=lambda i: hid[i])]
for t in tasks:
    t["support"] = [hid[int(x[1:])] if x.startswith("@") else x for x in t["support"]]
    a = t["answer"]
    if t["answer_type"] == "string":
        al = []
        if a in people:
            al.append(a.split()[-1])
        if a in COMP:
            al.append(a.split()[0])
        if al:
            t["aliases"] = al

kb = [json.loads(line) for line in open(OUT / "kb.jsonl", encoding="utf-8")]
with open(OUT / "hard_kb.jsonl", "w", encoding="utf-8") as f:
    for r in hard_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT / "hard_merged_kb.jsonl", "w", encoding="utf-8") as f:
    for r in kb + hard_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT / "hard_tasks.jsonl", "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(len(hard_rows), "hard passages;", len(tasks), "hard tasks;", counter)
